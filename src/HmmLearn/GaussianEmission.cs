using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;

namespace HmmLearn;

/// <summary>
/// The sufficient statistics of a Gaussian model: the base ones plus the weighted sums of the observations that the M-step needs
/// (hmmlearn's 'post', 'obs', 'obs**2' and 'obs*obs.T').
/// </summary>
public class GaussianStatistics : SufficientStatistics
{
    /// <summary>
    /// Allocates the base statistics, <see cref="Post"/> (n_components), <see cref="Obs"/> and <see cref="Obs2"/> (n_components x
    /// n_features) and, only for the Full and Tied covariance types, <see cref="ObsObsT"/> (n_components x n_features x n_features),
    /// all zeros. For Spherical and Diag, ObsObsT stays null.
    /// </summary>
    public GaussianStatistics(int nComponents, int nFeatures, CovarianceType covarianceType) : base(nComponents)
    {
        Post = new double[nComponents];
        Obs = new double[nComponents, nFeatures];
        Obs2 = new double[nComponents, nFeatures];
        if (covarianceType is CovarianceType.Full or CovarianceType.Tied)
            ObsObsT = new double[nComponents, nFeatures, nFeatures];
    }

    /// <summary>Post[c]: sum over the samples of the posterior probability of state c.</summary>
    public double[] Post { get; set; } = null!;

    /// <summary>Obs[c, f]: sum over the samples of posterior(c) * x[f].</summary>
    public double[,] Obs { get; set; } = null!;

    /// <summary>Obs2[c, f]: sum over the samples of posterior(c) * x[f]^2.</summary>
    public double[,] Obs2 { get; set; } = null!;

    /// <summary>ObsObsT[c, j, k]: sum over the samples of posterior(c) * x[j] * x[k]; null unless Full or Tied.</summary>
    public double[,,]? ObsObsT { get; set; }
}

/// <summary>
/// The Gaussian emission, shared by <see cref="GaussianHmm"/> and, later, the variational Gaussian model (port of BaseGaussianHMM
/// of hmmlearn/_emissions.py). The log likelihood itself is <see cref="Stats.LogMultivariateNormalDensity"/>.
/// </summary>
internal static class GaussianEmission
{
    /// <summary>
    /// Free scalar parameters per letter: 's' = nc - 1, 't' = nc * (nc - 1), 'm' = nc * nf, and 'c' = nc (Spherical), nc * nf (Diag),
    /// nc * nf * (nf + 1) / 2 (Full), nf * (nf + 1) / 2 (Tied).
    /// </summary>
    internal static IReadOnlyDictionary<char, int> NFitScalars(int nComponents, int nFeatures, CovarianceType covarianceType)
    {
        var c = covarianceType switch
        {
            CovarianceType.Spherical => nComponents,
            CovarianceType.Diag => nComponents * nFeatures,
            CovarianceType.Full => nComponents * nFeatures * (nFeatures + 1) / 2,
            _ => nFeatures * (nFeatures + 1) / 2,
        };
        return new Dictionary<char, int>
        {
            ['s'] = nComponents - 1,
            ['t'] = nComponents * (nComponents - 1),
            ['m'] = nComponents * nFeatures,
            ['c'] = c,
        };
    }

    /// <summary>
    /// Adds one sequence x (n_samples, n_features) with its state posteriors (n_samples, n_components) to the statistics.
    /// If <paramref name="needsMean"/>: Post += column sums of the posteriors, Obs += transpose(posteriors) * x.
    /// If <paramref name="needsCovars"/>: for Spherical and Diag, Obs2 += transpose(posteriors) * x^2 (element-wise square); for Full and
    /// Tied, ObsObsT[c] += sum over the samples of posteriors[i, c] * outer(x[i], x[i]).
    /// </summary>
    internal static void Accumulate(GaussianStatistics stats, double[,] x, double[,] posteriors, CovarianceType covarianceType,
        bool needsMean, bool needsCovars)
    {
        var n = x.GetLength(0);
        var nc = posteriors.GetLength(1);
        var nf = x.GetLength(1);
        if (needsMean)
        {
            for (var i = 0; i < n; i++)
                for (var c = 0; c < nc; c++)
                {
                    var w = posteriors[i, c];
                    stats.Post[c] += w;
                    for (var f = 0; f < nf; f++)
                        stats.Obs[c, f] += w * x[i, f];
                }
        }
        if (!needsCovars)
            return;
        if (covarianceType is CovarianceType.Spherical or CovarianceType.Diag)
        {
            for (var i = 0; i < n; i++)
                for (var c = 0; c < nc; c++)
                {
                    var w = posteriors[i, c];
                    for (var f = 0; f < nf; f++)
                        stats.Obs2[c, f] += w * x[i, f] * x[i, f];
                }
        }
        else
        {
            var obsObsT = stats.ObsObsT!;
            for (var i = 0; i < n; i++)
                for (var c = 0; c < nc; c++)
                {
                    var w = posteriors[i, c];
                    for (var j = 0; j < nf; j++)
                        for (var k = 0; k < nf; k++)
                            obsObsT[c, j, k] += w * x[i, j] * x[i, k];
                }
        }
    }

    /// <summary>
    /// One observation (n_features numbers) drawn from the Gaussian of <paramref name="state"/>: its mean plus L z, with z standard
    /// normal numbers from <paramref name="random"/> (Box-Muller) and L the lower Cholesky factor of fullCovars[state]. Throws
    /// <see cref="InvalidOperationException"/> if the matrix is not positive definite.
    /// </summary>
    internal static double[] GenerateSample(double[,] means, double[,,] fullCovars, int state, Random random)
    {
        var nf = means.GetLength(1);
        var covar = new double[nf, nf];
        for (var i = 0; i < nf; i++)
            for (var j = 0; j < nf; j++)
                covar[i, j] = fullCovars[state, i, j];

        var factor = Cholesky(covar) ?? throw new InvalidOperationException($"the covariance of state {state} is not positive definite");

        var z = new double[nf];
        var hasSpare = false;
        var spare = 0.0;
        for (var i = 0; i < nf; i++)
        {
            if (hasSpare)
            {
                z[i] = spare;
                hasSpare = false;
                continue;
            }
            var u1 = 1.0 - random.NextDouble(); // (0, 1], so log(u1) is finite
            var u2 = random.NextDouble();
            var radius = Math.Sqrt(-2.0 * Math.Log(u1));
            z[i] = radius * Math.Cos(2.0 * Math.PI * u2);
            spare = radius * Math.Sin(2.0 * Math.PI * u2);
            hasSpare = true;
        }

        var sample = new double[nf];
        for (var i = 0; i < nf; i++)
        {
            var value = means[state, i];
            for (var k = 0; k <= i; k++)
                value += factor[i, k] * z[k];
            sample[i] = value;
        }
        return sample;
    }

    /// <summary>
    /// Whether the symmetric matrix is positive definite: every eigenvalue strictly positive beyond a relative tolerance of 1e-12.
    /// Unlike a plain 'eigenvalue > 0' or Cholesky test, this rejects positive semi-definite matrices whose zero eigenvalue is computed
    /// as a tiny positive number (hmmlearn's _validate_covars rejects them with its Cholesky check).
    /// </summary>
    internal static bool IsPositiveDefinite(double[,] a)
    {
        var eigenvalues = Matrix<double>.Build.DenseOfArray(a).Evd(Symmetricity.Symmetric).EigenValues;
        var max = double.NegativeInfinity;
        for (var i = 0; i < eigenvalues.Count; i++)
            max = Math.Max(max, eigenvalues[i].Real);
        for (var i = 0; i < eigenvalues.Count; i++)
            if (eigenvalues[i].Real <= 1e-12 * max)
                return false;
        return true;
    }

    /// <summary>The lower Cholesky factor L (L L^T = a) of a symmetric positive definite matrix, or null if a is not positive definite.</summary>
    private static double[,]? Cholesky(double[,] a)
    {
        var n = a.GetLength(0);
        var factor = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var sum = a[i, j];
                for (var k = 0; k < j; k++)
                    sum -= factor[i, k] * factor[j, k];
                if (j == i)
                {
                    if (sum <= 0.0)
                        return null;
                    factor[i, i] = Math.Sqrt(sum);
                }
                else
                {
                    factor[i, j] = sum / factor[j, j];
                }
            }
        }
        return factor;
    }
}
