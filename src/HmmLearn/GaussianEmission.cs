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
        => throw new NotImplementedException();

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
        => throw new NotImplementedException();

    /// <summary>
    /// Adds one sequence x (n_samples, n_features) with its state posteriors (n_samples, n_components) to the statistics.
    /// If <paramref name="needsMean"/>: Post += column sums of the posteriors, Obs += transpose(posteriors) * x.
    /// If <paramref name="needsCovars"/>: for Spherical and Diag, Obs2 += transpose(posteriors) * x^2 (element-wise square); for Full and
    /// Tied, ObsObsT[c] += sum over the samples of posteriors[i, c] * outer(x[i], x[i]).
    /// </summary>
    internal static void Accumulate(GaussianStatistics stats, double[,] x, double[,] posteriors, CovarianceType covarianceType,
        bool needsMean, bool needsCovars) => throw new NotImplementedException();

    /// <summary>
    /// One observation (n_features numbers) drawn from the Gaussian of <paramref name="state"/>: its mean plus L z, with z standard
    /// normal numbers from <paramref name="random"/> (Box-Muller) and L a Cholesky factor of fullCovars[state]; if the matrix is not
    /// positive definite (but is positive semi-definite), a factor from its symmetric eigen decomposition, negative eigenvalues taken as 0.
    /// </summary>
    internal static double[] GenerateSample(double[,] means, double[,,] fullCovars, int state, Random random)
        => throw new NotImplementedException();
}
