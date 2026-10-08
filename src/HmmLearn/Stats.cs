namespace HmmLearn;

/// <summary>Port of hmmlearn/stats.py.</summary>
public static class Stats
{
    /// <summary>numpy's finfo(float).tiny, the smallest positive normal double: spherical and diag variances are raised to at least this.</summary>
    private const double Tiny = 2.2250738585072014E-308;

    /// <summary>hmmlearn's min_covar: added to the diagonal when the first Cholesky of a full or tied covariance fails.</summary>
    private const double MinCovar = 1.0E-7;

    /// <summary>
    /// Log probability density of each row of <paramref name="x"/> (n_samples, n_features) under each of the n_components
    /// Gaussians whose means are the rows of <paramref name="means"/> (n_components, n_features). Result shape (n_samples, n_components).
    /// Compact covariances use the uniform shapes of oracle/README.md: spherical (nc, 1, 1), diag (nc, 1, nf), full (nc, nf, nf),
    /// tied (1, nf, nf) (the one matrix serves every component).
    /// <list type="bullet">
    /// <item>Spherical and diag: variances are first raised to at least 2.2250738585072014E-308 (the smallest normal double, numpy's
    /// finfo(float).tiny); the density is -0.5 * (nf * log(2 pi) + sum log(variance) + sum (x - mean)^2 / variance);
    /// a spherical variance applies to every feature.</item>
    /// <item>Full and tied: Cholesky factor L of the covariance (lower triangular); if it is not positive definite, retry once with
    /// 1e-7 added to the diagonal; if that fails too, throw <see cref="ArgumentException"/>("'covars' must be symmetric, positive-definite");
    /// the density is -0.5 * (nf * log(2 pi) + sum y^2 + logdet) with L y = x - mean and logdet = 2 * sum log(L[i,i]).</item>
    /// </list>
    /// A covariance array whose shape does not match the covariance type, or x with a different number of features than the means,
    /// throws <see cref="ArgumentException"/>. A null <paramref name="x"/>, <paramref name="means"/> or <paramref name="covars"/>
    /// throws <see cref="ArgumentNullException"/>. Zero-size inputs are valid and never throw: x with 0 rows gives an empty
    /// (0, n_components) matrix; means with 0 rows (no component; covars shaped for nc = 0, tied (1, nf, nf)) give an (n_samples, 0)
    /// matrix; x and means with 0 features (covars shaped for nf = 0) give an (n_samples, n_components) matrix of zeros
    /// (the log density in dimension 0 is log 1).
    /// </summary>
    public static double[,] LogMultivariateNormalDensity(double[,] x, double[,] means, double[,,] covars,
        CovarianceType covarianceType = CovarianceType.Diag)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (means is null) throw new ArgumentNullException(nameof(means));
        if (covars is null) throw new ArgumentNullException(nameof(covars));

        var n = x.GetLength(0);
        var nc = means.GetLength(0);
        var nf = means.GetLength(1);
        if (x.GetLength(1) != nf)
            throw new ArgumentException($"x must have the same number of features as the means: {x.GetLength(1)} vs {nf}", nameof(x));

        var expected = covarianceType switch
        {
            CovarianceType.Spherical => (nc, 1, 1),
            CovarianceType.Diag => (nc, 1, nf),
            CovarianceType.Full => (nc, nf, nf),
            CovarianceType.Tied => (1, nf, nf),
            _ => throw new ArgumentOutOfRangeException(nameof(covarianceType)),
        };
        if ((covars.GetLength(0), covars.GetLength(1), covars.GetLength(2)) != expected)
            throw new ArgumentException(
                $"covars must have shape ({expected.Item1}, {expected.Item2}, {expected.Item3}) for {covarianceType}, " +
                $"got ({covars.GetLength(0)}, {covars.GetLength(1)}, {covars.GetLength(2)})", nameof(covars));

        return covarianceType switch
        {
            CovarianceType.Spherical => DiagLike(x, means, (c, _) => covars[c, 0, 0], n, nc, nf),
            CovarianceType.Diag => DiagLike(x, means, (c, f) => covars[c, 0, f], n, nc, nf),
            CovarianceType.Full => FullTied(x, means, covars, n, nc, nf, tied: false),
            CovarianceType.Tied => FullTied(x, means, covars, n, nc, nf, tied: true),
            _ => throw new ArgumentOutOfRangeException(nameof(covarianceType)),
        };
    }

    /// <summary>
    /// The spherical and diag densities: -0.5 * (nf * log(2 pi) + sum of log(variance) over the features + sum of (x - mean)^2 / variance),
    /// every variance raised to at least <see cref="Tiny"/> first; an overflow of (x - mean)^2 gives -infinity, as in numpy.
    /// </summary>
    private static double[,] DiagLike(double[,] x, double[,] means, Func<int, int, double> variance, int n, int nc, int nf)
    {
        var result = new double[n, nc];
        var constant = nf * Math.Log(2.0 * Math.PI);
        for (var c = 0; c < nc; c++)
        {
            for (var i = 0; i < n; i++)
            {
                var logSum = 0.0;
                var squaredSum = 0.0;
                for (var f = 0; f < nf; f++)
                {
                    var v = Math.Max(variance(c, f), Tiny);
                    var d = x[i, f] - means[c, f];
                    logSum += Math.Log(v);
                    squaredSum += d * d / v;
                }
                result[i, c] = -0.5 * (constant + logSum + squaredSum);
            }
        }
        return result;
    }

    /// <summary>
    /// The full and tied densities: -0.5 * (nf * log(2 pi) + sum of y^2 + logdet) with L y = x - mean, L the lower-triangular Cholesky
    /// factor of the component's covariance and logdet = 2 * sum log(L[i,i]). Tied uses its one matrix (component 0) for every component.
    /// </summary>
    private static double[,] FullTied(double[,] x, double[,] means, double[,,] covars, int n, int nc, int nf, bool tied)
    {
        var result = new double[n, nc];
        var constant = nf * Math.Log(2.0 * Math.PI);
        for (var c = 0; c < nc; c++)
        {
            var (factor, logDet) = Cholesky(ComponentMatrix(covars, tied ? 0 : c, nf));
            var y = new double[nf];
            for (var i = 0; i < n; i++)
            {
                var squaredSum = 0.0;
                for (var f = 0; f < nf; f++)
                {
                    var s = x[i, f] - means[c, f];
                    for (var k = 0; k < f; k++)
                        s -= factor[f, k] * y[k];
                    y[f] = s / factor[f, f];
                    squaredSum += y[f] * y[f];
                }
                result[i, c] = -0.5 * (constant + squaredSum + logDet);
            }
        }
        return result;
    }

    /// <summary>A copy of component <paramref name="k"/> of the compact tensor, shape (nf, nf); the caller's tensor is never modified.</summary>
    private static double[,] ComponentMatrix(double[,,] covars, int k, int nf)
    {
        var matrix = new double[nf, nf];
        for (var i = 0; i < nf; i++)
            for (var j = 0; j < nf; j++)
                matrix[i, j] = covars[k, i, j];
        return matrix;
    }

    /// <summary>
    /// The lower-triangular Cholesky factor of a covariance matrix (L L^T = covar) and logdet = 2 * sum log(L[i,i]);
    /// retries once with <see cref="MinCovar"/> added to the diagonal, and throws <see cref="ArgumentException"/> if it is still
    /// not positive definite.
    /// </summary>
    private static (double[,] Factor, double LogDet) Cholesky(double[,] covar)
    {
        var n = covar.GetLength(0);
        var factor = new double[n, n];
        if (TryFactor(covar, factor, out var logDet))
            return (factor, logDet);
        var withMinCovar = (double[,])covar.Clone();
        for (var i = 0; i < n; i++)
            withMinCovar[i, i] += MinCovar;
        if (TryFactor(withMinCovar, factor, out logDet))
            return (factor, logDet);
        throw new ArgumentException("'covars' must be symmetric, positive-definite");
    }

    /// <summary>
    /// The lower-triangular Cholesky factorization of <paramref name="a"/> in <paramref name="factor"/> (L L^T = a, only the lower
    /// triangle is read); false when a pivot is not strictly positive (the matrix is not positive definite).
    /// </summary>
    private static bool TryFactor(double[,] a, double[,] factor, out double logDet)
    {
        logDet = 0.0;
        var n = a.GetLength(0);
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var s = a[i, j];
                for (var k = 0; k < j; k++)
                    s -= factor[i, k] * factor[j, k];
                if (j == i)
                {
                    if (s <= 0.0)
                        return false;
                    factor[i, i] = Math.Sqrt(s);
                    logDet += 2.0 * Math.Log(factor[i, i]);
                }
                else
                {
                    factor[i, j] = s / factor[j, j];
                }
            }
        }
        return true;
    }
}
