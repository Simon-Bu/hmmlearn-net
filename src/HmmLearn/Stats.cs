namespace HmmLearn;

/// <summary>Port of hmmlearn/stats.py.</summary>
public static class Stats
{
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
    /// throws <see cref="ArgumentException"/>.
    /// </summary>
    public static double[,] LogMultivariateNormalDensity(double[,] x, double[,] means, double[,,] covars,
        CovarianceType covarianceType = CovarianceType.Diag) => throw new NotImplementedException();
}
