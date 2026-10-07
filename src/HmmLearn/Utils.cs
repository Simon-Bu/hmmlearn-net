namespace HmmLearn;

/// <summary>Port of hmmlearn/utils.py: in-place normalization helpers and the expansion of compact covariances.</summary>
public static class Utils
{
    /// <summary>
    /// Divides every element by the sum of all elements, in place. A sum of 0 is treated as 1 (the zeros stay zeros).
    /// </summary>
    public static void Normalize(double[] a) => throw new NotImplementedException();

    /// <summary>
    /// Normalizes in place. <paramref name="axis"/> null: the whole matrix is divided by the sum of all its elements;
    /// 0: each column is divided by its own sum; 1: each row is divided by its own sum. A sum of 0 is treated as 1.
    /// Any other axis throws <see cref="ArgumentOutOfRangeException"/>.
    /// </summary>
    public static void Normalize(double[,] a, int? axis = null) => throw new NotImplementedException();

    /// <summary>
    /// Normalizes in place so that sum(exp(a)) is 1: subtracts the log-sum-exp of all elements (computed stably: the maximum
    /// is factored out, and a maximum that is infinite is the log-sum-exp itself).
    /// </summary>
    public static void LogNormalize(double[] a) => throw new NotImplementedException();

    /// <summary>
    /// Same as <see cref="LogNormalize(double[])"/> for a matrix. <paramref name="axis"/> null: one log-sum-exp for all elements;
    /// 0: one per column; 1: one per row. If the chosen axis has length 1, every element is set to 0 (even -infinity).
    /// A row of -infinity gives NaN (-infinity minus -infinity). Any other axis throws <see cref="ArgumentOutOfRangeException"/>.
    /// </summary>
    public static void LogNormalize(double[,] a, int? axis = null) => throw new NotImplementedException();

    /// <summary>
    /// Expands compact covariance parameters (uniform shapes, see oracle/README.md) to full matrices of shape
    /// (<paramref name="nComponents"/>, <paramref name="nFeatures"/>, <paramref name="nFeatures"/>).
    /// Compact shapes: spherical (nc, 1, 1), diag (nc, 1, nf), full (nc, nf, nf), tied (1, nf, nf).
    /// Spherical gives variance times the identity, diag puts the variances on the diagonal, full keeps the matrices,
    /// tied repeats the single matrix nc times. A shape that does not match the covariance type throws <see cref="ArgumentException"/>.
    /// </summary>
    public static double[,,] FillCovars(double[,,] covars, CovarianceType covarianceType, int nComponents, int nFeatures)
        => throw new NotImplementedException();
}
