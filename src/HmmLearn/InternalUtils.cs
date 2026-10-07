namespace HmmLearn;

/// <summary>
/// Port of hmmlearn/_utils.py: the private helpers of the original package (public here so that the tests reach them).
/// Compact covariance parameters use the uniform shapes of oracle/README.md: spherical (nc, 1, 1), diag (nc, 1, nf),
/// full (nc, nf, nf), tied (1, nf, nf).
/// </summary>
public static class InternalUtils
{
    /// <summary>
    /// Natural logarithm of the absolute value of the determinant of a square matrix, or NaN if the determinant is negative,
    /// or -infinity if it is 0 (numpy.linalg.slogdet, then NaN for a negative sign). A matrix that is not square throws
    /// <see cref="ArgumentException"/>.
    /// </summary>
    public static double LogDet(double[,] a) => throw new NotImplementedException();

    /// <summary>
    /// Splits the rows of <paramref name="x"/> (n_samples, n_features) into consecutive sequences of the given lengths; each
    /// element of the result is a new matrix with the rows of one sequence. Null lengths: a list holding <paramref name="x"/> itself.
    /// Lengths that are negative or do not sum to the number of rows throw <see cref="ArgumentException"/>.
    /// </summary>
    public static List<double[,]> SplitXLengths(double[,] x, int[]? lengths) => throw new NotImplementedException();

    /// <summary>
    /// Basic checks of compact covariances; throws <see cref="ArgumentException"/> when hmmlearn raises ValueError.
    /// Shapes: spherical (nComponents, 1, 1), diag (nComponents, 1, nf), full (nComponents, nf, nf), tied (1, nf, nf), otherwise
    /// ArgumentException. Spherical and diag: every variance must be positive. Tied and full: every matrix must be symmetric
    /// (|a - b| &lt;= 1e-8 + 1e-5 * |b| between a[i,j] and its transpose b = a[j,i]) and positive definite (every eigenvalue &gt; 0).
    /// </summary>
    public static void ValidateCovars(double[,,] covars, CovarianceType covarianceType, int nComponents)
        => throw new NotImplementedException();

    /// <summary>
    /// Creates compact covariances for all states from one (nf, nf) template, in the uniform shapes above: spherical gives
    /// the mean of ALL entries of the matrix (off-diagonal ones included) for every state, diag the diagonal of the matrix,
    /// full the matrix itself, tied the matrix once. Result shapes: (nc, 1, 1), (nc, 1, nf), (nc, nf, nf) and (1, nf, nf).
    /// </summary>
    public static double[,,] DistributeCovarMatrixToMatchCovarianceType(double[,] tiedCv, CovarianceType covarianceType, int nComponents)
        => throw new NotImplementedException();
}
