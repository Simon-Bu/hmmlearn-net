using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;

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
    public static double LogDet(double[,] a)
    {
        var n = a.GetLength(0);
        if (a.GetLength(1) != n)
            throw new ArgumentException($"a must be square, got {n} x {a.GetLength(1)}", nameof(a));
        var u = (double[,])a.Clone();
        var logAbsDeterminant = 0.0;
        var sign = 1.0;
        for (var i = 0; i < n; i++)
        {
            var pivot = i;
            for (var r = i + 1; r < n; r++)
                if (Math.Abs(u[r, i]) > Math.Abs(u[pivot, i])) pivot = r;
            if (u[pivot, i] == 0.0)
                return double.NegativeInfinity; // a zero pivot makes the determinant 0
            if (pivot != i)
            {
                for (var c = i; c < n; c++)
                {
                    var t = u[i, c];
                    u[i, c] = u[pivot, c];
                    u[pivot, c] = t;
                }
                sign = -sign;
            }
            sign *= u[i, i] < 0.0 ? -1.0 : 1.0;
            logAbsDeterminant += Math.Log(Math.Abs(u[i, i]));
            for (var r = i + 1; r < n; r++)
            {
                var factor = u[r, i] / u[i, i];
                if (factor != 0.0)
                {
                    for (var c = i + 1; c < n; c++)
                        u[r, c] -= factor * u[i, c];
                    u[r, i] = 0.0;
                }
            }
        }
        return sign < 0.0 ? double.NaN : logAbsDeterminant;
    }

    /// <summary>
    /// Splits the rows of <paramref name="x"/> (n_samples, n_features) into consecutive sequences of the given lengths; each
    /// element of the result is a new matrix with the rows of one sequence. Null lengths: a list holding <paramref name="x"/> itself.
    /// Lengths that are negative or do not sum to the number of rows throw <see cref="ArgumentException"/>.
    /// </summary>
    public static List<double[,]> SplitXLengths(double[,] x, int[]? lengths)
    {
        if (lengths == null)
            return new List<double[,]> { x };
        var nRows = x.GetLength(0);
        var nColumns = x.GetLength(1);
        var total = 0;
        for (var i = 0; i < lengths.Length; i++)
        {
            if (lengths[i] < 0)
                throw new ArgumentException($"lengths[{i}] is {lengths[i]}, lengths must be >= 0", nameof(lengths));
            total += lengths[i];
        }
        if (total != nRows)
            throw new ArgumentException($"lengths {string.Join(", ", lengths)} sum to {total}, but x has {nRows} rows", nameof(lengths));
        var sequences = new List<double[,]>();
        var start = 0;
        for (var i = 0; i < lengths.Length; i++)
        {
            var length = lengths[i];
            var sequence = new double[length, nColumns];
            for (var r = 0; r < length; r++)
                for (var c = 0; c < nColumns; c++)
                    sequence[r, c] = x[start + r, c];
            sequences.Add(sequence);
            start += length;
        }
        return sequences;
    }

    /// <summary>
    /// Basic checks of compact covariances; throws <see cref="ArgumentException"/> when hmmlearn raises ValueError.
    /// Shapes: spherical (nComponents, 1, 1), diag (nComponents, 1, nf), full (nComponents, nf, nf), tied (1, nf, nf), otherwise
    /// ArgumentException. Spherical and diag: every variance must be positive. Tied and full: every matrix must be symmetric
    /// (|a - b| &lt;= 1e-8 + 1e-5 * |b| between a[i,j] and its transpose b = a[j,i]) and positive definite (every eigenvalue &gt; 0).
    /// </summary>
    public static void ValidateCovars(double[,,] covars, CovarianceType covarianceType, int nComponents)
    {
        var d0 = covars.GetLength(0);
        var d1 = covars.GetLength(1);
        var d2 = covars.GetLength(2);
        switch (covarianceType)
        {
            case CovarianceType.Spherical:
                if (d0 != nComponents || d1 != 1 || d2 != 1)
                    throw new ArgumentException($"'spherical' covars must have shape ({nComponents}, 1, 1), got ({d0}, {d1}, {d2})", nameof(covars));
                for (var k = 0; k < d0; k++)
                    if (covars[k, 0, 0] <= 0.0)
                        throw new ArgumentException($"'spherical' covars must be positive, got {covars[k, 0, 0]}", nameof(covars));
                break;
            case CovarianceType.Diag:
                if (d0 != nComponents || d1 != 1)
                    throw new ArgumentException($"'diag' covars must have shape ({nComponents}, 1, {d2}), got ({d0}, {d1}, {d2})", nameof(covars));
                for (var k = 0; k < d0; k++)
                    for (var f = 0; f < d2; f++)
                        if (covars[k, 0, f] <= 0.0)
                            throw new ArgumentException($"'diag' covars must be positive, got {covars[k, 0, f]}", nameof(covars));
                break;
            case CovarianceType.Full:
                if (d0 != nComponents || d1 != d2)
                    throw new ArgumentException($"'full' covars must have shape ({nComponents}, {d1}, {d1}), got ({d0}, {d1}, {d2})", nameof(covars));
                for (var k = 0; k < d0; k++)
                    CheckMatrix(covars, k);
                break;
            case CovarianceType.Tied:
                if (d0 != 1 || d1 != d2)
                    throw new ArgumentException($"'tied' covars must have shape (1, {d1}, {d1}), got ({d0}, {d1}, {d2})", nameof(covars));
                CheckMatrix(covars, 0);
                break;
        }
    }

    /// <summary>
    /// Creates compact covariances for all states from one (nf, nf) template, in the uniform shapes above: spherical gives
    /// the mean of ALL entries of the matrix (off-diagonal ones included) for every state, diag the diagonal of the matrix,
    /// full the matrix itself, tied the matrix once. Result shapes: (nc, 1, 1), (nc, 1, nf), (nc, nf, nf) and (1, nf, nf).
    /// </summary>
    public static double[,,] DistributeCovarMatrixToMatchCovarianceType(double[,] tiedCv, CovarianceType covarianceType, int nComponents)
    {
        var n = tiedCv.GetLength(0);
        switch (covarianceType)
        {
            case CovarianceType.Spherical:
            {
                var mean = 0.0;
                for (var i = 0; i < n; i++)
                    for (var j = 0; j < n; j++)
                        mean += tiedCv[i, j];
                mean /= n * n;
                var result = new double[nComponents, 1, 1];
                for (var k = 0; k < nComponents; k++)
                    result[k, 0, 0] = mean;
                return result;
            }
            case CovarianceType.Diag:
            {
                var result = new double[nComponents, 1, n];
                for (var k = 0; k < nComponents; k++)
                    for (var i = 0; i < n; i++)
                        result[k, 0, i] = tiedCv[i, i];
                return result;
            }
            case CovarianceType.Full:
            {
                var result = new double[nComponents, n, n];
                for (var k = 0; k < nComponents; k++)
                    for (var i = 0; i < n; i++)
                        for (var j = 0; j < n; j++)
                            result[k, i, j] = tiedCv[i, j];
                return result;
            }
            default: // CovarianceType.Tied: the enum has exactly these four values
            {
                var result = new double[1, n, n];
                for (var i = 0; i < n; i++)
                    for (var j = 0; j < n; j++)
                        result[0, i, j] = tiedCv[i, j];
                return result;
            }
        }
    }

    /// <summary>
    /// Symmetry (with the allclose tolerance |a[i,j] - a[j,i]| &lt;= 1e-8 + 1e-5 * |a[j,i]|) and positive definiteness
    /// (every eigenvalue &gt; 0) of one (nf, nf) matrix of the compact tensor.
    /// </summary>
    private static void CheckMatrix(double[,,] covars, int k)
    {
        var n = covars.GetLength(1);
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var tolerance = 1e-8 + 1e-5 * Math.Abs(covars[k, j, i]);
                if (Math.Abs(covars[k, i, j] - covars[k, j, i]) > tolerance)
                    throw new ArgumentException($"covars of component {k} must be symmetric, got {covars[k, i, j]} vs {covars[k, j, i]}", nameof(covars));
            }
        var matrix = Matrix<double>.Build.Dense(n, n);
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                matrix[i, j] = 0.5 * (covars[k, i, j] + covars[k, j, i]);
        var eigenvalues = matrix.Evd(Symmetricity.Symmetric).EigenValues;
        for (var i = 0; i < eigenvalues.Count; i++)
            if (eigenvalues[i].Real <= 0.0)
                throw new ArgumentException($"covars of component {k} must be positive definite, eigenvalue {i} is {eigenvalues[i].Real}", nameof(covars));
    }
}
