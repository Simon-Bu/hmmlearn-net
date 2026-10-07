namespace HmmLearn;

/// <summary>Port of hmmlearn/utils.py: in-place normalization helpers and the expansion of compact covariances.</summary>
public static class Utils
{
    /// <summary>
    /// Divides every element by the sum of all elements, in place. A sum of 0 is treated as 1 (the zeros stay zeros).
    /// </summary>
    public static void Normalize(double[] a)
    {
        var sum = 0.0;
        for (var i = 0; i < a.Length; i++)
            sum += a[i];
        if (sum == 0.0) sum = 1.0;
        for (var i = 0; i < a.Length; i++)
            a[i] /= sum;
    }

    /// <summary>
    /// Normalizes in place. <paramref name="axis"/> null: the whole matrix is divided by the sum of all its elements;
    /// 0: each column is divided by its own sum; 1: each row is divided by its own sum. A sum of 0 is treated as 1.
    /// Any other axis throws <see cref="ArgumentOutOfRangeException"/>.
    /// </summary>
    public static void Normalize(double[,] a, int? axis = null)
    {
        var rows = a.GetLength(0);
        var cols = a.GetLength(1);
        if (axis is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(axis), axis.Value, "axis must be null, 0 or 1");

        if (axis == 0)
        {
            for (var j = 0; j < cols; j++)
            {
                var sum = 0.0;
                for (var i = 0; i < rows; i++)
                    sum += a[i, j];
                if (sum == 0.0) sum = 1.0;
                for (var i = 0; i < rows; i++)
                    a[i, j] /= sum;
            }
            return;
        }
        if (axis == 1)
        {
            for (var i = 0; i < rows; i++)
            {
                var sum = 0.0;
                for (var j = 0; j < cols; j++)
                    sum += a[i, j];
                if (sum == 0.0) sum = 1.0;
                for (var j = 0; j < cols; j++)
                    a[i, j] /= sum;
            }
            return;
        }
        var total = 0.0;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                total += a[i, j];
        if (total == 0.0) total = 1.0;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                a[i, j] /= total;
    }

    /// <summary>
    /// Normalizes in place so that sum(exp(a)) is 1: subtracts the log-sum-exp of all elements (computed stably: the maximum
    /// is factored out, and a maximum that is infinite is the log-sum-exp itself). A single -infinity gives NaN, not 0
    /// (the length-1 rule of the matrix overload does not apply to vectors).
    /// </summary>
    public static void LogNormalize(double[] a)
    {
        var lse = LogSumExp(a);
        for (var i = 0; i < a.Length; i++)
            a[i] -= lse;
    }

    /// <summary>
    /// Same as <see cref="LogNormalize(double[])"/> for a matrix. <paramref name="axis"/> null: one log-sum-exp for all elements;
    /// 0: one per column; 1: one per row. If the chosen axis has length 1, every element is set to 0 (even -infinity).
    /// A row of -infinity gives NaN (-infinity minus -infinity). Any other axis throws <see cref="ArgumentOutOfRangeException"/>.
    /// </summary>
    public static void LogNormalize(double[,] a, int? axis = null)
    {
        var rows = a.GetLength(0);
        var cols = a.GetLength(1);
        if (axis is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(axis), axis.Value, "axis must be null, 0 or 1");

        if ((axis == 0 && rows == 1) || (axis == 1 && cols == 1))
        {
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++)
                    a[i, j] = 0.0;
            return;
        }
        if (axis == 0)
        {
            var column = new double[rows];
            for (var j = 0; j < cols; j++)
            {
                for (var i = 0; i < rows; i++)
                    column[i] = a[i, j];
                var columnLse = LogSumExp(column);
                for (var i = 0; i < rows; i++)
                    a[i, j] -= columnLse;
            }
            return;
        }
        if (axis == 1)
        {
            var row = new double[cols];
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                    row[j] = a[i, j];
                var rowLse = LogSumExp(row);
                for (var j = 0; j < cols; j++)
                    a[i, j] -= rowLse;
            }
            return;
        }
        var all = new double[rows * cols];
        var index = 0;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                all[index++] = a[i, j];
        var lse = LogSumExp(all);
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                a[i, j] -= lse;
    }

    /// <summary>
    /// Expands compact covariance parameters (uniform shapes, see oracle/README.md) to full matrices of shape
    /// (<paramref name="nComponents"/>, <paramref name="nFeatures"/>, <paramref name="nFeatures"/>).
    /// Compact shapes: spherical (nc, 1, 1), diag (nc, 1, nf), full (nc, nf, nf), tied (1, nf, nf).
    /// Spherical gives variance times the identity, diag puts the variances on the diagonal, full keeps the matrices,
    /// tied repeats the single matrix nc times. A shape that does not match the covariance type throws <see cref="ArgumentException"/>.
    /// </summary>
    public static double[,,] FillCovars(double[,,] covars, CovarianceType covarianceType, int nComponents, int nFeatures)
    {
        var result = new double[nComponents, nFeatures, nFeatures];
        switch (covarianceType)
        {
            case CovarianceType.Spherical:
                RequireShape(covars, covarianceType, nComponents, 1, 1);
                for (var k = 0; k < nComponents; k++)
                    for (var i = 0; i < nFeatures; i++)
                        result[k, i, i] = covars[k, 0, 0];
                break;
            case CovarianceType.Diag:
                RequireShape(covars, covarianceType, nComponents, 1, nFeatures);
                for (var k = 0; k < nComponents; k++)
                    for (var i = 0; i < nFeatures; i++)
                        result[k, i, i] = covars[k, 0, i];
                break;
            case CovarianceType.Full:
                RequireShape(covars, covarianceType, nComponents, nFeatures, nFeatures);
                for (var k = 0; k < nComponents; k++)
                    for (var i = 0; i < nFeatures; i++)
                        for (var j = 0; j < nFeatures; j++)
                            result[k, i, j] = covars[k, i, j];
                break;
            case CovarianceType.Tied:
                RequireShape(covars, covarianceType, 1, nFeatures, nFeatures);
                for (var k = 0; k < nComponents; k++)
                    for (var i = 0; i < nFeatures; i++)
                        for (var j = 0; j < nFeatures; j++)
                            result[k, i, j] = covars[0, i, j];
                break;
        }
        return result;
    }

    /// <summary>
    /// The stable log-sum-exp: the maximum is factored out; a maximum that is +infinity is the log-sum-exp itself,
    /// and a maximum that is -infinity (all entries -infinity) gives a log-sum-exp of -infinity (so subtracting it yields NaN).
    /// </summary>
    private static double LogSumExp(double[] values)
    {
        var max = double.NegativeInfinity;
        for (var i = 0; i < values.Length; i++)
            if (values[i] > max) max = values[i];
        if (max == double.PositiveInfinity)
            return double.PositiveInfinity;
        if (max == double.NegativeInfinity)
            return double.NegativeInfinity;
        var sum = 0.0;
        for (var i = 0; i < values.Length; i++)
            sum += Math.Exp(values[i] - max);
        return max + Math.Log(sum);
    }

    /// <summary>Throws <see cref="ArgumentException"/> when the compact tensor does not have the shape the covariance type requires.</summary>
    private static void RequireShape(double[,,] covars, CovarianceType covarianceType, int d0, int d1, int d2)
    {
        if (covars.GetLength(0) != d0 || covars.GetLength(1) != d1 || covars.GetLength(2) != d2)
            throw new ArgumentException(
                $"covars must have shape ({d0}, {d1}, {d2}) for {covarianceType}, got ({covars.GetLength(0)}, {covars.GetLength(1)}, {covars.GetLength(2)})");
    }
}
