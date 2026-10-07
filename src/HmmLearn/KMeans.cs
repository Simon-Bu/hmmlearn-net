namespace HmmLearn;

/// <summary>
/// A minimal, deterministic k-means (Lloyd's algorithm) for the default initializations of the Gaussian and mixture models. It
/// replaces scikit-learn's KMeans, which hmmlearn uses; it is not meant to reproduce its results, only to find sensible clusters.
/// </summary>
public static class KMeans
{
    /// <summary>
    /// Clusters the rows of <paramref name="x"/> (n_samples, n_features) into <paramref name="k"/> groups and returns the centers
    /// (k, n_features) and the label (0..k-1) of each sample. Deterministic: the same x and k always give the same result.
    /// <list type="number">
    /// <item>Seeding by the farthest-point rule: the first center is the first row of x; each next center is the row whose distance to
    /// its nearest chosen center is the largest (the smallest index wins a tie), so the seeds are spread out.</item>
    /// <item>Lloyd iterations, at most <paramref name="maxIter"/>: assign each sample to the nearest center (squared Euclidean distance,
    /// the smallest index wins a tie), then move each center to the mean of its samples; a center that lost all its samples moves to the
    /// sample that is farthest from its own center (the smallest index wins a tie) and takes it. Stop when no label changes or when the
    /// largest displacement of a center is below <paramref name="tol"/>.</item>
    /// </list>
    /// Throws <see cref="ArgumentException"/> if x has no rows or columns, contains NaN or infinity, k is below 1 or above the number of rows.
    /// </summary>
    public static (double[,] Centers, int[] Labels) Fit(double[,] x, int k, int maxIter = 300, double tol = 1e-8)
    {
        Validate(x, k, maxIter);
        var rows = x.GetLength(0);
        var features = x.GetLength(1);

        var centers = Seed(x, k, features);
        var labels = Assign(x, centers, rows, features);
        for (var iter = 0; iter < maxIter; iter++)
        {
            var newCenters = Move(x, centers, labels, rows, features);
            var newLabels = Assign(x, newCenters, rows, features);
            var stable = SameLabels(newLabels, labels) || MaxDisplacement(centers, newCenters, features) <= tol;
            centers = newCenters;
            labels = newLabels;
            if (stable)
                break;
        }
        return (centers, labels);
    }

    /// <summary>
    /// Checks the arguments: x must have at least one row and one column and no NaN or infinity, k must be between 1 and the number
    /// of rows, and maxIter must be non-negative; throws <see cref="ArgumentException"/> otherwise.
    /// </summary>
    private static void Validate(double[,] x, int k, int maxIter)
    {
        var rows = x.GetLength(0);
        var features = x.GetLength(1);
        if (rows == 0 || features == 0)
            throw new ArgumentException("x must have at least one row and one column.");
        for (var i = 0; i < rows; i++)
            for (var f = 0; f < features; f++)
                if (!double.IsFinite(x[i, f]))
                    throw new ArgumentException("x must not contain NaN or infinity.");
        if (k < 1 || k > rows)
            throw new ArgumentException("k must be between 1 and the number of rows of x.");
        if (maxIter < 0)
            throw new ArgumentException("maxIter must be non-negative.");
    }

    /// <summary>
    /// Seeds the k centers by the farthest-point rule: the first center is the first row of x, and each next center is the row whose
    /// distance to its nearest already chosen center is the largest (the smallest row index wins a tie).
    /// </summary>
    private static double[,] Seed(double[,] x, int k, int features)
    {
        var rows = x.GetLength(0);
        var centers = new double[k, features];
        for (var f = 0; f < features; f++)
            centers[0, f] = x[0, f];
        for (var c = 1; c < k; c++)
        {
            var best = 0;
            var bestDistance = -1.0;
            for (var i = 0; i < rows; i++)
            {
                var distance = NearestSquaredDistance(x, i, centers, c, features);
                if (distance > bestDistance)
                {
                    bestDistance = distance;
                    best = i;
                }
            }
            for (var f = 0; f < features; f++)
                centers[c, f] = x[best, f];
        }
        return centers;
    }

    /// <summary>
    /// Assigns every row of x to the nearest center (squared Euclidean distance, the smallest center index wins a tie).
    /// </summary>
    private static int[] Assign(double[,] x, double[,] centers, int rows, int features)
    {
        var k = centers.GetLength(0);
        var labels = new int[rows];
        for (var i = 0; i < rows; i++)
        {
            var best = 0;
            var bestDistance = SquaredDistance(x, i, centers, 0, features);
            for (var c = 1; c < k; c++)
            {
                var distance = SquaredDistance(x, i, centers, c, features);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    best = c;
                }
            }
            labels[i] = best;
        }
        return labels;
    }

    /// <summary>
    /// Moves each center to the mean of the samples assigned to it; a center with no sample moves to the sample that is farthest from
    /// its own position (the smallest row index wins a tie), so the final assignment takes it for that center.
    /// </summary>
    private static double[,] Move(double[,] x, double[,] centers, int[] labels, int rows, int features)
    {
        var k = centers.GetLength(0);
        var newCenters = new double[k, features];
        for (var c = 0; c < k; c++)
        {
            var count = 0;
            for (var i = 0; i < rows; i++)
            {
                if (labels[i] != c)
                    continue;
                count++;
                for (var f = 0; f < features; f++)
                    newCenters[c, f] += x[i, f];
            }
            if (count > 0)
            {
                for (var f = 0; f < features; f++)
                    newCenters[c, f] /= count;
                continue;
            }
            var best = 0;
            var bestDistance = -1.0;
            for (var i = 0; i < rows; i++)
            {
                var distance = SquaredDistance(x, i, centers, c, features);
                if (distance > bestDistance)
                {
                    bestDistance = distance;
                    best = i;
                }
            }
            for (var f = 0; f < features; f++)
                newCenters[c, f] = x[best, f];
        }
        return newCenters;
    }

    /// <summary>The largest Euclidean displacement of one center between two center matrices.</summary>
    private static double MaxDisplacement(double[,] oldCenters, double[,] newCenters, int features)
    {
        var max = 0.0;
        for (var c = 0; c < oldCenters.GetLength(0); c++)
        {
            var sum = 0.0;
            for (var f = 0; f < features; f++)
            {
                var diff = newCenters[c, f] - oldCenters[c, f];
                sum += diff * diff;
            }
            var displacement = Math.Sqrt(sum);
            if (displacement > max)
                max = displacement;
        }
        return max;
    }

    /// <summary>Whether two label vectors are identical element by element.</summary>
    private static bool SameLabels(int[] a, int[] b)
    {
        for (var i = 0; i < a.Length; i++)
            if (a[i] != b[i])
                return false;
        return true;
    }

    /// <summary>The squared Euclidean distance between a row of x and a center row.</summary>
    private static double SquaredDistance(double[,] x, int row, double[,] centers, int center, int features)
    {
        var sum = 0.0;
        for (var f = 0; f < features; f++)
        {
            var diff = x[row, f] - centers[center, f];
            sum += diff * diff;
        }
        return sum;
    }

    /// <summary>The squared Euclidean distance from a row of x to its nearest of the first <paramref name="count"/> centers.</summary>
    private static double NearestSquaredDistance(double[,] x, int row, double[,] centers, int count, int features)
    {
        var best = double.PositiveInfinity;
        for (var c = 0; c < count; c++)
        {
            var distance = SquaredDistance(x, row, centers, c, features);
            if (distance < best)
                best = distance;
        }
        return best;
    }
}
