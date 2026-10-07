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
        => throw new NotImplementedException();
}
