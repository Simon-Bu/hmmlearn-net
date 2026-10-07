using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm12;

/// <summary>
/// Acceptance tests for <see cref="KMeans.Fit"/> (ticket HMM-12, criterion C1): on data made of three or four well
/// separated blobs (Box-Muller standard normals around fixed centers, drawn from a Random with the seed 12345), Fit puts
/// the samples of each blob in the same cluster and the samples of different blobs in different clusters, and its centers
/// are close to the true blob means (a few standard errors). Partitions are compared, never the label numbers.
/// </summary>
public class SeparatedClustersTests
{
    /// <summary>One standard normal by Box-Muller (cosine branch); u1 is drawn in (0, 1] so the log is finite.</summary>
    private static double NextGaussian(Random rng)
    {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// per samples per blob, drawn from a standard normal around each center (in blob order) with a Random(12345):
    /// returns the data and the blob index of every row.
    /// </summary>
    private static (double[,] X, int[] Blob) Blobs(double[][] centers, int per)
    {
        var rng = new Random(12345);
        var features = centers[0].Length;
        var x = new double[centers.Length * per, features];
        var blob = new int[centers.Length * per];
        var row = 0;
        for (var b = 0; b < centers.Length; b++)
        {
            for (var s = 0; s < per; s++)
            {
                for (var f = 0; f < features; f++)
                    x[row, f] = centers[b][f] + NextGaussian(rng);
                blob[row] = b;
                row++;
            }
        }
        return (x, blob);
    }

    private static double SquaredDistance(double[,] x, int row, double[] center)
    {
        var d = 0.0;
        for (var f = 0; f < x.GetLength(1); f++)
        {
            var diff = x[row, f] - center[f];
            d += diff * diff;
        }
        return d;
    }

    /// <summary>
    /// Precondition of the test data: the blobs are well separated, i.e. every sample is strictly closer to its own blob
    /// center than to the center of any other blob.
    /// </summary>
    private static void AssertWellSeparated(double[,] x, int[] blob, double[][] centers)
    {
        for (var i = 0; i < x.GetLength(0); i++)
        {
            var own = SquaredDistance(x, i, centers[blob[i]]);
            for (var b = 0; b < centers.Length; b++)
            {
                if (b == blob[i]) continue;
                var other = SquaredDistance(x, i, centers[b]);
                Assert.True(own < other,
                    $"sample {i} of blob {blob[i]} is not strictly closest to its own blob center (squared distances: own {own:R}, blob {b} {other:R})");
            }
        }
    }

    private static int FirstSampleOf(int[] blob, int value)
    {
        for (var i = 0; i < blob.Length; i++)
            if (blob[i] == value)
                return i;
        throw new InvalidOperationException($"blob {value} has no sample");
    }

    /// <summary>
    /// Samples of one blob all carry the same label, and the labels of two different blobs are different (the label
    /// numbers themselves are never asserted).
    /// </summary>
    private static void AssertPartitionMatchesBlobs(int[] labels, int[] blob, int k)
    {
        for (var i = 0; i < labels.Length; i++)
            Assert.InRange(labels[i], 0, k - 1);
        for (var a = 0; a < k; a++)
        {
            var labelA = labels[FirstSampleOf(blob, a)];
            for (var i = 0; i < blob.Length; i++)
                if (blob[i] == a)
                    Assert.True(labelA == labels[i], $"sample {i} of blob {a} does not carry blob {a}'s label");
            for (var b = a + 1; b < k; b++)
                Assert.True(labelA != labels[FirstSampleOf(blob, b)],
                    $"blob {a} and blob {b} were merged into one cluster");
        }
    }

    /// <summary>
    /// Some permutation of the centers matches the means of the blob samples with a per-feature error of at most 5
    /// standard errors (the standard deviation of the drawn normals is 1 by construction, so se = 1/sqrt(per)).
    /// </summary>
    private static void AssertCentersCloseToBlobMeans(double[,] centers, double[,] x, int[] blob, int per)
    {
        var k = x.GetLength(0) / per;
        var features = x.GetLength(1);
        var means = new double[k, features];
        for (var b = 0; b < k; b++)
        {
            var count = 0;
            for (var i = 0; i < x.GetLength(0); i++)
                if (blob[i] == b)
                {
                    for (var f = 0; f < features; f++)
                        means[b, f] += x[i, f];
                    count++;
                }
            Assert.True(count > 0, $"blob {b} has no sample");
            for (var f = 0; f < features; f++)
                means[b, f] /= count;
        }

        var standardError = 1.0 / Math.Sqrt(per);
        var best = double.PositiveInfinity;
        foreach (var match in Permutations(k))
        {
            var worst = 0.0;
            for (var c = 0; c < k; c++)
                for (var f = 0; f < features; f++)
                    worst = Math.Max(worst, Math.Abs(centers[c, f] - means[match[c], f]));
            best = Math.Min(best, worst);
        }
        Assert.True(best <= 5.0 * standardError,
            $"no permutation of the centers matches the blob means within 5 standard errors: worst per-feature error {best:R}, tolerance {5.0 * standardError:R}");
    }

    /// <summary>All permutations of 0..k-1 (k <= 4 in this ticket).</summary>
    private static IEnumerable<int[]> Permutations(int k)
    {
        var items = new int[k];
        for (var i = 0; i < k; i++)
            items[i] = i;
        yield return (int[])items.Clone();
        while (NextPermutation(items))
            yield return (int[])items.Clone();
    }

    private static bool NextPermutation(int[] a)
    {
        var i = a.Length - 2;
        while (i >= 0 && a[i] >= a[i + 1])
            i--;
        if (i < 0)
            return false;
        var j = a.Length - 1;
        while (a[j] <= a[i])
            j--;
        (a[i], a[j]) = (a[j], a[i]);
        Array.Reverse(a, i + 1, a.Length - i - 1);
        return true;
    }

    /// <summary>
    /// Three well separated 2D blobs (30 samples each) end up as three clusters, one per blob, with centers near the
    /// blob means.
    /// </summary>
    [Fact]
    public void Three_well_separated_2d_blobs_end_up_as_three_clusters_with_centers_near_the_blob_means()
    {
        var centers = new[] { new[] { 0.0, 0.0 }, new[] { 10.0, 0.0 }, new[] { 0.0, 10.0 } };
        var (x, blob) = Blobs(centers, 30);
        AssertWellSeparated(x, blob, centers);

        var (result, labels) = KMeans.Fit(x, 3);

        Assert.Equal(3, result.GetLength(0));
        Assert.Equal(2, result.GetLength(1));
        Assert.Equal(x.GetLength(0), labels.Length);
        AssertPartitionMatchesBlobs(labels, blob, 3);
        AssertCentersCloseToBlobMeans(result, x, blob, 30);
    }

    /// <summary>
    /// Four well separated 3D blobs (25 samples each) end up as four clusters, one per blob, with centers near the blob
    /// means.
    /// </summary>
    [Fact]
    public void Four_well_separated_3d_blobs_end_up_as_four_clusters_with_centers_near_the_blob_means()
    {
        var centers = new[]
        {
            new[] { 0.0, 0.0, 0.0 }, new[] { 8.0, 0.0, 8.0 }, new[] { 0.0, 8.0, 8.0 }, new[] { 8.0, 8.0, 0.0 }
        };
        var (x, blob) = Blobs(centers, 25);
        AssertWellSeparated(x, blob, centers);

        var (result, labels) = KMeans.Fit(x, 4);

        Assert.Equal(4, result.GetLength(0));
        Assert.Equal(3, result.GetLength(1));
        Assert.Equal(x.GetLength(0), labels.Length);
        AssertPartitionMatchesBlobs(labels, blob, 4);
        AssertCentersCloseToBlobMeans(result, x, blob, 25);
    }
}
