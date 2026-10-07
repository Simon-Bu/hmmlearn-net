using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm12;

/// <summary>
/// Acceptance tests for <see cref="KMeans.Fit"/> (ticket HMM-12, criterion C2): at the end each center is the mean of the
/// samples that carry its label, each sample is at least as close to its own center as to any other, the result is
/// deterministic (the same input gives the same output), and x is left untouched.
/// </summary>
public class LloydInvariantsTests
{
    /// <summary>One standard normal by Box-Muller (cosine branch); u1 is drawn in (0, 1] so the log is finite.</summary>
    private static double NextGaussian(Random rng)
    {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Three well separated 2D blobs (30 samples each), drawn from a standard normal around (0, 0), (10, 2) and (2, 10)
    /// with a Random(12345): data on which Lloyd's algorithm converges to a fixed point.
    /// </summary>
    private static (double[,] X, int[] Blob) Data()
    {
        var centers = new[] { new[] { 0.0, 0.0 }, new[] { 10.0, 2.0 }, new[] { 2.0, 10.0 } };
        const int per = 30;
        var rng = new Random(12345);
        var x = new double[centers.Length * per, centers[0].Length];
        var blob = new int[centers.Length * per];
        var row = 0;
        for (var b = 0; b < centers.Length; b++)
        {
            for (var s = 0; s < per; s++)
            {
                for (var f = 0; f < centers[0].Length; f++)
                    x[row, f] = centers[b][f] + NextGaussian(rng);
                blob[row] = b;
                row++;
            }
        }
        return (x, blob);
    }

    /// <summary>Precondition: every sample is strictly closer to its own blob center than to any other blob center.</summary>
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

    /// <summary>Squared Euclidean distance from a row of x to a row of centers.</summary>
    private static double SquaredDistance(double[,] x, int row, double[,] centers, int center)
    {
        var d = 0.0;
        for (var f = 0; f < x.GetLength(1); f++)
        {
            var diff = x[row, f] - centers[center, f];
            d += diff * diff;
        }
        return d;
    }

    /// <summary>Two label vectors are identical element by element.</summary>
    private static void AssertSameLabels(int[] a, int[] b, string what)
    {
        Assert.True(a.Length == b.Length, $"{what}: lengths differ ({a.Length} vs {b.Length})");
        for (var i = 0; i < a.Length; i++)
            Assert.True(a[i] == b[i], $"{what}: label of sample {i} differs ({a[i]} vs {b[i]})");
    }

    /// <summary>
    /// Each returned center is the mean of the samples that carry its label (computed over those samples in code), and no
    /// cluster is empty.
    /// </summary>
    [Fact]
    public void Every_center_is_the_mean_of_the_samples_carrying_its_label()
    {
        var (x, blob) = Data();
        AssertWellSeparated(x, blob, new[] { new[] { 0.0, 0.0 }, new[] { 10.0, 2.0 }, new[] { 2.0, 10.0 } });

        var (centers, labels) = KMeans.Fit(x, 3);

        var rows = x.GetLength(0);
        var features = x.GetLength(1);
        for (var c = 0; c < 3; c++)
        {
            var count = 0;
            for (var i = 0; i < rows; i++)
                if (labels[i] == c)
                    count++;
            Assert.True(count > 0, $"cluster {c} has no sample");
            for (var f = 0; f < features; f++)
            {
                var sum = 0.0;
                for (var i = 0; i < rows; i++)
                    if (labels[i] == c)
                        sum += x[i, f];
                Close.Equal(sum / count, centers[c, f], $"center {c}, feature {f}, mean of its {count} samples");
            }
        }
    }

    /// <summary>
    /// Each sample is at least as close (squared Euclidean distance) to its own center as to any other center.
    /// </summary>
    [Fact]
    public void Every_sample_is_at_least_as_close_to_its_own_center_as_to_any_other()
    {
        var (x, blob) = Data();
        AssertWellSeparated(x, blob, new[] { new[] { 0.0, 0.0 }, new[] { 10.0, 2.0 }, new[] { 2.0, 10.0 } });

        var (centers, labels) = KMeans.Fit(x, 3);

        for (var i = 0; i < x.GetLength(0); i++)
        {
            var own = SquaredDistance(x, i, centers, labels[i]);
            for (var c = 0; c < 3; c++)
            {
                if (c == labels[i]) continue;
                var other = SquaredDistance(x, i, centers, c);
                Assert.True(own <= other + 1e-9,
                    $"sample {i} (label {labels[i]}) is farther from its own center (squared distance {own:R}) than from center {c} ({other:R})");
            }
        }
    }

    /// <summary>
    /// The same input gives the same output: two calls with the same x (default settings, and explicit maxIter/tol) give
    /// identical labels and bit-identical centers.
    /// </summary>
    [Fact]
    public void The_result_is_deterministic_for_the_same_input()
    {
        var (x, _) = Data();
        var (centers1, labels1) = KMeans.Fit(x, 3);
        var (centers2, labels2) = KMeans.Fit(x, 3);

        AssertSameLabels(labels1, labels2, "labels of two runs with the same input");
        for (var c = 0; c < centers1.GetLength(0); c++)
            for (var f = 0; f < centers1.GetLength(1); f++)
                Assert.True(centers1[c, f] == centers2[c, f],
                    $"center {c},{f}: two runs with the same input gave {centers1[c, f]:R} and {centers2[c, f]:R}");

        var (centers3, labels3) = KMeans.Fit(x, 3, maxIter: 100, tol: 1e-5);
        var (centers4, labels4) = KMeans.Fit(x, 3, maxIter: 100, tol: 1e-5);
        AssertSameLabels(labels3, labels4, "labels of two runs with the same input and explicit maxIter/tol");
        for (var c = 0; c < centers3.GetLength(0); c++)
            for (var f = 0; f < centers3.GetLength(1); f++)
                Assert.True(centers3[c, f] == centers4[c, f],
                    $"center {c},{f}: two runs with the same input and explicit maxIter/tol gave {centers3[c, f]:R} and {centers4[c, f]:R}");
    }

    /// <summary>x is left untouched: every element of the input matrix is bit-identical to what it was before Fit.</summary>
    [Fact]
    public void The_input_matrix_is_left_untouched()
    {
        var x = new double[,]
        {
            { 0.0, 0.0 }, { 1.0, 0.0 }, { 9.0, 1.0 }, { 10.0, 2.0 }, { -3.0, 5.0 }
        };
        var copy = (double[,])x.Clone();

        KMeans.Fit(x, 2);

        for (var i = 0; i < x.GetLength(0); i++)
            for (var f = 0; f < x.GetLength(1); f++)
                Assert.True(copy[i, f] == x[i, f], $"x[{i},{f}] was modified: was {copy[i, f]:R}, now {x[i, f]:R}");
    }
}
