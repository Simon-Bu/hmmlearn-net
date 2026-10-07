using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm12;

/// <summary>
/// Acceptance tests for <see cref="KMeans.Fit"/> (ticket HMM-12, criterion C3): k = 1 gives the mean of the data, k = n
/// gives every distinct sample its own center, duplicate points do not break the algorithm, maxIter = 0 returns the
/// farthest-point seeds and the labels of the nearest seed, a center left without samples is recovered, and invalid
/// arguments throw ArgumentException.
/// </summary>
public class EdgeCaseTests
{
    /// <summary>
    /// k = 1: every sample carries the only label and the single center is the mean of all the data (the mean is computed
    /// from the test's own rows in code).
    /// </summary>
    [Fact]
    public void k_one_gives_the_mean_of_the_data()
    {
        var x = new double[,]
        {
            { 0.0, 0.0 }, { 2.0, 0.0 }, { 1.0, 1.0 }, { 1.0, -1.0 }
        };

        var (centers, labels) = KMeans.Fit(x, 1);

        Assert.Equal(x.GetLength(0), labels.Length);
        for (var i = 0; i < labels.Length; i++)
            Assert.Equal(0, labels[i]);

        var sum0 = 0.0;
        var sum1 = 0.0;
        for (var i = 0; i < x.GetLength(0); i++)
        {
            sum0 += x[i, 0];
            sum1 += x[i, 1];
        }
        Close.Equal(sum0 / x.GetLength(0), centers[0, 0], "center feature 0, the mean of the data");
        Close.Equal(sum1 / x.GetLength(0), centers[0, 1], "center feature 1, the mean of the data");
    }

    /// <summary>
    /// k = n with distinct samples: every sample gets its own center. Each center equals exactly one input row, and that
    /// row carries the center's index as its label.
    /// </summary>
    [Fact]
    public void k_equal_to_n_with_distinct_samples_gives_each_sample_its_own_center()
    {
        var x = new double[,]
        {
            { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }
        };

        var (centers, labels) = KMeans.Fit(x, 3);

        for (var c = 0; c < 3; c++)
        {
            var owner = -1;
            for (var i = 0; i < x.GetLength(0); i++)
            {
                if (Close.IsClose(x[i, 0], centers[c, 0]) && Close.IsClose(x[i, 1], centers[c, 1]))
                {
                    Assert.True(owner < 0, $"center {c} equals both row {owner} and row {i}");
                    owner = i;
                }
            }
            Assert.True(owner >= 0, $"no input row matches center {c}");
            Assert.True(c == labels[owner], $"the label of row {owner}, whose own center is {c}");
        }
    }

    /// <summary>
    /// Duplicate points do not break the algorithm: two copies of each of two distinct points, k = 2, put the copies of
    /// each point in the same cluster and give each cluster the (common) point as its center.
    /// </summary>
    [Fact]
    public void Duplicate_points_do_not_break_the_algorithm()
    {
        var x = new double[,]
        {
            { 0.0, 0.0 }, { 0.0, 0.0 }, { 3.0, 4.0 }, { 3.0, 4.0 }
        };

        var (centers, labels) = KMeans.Fit(x, 2);

        Assert.True(labels[0] == labels[1], "the two identical (0, 0) samples must share a label");
        Assert.True(labels[2] == labels[3], "the two identical (3, 4) samples must share a label");
        Assert.True(labels[0] != labels[2], "the two distinct points must end up in different clusters");
        Close.Equal(0.0, centers[labels[0], 0], "center of the (0, 0) cluster, feature 0");
        Close.Equal(0.0, centers[labels[0], 1], "center of the (0, 0) cluster, feature 1");
        Close.Equal(3.0, centers[labels[2], 0], "center of the (3, 4) cluster, feature 0");
        Close.Equal(4.0, centers[labels[2], 1], "center of the (3, 4) cluster, feature 1");
    }

    /// <summary>
    /// All samples identical with k = 1: the single center is the common point and every sample carries label 0.
    /// </summary>
    [Fact]
    public void All_identical_points_with_k_one_give_the_common_point()
    {
        var x = new double[,]
        {
            { 2.0, 3.0 }, { 2.0, 3.0 }, { 2.0, 3.0 }
        };

        var (centers, labels) = KMeans.Fit(x, 1);

        for (var i = 0; i < labels.Length; i++)
            Assert.Equal(0, labels[i]);
        Close.Equal(2.0, centers[0, 0], "center feature 0");
        Close.Equal(3.0, centers[0, 1], "center feature 1");
    }

    /// <summary>
    /// maxIter = 0 returns the farthest-point seeds and the labels of the nearest seed. Seeds: (0, 0) = x[0], then the row
    /// farthest from it, (4, 0); the labels are the nearest-seed assignment on integer coordinates.
    /// </summary>
    [Fact]
    public void max_iter_zero_returns_the_farthest_point_seeds_and_the_labels_of_the_nearest_seed()
    {
        var x = new double[,]
        {
            { 0.0, 0.0 }, { 1.0, 0.0 }, { 3.0, 0.0 }, { 4.0, 0.0 }
        };

        var (centers, labels) = KMeans.Fit(x, 2, maxIter: 0);

        Close.Equal(0.0, centers[0, 0], "seed 0, feature 0 (x[0])");
        Close.Equal(0.0, centers[0, 1], "seed 0, feature 1 (x[0])");
        Close.Equal(4.0, centers[1, 0], "seed 1, feature 0 (the row farthest from x[0])");
        Close.Equal(0.0, centers[1, 1], "seed 1, feature 1 (the row farthest from x[0])");
        Assert.Equal(new[] { 0, 0, 1, 1 }, labels);
    }

    /// <summary>
    /// maxIter = 0 with ties: (1, 0) and (0, 1) are equidistant from (0, 0), so the smallest row index (row 1) wins the
    /// seeding; then (0, 1) is closer to seed 0 (distance 1) than to seed 1 (distance sqrt(2)).
    /// </summary>
    [Fact]
    public void max_iter_zero_breaks_seeding_ties_by_the_smallest_row_index()
    {
        var x = new double[,]
        {
            { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }
        };

        var (centers, labels) = KMeans.Fit(x, 2, maxIter: 0);

        Close.Equal(0.0, centers[0, 0], "seed 0, feature 0 (x[0])");
        Close.Equal(0.0, centers[0, 1], "seed 0, feature 1 (x[0])");
        Close.Equal(1.0, centers[1, 0], "seed 1, feature 0 (row 1 wins the tie with row 2)");
        Close.Equal(0.0, centers[1, 1], "seed 1, feature 1 (row 1 wins the tie with row 2)");
        Assert.Equal(new[] { 0, 1, 0 }, labels);
    }

    /// <summary>
    /// A center left without samples is recovered: four 0's and four 10's with k = 3 (1D). The farthest-point seeds are
    /// (0, 10, 0), so one cluster is empty from the start; the algorithm must not break and must end in a consistent state:
    /// valid labels, the 10's in one cluster with center 10, every 0-sample labeled with a center sitting at 0, every
    /// center sitting on a data point, and each sample at least as close to its own center as to any other.
    /// </summary>
    [Fact]
    public void A_center_left_without_samples_is_recovered()
    {
        var x = new double[8, 1]
        {
            { 0.0 }, { 0.0 }, { 0.0 }, { 0.0 }, { 10.0 }, { 10.0 }, { 10.0 }, { 10.0 }
        };

        var (centers, labels) = KMeans.Fit(x, 3);

        for (var i = 0; i < labels.Length; i++)
            Assert.InRange(labels[i], 0, 2);

        // the 10's form one cluster, whose center is the mean of the 10's (10)
        for (var i = 5; i < 8; i++)
            Assert.True(labels[4] == labels[i], $"row {i} (value 10) must share the cluster of row 4");
        Close.Equal(10.0, centers[labels[4], 0], "center of the 10's cluster");

        // every 0-sample is labeled with a center sitting at 0 (the mean of the 0's)
        for (var i = 0; i < 4; i++)
            Close.Equal(0.0, centers[labels[i], 0], $"center of the cluster of row {i} (value 0)");

        // every center sits on a data point (the recovered empty cluster moved to a sample), and both values survive
        var sawZero = false;
        var sawTen = false;
        for (var c = 0; c < 3; c++)
        {
            if (Close.IsClose(0.0, centers[c, 0]))
                sawZero = true;
            else if (Close.IsClose(10.0, centers[c, 0]))
                sawTen = true;
            else
                Assert.Fail($"center {c} sits at {centers[c, 0]:R}, which is not a data point");
        }
        Assert.True(sawZero && sawTen, $"the centers must keep both values of the data: {string.Join(", ", new[] { centers[0, 0], centers[1, 0], centers[2, 0] })}");

        // the labels are consistent with the returned centers
        for (var i = 0; i < x.GetLength(0); i++)
        {
            var own = (x[i, 0] - centers[labels[i], 0]) * (x[i, 0] - centers[labels[i], 0]);
            for (var c = 0; c < 3; c++)
            {
                if (c == labels[i]) continue;
                var other = (x[i, 0] - centers[c, 0]) * (x[i, 0] - centers[c, 0]);
                Assert.True(own <= other + 1e-9,
                    $"sample {i} (value {x[i, 0]:R}, label {labels[i]}) is farther from its own center (squared distance {own:R}) than from center {c} ({other:R})");
            }
        }
    }

    /// <summary>
    /// Invalid arguments throw ArgumentException (exact type): x without rows or columns, NaN or infinity in x, k below 1
    /// or above the number of rows, and maxIter below 0.
    /// </summary>
    [Theory]
    [InlineData(0)] // x has no rows (0 x 3)
    [InlineData(1)] // x has no columns (3 x 0)
    [InlineData(2)] // NaN in x
    [InlineData(3)] // +infinity in x
    [InlineData(4)] // -infinity in x
    [InlineData(5)] // k below 1
    [InlineData(6)] // k above the number of rows
    [InlineData(7)] // maxIter below 0
    public void Invalid_arguments_throw_argument_exception(int caseId)
    {
        var call = caseId switch
        {
            0 => new Action(() => KMeans.Fit(new double[0, 3], 2)),
            1 => new Action(() => KMeans.Fit(new double[3, 0], 1)),
            2 => new Action(() => KMeans.Fit(new double[,] { { 1.0, double.NaN }, { 2.0, 3.0 } }, 2)),
            3 => new Action(() => KMeans.Fit(new double[,] { { 1.0, double.PositiveInfinity }, { 2.0, 3.0 } }, 2)),
            4 => new Action(() => KMeans.Fit(new double[,] { { double.NegativeInfinity, 1.0 }, { 2.0, 3.0 } }, 2)),
            5 => new Action(() => KMeans.Fit(new double[,] { { 1.0 }, { 2.0 } }, 0)),
            6 => new Action(() => KMeans.Fit(new double[,] { { 1.0 }, { 2.0 } }, 3)),
            _ => new Action(() => KMeans.Fit(new double[,] { { 1.0 }, { 2.0 } }, 2, maxIter: -1)),
        };
        Assert.Throws<ArgumentException>(call);
    }
}
