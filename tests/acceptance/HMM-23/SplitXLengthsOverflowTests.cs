using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm23;

/// <summary>
/// Acceptance tests for <see cref="InternalUtils.SplitXLengths"/> (ticket HMM-23, criterion C1): a sum of lengths that wraps
/// around 2^31 in 32 bits must not pass the sum check. Matrices have zero columns so that no test allocates memory.
/// </summary>
public class SplitXLengthsOverflowTests
{
    /// <summary>Four lengths of 2^30 sum to 2^32, which wraps to 0 = the number of rows of x.</summary>
    [Fact]
    public void Sum_wrapping_to_zero_rows_throws_ArgumentException()
    {
        var lengths = new[] { 1073741824, 1073741824, 1073741824, 1073741824 };
        Assert.Throws<ArgumentException>(() => InternalUtils.SplitXLengths(new double[0, 0], lengths));
    }

    /// <summary>The true sum 2^32 + 2 wraps to 2 = the number of rows of x.</summary>
    [Fact]
    public void Sum_wrapping_to_two_rows_throws_ArgumentException()
    {
        var lengths = new[] { int.MaxValue, int.MaxValue, 4 };
        Assert.Throws<ArgumentException>(() => InternalUtils.SplitXLengths(new double[2, 0], lengths));
    }

    /// <summary>Control: a valid split keeps its documented result.</summary>
    [Fact]
    public void Valid_split_gives_new_matrices_with_the_right_rows()
    {
        var x = new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
        var result = InternalUtils.SplitXLengths(x, new[] { 1, 0, 2 });
        Assert.Equal(3, result.Count);
        Close.Equal(new double[,] { { 1, 2 } }, result[0], "first sequence");
        Close.Equal(new double[0, 2], result[1], "empty sequence");
        Close.Equal(new double[,] { { 3, 4 }, { 5, 6 } }, result[2], "last sequence");
    }

    /// <summary>Control: a negative length throws ArgumentException.</summary>
    [Fact]
    public void Negative_length_throws_ArgumentException()
    {
        Assert.Throws<ArgumentException>(() => InternalUtils.SplitXLengths(new double[2, 0], new[] { 3, -1 }));
    }

    /// <summary>Control: a plain wrong sum throws ArgumentException.</summary>
    [Fact]
    public void Plain_wrong_sum_throws_ArgumentException()
    {
        Assert.Throws<ArgumentException>(() => InternalUtils.SplitXLengths(new double[2, 0], new[] { 1, 2 }));
    }
}
