using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm2;

/// <summary>
/// Acceptance tests for <see cref="InternalUtils.SplitXLengths"/> (ticket HMM-2, criterion C2): every case of
/// <c>split_lengths_cases</c> in oracle/golden/internal_utils.json, plus hand-checked edges (null lengths gives
/// the input itself, new matrices per sequence, a zero length, and ArgumentException for negative lengths or a
/// sum that is not the number of rows).
/// </summary>
public class SplitXLengthsTests
{
    /// <summary>Every golden case is reproduced: null lengths, one or several sequences, unit sequences, the wrong sum.</summary>
    [Fact]
    public void Every_split_lengths_case_of_the_golden_file_is_reproduced()
    {
        var cases = Golden.Load("internal_utils").GetProperty("split_lengths_cases").Items().ToList();
        Assert.NotEmpty(cases);
        foreach (var c in cases)
        {
            var name = c.GetProperty("name").AsString();
            var x = c.GetProperty("X").AsMatrix();
            var lengthsElement = c.GetProperty("lengths");
            var lengths = lengthsElement.ValueKind == JsonValueKind.Null ? null : lengthsElement.AsInts();
            if (c.Has("raises"))
            {
                var thrown = Record.Exception(() => InternalUtils.SplitXLengths(x, lengths));
                Assert.True(thrown?.GetType() == typeof(ArgumentException), $"{name}: expected ArgumentException, got {thrown?.GetType().Name ?? "no exception"}");
            }
            else
            {
                var expected = c.GetProperty("expected");
                var result = InternalUtils.SplitXLengths(x, lengths);
                Assert.Equal(expected.GetArrayLength(), result.Count);
                for (var i = 0; i < result.Count; i++)
                    Close.Equal(expected.EnumerateArray().ElementAt(i).AsMatrix(), result[i], $"{name} sequence {i}");
            }
        }
    }

    /// <summary>Null lengths give a list holding the input itself (the same reference, not a copy).</summary>
    [Fact]
    public void Split_x_lengths_with_null_lengths_returns_a_list_holding_the_input_itself()
    {
        var x = new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } };
        var result = InternalUtils.SplitXLengths(x, null);
        Assert.Single(result);
        Assert.Same(x, result[0]);
    }

    /// <summary>Each sequence is a new matrix with its rows, in order (hand-checked 4 rows, lengths 1 and 3);
    /// changing a sequence does not change the input.</summary>
    [Fact]
    public void Split_x_lengths_copies_the_rows_of_each_sequence_into_new_matrices()
    {
        var x = new double[,] { { 0.0, 1.0 }, { 2.0, 3.0 }, { 4.0, 5.0 }, { 6.0, 7.0 } };
        var result = InternalUtils.SplitXLengths(x, new[] { 1, 3 });
        Assert.Equal(2, result.Count);
        Close.Equal(new double[,] { { 0.0, 1.0 } }, result[0], "first sequence");
        Close.Equal(new double[,] { { 2.0, 3.0 }, { 4.0, 5.0 }, { 6.0, 7.0 } }, result[1], "second sequence");

        result[0][0, 0] = -1.0;
        result[1][2, 1] = -1.0;
        Close.Equal(new double[,] { { 0.0, 1.0 }, { 2.0, 3.0 }, { 4.0, 5.0 }, { 6.0, 7.0 } }, x, "input unchanged");
    }

    /// <summary>A zero length is not negative: it gives an empty sequence (hand-checked [0, 3] on 3 rows).</summary>
    [Fact]
    public void Split_x_lengths_a_zero_length_gives_an_empty_sequence()
    {
        var x = new double[,] { { 0.0, 1.0 }, { 2.0, 3.0 }, { 4.0, 5.0 } };
        var result = InternalUtils.SplitXLengths(x, new[] { 0, 3 });
        Assert.Equal(2, result.Count);
        Close.Equal(new double[0, 2], result[0], "empty sequence");
        Close.Equal(x, result[1], "rest of the rows");
    }

    /// <summary>A negative length throws ArgumentException, even when the lengths still sum to the number of rows.</summary>
    [Fact]
    public void Split_x_lengths_rejects_negative_lengths_even_when_they_sum_to_the_rows()
    {
        Assert.Throws<ArgumentException>(() => InternalUtils.SplitXLengths(new double[2, 1], new[] { -1, 3 }));
        Assert.Throws<ArgumentException>(() => InternalUtils.SplitXLengths(new double[4, 1], new[] { -1, 5 }));
    }

    /// <summary>Lengths that do not sum to the number of rows throw ArgumentException.</summary>
    [Theory]
    [InlineData(5, 2, 2)]  // sum 4 < 5
    [InlineData(5, 6)]     // sum 6 > 5
    [InlineData(4, 1, 1)]  // sum 2 < 4
    [InlineData(0, 1)]     // one sequence asked, no rows
    public void Split_x_lengths_rejects_lengths_that_do_not_sum_to_the_number_of_rows(int nRows, params int[] lengths)
    {
        Assert.Throws<ArgumentException>(() => InternalUtils.SplitXLengths(new double[nRows, 2], lengths));
    }

    /// <summary>Empty lengths sum to 0: they throw for a non-empty matrix and give no sequence for an empty one.</summary>
    [Fact]
    public void Split_x_lengths_empty_lengths_sum_to_zero()
    {
        Assert.Throws<ArgumentException>(() => InternalUtils.SplitXLengths(new double[3, 2], Array.Empty<int>()));
        Assert.Empty(InternalUtils.SplitXLengths(new double[0, 2], Array.Empty<int>()));
    }
}
