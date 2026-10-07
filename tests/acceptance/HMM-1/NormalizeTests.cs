using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm1;

/// <summary>
/// Acceptance tests for <see cref="Utils.Normalize"/> (ticket HMM-1, criterion C1): every case of
/// <c>normalize_cases</c> in oracle/golden/utils.json, plus hand-checked edges (in-place, zero sums, bad axis).
/// </summary>
public class NormalizeTests
{
    /// <summary>Every golden case (vector or matrix, axis null, 0 or 1) is reproduced, in place, on the very array that was passed.</summary>
    [Fact]
    public void Every_normalize_case_of_the_golden_file_is_reproduced_in_place()
    {
        var cases = Golden.Load("utils").GetProperty("normalize_cases").Items().ToList();
        Assert.NotEmpty(cases);
        foreach (var c in cases)
        {
            var name = c.GetProperty("name").AsString();
            var axis = AxisOf(c);
            var input = c.GetProperty("input");
            if (input.EnumerateArray().First().ValueKind != JsonValueKind.Array)
            {
                Assert.True(axis is null, $"case {name}: a vector case must have a null axis");
                var vector = input.AsVector();
                Utils.Normalize(vector);
                Close.Equal(c.GetProperty("expected").AsVector(), vector, name);
            }
            else
            {
                var matrix = input.AsMatrix();
                Utils.Normalize(matrix, axis);
                Close.Equal(c.GetProperty("expected").AsMatrix(), matrix, name);
            }
        }
    }

    /// <summary>The array that is passed is the one that comes out normalized (no copy), and the total sum becomes 1.</summary>
    [Fact]
    public void Normalize_vector_divides_by_the_total_sum_in_place()
    {
        var a = new double[] { 1.0, 2.0, 3.0 };
        var reference = a;
        Utils.Normalize(a);
        Close.Equal(new double[] { 1.0 / 6, 2.0 / 6, 3.0 / 6 }, reference, "in-place vector");
        Close.Equal(1.0, reference.Sum(), "total sum");
    }

    /// <summary>A vector whose sum is 0 (all zeros, or cancelling values) stays unchanged and produces no NaN.</summary>
    [Fact]
    public void Normalize_vector_zero_sum_leaves_the_values_and_no_nan_appears()
    {
        var zeros = new double[] { 0.0, 0.0, 0.0 };
        Utils.Normalize(zeros);
        Close.Equal(new double[] { 0.0, 0.0, 0.0 }, zeros, "all zeros");
        Assert.All(zeros, v => Assert.False(double.IsNaN(v), "no NaN for a zero sum"));

        var cancelling = new double[] { 1.0, -1.0 };
        Utils.Normalize(cancelling);
        Close.Equal(new double[] { 1.0, -1.0 }, cancelling, "values cancelling to a zero sum");
    }

    /// <summary>Axis null (explicit or by default) divides the whole matrix by the sum of all its elements.</summary>
    [Fact]
    public void Normalize_matrix_whole_divides_by_the_total_sum_explicitly_and_by_default()
    {
        var whole = new double[,] { { 2.0, 2.0 }, { 1.0, 3.0 } };
        Utils.Normalize(whole, null);
        Close.Equal(new double[,] { { 0.25, 0.25 }, { 0.125, 0.375 } }, whole, "axis null");

        var byDefault = new double[,] { { 2.0, 2.0 }, { 1.0, 3.0 } };
        Utils.Normalize(byDefault);
        Close.Equal(new double[,] { { 0.25, 0.25 }, { 0.125, 0.375 } }, byDefault, "default axis");
    }

    /// <summary>Axis 0 normalizes each column to sum to 1, axis 1 each row (hand-checked 2x2).</summary>
    [Fact]
    public void Normalize_matrix_axis_zero_and_axis_one_normalize_columns_and_rows()
    {
        var byColumn = new double[,] { { 2.0, 2.0 }, { 1.0, 3.0 } };
        Utils.Normalize(byColumn, 0);
        Close.Equal(new double[,] { { 2.0 / 3, 2.0 / 5 }, { 1.0 / 3, 3.0 / 5 } }, byColumn, "axis 0");
        Close.Equal(1.0, byColumn[0, 0] + byColumn[1, 0], "column 0 sum");
        Close.Equal(1.0, byColumn[0, 1] + byColumn[1, 1], "column 1 sum");

        var byRow = new double[,] { { 2.0, 2.0 }, { 1.0, 3.0 } };
        Utils.Normalize(byRow, 1);
        Close.Equal(new double[,] { { 0.5, 0.5 }, { 0.25, 0.75 } }, byRow, "axis 1");
        Close.Equal(1.0, byRow[0, 0] + byRow[0, 1], "row 0 sum");
        Close.Equal(1.0, byRow[1, 0] + byRow[1, 1], "row 1 sum");
    }

    /// <summary>A zero sum is treated as 1 on every axis (the port guards axis 0 too, where the Python original would give NaN):
    /// the zeros stay zeros, the cancelling row stays, and no NaN appears.</summary>
    [Fact]
    public void Normalize_matrix_a_zero_sum_keeps_the_zeros_on_both_axes()
    {
        var byRow = new double[,] { { 0.0, 0.0 }, { 1.0, -1.0 }, { 2.0, 2.0 } };
        Utils.Normalize(byRow, 1);
        Close.Equal(new double[,] { { 0.0, 0.0 }, { 1.0, -1.0 }, { 0.5, 0.5 } }, byRow, "axis 1");
        Assert.All(byRow.Cast<double>(), v => Assert.False(double.IsNaN(v), "no NaN for zero row sums"));

        var byColumn = new double[,] { { 0.0, 1.0 }, { 0.0, 1.0 } };
        Utils.Normalize(byColumn, 0);
        Close.Equal(new double[,] { { 0.0, 0.5 }, { 0.0, 0.5 } }, byColumn, "axis 0: zero column");
        Assert.All(byColumn.Cast<double>(), v => Assert.False(double.IsNaN(v), "no NaN for zero column sums"));

        var whole = new double[,] { { 0.0, 0.0 }, { 0.0, 0.0 } };
        Utils.Normalize(whole);
        Close.Equal(new double[,] { { 0.0, 0.0 }, { 0.0, 0.0 } }, whole, "axis null: all zeros");
    }

    /// <summary>Any axis other than null, 0 or 1 throws ArgumentOutOfRangeException and leaves the matrix untouched.</summary>
    [Theory]
    [InlineData(-1)]
    [InlineData(2)]
    [InlineData(5)]
    public void Normalize_matrix_a_bad_axis_throws_argument_out_of_range(int axis)
    {
        var a = new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } };
        Assert.Throws<ArgumentOutOfRangeException>(() => Utils.Normalize(a, axis));
        Close.Equal(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }, a, "unchanged on exception");
    }

    private static int? AxisOf(JsonElement c)
    {
        var axis = c.GetProperty("axis");
        return axis.ValueKind == JsonValueKind.Null ? null : axis.GetInt32();
    }
}
