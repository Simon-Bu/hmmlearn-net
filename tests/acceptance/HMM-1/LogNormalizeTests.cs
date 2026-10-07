using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm1;

/// <summary>
/// Acceptance tests for <see cref="Utils.LogNormalize"/> (ticket HMM-1, criterion C2): every case of
/// <c>log_normalize_cases</c> in oracle/golden/utils.json, plus hand-checked edges (stability, in place,
/// length-1 axes, -infinity, an infinite maximum, bad axis).
/// </summary>
public class LogNormalizeTests
{
    /// <summary>Every golden case (vector or matrix, axis null, 0 or 1, including the -infinity cases) is reproduced,
    /// in place, on the very array that was passed.</summary>
    [Fact]
    public void Every_log_normalize_case_of_the_golden_file_is_reproduced_in_place()
    {
        var cases = Golden.Load("utils").GetProperty("log_normalize_cases").Items().ToList();
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
                Utils.LogNormalize(vector);
                Close.Equal(c.GetProperty("expected").AsVector(), vector, name);
            }
            else
            {
                var matrix = input.AsMatrix();
                Utils.LogNormalize(matrix, axis);
                Close.Equal(c.GetProperty("expected").AsMatrix(), matrix, name);
            }
        }
    }

    /// <summary>The array that is passed is the one that comes out normalized (no copy).</summary>
    [Fact]
    public void LogNormalize_vector_subtracts_the_logsumexp_in_place()
    {
        var a = new double[] { 0.0, 0.0 };
        var reference = a;
        Utils.LogNormalize(a);
        Close.Equal(new double[] { -Math.Log(2.0), -Math.Log(2.0) }, reference, "in-place vector");
        Close.Equal(1.0, reference.Sum(v => Math.Exp(v)), "sum of exp");
    }

    /// <summary>The log-sum-exp is computed stably: the maximum is factored out, so neither overflow (large values)
    /// nor underflow (small values) changes the result.</summary>
    [Fact]
    public void LogNormalize_vector_is_stable_for_extreme_values()
    {
        var large = new double[] { 1000.0, 1000.0 };
        Utils.LogNormalize(large);
        Close.Equal(new double[] { 1000.0 - Math.Log(2.0), 1000.0 - Math.Log(2.0) }, large, "large values");

        var small = new double[] { -1000.0, -1000.0 };
        Utils.LogNormalize(small);
        Close.Equal(new double[] { -Math.Log(2.0), -Math.Log(2.0) }, small, "small values");
    }

    /// <summary>The vector overload has no axis, so the length-1 rule of the matrix overload does not apply: the
    /// log-sum-exp of one element is the element itself (finite: 0; a lone -infinity: -infinity - -infinity = NaN).</summary>
    [Fact]
    public void LogNormalize_vector_of_one_element_and_of_negative_infinity()
    {
        var one = new double[] { -5.0 };
        Utils.LogNormalize(one);
        Close.Equal(new double[] { 0.0 }, one, "single finite element");

        var minusInfinity = new double[] { double.NegativeInfinity };
        Utils.LogNormalize(minusInfinity);
        Assert.True(double.IsNaN(minusInfinity[0]), "a lone -infinity gives NaN, not 0");
    }

    /// <summary>Axis 1 subtracts the log-sum-exp of each row, axis 0 the one of each column, and the default axis
    /// the one of the whole matrix (hand-checked 2x2 with values 0..3).</summary>
    [Fact]
    public void LogNormalize_matrix_axis_zero_and_axis_one_subtract_the_column_and_row_logsumexp()
    {
        var rowSpread = Math.Log(1.0 + Math.E); // LSE of [0,1] = LSE of [2,3] - 2
        var byRow = new double[,] { { 0.0, 1.0 }, { 2.0, 3.0 } };
        Utils.LogNormalize(byRow, 1);
        Close.Equal(new double[,] { { -rowSpread, 1.0 - rowSpread }, { -rowSpread, 1.0 - rowSpread } }, byRow, "axis 1");
        Close.Equal(1.0, Math.Exp(byRow[0, 0]) + Math.Exp(byRow[0, 1]), "row 0 sum of exp");
        Close.Equal(1.0, Math.Exp(byRow[1, 0]) + Math.Exp(byRow[1, 1]), "row 1 sum of exp");

        var columnSpread = Math.Log(1.0 + Math.Exp(-2.0)); // both columns spread by 2
        var byColumn = new double[,] { { 0.0, 1.0 }, { 2.0, 3.0 } };
        Utils.LogNormalize(byColumn, 0);
        Close.Equal(new double[,]
        {
            { -2.0 - columnSpread, -2.0 - columnSpread },
            { -columnSpread, -columnSpread },
        }, byColumn, "axis 0");

        var wholeLse = 3.0 + Math.Log(1.0 + Math.Exp(-1.0) + Math.Exp(-2.0) + Math.Exp(-3.0)); // factored out by the max
        var whole = new double[,] { { 0.0, 1.0 }, { 2.0, 3.0 } };
        Utils.LogNormalize(whole);
        Close.Equal(new double[,]
        {
            { -wholeLse, 1.0 - wholeLse },
            { 2.0 - wholeLse, 3.0 - wholeLse },
        }, whole, "default axis (whole matrix)");
        Close.Equal(1.0, whole.Cast<double>().Sum(v => Math.Exp(v)), "sum of exp");
    }

    /// <summary>If the chosen axis has length 1, every element becomes 0 (a single row for axis 0, a single column for axis 1).</summary>
    [Fact]
    public void LogNormalize_matrix_a_single_element_axis_sets_everything_to_zero()
    {
        var oneColumn = new double[,] { { -3.0 }, { 4.5 } };
        Utils.LogNormalize(oneColumn, 1);
        Close.Equal(new double[,] { { 0.0 }, { 0.0 } }, oneColumn, "axis 1 of a (2,1) matrix");

        var oneRow = new double[,] { { -2.0, 7.0 } };
        Utils.LogNormalize(oneRow, 0);
        Close.Equal(new double[,] { { 0.0, 0.0 } }, oneRow, "axis 0 of a (1,2) matrix");
    }

    /// <summary>-infinity entries: in a row (or column) with a finite maximum they stay -infinity; a whole row or
    /// column of -infinity has log-sum-exp -infinity, and -infinity - -infinity gives NaN.</summary>
    [Fact]
    public void LogNormalize_matrix_minus_infinity_entries_follow_the_documented_rules()
    {
        var rows = new double[,] { { double.NegativeInfinity, 0.0 }, { 0.0, -1.0 } };
        Utils.LogNormalize(rows, 1);
        var rowLse = Math.Log(1.0 + Math.Exp(-1.0)); // LSE of [0, -1]
        Close.Equal(new double[,] { { double.NegativeInfinity, -rowLse }, { -rowLse, -1.0 - rowLse } }, rows, "row with -infinity");

        var columns = new double[,] { { double.NegativeInfinity, 0.0 }, { double.NegativeInfinity, -1.0 } };
        Utils.LogNormalize(columns, 0);
        Close.Equal(new double[,] { { double.NaN, -rowLse }, { double.NaN, -1.0 - rowLse } }, columns, "column of -infinity gives NaN");
    }

    /// <summary>If the maximum is infinite it is the log-sum-exp itself: +infinity - +infinity is NaN, and finite - +infinity is -infinity.</summary>
    [Fact]
    public void LogNormalize_an_infinite_maximum_is_the_logsumexp_itself()
    {
        var vector = new double[] { double.PositiveInfinity, 5.0 };
        Utils.LogNormalize(vector);
        Close.Equal(new double[] { double.NaN, double.NegativeInfinity }, vector, "vector with +infinity");

        var matrix = new double[,] { { 5.0, double.PositiveInfinity }, { 1.0, 2.0 } };
        Utils.LogNormalize(matrix);
        Close.Equal(new double[,] { { double.NegativeInfinity, double.NaN }, { double.NegativeInfinity, double.NegativeInfinity } }, matrix, "matrix with +infinity");
    }

    /// <summary>Any axis other than null, 0 or 1 throws ArgumentOutOfRangeException and leaves the matrix untouched.</summary>
    [Theory]
    [InlineData(-1)]
    [InlineData(2)]
    [InlineData(5)]
    public void LogNormalize_matrix_a_bad_axis_throws_argument_out_of_range(int axis)
    {
        var a = new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } };
        Assert.Throws<ArgumentOutOfRangeException>(() => Utils.LogNormalize(a, axis));
        Close.Equal(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }, a, "unchanged on exception");
    }

    private static int? AxisOf(JsonElement c)
    {
        var axis = c.GetProperty("axis");
        return axis.ValueKind == JsonValueKind.Null ? null : axis.GetInt32();
    }
}
