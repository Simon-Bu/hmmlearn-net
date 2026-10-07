using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm2;

/// <summary>
/// Acceptance tests for <see cref="InternalUtils.LogDet"/> (ticket HMM-2, criterion C1): every case of
/// <c>logdet_cases</c> in oracle/golden/internal_utils.json, plus hand-checked edges (known determinants,
/// NaN for a negative determinant, -infinity for a singular matrix, non-square throws ArgumentException).
/// </summary>
public class LogDetTests
{
    /// <summary>Every golden case is reproduced (positive determinants, the negative determinant, the singular matrix).</summary>
    [Fact]
    public void Every_logdet_case_of_the_golden_file_is_reproduced()
    {
        var cases = Golden.Load("internal_utils").GetProperty("logdet_cases").Items().ToList();
        Assert.NotEmpty(cases);
        foreach (var c in cases)
        {
            var name = c.GetProperty("name").AsString();
            var matrix = c.GetProperty("matrix").AsMatrix();
            Close.Equal(c.GetProperty("expected").AsDouble(), InternalUtils.LogDet(matrix), name);
        }
    }

    /// <summary>Identity and triangular matrices with a known product of the diagonal: 0, ln 4, ln 8 (hand-checked).</summary>
    [Fact]
    public void Log_det_of_diagonal_and_triangular_matrices_is_the_log_of_the_product_of_the_diagonal()
    {
        Close.Equal(0.0, InternalUtils.LogDet(new double[,] { { 1.0 } }), "1x1 identity");
        Close.Equal(Math.Log(4.0), InternalUtils.LogDet(new double[,] { { 2.0, 0.0 }, { 0.0, 2.0 } }), "diag(2, 2)");
        Close.Equal(Math.Log(8.0), InternalUtils.LogDet(new double[,]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 2.0, 0.0 },
            { 0.0, 0.0, 4.0 }
        }), "diag(1, 2, 4)");
        Close.Equal(Math.Log(8.0), InternalUtils.LogDet(new double[,]
        {
            { 1.0, 3.0, 5.0 },
            { 0.0, 2.0, 7.0 },
            { 0.0, 0.0, 4.0 }
        }), "upper triangular, diag(1, 2, 4)");
    }

    /// <summary>A matrix with determinant exactly 1 (2*1 - 1*1) has log det 0 (hand-checked).</summary>
    [Fact]
    public void Log_det_of_a_matrix_with_unit_determinant_is_zero()
    {
        Close.Equal(0.0, InternalUtils.LogDet(new double[,] { { 2.0, 1.0 }, { 1.0, 1.0 } }), "det = 2*1 - 1*1 = 1");
    }

    /// <summary>A negative determinant gives NaN, like hmmlearn (hand-checked det -1 matrices).</summary>
    [Fact]
    public void Log_det_of_a_negative_determinant_gives_nan()
    {
        Close.Equal(double.NaN, InternalUtils.LogDet(new double[,] { { -1.0, 0.0 }, { 0.0, 1.0 } }), "det = -1");
        Close.Equal(double.NaN, InternalUtils.LogDet(new double[,] { { 0.0, 1.0 }, { 1.0, 0.0 } }), "det = -1 (rows swapped)");
    }

    /// <summary>A singular matrix (determinant 0) gives -infinity (hand-checked 1x1 and a rank-1 2x2).</summary>
    [Fact]
    public void Log_det_of_a_singular_matrix_gives_minus_infinity()
    {
        Close.Equal(double.NegativeInfinity, InternalUtils.LogDet(new double[,] { { 0.0 } }), "1x1 zero");
        Close.Equal(double.NegativeInfinity, InternalUtils.LogDet(new double[,] { { 1.0, 2.0 }, { 2.0, 4.0 } }), "rank-1 2x2");
    }

    /// <summary>A matrix that is not square throws ArgumentException.</summary>
    [Theory]
    [InlineData(2, 3)]
    [InlineData(3, 2)]
    [InlineData(1, 2)]
    public void Log_det_of_a_non_square_matrix_throws_argument(int rows, int columns)
    {
        var matrix = new double[rows, columns];
        Assert.Throws<ArgumentException>(() => InternalUtils.LogDet(matrix));
    }
}
