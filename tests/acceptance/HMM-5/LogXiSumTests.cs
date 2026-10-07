using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm5;

/// <summary>
/// Acceptance tests for <see cref="Hmmc.ComputeLogXiSum"/> (ticket HMM-5, criterion C2): every log_xi_sum expectation of
/// oracle/golden/hmmc.json with the oracle's own log lattices (including the -infinity entries and the single-sample case,
/// which gives a matrix of -infinity), the total mass of the exponentiated xi sum, and hand-written shape errors.
/// </summary>
public class LogXiSumTests
{
    private static List<JsonElement> Cases() => Golden.Load("hmmc").GetProperty("cases").Items().ToList();

    private static double[,] Compute(JsonElement c) => Hmmc.ComputeLogXiSum(
        c.GetProperty("forward_log").GetProperty("fwdlattice").AsMatrix(),
        c.GetProperty("transmat").AsMatrix(),
        c.GetProperty("backward_log").GetProperty("bwdlattice").AsMatrix(),
        c.GetProperty("log_frameprob").AsMatrix());

    /// <summary>Every log_xi_sum expectation of the golden file is reproduced with the oracle's own inputs: the log forward
    /// lattice, the (linear) transition matrix, the log backward lattice and the log frame probabilities of the same case
    /// (including the one-sample case, which gives a matrix of -infinity). Cases whose log probability is -infinity carry no
    /// log_xi_sum in the golden file and are skipped.</summary>
    [Fact]
    public void Every_log_xi_sum_expectation_of_the_golden_file_is_reproduced()
    {
        var cases = Cases();
        Assert.NotEmpty(cases);
        var checkedCases = 0;
        foreach (var c in cases)
        {
            if (!c.Has("log_xi_sum")) continue;
            var name = c.GetProperty("name").AsString();
            Close.Equal(c.GetProperty("log_xi_sum").AsMatrix(), Compute(c), $"{name}: log_xi_sum");
            checkedCases++;
        }
        Assert.True(checkedCases >= 9, $"expected at least 9 log_xi_sum cases, saw {checkedCases}");
    }

    /// <summary>The golden file carries -infinity entries in its log_xi_sum matrices (impossible transitions); they must come
    /// back exactly as -infinity (never NaN or a large finite number), and no NaN may appear anywhere in the results.</summary>
    [Fact]
    public void Minus_infinity_entries_of_the_golden_file_are_reproduced_without_NaN()
    {
        var infinityEntries = 0;
        foreach (var c in Cases())
        {
            if (!c.Has("log_xi_sum")) continue;
            var name = c.GetProperty("name").AsString();
            var expected = c.GetProperty("log_xi_sum").AsMatrix();
            var actual = Compute(c);
            var rows = expected.GetLength(0);
            var cols = expected.GetLength(1);
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    if (expected[i, j] == double.NegativeInfinity)
                    {
                        infinityEntries++;
                        Assert.True(
                            actual[i, j] == double.NegativeInfinity,
                            $"{name}: log_xi_sum[{i}, {j}] must be -infinity, got {actual[i, j]:R}");
                    }
                    Assert.False(double.IsNaN(actual[i, j]), $"{name}: no NaN at log_xi_sum[{i}, {j}]");
                }
            }
        }
        Assert.True(infinityEntries >= 1, "the golden file must contain at least one -infinity log_xi_sum entry");
    }

    /// <summary>A single sample has no transitions: the log sum starts at -infinity and no term is added, so every entry of
    /// the result is exactly -infinity (not merely very negative).</summary>
    [Fact]
    public void The_single_sample_case_gives_a_matrix_of_exactly_minus_infinity()
    {
        var c = Assert.Single(Cases(), x => x.GetProperty("name").AsString() == "one-sample");
        Assert.All(Compute(c).Cast<double>(), v => Assert.Equal(double.NegativeInfinity, v));
    }

    /// <summary>Exponentiating the entries of the log xi sum gives the probabilities that the transition (i, j) occurs at some
    /// time (an -infinity entry contributes 0); the joint mass over all states is one per time, so the exponentiated matrix
    /// sums to n_samples - 1, computed from the case's own log_frameprob shape.</summary>
    [Fact]
    public void The_exponentiated_matrix_of_every_golden_case_has_total_mass_of_the_number_of_transitions()
    {
        var checkedCases = 0;
        foreach (var c in Cases())
        {
            if (!c.Has("log_xi_sum")) continue;
            var name = c.GetProperty("name").AsString();
            var ns = c.GetProperty("log_frameprob").GetArrayLength();
            var total = 0.0;
            foreach (var v in Compute(c).Cast<double>()) total += Math.Exp(v);
            Close.Equal((double)(ns - 1), total, $"{name}: total mass");
            checkedCases++;
        }
        Assert.True(checkedCases >= 9, $"expected at least 9 total-mass cases, saw {checkedCases}");
    }

    // Hand-written shape errors (criterion C2): the consistent base is fwdlattice (3, 2), transmat (2, 2), bwdlattice (3, 2),
    // log_frameprob (3, 2).

    private static double[,] Fwd32() => new double[,] { { -2.0, -0.1 }, { -0.5, -1.2 }, { -0.3, -2.5 } };

    private static double[,] Trans2() => new double[,] { { 0.8, 0.2 }, { 0.3, 0.7 } };

    private static double[,] Bwd32() => new double[,] { { -1.0, -1.0 }, { -1.6, -0.2 }, { -0.7, -0.7 } };

    private static double[,] LogFrame32() => new double[,] { { -0.5, -0.9 }, { -0.7, -0.7 }, { -0.1, -2.3 } };

    private static void AssertShapeMismatch(double[,] fwdlattice, double[,] transmat, double[,] bwdlattice, double[,] logFrameprob)
    {
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.ComputeLogXiSum(fwdlattice, transmat, bwdlattice, logFrameprob));
        Assert.Equal("shape mismatch", ex.Message);
    }

    /// <summary>A fwdlattice whose shape is not (n_samples, nc) of the log_frameprob throws ArgumentException "shape mismatch".</summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(4, 2)]
    [InlineData(3, 3)]
    public void A_fwdlattice_of_the_wrong_shape_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(new double[rows, columns], Trans2(), Bwd32(), LogFrame32());
    }

    /// <summary>A bwdlattice whose shape is not (n_samples, nc) of the log_frameprob throws ArgumentException "shape mismatch".</summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(4, 2)]
    [InlineData(3, 3)]
    public void A_bwdlattice_of_the_wrong_shape_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(Fwd32(), Trans2(), new double[rows, columns], LogFrame32());
    }

    /// <summary>A transmat that is not nc x nc (nc taken from the log_frameprob) throws ArgumentException "shape mismatch".</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 1)]
    [InlineData(3, 2)]
    [InlineData(2, 3)]
    [InlineData(3, 3)]
    public void A_transmat_that_is_not_nc_by_nc_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(Fwd32(), new double[rows, columns], Bwd32(), LogFrame32());
    }

    /// <summary>A log_frameprob whose shape is not (n_samples, nc) of the fwdlattice throws ArgumentException "shape mismatch".</summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(4, 2)]
    [InlineData(3, 3)]
    public void A_log_frameprob_of_the_wrong_shape_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(Fwd32(), Trans2(), Bwd32(), new double[rows, columns]);
    }
}
