using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm5;

/// <summary>
/// Acceptance tests for <see cref="Hmmc.ComputeScalingXiSum"/> (ticket HMM-5, criterion C1): every scaling_xi_sum expectation of
/// oracle/golden/hmmc.json with the oracle's own lattices (including the single-sample case, which gives zeros), the total
/// mass of the xi sum, and hand-written shape errors.
/// </summary>
public class ScalingXiSumTests
{
    private static List<JsonElement> Cases() => Golden.Load("hmmc").GetProperty("cases").Items().ToList();

    private static double[,] Compute(JsonElement c) => Hmmc.ComputeScalingXiSum(
        c.GetProperty("forward_scaling").GetProperty("fwdlattice").AsMatrix(),
        c.GetProperty("transmat").AsMatrix(),
        c.GetProperty("backward_scaling").GetProperty("bwdlattice").AsMatrix(),
        c.GetProperty("frameprob").AsMatrix());

    /// <summary>Every scaling_xi_sum expectation of the golden file is reproduced with the oracle's own inputs: the scaled
    /// forward lattice, the transition matrix, the scaled backward lattice and the frame probabilities of the same case
    /// (including the one-sample case, which gives a matrix of zeros). Cases whose scaling forward pass raised carry no
    /// scaling_xi_sum in the golden file and are skipped.</summary>
    [Fact]
    public void Every_scaling_xi_sum_expectation_of_the_golden_file_is_reproduced()
    {
        var cases = Cases();
        Assert.NotEmpty(cases);
        var checkedCases = 0;
        foreach (var c in cases)
        {
            if (!c.Has("scaling_xi_sum")) continue;
            var name = c.GetProperty("name").AsString();
            Close.Equal(c.GetProperty("scaling_xi_sum").AsMatrix(), Compute(c), $"{name}: scaling_xi_sum");
            checkedCases++;
        }
        Assert.True(checkedCases >= 9, $"expected at least 9 scaling_xi_sum cases, saw {checkedCases}");
    }

    /// <summary>A single sample has no transitions: the sum over the empty set of times is exactly the neutral element, so
    /// every entry of the result is exactly 0.0 (not merely close to zero).</summary>
    [Fact]
    public void The_single_sample_case_gives_a_matrix_of_exactly_zeros()
    {
        var c = Assert.Single(Cases(), x => x.GetProperty("name").AsString() == "one-sample");
        Assert.All(Compute(c).Cast<double>(), v => Assert.Equal(0.0, v));
    }

    /// <summary>The entries of the xi sum are the posterior probabilities that the transition (i, j) occurs at some time; the
    /// joint mass over all states is one per time, so the whole matrix sums to n_samples - 1, computed from the case's own
    /// frameprob shape (for the one-sample case that total is exactly 0).</summary>
    [Fact]
    public void The_matrix_of_every_golden_case_has_total_mass_of_the_number_of_transitions()
    {
        var checkedCases = 0;
        foreach (var c in Cases())
        {
            if (!c.Has("scaling_xi_sum")) continue;
            var name = c.GetProperty("name").AsString();
            var ns = c.GetProperty("frameprob").GetArrayLength();
            var total = 0.0;
            foreach (var v in Compute(c).Cast<double>()) total += v;
            Close.Equal((double)(ns - 1), total, $"{name}: total mass");
            checkedCases++;
        }
        Assert.True(checkedCases >= 9, $"expected at least 9 total-mass cases, saw {checkedCases}");
    }

    // Hand-written shape errors (criterion C1): the consistent base is fwdlattice (3, 2), transmat (2, 2), bwdlattice (3, 2),
    // frameprob (3, 2).

    private static double[,] Fwd32() => new double[,] { { 0.1, 0.9 }, { 0.4, 0.6 }, { 0.7, 0.3 } };

    private static double[,] Trans2() => new double[,] { { 0.8, 0.2 }, { 0.3, 0.7 } };

    private static double[,] Bwd32() => new double[,] { { 0.5, 0.5 }, { 0.2, 0.8 }, { 1.0, 1.0 } };

    private static double[,] Frame32() => new double[,] { { 0.6, 0.4 }, { 0.5, 0.5 }, { 0.9, 0.1 } };

    private static void AssertShapeMismatch(double[,] fwdlattice, double[,] transmat, double[,] bwdlattice, double[,] frameprob)
    {
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.ComputeScalingXiSum(fwdlattice, transmat, bwdlattice, frameprob));
        Assert.Equal("shape mismatch", ex.Message);
    }

    /// <summary>A fwdlattice whose shape is not (n_samples, nc) of the frameprob throws ArgumentException "shape mismatch".</summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(4, 2)]
    [InlineData(3, 3)]
    public void A_fwdlattice_of_the_wrong_shape_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(new double[rows, columns], Trans2(), Bwd32(), Frame32());
    }

    /// <summary>A bwdlattice whose shape is not (n_samples, nc) of the frameprob throws ArgumentException "shape mismatch".</summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(4, 2)]
    [InlineData(3, 3)]
    public void A_bwdlattice_of_the_wrong_shape_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(Fwd32(), Trans2(), new double[rows, columns], Frame32());
    }

    /// <summary>A transmat that is not nc x nc (nc taken from the frameprob) throws ArgumentException "shape mismatch".</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 1)]
    [InlineData(3, 2)]
    [InlineData(2, 3)]
    [InlineData(3, 3)]
    public void A_transmat_that_is_not_nc_by_nc_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(Fwd32(), new double[rows, columns], Bwd32(), Frame32());
    }

    /// <summary>A frameprob whose shape is not (n_samples, nc) of the fwdlattice throws ArgumentException "shape mismatch".</summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(4, 2)]
    [InlineData(3, 3)]
    public void A_frameprob_of_the_wrong_shape_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(Fwd32(), Trans2(), Bwd32(), new double[rows, columns]);
    }
}
