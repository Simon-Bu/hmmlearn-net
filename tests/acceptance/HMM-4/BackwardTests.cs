using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm4;

/// <summary>
/// Acceptance tests for <see cref="Hmmc.BackwardScaling"/> and <see cref="Hmmc.BackwardLog"/> (ticket HMM-4, criterion C3):
/// every backward expectation of oracle/golden/hmmc.json with the oracle's own inputs (the scaling vector of the same case
/// for the scaling flavour), cross-consistency with the forward passes, and hand-written shape errors (including a scaling
/// vector of the wrong length).
/// </summary>
public class BackwardTests
{
    private static List<JsonElement> Cases() => Golden.Load("hmmc").GetProperty("cases").Items().ToList();

    /// <summary>Every backward_scaling expectation of the golden file is reproduced, using the oracle's own inputs and the
    /// scaling vector of the same case.</summary>
    [Fact]
    public void Every_backward_scaling_expectation_of_the_golden_file_is_reproduced()
    {
        var cases = Cases();
        Assert.NotEmpty(cases);
        var checkedCases = 0;
        foreach (var c in cases)
        {
            if (!c.Has("backward_scaling")) continue;
            var name = c.GetProperty("name").AsString();
            Assert.True(c.Has("forward_scaling"), $"case {name} must carry the scaling vector of the same case");
            var startprob = c.GetProperty("startprob").AsVector();
            var transmat = c.GetProperty("transmat").AsMatrix();
            var frameprob = c.GetProperty("frameprob").AsMatrix();
            var scaling = c.GetProperty("forward_scaling").GetProperty("scaling").AsVector();
            var bwdLattice = Hmmc.BackwardScaling(startprob, transmat, frameprob, scaling);
            Close.Equal(c.GetProperty("backward_scaling").GetProperty("bwdlattice").AsMatrix(), bwdLattice, $"{name}: bwdlattice");
            checkedCases++;
        }
        Assert.True(checkedCases >= 9, $"expected at least 9 backward_scaling cases, saw {checkedCases}");
    }

    /// <summary>Every backward_log expectation of the golden file is reproduced, with the oracle's own log frame
    /// probabilities (the -infinity rows included).</summary>
    [Fact]
    public void Every_backward_log_expectation_of_the_golden_file_is_reproduced()
    {
        var cases = Cases();
        Assert.NotEmpty(cases);
        var checkedCases = 0;
        foreach (var c in cases)
        {
            if (!c.Has("backward_log")) continue;
            var name = c.GetProperty("name").AsString();
            var bwdLattice = Hmmc.BackwardLog(
                c.GetProperty("startprob").AsVector(),
                c.GetProperty("transmat").AsMatrix(),
                c.GetProperty("log_frameprob").AsMatrix());
            Close.Equal(c.GetProperty("backward_log").GetProperty("bwdlattice").AsMatrix(), bwdLattice, $"{name}: bwdlattice");
            checkedCases++;
        }
        Assert.True(checkedCases >= 10, $"expected at least 10 backward_log cases, saw {checkedCases}");
    }

    /// <summary>Cross-consistency on the oracle's own inputs: for every sample t, the row-wise product of the scaled forward
    /// and backward lattices sums to the scaling factor of t (alpha_t * beta_t is the joint distribution over the state at t,
    /// and the two scaling conventions multiply to exactly that factor).</summary>
    [Fact]
    public void The_scaled_forward_and_backward_lattices_are_consistent_on_every_golden_case()
    {
        var checkedCases = 0;
        foreach (var c in Cases())
        {
            if (!c.Has("forward_scaling")) continue;
            var expected = c.GetProperty("forward_scaling");
            if (expected.Has("raises")) continue;
            var name = c.GetProperty("name").AsString();
            var startprob = c.GetProperty("startprob").AsVector();
            var transmat = c.GetProperty("transmat").AsMatrix();
            var frameprob = c.GetProperty("frameprob").AsMatrix();
            var (_, fwdLattice, scaling) = Hmmc.ForwardScaling(startprob, transmat, frameprob);
            var bwdLattice = Hmmc.BackwardScaling(startprob, transmat, frameprob, scaling);
            for (var t = 0; t < frameprob.GetLength(0); t++)
            {
                var sum = 0.0;
                for (var i = 0; i < frameprob.GetLength(1); i++) sum += fwdLattice[t, i] * bwdLattice[t, i];
                Close.Equal(scaling[t], sum, $"{name}: t={t}");
            }
            checkedCases++;
        }
        Assert.True(checkedCases >= 9, $"expected at least 9 consistent cases, saw {checkedCases}");
    }

    /// <summary>Cross-consistency in log space: for every sample t, the log-sum-exp of (fwd[t] + bwd[t]) is the log
    /// probability of the whole sequence (for the impossible observation it is -infinity on every row).</summary>
    [Fact]
    public void The_log_lattices_are_consistent_with_the_log_probability_on_every_golden_case()
    {
        var checkedCases = 0;
        foreach (var c in Cases())
        {
            if (!c.Has("forward_log") || !c.Has("backward_log")) continue;
            var name = c.GetProperty("name").AsString();
            var startprob = c.GetProperty("startprob").AsVector();
            var transmat = c.GetProperty("transmat").AsMatrix();
            var logFrameprob = c.GetProperty("log_frameprob").AsMatrix();
            var (logProb, fwdLattice) = Hmmc.ForwardLog(startprob, transmat, logFrameprob);
            var bwdLattice = Hmmc.BackwardLog(startprob, transmat, logFrameprob);
            for (var t = 0; t < logFrameprob.GetLength(0); t++)
            {
                var row = new double[logFrameprob.GetLength(1)];
                for (var i = 0; i < row.Length; i++) row[i] = fwdLattice[t, i] + bwdLattice[t, i];
                Close.Equal(logProb, Hmmc.LogSumExp(row), $"{name}: t={t}");
            }
            checkedCases++;
        }
        Assert.True(checkedCases >= 10, $"expected at least 10 consistent cases, saw {checkedCases}");
    }

    /// <summary>Hand-written inputs, last row only: the backward lattice is initialized to the last scaling factor in the
    /// scaling flavour and to 0 in the log flavour.</summary>
    [Fact]
    public void The_last_row_of_the_backward_lattices_is_the_documented_initialization()
    {
        var startprob = new double[] { 0.4, 0.6 };
        var transmat = new double[,] { { 0.7, 0.3 }, { 0.1, 0.9 } };
        var frameprob = new double[,] { { 0.25, 0.75 }, { 0.6, 0.4 }, { 0.9, 0.1 } };
        var (_, _, scaling) = Hmmc.ForwardScaling(startprob, transmat, frameprob);
        var bwdScaling = Hmmc.BackwardScaling(startprob, transmat, frameprob, scaling);
        Close.Equal(scaling[2], bwdScaling[2, 0], "bwd[2, 0] (scaling)");
        Close.Equal(scaling[2], bwdScaling[2, 1], "bwd[2, 1] (scaling)");

        var logFrameprob = new double[,] { { -1.4, -0.29 }, { -0.51, -0.92 }, { -0.105, -2.30 } };
        var bwdLog = Hmmc.BackwardLog(startprob, transmat, logFrameprob);
        Close.Equal(0.0, bwdLog[2, 0], "bwd[2, 0] (log)");
        Close.Equal(0.0, bwdLog[2, 1], "bwd[2, 1] (log)");
    }

    // Hand-written shape errors (criterion C3): consistent base case is startprob (2,), transmat (2, 2), frameprob (3, 2),
    // scaling (3,).

    private static double[] Start2() => new double[] { 0.3, 0.7 };

    private static double[,] Trans2() => new double[,] { { 0.9, 0.1 }, { 0.2, 0.8 } };

    private static double[,] Frame3() => new double[,] { { 0.4, 0.6 }, { 0.5, 0.5 }, { 0.1, 0.9 } };

    private static double[] Scaling3() => new double[] { 2.0, 2.5, 3.0 };

    private static double[,] LogFrame3() => new double[,] { { -0.9, -0.5 }, { -0.7, -1.1 }, { -0.2, -1.8 } };

    private static void AssertBackwardScalingShapeMismatch(double[] startprob, double[,] transmat, double[,] frameprob, double[] scaling)
    {
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.BackwardScaling(startprob, transmat, frameprob, scaling));
        Assert.Equal("shape mismatch", ex.Message);
    }

    private static void AssertBackwardLogShapeMismatch(double[] startprob, double[,] transmat, double[,] logFrameprob)
    {
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.BackwardLog(startprob, transmat, logFrameprob));
        Assert.Equal("shape mismatch", ex.Message);
    }

    /// <summary>A startprob whose length is not the number of components (it only takes part in the shape check) throws
    /// ArgumentException in both flavours.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    public void A_startprob_of_the_wrong_length_throws_shape_mismatch_in_both_flavours(int length)
    {
        AssertBackwardScalingShapeMismatch(new double[length], Trans2(), Frame3(), Scaling3());
        AssertBackwardLogShapeMismatch(new double[length], Trans2(), LogFrame3());
    }

    /// <summary>A transmat that is not nc x nc (either dimension wrong, nc taken from the frameprob) throws ArgumentException
    /// in both flavours.</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 1)]
    [InlineData(3, 2)]
    [InlineData(2, 3)]
    [InlineData(3, 3)]
    public void A_transmat_that_is_not_nc_by_nc_throws_shape_mismatch_in_both_flavours(int rows, int columns)
    {
        AssertBackwardScalingShapeMismatch(Start2(), new double[rows, columns], Frame3(), Scaling3());
        AssertBackwardLogShapeMismatch(Start2(), new double[rows, columns], LogFrame3());
    }

    /// <summary>A scaling vector of the wrong length (the criterion names this case explicitly; ns is 3 here) throws
    /// ArgumentException.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void A_scaling_vector_of_the_wrong_length_throws_shape_mismatch(int length)
    {
        AssertBackwardScalingShapeMismatch(Start2(), Trans2(), Frame3(), new double[length]);
    }

    /// <summary>At least one sample is required: an empty frameprob (0 rows) throws ArgumentException in both flavours.</summary>
    [Fact]
    public void An_empty_frameprob_with_no_samples_throws_shape_mismatch_in_both_flavours()
    {
        AssertBackwardScalingShapeMismatch(Start2(), Trans2(), new double[0, 2], new double[0]);
        AssertBackwardLogShapeMismatch(Start2(), Trans2(), new double[0, 2]);
    }

    /// <summary>A frameprob whose number of components differs from the length of the startprob (and the transmat) throws
    /// ArgumentException in both flavours.</summary>
    [Fact]
    public void A_frameprob_whose_component_count_does_not_agree_throws_shape_mismatch_in_both_flavours()
    {
        AssertBackwardScalingShapeMismatch(Start2(), Trans2(), new double[3, 3], Scaling3());
        AssertBackwardLogShapeMismatch(Start2(), Trans2(), new double[3, 3]);
    }
}
