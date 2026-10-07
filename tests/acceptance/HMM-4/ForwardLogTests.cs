using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm4;

/// <summary>
/// Acceptance tests for <see cref="Hmmc.ForwardLog"/> (ticket HMM-4, criterion C2): every forward_log expectation of
/// oracle/golden/hmmc.json with the oracle's own inputs (including the -infinity log probabilities), -infinity flowing
/// through without NaN, and hand-written shape errors.
/// </summary>
public class ForwardLogTests
{
    private static List<JsonElement> Cases() => Golden.Load("hmmc").GetProperty("cases").Items().ToList();

    /// <summary>Every forward_log expectation of the golden file is reproduced: log_prob and the lattice, with -infinity
    /// entries compared as -infinity (never NaN).</summary>
    [Fact]
    public void Every_forward_log_expectation_of_the_golden_file_is_reproduced()
    {
        var cases = Cases();
        Assert.NotEmpty(cases);
        var checkedCases = 0;
        foreach (var c in cases)
        {
            if (!c.Has("forward_log")) continue;
            var name = c.GetProperty("name").AsString();
            var (logProb, fwdLattice) = Hmmc.ForwardLog(
                c.GetProperty("startprob").AsVector(),
                c.GetProperty("transmat").AsMatrix(),
                c.GetProperty("log_frameprob").AsMatrix());
            var expected = c.GetProperty("forward_log");
            Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb, $"{name}: log_prob");
            Close.Equal(expected.GetProperty("fwdlattice").AsMatrix(), fwdLattice, $"{name}: fwdlattice");
            checkedCases++;
        }
        Assert.True(checkedCases >= 10, $"expected at least 10 forward_log cases, saw {checkedCases}");
    }

    /// <summary>The criterion requires -infinity log probabilities without NaN: the zero-frame-probabilities case (zeros in
    /// the frameprob, i.e. -Infinity rows in the log frameprob) and the impossible-observation case (log_prob = -Infinity)
    /// must both come out NaN-free.</summary>
    [Fact]
    public void Minus_infinity_expectations_are_reproduced_without_NaN()
    {
        foreach (var name in new[] { "zero-frame-probabilities", "impossible-observation" })
        {
            var c = Assert.Single(Cases(), x => x.GetProperty("name").AsString() == name);
            var (logProb, fwdLattice) = Hmmc.ForwardLog(
                c.GetProperty("startprob").AsVector(),
                c.GetProperty("transmat").AsMatrix(),
                c.GetProperty("log_frameprob").AsMatrix());
            Assert.False(double.IsNaN(logProb), $"{name}: log_prob must not be NaN");
            Assert.All(fwdLattice.Cast<double>(), v => Assert.False(double.IsNaN(v), $"{name}: no NaN in the lattice"));
        }
    }

    /// <summary>The impossible observation gives exactly log_prob = -infinity: the last lattice row is all -infinity and
    /// LogSumExp of it must return -infinity as is.</summary>
    [Fact]
    public void The_impossible_observation_has_minus_infinity_log_prob()
    {
        var c = Assert.Single(Cases(), x => x.GetProperty("name").AsString() == "impossible-observation");
        var (logProb, _) = Hmmc.ForwardLog(
            c.GetProperty("startprob").AsVector(),
            c.GetProperty("transmat").AsMatrix(),
            c.GetProperty("log_frameprob").AsMatrix());
        Assert.Equal(double.NegativeInfinity, logProb);
    }

    /// <summary>Hand-written: -infinity flows through the whole pass without NaN. The identity transition matrix keeps the
    /// states separate, sample 1 is impossible under every state, so from there on the lattice is all -infinity (LogSumExp
    /// of an all -infinity row must return -infinity, not NaN) and the log probability is -infinity.</summary>
    [Fact]
    public void Minus_infinity_flows_through_the_pass_without_NaN()
    {
        var startprob = new double[] { 0.5, 0.5 };
        var transmat = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var logFrameprob = new double[,]
        {
            { 0.0, double.NegativeInfinity },
            { double.NegativeInfinity, double.NegativeInfinity },
            { 0.0, 0.0 }
        };
        var (logProb, fwdLattice) = Hmmc.ForwardLog(startprob, transmat, logFrameprob);
        var expected = new double[,]
        {
            { Math.Log(0.5), double.NegativeInfinity },
            { double.NegativeInfinity, double.NegativeInfinity },
            { double.NegativeInfinity, double.NegativeInfinity }
        };
        Close.Equal(expected, fwdLattice, "fwdlattice");
        Assert.Equal(double.NegativeInfinity, logProb);
        Assert.All(fwdLattice.Cast<double>(), v => Assert.False(double.IsNaN(v), "no NaN anywhere"));
    }

    // Hand-written shape errors (criterion C2): consistent base case is startprob (2,), transmat (2, 2), logFrameprob (3, 2).

    private static double[] Start2() => new double[] { 0.3, 0.7 };

    private static double[,] Trans2() => new double[,] { { 0.9, 0.1 }, { 0.2, 0.8 } };

    private static double[,] LogFrame3() => new double[,] { { -0.9, -0.5 }, { -0.7, -1.1 }, { -0.2, -1.8 } };

    private static void AssertShapeMismatch(double[] startprob, double[,] transmat, double[,] logFrameprob)
    {
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.ForwardLog(startprob, transmat, logFrameprob));
        Assert.Equal("shape mismatch", ex.Message);
    }

    /// <summary>A startprob whose length is not the number of components of the log frameprob throws ArgumentException.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    public void A_startprob_of_the_wrong_length_throws_shape_mismatch(int length)
    {
        AssertShapeMismatch(new double[length], Trans2(), LogFrame3());
    }

    /// <summary>A transmat that is not nc x nc (either dimension wrong, nc taken from the log frameprob) throws
    /// ArgumentException.</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 1)]
    [InlineData(3, 2)]
    [InlineData(2, 3)]
    [InlineData(3, 3)]
    public void A_transmat_that_is_not_nc_by_nc_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(Start2(), new double[rows, columns], LogFrame3());
    }

    /// <summary>At least one sample is required: an empty log frameprob (0 rows) throws ArgumentException.</summary>
    [Fact]
    public void An_empty_log_frameprob_with_no_samples_throws_shape_mismatch()
    {
        AssertShapeMismatch(Start2(), Trans2(), new double[0, 2]);
    }

    /// <summary>A log frameprob whose number of components differs from the length of the startprob (and the transmat)
    /// throws ArgumentException.</summary>
    [Fact]
    public void A_log_frameprob_whose_component_count_does_not_agree_throws_shape_mismatch()
    {
        AssertShapeMismatch(Start2(), Trans2(), new double[3, 3]);
    }
}
