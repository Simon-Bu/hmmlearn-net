using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm4;

/// <summary>
/// Acceptance tests for <see cref="Hmmc.ForwardScaling"/> (ticket HMM-4, criterion C1): every forward_scaling expectation of
/// oracle/golden/hmmc.json with the oracle's own inputs (the Wikipedia example, zeros in the transition matrix and a single
/// sample are among them), the underflow exception for the cases marked raises, the documented invariants, and
/// hand-written shape errors.
/// </summary>
public class ForwardScalingTests
{
    private static readonly string UnderflowMessage =
        "forward pass failed with underflow; consider using implementation='log' instead";

    private static List<JsonElement> Cases() => Golden.Load("hmmc").GetProperty("cases").Items().ToList();

    /// <summary>Every forward_scaling expectation of the golden file is reproduced: log_prob, the scaled lattice and the
    /// scaling vector; the cases marked raises throw the documented ArithmeticException.</summary>
    [Fact]
    public void Every_forward_scaling_expectation_of_the_golden_file_is_reproduced()
    {
        var cases = Cases();
        Assert.NotEmpty(cases);
        var numericCases = 0;
        var raisingCases = 0;
        foreach (var c in cases)
        {
            if (!c.Has("forward_scaling")) continue;
            var name = c.GetProperty("name").AsString();
            var startprob = c.GetProperty("startprob").AsVector();
            var transmat = c.GetProperty("transmat").AsMatrix();
            var frameprob = c.GetProperty("frameprob").AsMatrix();
            var expected = c.GetProperty("forward_scaling");
            if (expected.Has("raises"))
            {
                Assert.True(expected.GetProperty("raises").AsBool(), $"case {name} must be marked as raising");
                var ex = Assert.Throws<ArithmeticException>(() => Hmmc.ForwardScaling(startprob, transmat, frameprob));
                Assert.Equal(UnderflowMessage, ex.Message);
                raisingCases++;
                continue;
            }
            var (logProb, fwdLattice, scaling) = Hmmc.ForwardScaling(startprob, transmat, frameprob);
            Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb, $"{name}: log_prob");
            Close.Equal(expected.GetProperty("fwdlattice").AsMatrix(), fwdLattice, $"{name}: fwdlattice");
            Close.Equal(expected.GetProperty("scaling").AsVector(), scaling, $"{name}: scaling");
            numericCases++;
        }
        Assert.True(numericCases >= 9, $"expected at least 9 numeric forward_scaling cases, saw {numericCases}");
        Assert.True(raisingCases >= 1, $"expected at least 1 raising forward_scaling case, saw {raisingCases}");
    }

    /// <summary>The criterion names the Wikipedia example, zeros in the transition matrix and a single sample: all three are
    /// golden cases, and each one is reproduced end to end (log_prob, lattice and scaling).</summary>
    [Theory]
    [InlineData("wikipedia")]
    [InlineData("left-to-right-with-zeros")]
    [InlineData("one-sample")]
    public void The_cases_named_in_the_criterion_are_reproduced(string name)
    {
        var c = Assert.Single(Cases(), x => x.GetProperty("name").AsString() == name);
        var expected = c.GetProperty("forward_scaling");
        Assert.True(expected.Has("log_prob"), $"case {name} must carry a numeric forward_scaling expectation");
        var (logProb, fwdLattice, scaling) = Hmmc.ForwardScaling(
            c.GetProperty("startprob").AsVector(),
            c.GetProperty("transmat").AsMatrix(),
            c.GetProperty("frameprob").AsMatrix());
        Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb, $"{name}: log_prob");
        Close.Equal(expected.GetProperty("fwdlattice").AsMatrix(), fwdLattice, $"{name}: fwdlattice");
        Close.Equal(expected.GetProperty("scaling").AsVector(), scaling, $"{name}: scaling");
    }

    /// <summary>The impossible observation (a sample with probability 0 under every state) throws the documented
    /// ArithmeticException, with the exact documented message.</summary>
    [Fact]
    public void The_impossible_observation_case_throws_the_documented_underflow_error()
    {
        var c = Assert.Single(Cases(), x => x.GetProperty("name").AsString() == "impossible-observation");
        Assert.True(c.GetProperty("forward_scaling").Has("raises"), "impossible-observation must be marked as raising");
        var ex = Assert.Throws<ArithmeticException>(() => Hmmc.ForwardScaling(
            c.GetProperty("startprob").AsVector(),
            c.GetProperty("transmat").AsMatrix(),
            c.GetProperty("frameprob").AsMatrix()));
        Assert.Equal(UnderflowMessage, ex.Message);
    }

    /// <summary>Hand-written inputs: every row of the scaled lattice sums to 1 (the row is rescaled by 1 / (row sum) at every
    /// step), and the returned log probability is minus the sum of the logs of the scales.</summary>
    [Fact]
    public void Rows_of_the_scaled_lattice_sum_to_one_and_log_prob_is_minus_the_sum_of_the_logs_of_the_scales()
    {
        var startprob = new double[] { 0.2, 0.3, 0.5 };
        var transmat = new double[,]
        {
            { 0.1, 0.6, 0.3 },
            { 0.4, 0.2, 0.4 },
            { 0.9, 0.05, 0.05 }
        };
        var frameprob = new double[,]
        {
            { 0.25, 0.50, 0.25 },
            { 0.10, 0.80, 0.10 },
            { 0.70, 0.20, 0.10 },
            { 0.40, 0.40, 0.20 }
        };
        var (logProb, fwdLattice, scaling) = Hmmc.ForwardScaling(startprob, transmat, frameprob);
        for (var t = 0; t < frameprob.GetLength(0); t++)
        {
            var sum = 0.0;
            for (var i = 0; i < startprob.Length; i++) sum += fwdLattice[t, i];
            Close.Equal(1.0, sum, $"row {t} of the scaled lattice");
        }
        Close.Equal(-scaling.Sum(s => Math.Log(s)), logProb, "log_prob = -sum(log(scale))");
    }

    // Hand-written shape errors (criterion C1): consistent base case is startprob (2,), transmat (2, 2), frameprob (3, 2).

    private static double[] Start2() => new double[] { 0.3, 0.7 };

    private static double[,] Trans2() => new double[,] { { 0.9, 0.1 }, { 0.2, 0.8 } };

    private static double[,] Frame3() => new double[,] { { 0.4, 0.6 }, { 0.5, 0.5 }, { 0.1, 0.9 } };

    private static void AssertShapeMismatch(double[] startprob, double[,] transmat, double[,] frameprob)
    {
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.ForwardScaling(startprob, transmat, frameprob));
        Assert.Equal("shape mismatch", ex.Message);
    }

    /// <summary>A startprob whose length is not the number of components of the frameprob throws ArgumentException.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    public void A_startprob_of_the_wrong_length_throws_shape_mismatch(int length)
    {
        AssertShapeMismatch(new double[length], Trans2(), Frame3());
    }

    /// <summary>A transmat that is not nc x nc (either dimension wrong, nc taken from the frameprob) throws ArgumentException.</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 1)]
    [InlineData(3, 2)]
    [InlineData(2, 3)]
    [InlineData(3, 3)]
    public void A_transmat_that_is_not_nc_by_nc_throws_shape_mismatch(int rows, int columns)
    {
        AssertShapeMismatch(Start2(), new double[rows, columns], Frame3());
    }

    /// <summary>At least one sample is required: an empty frameprob (0 rows) throws ArgumentException.</summary>
    [Fact]
    public void An_empty_frameprob_with_no_samples_throws_shape_mismatch()
    {
        AssertShapeMismatch(Start2(), Trans2(), new double[0, 2]);
    }

    /// <summary>A frameprob whose number of components differs from the length of the startprob (and the transmat) throws
    /// ArgumentException.</summary>
    [Fact]
    public void A_frameprob_whose_component_count_does_not_agree_throws_shape_mismatch()
    {
        AssertShapeMismatch(Start2(), Trans2(), new double[3, 3]);
    }
}
