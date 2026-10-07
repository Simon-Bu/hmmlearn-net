using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm5;

/// <summary>
/// Acceptance tests for <see cref="Hmmc.Viterbi"/> (ticket HMM-5, criterion C3): every viterbi expectation of
/// oracle/golden/hmmc.json whose case name does not start with "viterbi-" (including the Wikipedia example and the
/// impossible observation), plus hand-written shape errors. The tie-breaking cases are covered by
/// <see cref="ViterbiTieBreakingTests"/>.
/// </summary>
public class ViterbiTests
{
    private static List<JsonElement> Cases() => Golden.Load("hmmc").GetProperty("cases").Items().ToList();

    private static List<JsonElement> NonTieCases() =>
        Cases().Where(c => !c.GetProperty("name").AsString().StartsWith("viterbi-")).ToList();

    private static (double LogProb, int[] States) Compute(JsonElement c) => Hmmc.Viterbi(
        c.GetProperty("startprob").AsVector(),
        c.GetProperty("transmat").AsMatrix(),
        c.GetProperty("log_frameprob").AsMatrix());

    /// <summary>Every viterbi expectation of the golden file whose case name does not start with "viterbi-" is reproduced: the
    /// log probability and the state sequence, using the oracle's own startprob, transmat and log_frameprob. This includes the
    /// Wikipedia example and the impossible observation (whose log probability is -infinity).</summary>
    [Fact]
    public void Every_non_tie_viterbi_expectation_of_the_golden_file_is_reproduced()
    {
        var cases = NonTieCases();
        Assert.NotEmpty(cases);
        var checkedCases = 0;
        foreach (var c in cases)
        {
            if (!c.Has("viterbi")) continue;
            var name = c.GetProperty("name").AsString();
            var expected = c.GetProperty("viterbi");
            var (logProb, states) = Compute(c);
            Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb, $"{name}: log_prob");
            var expectedStates = expected.GetProperty("states").AsInts();
            Assert.True(
                expectedStates.SequenceEqual(states),
                $"{name}: states expected [{string.Join(", ", expectedStates)}], got [{string.Join(", ", states)}]");
            checkedCases++;
        }
        Assert.True(checkedCases >= 8, $"expected at least 8 non-tie viterbi cases, saw {checkedCases}");
    }

    /// <summary>The Wikipedia example from reference/hmmlearn-0.3.3/src/hmmlearn/tests/test_base.py (test_do_viterbi_pass): the
    /// most likely state sequence is [0, 0, 1, 0, 0]; the exact log probability is the golden file's (its rounded reference
    /// value -4.4590 is outside the oracle tolerance of the full-precision value, so it is not retyped here).</summary>
    [Fact]
    public void The_wikipedia_example_gives_the_reference_state_sequence_and_log_prob()
    {
        var c = Assert.Single(Cases(), x => x.GetProperty("name").AsString() == "wikipedia");
        var expected = c.GetProperty("viterbi");
        var (logProb, states) = Compute(c);
        Assert.Equal(new int[] { 0, 0, 1, 0, 0 }, states);
        Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb, "log_prob");
    }

    /// <summary>The impossible observation has an exactly -infinity log probability (the maximum of an all -infinity last
    /// lattice row) and the state sequence of the golden file.</summary>
    [Fact]
    public void The_impossible_observation_has_minus_infinity_log_prob()
    {
        var c = Assert.Single(Cases(), x => x.GetProperty("name").AsString() == "impossible-observation");
        var expected = c.GetProperty("viterbi");
        var (logProb, states) = Compute(c);
        Assert.Equal(double.NegativeInfinity, logProb);
        Assert.Equal(expected.GetProperty("states").AsInts(), states);
    }

    // Hand-written shape errors (criterion C3): the consistent base is startprob (2,), transmat (2, 2), log_frameprob (3, 2).

    private static double[] Start2() => new double[] { 0.6, 0.4 };

    private static double[,] Trans2() => new double[,] { { 0.8, 0.2 }, { 0.3, 0.7 } };

    private static double[,] LogFrame32() => new double[,] { { -0.5, -0.9 }, { -0.7, -0.7 }, { -0.1, -2.3 } };

    /// <summary>A startprob whose length is not nc (nc taken from the log_frameprob) throws ArgumentException "shape mismatch".</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    public void A_startprob_of_the_wrong_length_throws_shape_mismatch(int length)
    {
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.Viterbi(new double[length], Trans2(), LogFrame32()));
        Assert.Equal("shape mismatch", ex.Message);
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
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.Viterbi(Start2(), new double[rows, columns], LogFrame32()));
        Assert.Equal("shape mismatch", ex.Message);
    }

    /// <summary>An empty log_frameprob (no samples at all) throws ArgumentException "shape mismatch" ("at least one sample").</summary>
    [Fact]
    public void An_empty_log_frameprob_throws_shape_mismatch()
    {
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.Viterbi(Start2(), Trans2(), new double[0, 2]));
        Assert.Equal("shape mismatch", ex.Message);
    }

    /// <summary>A log_frameprob whose component count does not agree with the startprob throws ArgumentException "shape mismatch".</summary>
    [Fact]
    public void A_log_frameprob_whose_component_count_does_not_agree_throws_shape_mismatch()
    {
        var ex = Assert.Throws<ArgumentException>(() => Hmmc.Viterbi(Start2(), Trans2(), new double[3, 3]));
        Assert.Equal("shape mismatch", ex.Message);
    }
}
