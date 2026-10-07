using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm5;

/// <summary>
/// Acceptance tests for the Viterbi tie-breaking of <see cref="Hmmc.Viterbi"/> (ticket HMM-5, criterion C4): the C++ compares
/// (value, index) pairs, so the final state is the FIRST index of the maximum of the last lattice row while going backwards
/// the LARGEST index among equal maxima wins (an all -infinity candidate list included). The golden cases "viterbi-ties" and
/// "viterbi-partial-ties" encode these quirks, and a hand-written fully-tied uniform model pins the rule on trivial inputs.
/// </summary>
public class ViterbiTieBreakingTests
{
    private static List<JsonElement> Cases() => Golden.Load("hmmc").GetProperty("cases").Items().ToList();

    private static List<JsonElement> TieCases() =>
        Cases().Where(c => c.GetProperty("name").AsString().StartsWith("viterbi-")).ToList();

    private static (double LogProb, int[] States) Compute(JsonElement c) => Hmmc.Viterbi(
        c.GetProperty("startprob").AsVector(),
        c.GetProperty("transmat").AsMatrix(),
        c.GetProperty("log_frameprob").AsMatrix());

    /// <summary>Every viterbi expectation of the golden file whose case name starts with "viterbi-" (the deliberate tie cases)
    /// is reproduced: the log probability and the state sequence, using the oracle's own inputs.</summary>
    [Fact]
    public void Every_viterbi_tie_case_of_the_golden_file_is_reproduced()
    {
        var cases = TieCases();
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
        Assert.True(checkedCases >= 2, $"expected at least 2 viterbi tie cases, saw {checkedCases}");
    }

    /// <summary>The "viterbi-ties" case (every path equally likely in a 3-state model) is reproduced exactly.</summary>
    [Fact]
    public void The_viterbi_ties_case_is_reproduced()
    {
        var c = Assert.Single(TieCases(), x => x.GetProperty("name").AsString() == "viterbi-ties");
        var expected = c.GetProperty("viterbi");
        var (logProb, states) = Compute(c);
        Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb, "log_prob");
        Assert.Equal(expected.GetProperty("states").AsInts(), states);
    }

    /// <summary>The "viterbi-partial-ties" case (some, but not all, candidate lists tied) is reproduced exactly.</summary>
    [Fact]
    public void The_viterbi_partial_ties_case_is_reproduced()
    {
        var c = Assert.Single(TieCases(), x => x.GetProperty("name").AsString() == "viterbi-partial-ties");
        var expected = c.GetProperty("viterbi");
        var (logProb, states) = Compute(c);
        Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb, "log_prob");
        Assert.Equal(expected.GetProperty("states").AsInts(), states);
    }

    /// <summary>Hand-written, trivial inputs (zeros and the power of two 0.5): a model where every path is equally likely —
    /// uniform startprob and transmat, zero log frameprob (emission 1). Every candidate list is tied, so going backwards the
    /// LARGEST index wins at every time and the final state is the FIRST index of the maximum: the states are
    /// [nc-1, ..., nc-1, 0] and the log probability is log((1/nc)^n_samples), both computed from the inputs of this test.</summary>
    [Fact]
    public void A_fully_tied_model_gives_the_largest_index_backwards_and_the_first_index_last()
    {
        const int ns = 3;
        const int nc = 2;
        var startprob = new double[nc];
        var transmat = new double[nc, nc];
        for (var i = 0; i < nc; i++)
        {
            startprob[i] = 0.5;
            for (var j = 0; j < nc; j++) transmat[i, j] = 0.5;
        }
        var logFrameprob = new double[ns, nc]; // all zeros: emission probability 1
        var (logProb, states) = Hmmc.Viterbi(startprob, transmat, logFrameprob);
        var expectedStates = new int[ns];
        for (var t = 0; t < ns - 1; t++) expectedStates[t] = nc - 1;
        expectedStates[ns - 1] = 0;
        Assert.Equal(expectedStates, states);
        Close.Equal(-(double)ns * Math.Log((double)nc), logProb, "log_prob");
    }
}
