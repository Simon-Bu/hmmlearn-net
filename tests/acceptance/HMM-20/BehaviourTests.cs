using System.Collections.Generic;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm20;

/// <summary>
/// Acceptance tests for the behaviour of <see cref="VariationalCategoricalHmm"/> (ticket HMM-20, criterion C4): the variational
/// lower bound never decreases during Fit on Beal-style data (sequences sampled from a small three-state cyclic categorical
/// model) with a random initialization and a fixed seed, with both implementations; sequences of length 1 are accepted; the log
/// and the scaling implementations agree on the same random initialization; and a fit rejects fractional, negative and oversized
/// symbols.
/// </summary>
public class BehaviourTests
{
    private static ExposedModel Model(int nComponents, int? nFeatures, int randomState, int nIter, double tol,
        Implementation implementation, string @params = "ste", string initParams = "ste") =>
        new(nComponents, null, null, null, nFeatures, DecoderAlgorithm.Viterbi, randomState, nIter, tol, false,
            @params, initParams, implementation);

    /// <summary>
    /// Beal-style data of a three-state cyclic model: startprob (1/3, 1/3, 1/3), transition matrix [[0,1,0],[0,0,1],[1,0,0]]
    /// (a deterministic cycle) and identity emission probabilities (the emitted symbol is the state), so a sequence is a cycle of
    /// period 3 that starts from a uniformly drawn state.
    /// </summary>
    private static (double[,] X, int[] Lengths) BealData(int seed, int[] lengths)
    {
        var random = new Random(seed);
        var total = 0;
        foreach (var length in lengths)
            total += length;
        var x = new double[total, 1];
        var offset = 0;
        foreach (var length in lengths)
        {
            var u = random.NextDouble();
            var start = u < 1.0 / 3.0 ? 0 : u < 2.0 / 3.0 ? 1 : 2;
            for (var t = 0; t < length; t++)
                x[offset + t, 0] = (start + t) % 3;
            offset += length;
        }
        return (x, lengths);
    }

    /// <summary>The lower bound (Monitor.History) must never decrease by more than 1e-9 relative from one iteration to the next.</summary>
    private static void AssertNonDecreasing(IReadOnlyList<double> history, string what)
    {
        Assert.True(history.Count >= 2, $"{what}: expected at least two reports, got {history.Count}");
        for (var i = 1; i < history.Count; i++)
        {
            var maxAllowedDecrease = 1e-9 * Math.Max(1.0, Math.Abs(history[i - 1]));
            Assert.True(history[i] >= history[i - 1] - maxAllowedDecrease,
                $"{what}: the lower bound decreased from {history[i - 1]:R} (iteration {i - 1}) to {history[i]:R} (iteration {i})");
        }
    }

    /// <summary>Every entry is finite and every row sums to 1.</summary>
    private static void AssertRowsSumToOneAndFinite(double[,] m, string what)
    {
        for (var i = 0; i < m.GetLength(0); i++)
        {
            var sum = 0.0;
            for (var j = 0; j < m.GetLength(1); j++)
            {
                Assert.True(!double.IsNaN(m[i, j]) && !double.IsInfinity(m[i, j]), $"{what}[{i}, {j}] must be finite");
                sum += m[i, j];
            }
            Close.Equal(1.0, sum, $"{what} row {i} must sum to 1");
        }
    }

    /// <summary>On Beal-style data with a random initialization and a fixed seed, the lower bound never decreases during a fit,
    /// with both implementations, and the fitted parameters are valid.</summary>
    [Theory]
    [InlineData(Implementation.Log)]
    [InlineData(Implementation.Scaling)]
    public void lower_bound_never_decreases_during_fit(Implementation implementation)
    {
        var (x, lengths) = BealData(1984, new[] { 100, 100, 100, 100, 100, 100, 100 });
        var model = Model(3, null, 1984, 50, 1e-12, implementation);

        model.Fit(x, lengths);

        AssertNonDecreasing(model.Monitor.History, $"lower bound ({implementation})");
        AssertRowsSumToOneAndFinite(model.EmissionProb!, "the emissionprob");
        AssertRowsSumToOneAndFinite(model.TransMat!, "the transmat");
        var startSum = 0.0;
        foreach (var p in model.StartProb!)
            startSum += p;
        Close.Equal(1.0, startSum, "the startprob must sum to 1");
    }

    /// <summary>A fit accepts sequences of length 1: on Beal-style data that includes unit sequences the lower bound never
    /// decreases and the fitted parameters are valid.</summary>
    [Fact]
    public void fit_accepts_sequences_of_length_one()
    {
        var (x, lengths) = BealData(1984, new[] { 100, 100, 100, 100, 100, 100, 100, 1, 1, 1 });
        var model = Model(3, null, 1984, 20, 1e-12, Implementation.Log);

        model.Fit(x, lengths);

        AssertNonDecreasing(model.Monitor.History, "lower bound with length-1 sequences");
        Assert.NotNull(model.EmissionProb);
        AssertRowsSumToOneAndFinite(model.EmissionProb!, "the emissionprob");
        AssertRowsSumToOneAndFinite(model.TransMat!, "the transmat");
    }

    /// <summary>The log and the scaling implementations, started from the same random initialization (fixed seed) on the same
    /// Beal-style data, fit to the same parameters (oracle tolerance).</summary>
    [Fact]
    public void log_and_scaling_implementations_agree_on_the_same_random_initialization()
    {
        var (x, lengths) = BealData(17, new[] { 40, 40, 40 });
        var logModel = Model(3, null, 17, 10, 1e-12, Implementation.Log);
        var scalingModel = Model(3, null, 17, 10, 1e-12, Implementation.Scaling);

        logModel.Fit(x, lengths);
        scalingModel.Fit(x, lengths);

        Close.Equal(logModel.StartProbPosterior!, scalingModel.StartProbPosterior!, "the startprob posterior");
        Close.Equal(logModel.TransMatPosterior!, scalingModel.TransMatPosterior!, "the transmat posterior");
        Close.Equal(logModel.EmissionProbPosterior!, scalingModel.EmissionProbPosterior!, "the emission posterior");
        Close.Equal(logModel.StartProb!, scalingModel.StartProb!, "the startprob");
        Close.Equal(logModel.TransMat!, scalingModel.TransMat!, "the transmat");
        Close.Equal(logModel.EmissionProb!, scalingModel.EmissionProb!, "the emissionprob");
    }

    /// <summary>A fit rejects the symbols that CategoricalEmission.CheckAndSetNFeatures rejects: fractional, negative, and larger
    /// than NFeatures - 1 when NFeatures is given.</summary>
    [Theory]
    [InlineData(1.5, null)]
    [InlineData(-1.0, null)]
    [InlineData(3.0, 3)]
    public void fit_rejects_invalid_symbols(double symbol, int? nFeatures)
    {
        var model = Model(2, nFeatures, 1, 1, 1e-2, Implementation.Log);
        var x = new double[2, 1] { { 0.0 }, { symbol } };

        Assert.Throws<ArgumentException>(() => model.Fit(x, new[] { 2 }));
    }
}
