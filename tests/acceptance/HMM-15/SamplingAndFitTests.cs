using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm15;

/// <summary>
/// Acceptance tests for the sampling of <see cref="PoissonHmm"/> (ticket HMM-15, criterion C4): the observations are
/// non-negative integer counts, the per-state mean counts stay within 5 standard deviations of the rate of the state they were
/// drawn from (the transmat is uniform, so the states are independent and the per-state sample size is large), a zero rate draws
/// exactly 0, and the model is reproducible with a fixed Random or RandomState. All the true parameters are the own inputs of
/// the test (never oracle numbers). Also: a fit from a random initialization starts from valid positive rates, is reproducible
/// with the same seed, and the log likelihood never decreases from one single-iteration fit to the next.
/// </summary>
public class SamplingAndFitTests
{
    /// <summary>The true parameters of the test's own model (2 components, 3 features, uniform transmat).</summary>
    private static (double[] StartProb, double[,] TransMat, double[,] Lambdas) TrueParameters() =>
        (new double[] { 0.6, 0.4 },
            new double[,] { { 0.5, 0.5 }, { 0.5, 0.5 } },
            new double[,] { { 2.5, 1.0, 5.0 }, { 0.5, 3.5, 1.5 } });

    /// <summary>The true model, fully parameterized (InitParams "" so the given parameters are used as-is).</summary>
    private static PoissonHmm TrueModel(Implementation implementation)
    {
        var (startProb, transMat, lambdas) = TrueParameters();
        var model = new PoissonHmm(
            2, 1.0, 1.0, 0.0, 0.0, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "stl", "", implementation);
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.Lambdas = lambdas;
        return model;
    }

    /// <summary>500 sampled counts (the own true model, a fixed seed) split in 2 sequences of 250, used by the fit tests.</summary>
    private static (double[,] X, int[] Lengths) FitData()
    {
        var (x, _) = TrueModel(Implementation.Log).Sample(500, random: new Random(12345));
        return (x, new[] { 250, 250 });
    }

    [Fact]
    public void sample_counts_match_the_lambdas_of_their_state_within_five_standard_deviations()
    {
        const int nSamples = 5000;
        var (_, _, lambdas) = TrueParameters();
        var (x, states) = TrueModel(Implementation.Log).Sample(nSamples, random: new Random(12345));

        Assert.Equal(nSamples, x.GetLength(0));
        Assert.Equal(lambdas.GetLength(1), x.GetLength(1));
        Assert.Equal(nSamples, states.Length);

        var nFeatures = lambdas.GetLength(1);
        var nStates = lambdas.GetLength(0);
        var samplesPerState = new int[nStates];
        var countsPerState = new long[nStates, nFeatures];
        for (var i = 0; i < nSamples; i++)
        {
            for (var f = 0; f < nFeatures; f++)
            {
                Assert.True(x[i, f] >= 0.0 && x[i, f] == Math.Floor(x[i, f]),
                    $"sample [{i},{f}] must be a non-negative integer count, got {x[i, f]:R}");
            }
            Assert.InRange(states[i], 0, nStates - 1);
            samplesPerState[states[i]]++;
            for (var f = 0; f < nFeatures; f++)
                countsPerState[states[i], f] += (long)x[i, f];
        }

        // The transmat is uniform: every position is state s with probability 1/2, so both states are sampled ~n/2 times.
        for (var s = 0; s < nStates; s++)
            Assert.True(samplesPerState[s] >= 1000,
                $"state {s} was sampled {samplesPerState[s]} times; with the uniform transmat it must be about {nSamples / 2}");

        for (var s = 0; s < nStates; s++)
            for (var f = 0; f < nFeatures; f++)
            {
                var lambda = lambdas[s, f];
                var mean = countsPerState[s, f] / (double)samplesPerState[s];
                var bound = 5.0 * Math.Sqrt(lambda / samplesPerState[s]);
                Assert.True(Math.Abs(mean - lambda) <= bound,
                    $"state {s}, feature {f}: the mean count {mean} deviates from the rate {lambda} by more than 5 standard deviations (bound {bound})");
            }
    }

    [Fact]
    public void sample_is_reproducible_with_a_fixed_random_or_random_state()
    {
        var (x1, states1) = TrueModel(Implementation.Log).Sample(37, random: new Random(12345));
        var (x2, states2) = TrueModel(Implementation.Log).Sample(37, random: new Random(12345));
        Close.Equal(x1, x2, "the observations of two samples with the same random");
        Assert.Equal(states1, states2);

        var byState = TrueModel(Implementation.Log);
        byState.RandomState = 12345;
        var (x3, states3) = byState.Sample(37);

        var byState2 = TrueModel(Implementation.Log);
        byState2.RandomState = 12345;
        var (x4, states4) = byState2.Sample(37);

        Close.Equal(x3, x4, "the observations of two samples with the same RandomState");
        Assert.Equal(states3, states4);
    }

    /// <summary>GenerateSampleFromState draws one Poisson per feature: a zero rate draws exactly 0 (degenerate) and the
    /// non-degenerate rate matches its mean within 5 standard deviations over many draws.</summary>
    [Fact]
    public void generate_sample_from_state_draws_one_poisson_count_per_feature()
    {
        const int nDraws = 20000;
        const double rate = 2.5;
        var model = new GenerateSampleHarness(new double[,] { { 0.0, rate, 0.0 }, { 1.0, 1.0, 1.0 } });

        for (var i = 0; i < 100; i++)
        {
            var drawn = model.Draw(0, new Random(12345 + i));
            Assert.Equal(3, drawn.Length);
            Assert.True(drawn[0] == 0.0 && drawn[2] == 0.0,
                $"a zero rate must draw exactly 0, got [{drawn[0]:R}, {drawn[1]:R}, {drawn[2]:R}]");
        }

        // One generator for all the draws (consecutive seeds of System.Random give correlated first numbers).
        var random = new Random(12345);
        var total = 0.0;
        for (var i = 0; i < nDraws; i++)
            total += model.Draw(0, random)[1];
        var mean = total / nDraws;
        var bound = 5.0 * Math.Sqrt(rate / nDraws);
        Assert.True(Math.Abs(mean - rate) <= bound,
            $"the mean of {nDraws} Poisson({rate}) draws is {mean}, outside 5 standard deviations (bound {bound})");
    }

    [Theory]
    [InlineData(Implementation.Log)]
    [InlineData(Implementation.Scaling)]
    public void a_fit_from_a_random_initialization_starts_from_valid_positive_rates(Implementation implementation)
    {
        var (x, lengths) = FitData();
        var model = new PoissonHmm(
            2, 1.0, 1.0, 0.0, 0.0, DecoderAlgorithm.Viterbi, 12345, 0, 1e-2, false, "stl", "stl", implementation);

        model.Fit(x, lengths);

        Assert.True(model.NFeatures == 3, $"n_features: expected 3, got {model.NFeatures}");
        Assert.True(model.StartProb!.All(v => v >= 0.0), "startprob has a negative entry after the random initialization");
        Assert.True(Math.Abs(model.StartProb.Sum() - 1.0) <= 1e-8, "startprob does not sum to 1 after the random initialization");
        Assert.True(model.TransMat!.Cast<double>().All(v => v >= 0.0), "transmat has a negative entry after the random initialization");
        Assert.True(RowsSumToOne(model.TransMat), "transmat is not row-stochastic after the random initialization");
        Assert.NotNull(model.Lambdas);
        Assert.Equal(2, model.Lambdas!.GetLength(0));
        Assert.Equal(3, model.Lambdas.GetLength(1));
        Assert.True(model.Lambdas.Cast<double>().All(v => v > 0.0),
            "lambdas has a non-positive entry after the random initialization");
    }

    [Theory]
    [InlineData(Implementation.Log)]
    [InlineData(Implementation.Scaling)]
    public void a_fit_from_a_random_initialization_is_reproducible_with_the_same_seed(Implementation implementation)
    {
        var (x, lengths) = FitData();
        var m1 = new PoissonHmm(2, 1.0, 1.0, 0.0, 0.0, DecoderAlgorithm.Viterbi, 12345, 0, 1e-2, false, "stl", "stl", implementation);
        var m2 = new PoissonHmm(2, 1.0, 1.0, 0.0, 0.0, DecoderAlgorithm.Viterbi, 12345, 0, 1e-2, false, "stl", "stl", implementation);

        m1.Fit(x, lengths);
        m2.Fit(x, lengths);

        Close.Equal(m1.StartProb!, m2.StartProb!, "the startprob of two fits with the same seed");
        Close.Equal(m1.TransMat!, m2.TransMat!, "the transmat of two fits with the same seed");
        Close.Equal(m1.Lambdas!, m2.Lambdas!, "the lambdas of two fits with the same seed");
        Assert.True(m1.NFeatures == m2.NFeatures, $"n_features: {m1.NFeatures} and {m2.NFeatures} differ");
    }

    /// <summary>EM never decreases the log likelihood: starting from a random initialization (1 iteration), every further
    /// single-iteration fit (no re-init) must keep the score at least where the previous one left it, up to 1e-9 relative.</summary>
    [Theory]
    [InlineData(Implementation.Log)]
    [InlineData(Implementation.Scaling)]
    public void the_log_likelihood_never_decreases_from_one_single_iteration_fit_to_the_next(Implementation implementation)
    {
        var (x, lengths) = FitData();
        var model = new PoissonHmm(2, 1.0, 1.0, 0.0, 0.0, DecoderAlgorithm.Viterbi, 12345, 1, 1e-2, false, "stl", "stl", implementation);

        model.Fit(x, lengths);
        var logLikelihood = model.Score(x, lengths);

        for (var i = 1; i <= 10; i++)
        {
            model.InitParams = "";
            model.Fit(x, lengths);
            var next = model.Score(x, lengths);
            Assert.True(next >= logLikelihood - 1e-9 * Math.Abs(logLikelihood),
                $"iteration {i} ({implementation}): the log likelihood decreased from {logLikelihood} to {next} (more than 1e-9 relative)");
            logLikelihood = next;
        }
    }

    private static bool RowsSumToOne(double[,] a)
    {
        for (var i = 0; i < a.GetLength(0); i++)
        {
            var sum = 0.0;
            for (var j = 0; j < a.GetLength(1); j++)
                sum += a[i, j];
            if (Math.Abs(sum - 1.0) > 1e-8)
                return false;
        }
        return true;
    }

    /// <summary>A model exposing GenerateSampleFromState, ready for direct draws (fixed parameters, no initialization).</summary>
    private sealed class GenerateSampleHarness : PoissonHmm
    {
        public GenerateSampleHarness(double[,] lambdas)
            : base(2, 1.0, 1.0, 0.0, 0.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "stl", "", Implementation.Log)
        {
            StartProb = new[] { 0.5, 0.5 };
            TransMat = new double[,] { { 0.5, 0.5 }, { 0.5, 0.5 } };
            Lambdas = lambdas;
        }

        public double[] Draw(int state, Random random) => GenerateSampleFromState(state, random);
    }
}
