using System.Text.Json;
using MathNet.Numerics;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm11;

/// <summary>
/// Acceptance tests for the inference of <see cref="MultinomialHmm"/> (ticket HMM-11, criterion C1): Score, ScoreSamples,
/// Decode (Viterbi and MAP), Predict and PredictProba reproduce every inference case of oracle/golden/multinomial.json with
/// both implementations (fixed trials, one trial that is the categorical example, trial counts that vary per sample).
/// The likelihood of a sample of counts c with total n under a state of probabilities p is the multinomial probability
/// n! / (prod_j c_j!) * prod_j p_j^c_j with 0^0 = 1; the log likelihood is computed directly with log-gamma, where
/// c_j * log(p_j) is 0 for c_j = 0 and -infinity for a positive c_j with p_j = 0.
/// </summary>
public class InferenceParityTests
{
    /// <summary>Every inference case of the oracle, once per implementation.</summary>
    public static IEnumerable<object[]> InferenceCases =>
        Golden.Load("multinomial").GetProperty("inference_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(impl => new object[] { name, impl }));

    private static JsonElement InferenceCase(string name) =>
        Assert.Single(Golden.Load("multinomial").GetProperty("inference_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    /// <summary>Builds the model of the oracle case (init_params "" : nothing is initialized, the three arrays are the
    /// given parameters) with the given implementation; a JSON null n_trials leaves NTrials null.</summary>
    private static MultinomialHmm Model(JsonElement c, Implementation implementation)
    {
        var model = new MultinomialHmm(
            c.GetProperty("n_components").AsInt(),
            c.Has("n_trials") ? (int?)c.GetProperty("n_trials").AsInt() : null,
            1.0, 1.0, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "ste", "", implementation);
        model.StartProb = c.GetProperty("startprob").AsVector();
        model.TransMat = c.GetProperty("transmat").AsMatrix();
        model.EmissionProb = c.GetProperty("emissionprob").AsMatrix();
        return model;
    }

    /// <summary>Score, ScoreSamples, Decode (Viterbi and MAP), Predict and PredictProba reproduce the expected values of the
    /// oracle case, with both implementations.</summary>
    [Theory]
    [MemberData(nameof(InferenceCases))]
    public void inference_reproduces_the_oracle(string name, Implementation implementation)
    {
        var c = InferenceCase(name);
        var model = Model(c, implementation);
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var expected = c.GetProperty("expected");

        Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths),
            $"{name}: score with the {implementation} implementation");

        var (logProb, posteriors) = model.ScoreSamples(x, lengths);
        Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb,
            $"{name}: score_samples log_prob with the {implementation} implementation");
        Close.Equal(expected.GetProperty("posteriors").AsMatrix(), posteriors,
            $"{name}: score_samples posteriors with the {implementation} implementation");

        var (viterbiLogProb, viterbiStates) = model.Decode(x, lengths, DecoderAlgorithm.Viterbi);
        Close.Equal(expected.GetProperty("viterbi").GetProperty("log_prob").AsDouble(), viterbiLogProb,
            $"{name}: viterbi log_prob with the {implementation} implementation");
        Assert.Equal(expected.GetProperty("viterbi").GetProperty("states").AsInts(), viterbiStates);

        var (mapLogProb, mapStates) = model.Decode(x, lengths, DecoderAlgorithm.Map);
        Close.Equal(expected.GetProperty("map").GetProperty("log_prob").AsDouble(), mapLogProb,
            $"{name}: map log_prob with the {implementation} implementation");
        Assert.Equal(expected.GetProperty("map").GetProperty("states").AsInts(), mapStates);

        Assert.Equal(expected.GetProperty("viterbi").GetProperty("states").AsInts(), model.Predict(x, lengths));
        Close.Equal(expected.GetProperty("posteriors").AsMatrix(), model.PredictProba(x, lengths),
            $"{name}: predict_proba with the {implementation} implementation");
    }

    /// <summary>Exposes the protected <see cref="MultinomialHmm.ComputeLikelihood"/> and
    /// <see cref="MultinomialHmm.ComputeLogLikelihood"/> of the model under test.</summary>
    private sealed class LikelihoodHarness : MultinomialHmm
    {
        public LikelihoodHarness(int? nTrials, double[,] emissionProb)
            : base(2, nTrials, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "ste", "", Implementation.Log)
        {
            StartProb = new[] { 0.5, 0.5 };
            TransMat = new double[,] { { 0.5, 0.5 }, { 0.5, 0.5 } };
            EmissionProb = emissionProb;
        }

        public double[,] LikelihoodDirect(double[,] x) => ComputeLikelihood(x);
        public double[,] LogLikelihoodDirect(double[,] x) => ComputeLogLikelihood(x);
    }

    /// <summary>A sample with no count at all has likelihood exactly 1 (0! / 0! * prod p^0, with 0^0 = 1) and log likelihood
    /// exactly 0, under every state.</summary>
    [Fact]
    public void a_zero_count_sample_has_likelihood_one_and_log_likelihood_zero_under_every_state()
    {
        var emissionProb = new double[,] { { 0.2, 0.3, 0.5 }, { 0.6, 0.3, 0.1 } };
        var model = new LikelihoodHarness(null, emissionProb);
        var x = new double[,] { { 0, 0, 0 } };

        var likelihood = model.LikelihoodDirect(x);
        var logLikelihood = model.LogLikelihoodDirect(x);

        Assert.True(likelihood.Cast<double>().All(v => v == 1.0),
            $"the likelihood of the zero-count sample must be exactly 1 under every state, got [{Join(likelihood)}]");
        Assert.True(logLikelihood.Cast<double>().All(v => v == 0.0),
            $"the log likelihood of the zero-count sample must be exactly 0 under every state, got [{Join(logLikelihood)}]");
    }

    /// <summary>The likelihood of each sample under each state is the multinomial probability of its counts, computed here
    /// with integer factorials and Math.Pow; the probabilities are powers of two, so every value is exact in binary and the
    /// comparison checks the formula (binomial coefficient and powers) itself.</summary>
    [Fact]
    public void the_likelihood_is_the_multinomial_formula_of_the_counts()
    {
        var emissionProb = new double[,] { { 0.5, 0.25, 0.125, 0.125 }, { 0.125, 0.125, 0.5, 0.25 } };
        var model = new LikelihoodHarness(4, emissionProb);
        var x = new double[,] { { 2, 2, 0, 0 }, { 1, 1, 2, 0 } };

        var result = model.LikelihoodDirect(x);

        var expected = new double[x.GetLength(0), emissionProb.GetLength(0)];
        for (var i = 0; i < x.GetLength(0); i++)
            for (var c = 0; c < emissionProb.GetLength(0); c++)
            {
                var probRow = new double[x.GetLength(1)];
                for (var j = 0; j < x.GetLength(1); j++)
                    probRow[j] = emissionProb[c, j];
                expected[i, c] = MultinomialProbability(x, i, probRow);
            }
        Close.Equal(expected, result, "likelihood");
    }

    /// <summary>A sample with one trial reduces to the categorical case: the likelihood is the emission probability of the
    /// single symbol of the sample, counted from the test's own inputs.</summary>
    [Fact]
    public void a_one_trial_sample_has_the_categorical_likelihood()
    {
        var emissionProb = new double[,] { { 0.2, 0.3, 0.5 }, { 0.6, 0.3, 0.1 } };
        var model = new LikelihoodHarness(1, emissionProb);
        var x = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

        var result = model.LikelihoodDirect(x);

        var expected = new double[x.GetLength(0), emissionProb.GetLength(0)];
        for (var i = 0; i < x.GetLength(0); i++)
        {
            var symbol = 0;
            for (var j = 0; j < x.GetLength(1); j++)
                if (x[i, j] > 0.0)
                    symbol = j;
            for (var c = 0; c < emissionProb.GetLength(0); c++)
                expected[i, c] = emissionProb[c, symbol];
        }
        Close.Equal(expected, result, "one-trial likelihood");
    }

    /// <summary>The log likelihood is the log of the same probability: 0 * log(0) = 0 (a zero count on a zero probability
    /// stays finite and equals the log of the likelihood), and a positive count on a zero probability is exactly
    /// -infinity.</summary>
    [Fact]
    public void the_log_likelihood_handles_zero_counts_and_zero_probabilities()
    {
        var emissionProb = new double[,] { { 0.5, 0.5, 0.0 }, { 0.0, 0.5, 0.5 } };
        var model = new LikelihoodHarness(3, emissionProb);
        var x = new double[,] { { 1, 2, 0 }, { 0, 0, 1 } };

        var likelihood = model.LikelihoodDirect(x);
        var logLikelihood = model.LogLikelihoodDirect(x);

        // {1,2,0} under state 0: c2 = 0 with p2 = 0 -> 0 * log(0) = 0 -> finite, the log of the likelihood.
        Assert.True(double.IsFinite(logLikelihood[0, 0]),
            $"the log likelihood of {{1,2,0}} under state 0 must be finite (0 * log(0) = 0), got {logLikelihood[0, 0]:R}");
        Close.Equal(Math.Log(likelihood[0, 0]), logLikelihood[0, 0], "log likelihood of {1,2,0} under state 0");
        // {1,2,0} under state 1: c0 = 1 > 0 with p0 = 0 -> -infinity.
        Assert.True(double.IsNegativeInfinity(logLikelihood[0, 1]),
            $"the log likelihood of {{1,2,0}} under state 1 must be -infinity, got {logLikelihood[0, 1]:R}");
        // {0,0,1} under state 0: c2 = 1 > 0 with p2 = 0 -> -infinity.
        Assert.True(double.IsNegativeInfinity(logLikelihood[1, 0]),
            $"the log likelihood of {{0,0,1}} under state 0 must be -infinity, got {logLikelihood[1, 0]:R}");
        // {0,0,1} under state 1: c0 = 0 with p0 = 0 -> 0 * log(0) = 0 -> the log of the likelihood 0.5.
        Close.Equal(Math.Log(emissionProb[1, 2]), logLikelihood[1, 1], "log likelihood of {0,0,1} under state 1");
    }

    /// <summary>The log likelihood is computed directly with log-gamma: it stays finite and equals the log-gamma formula
    /// even where the product of probabilities would underflow to 0.</summary>
    [Fact]
    public void the_log_likelihood_stays_finite_where_the_product_underflows()
    {
        var emissionProb = new double[,] { { 0.5, 0.5 }, { 0.5, 0.5 } };
        var model = new LikelihoodHarness(10000, emissionProb);
        var x = new double[,] { { 5000, 5000 } };

        var logLikelihood = model.LogLikelihoodDirect(x);

        var n = x[0, 0] + x[0, 1];
        var expected = SpecialFunctions.GammaLn(n + 1.0);
        for (var j = 0; j < x.GetLength(1); j++)
            expected += x[0, j] * Math.Log(emissionProb[0, j]) - SpecialFunctions.GammaLn(x[0, j] + 1.0);
        Assert.True(double.IsFinite(logLikelihood[0, 0]),
            $"the log likelihood of {{5000,5000}} must be finite, got {logLikelihood[0, 0]:R}");
        Close.Equal(expected, logLikelihood[0, 0], "log likelihood with an underflowing product");
    }

    /// <summary>n! / (prod c_j!) * prod p_j^c_j with the coefficient computed in exact integer arithmetic.</summary>
    private static double MultinomialProbability(double[,] x, int row, double[] p)
    {
        var n = 0L;
        for (var j = 0; j < x.GetLength(1); j++)
            n += (long)x[row, j];
        var coefficient = Factorial(n);
        for (var j = 0; j < x.GetLength(1); j++)
            coefficient /= Factorial((long)x[row, j]);
        var prob = (double)coefficient;
        for (var j = 0; j < x.GetLength(1); j++)
            prob *= Math.Pow(p[j], (int)x[row, j]);
        return prob;
    }

    private static long Factorial(long n)
    {
        var result = 1L;
        for (var i = 2; i <= n; i++)
            result *= i;
        return result;
    }

    private static string Join(double[,] a) => string.Join(", ", a.Cast<double>());
}
