using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm10;

/// <summary>
/// Acceptance tests for the inference of <see cref="CategoricalHmm"/> (ticket HMM-10, criterion C1): Score, ScoreSamples,
/// Decode (Viterbi and MAP), Predict and PredictProba reproduce every inference case of oracle/golden/categorical.json with
/// both implementations, including the Wikipedia rainy/sunny example, several sequences, a unit sequence and a model with
/// structural zeros. The likelihood of a sample under a state is the EmissionProb of its symbol; its log is the Math.Log of
/// it (the log of a zero probability is -infinity).
/// </summary>
public class InferenceParityTests
{
    /// <summary>Every inference case of the oracle, once per implementation.</summary>
    public static IEnumerable<object[]> InferenceCases =>
        Golden.Load("categorical").GetProperty("inference_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(impl => new object[] { name, impl }));

    private static JsonElement InferenceCase(string name) =>
        Assert.Single(Golden.Load("categorical").GetProperty("inference_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    /// <summary>Builds the model of the oracle case (init_params "" : nothing is initialized, the three arrays are the
    /// given parameters) with the given implementation.</summary>
    private static CategoricalHmm Model(JsonElement c, Implementation implementation)
    {
        var model = new CategoricalHmm(
            c.GetProperty("n_components").AsInt(), 1.0, 1.0, 1.0, c.GetProperty("n_features").AsInt(),
            DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "ste", "", implementation);
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

    /// <summary>The likelihood of a sample under a state is the EmissionProb of its symbol, including a zero probability.</summary>
    [Fact]
    public void the_likelihood_of_a_sample_is_the_emission_probability_of_its_symbol()
    {
        var emissionProb = new double[,] { { 0.1, 0.4, 0.5 }, { 0.6, 0.3, 0.0 } };
        var x = new double[,] { { 0 }, { 2 }, { 1 }, { 0 } };

        var result = CategoricalEmission.Likelihood(emissionProb, x);

        var expected = new double[x.GetLength(0), emissionProb.GetLength(0)];
        for (var i = 0; i < x.GetLength(0); i++)
            for (var c = 0; c < emissionProb.GetLength(0); c++)
                expected[i, c] = emissionProb[c, (int)x[i, 0]];
        Close.Equal(expected, result, "likelihood");
    }

    /// <summary>The log likelihood is the Math.Log of the emission probability of the symbol of each sample under each state;
    /// the log of a zero probability is -infinity.</summary>
    [Fact]
    public void the_log_likelihood_is_the_log_of_the_emission_probability_of_each_sample()
    {
        var model = new LogLikelihoodHarness
        {
            StartProb = new[] { 0.5, 0.5 },
            TransMat = new double[,] { { 0.6, 0.4 }, { 0.4, 0.6 } },
            EmissionProb = new double[,] { { 0.5, 0.5, 0.0 }, { 0.25, 0.25, 0.5 } },
        };
        var x = new double[,] { { 1 }, { 2 }, { 0 } };

        var result = model.ComputeLogLikelihoodDirect(x);

        var expected = new double[x.GetLength(0), model.NComponents];
        for (var i = 0; i < x.GetLength(0); i++)
            for (var c = 0; c < model.NComponents; c++)
                expected[i, c] = Math.Log(model.EmissionProb![c, (int)x[i, 0]]);
        Close.Equal(expected, result, "log likelihood");
    }

    /// <summary>Exposes the protected <see cref="CategoricalHmm.ComputeLogLikelihood"/> of the model under test.</summary>
    private sealed class LogLikelihoodHarness : CategoricalHmm
    {
        public LogLikelihoodHarness()
            : base(2, 1.0, 1.0, 1.0, 3, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "ste", "", Implementation.Log)
        {
        }

        public double[,] ComputeLogLikelihoodDirect(double[,] x) => ComputeLogLikelihood(x);
    }
}
