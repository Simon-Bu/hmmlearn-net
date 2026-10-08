using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm15;

/// <summary>
/// Criterion C1 of ticket HMM-15: the inference of <see cref="PoissonHmm"/> (Score, ScoreSamples, Decode (Viterbi and Map),
/// Predict, PredictProba) reproduces every inference case of oracle/golden/poisson.json with both implementations (several
/// sequences, zero counts). The golden file is the single source of truth for every expected number; the model is built with
/// InitParams "" so that the given parameters are used as-is, and the parameters and the observations are passed to the model
/// as copies so that a mutation of the caller's arrays is detected.
/// </summary>
public class InferenceParityTests
{
    public static IEnumerable<object[]> InferenceCases =>
        Golden.Load("poisson").GetProperty("inference_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(implementation => new object[] { name, implementation }));

    private static JsonElement Case(string name) =>
        Assert.Single(Golden.Load("poisson").GetProperty("inference_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    /// <summary>The model of the case: default priors, InitParams "" (the given parameters are used as-is).</summary>
    private static PoissonHmm Model(JsonElement c, Implementation implementation)
    {
        var model = new PoissonHmm(
            nComponents: c.GetProperty("n_components").AsInt(),
            startProbPrior: 1.0,
            transMatPrior: 1.0,
            lambdasPrior: 0.0,
            lambdasWeight: 0.0,
            algorithm: DecoderAlgorithm.Viterbi,
            randomState: null,
            nIter: 10,
            tol: 1e-2,
            verbose: false,
            @params: "stl",
            initParams: "",
            implementation);
        model.StartProb = c.GetProperty("startprob").AsVector();
        model.TransMat = c.GetProperty("transmat").AsMatrix();
        model.Lambdas = c.GetProperty("lambdas").AsMatrix();
        return model;
    }

    [Theory]
    [MemberData(nameof(InferenceCases))]
    public void inference_reproduces_the_oracle(string name, Implementation implementation)
    {
        var c = Case(name);
        var what = $"{name} ({implementation})";
        var model = Model(c, implementation);
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var expected = c.GetProperty("expected");
        var x0 = (double[,])x.Clone();
        var startProb0 = (double[])model.StartProb!.Clone();
        var transMat0 = (double[,])model.TransMat!.Clone();
        var lambdas0 = (double[,])model.Lambdas!.Clone();

        // Score
        Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths), $"{what} Score");

        // ScoreSamples: the log probability and the per-sample posteriors
        var (logProb, posteriors) = model.ScoreSamples(x, lengths);
        Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb, $"{what} ScoreSamples log probability");
        Close.Equal(expected.GetProperty("posteriors").AsMatrix(), posteriors, $"{what} ScoreSamples posteriors");

        // Decode with Viterbi
        var (viterbiLogProb, viterbiStates) = model.Decode(x, lengths, DecoderAlgorithm.Viterbi);
        Close.Equal(expected.GetProperty("viterbi").GetProperty("log_prob").AsDouble(), viterbiLogProb, $"{what} Viterbi log probability");
        Assert.Equal(expected.GetProperty("viterbi").GetProperty("states").AsInts(), viterbiStates);

        // Decode with Map
        var (mapLogProb, mapStates) = model.Decode(x, lengths, DecoderAlgorithm.Map);
        Close.Equal(expected.GetProperty("map").GetProperty("log_prob").AsDouble(), mapLogProb, $"{what} MAP log probability");
        Assert.Equal(expected.GetProperty("map").GetProperty("states").AsInts(), mapStates);

        // Predict: the model's default algorithm is Viterbi
        Assert.Equal(expected.GetProperty("viterbi").GetProperty("states").AsInts(), model.Predict(x, lengths));

        // PredictProba: the per-sample posteriors
        Close.Equal(expected.GetProperty("posteriors").AsMatrix(), model.PredictProba(x, lengths), $"{what} PredictProba");

        // The caller's arrays are not mutated by the inference
        Close.Equal(x0, x, $"{what} caller's x");
        Close.Equal(startProb0, model.StartProb, $"{what} caller's startprob");
        Close.Equal(transMat0, model.TransMat, $"{what} caller's transmat");
        Close.Equal(lambdas0, model.Lambdas, $"{what} caller's lambdas");
    }
}
