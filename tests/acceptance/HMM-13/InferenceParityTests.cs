using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm13;

/// <summary>
/// Criterion C1 of ticket HMM-13: the inference of <see cref="GaussianHmm"/> (Score, ScoreSamples, Decode (Viterbi and Map),
/// Predict, PredictProba) reproduces every inference case of all four oracle files (spherical, diag, tied, full covariances) for
/// both implementations, over several sequences including a unit-length sequence. The golden file is the single source of truth
/// for every expected number; the model is built with InitParams "" so that the given parameters are used as-is, and the default
/// decoding algorithm is Viterbi so that Predict matches the golden Viterbi states.
/// </summary>
public class InferenceParityTests
{
    private static readonly string[] GoldenFiles = { "gaussian_spherical", "gaussian_diag", "gaussian_tied", "gaussian_full" };

    public static IEnumerable<object[]> InferenceCases =>
        GoldenFiles.SelectMany(file =>
            Golden.Load(file).GetProperty("inference_cases").Items()
                .Select(c => c.GetProperty("name").AsString())
                .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                    .Select(implementation => new object[] { file, name, implementation })));

    private static JsonElement InferenceCase(string file, string name) =>
        Assert.Single(Golden.Load(file).GetProperty("inference_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    private static GaussianHmm Model(JsonElement c, Implementation implementation)
    {
        var type = Enum.Parse<CovarianceType>(c.GetProperty("covariance_type").AsString(), ignoreCase: true);
        var model = new GaussianHmm(
            c.GetProperty("n_components").AsInt(),
            type,
            1e-3, 1.0, 1.0, 0, 0, 1e-2, 1,
            DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "stmc", "", implementation);
        model.StartProb = c.GetProperty("startprob").AsVector();
        model.TransMat = c.GetProperty("transmat").AsMatrix();
        model.Means = c.GetProperty("means").AsMatrix();
        model.Covars = c.GetProperty("covars").AsTensor3();
        return model;
    }

    [Theory]
    [MemberData(nameof(InferenceCases))]
    public void inference_reproduces_the_oracle(string file, string name, Implementation implementation)
    {
        var c = InferenceCase(file, name);
        var what = $"{file}/{name} ({implementation})";
        var model = Model(c, implementation);
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var expected = c.GetProperty("expected");

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
    }
}
