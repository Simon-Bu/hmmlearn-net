using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm16;

/// <summary>
/// Criterion C1 of ticket HMM-16: inference of <see cref="GmmHmm"/> (<see cref="AbstractHmm.Score"/>,
/// <see cref="AbstractHmm.ScoreSamples"/>, <see cref="AbstractHmm.Decode"/> with Viterbi and Map, <see cref="AbstractHmm.Predict"/>
/// and <see cref="AbstractHmm.PredictProba"/>) reproduces every inference case of the four oracle files (spherical, diag, tied and
/// full covariances; two states of two mixture components) under both implementations, over several sequences including a
/// length-1 sequence. The golden files are the only source of the expected numbers; the model is built with InitParams "" so that
/// the given parameters are used as is, and the default decoding algorithm is Viterbi so that Predict matches the golden Viterbi
/// states.
/// </summary>
public class InferenceParityTests
{
    private static readonly string[] GoldenFiles = { "gmm_spherical", "gmm_diag", "gmm_tied", "gmm_full" };

    public static IEnumerable<object[]> InferenceCases =>
        GoldenFiles.SelectMany(file =>
            Golden.Load(file).GetProperty("inference_cases").Items()
                .Select(c => c.GetProperty("name").AsString())
                .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                    .Select(implementation => new object[] { file, name, implementation })));

    private static JsonElement InferenceCase(string file, string name) =>
        Assert.Single(Golden.Load(file).GetProperty("inference_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    private static GmmHmm Model(JsonElement c, Implementation implementation)
    {
        var type = Enum.Parse<CovarianceType>(c.GetProperty("covariance_type").AsString(), ignoreCase: true);
        var model = new GmmHmm(
            nComponents: c.GetProperty("n_components").AsInt(),
            nMix: c.GetProperty("n_mix").AsInt(),
            covarianceType: type,
            initParams: "",
            implementation: implementation);
        model.StartProb = c.GetProperty("startprob").AsVector();
        model.TransMat = c.GetProperty("transmat").AsMatrix();
        model.Weights = c.GetProperty("weights").AsMatrix();
        model.Means = c.GetProperty("means").AsTensor3();
        model.Covars = c.GetProperty("covars").AsTensor4();
        model.NFeatures = c.GetProperty("n_features").AsInt();
        return model;
    }

    [Theory]
    [MemberData(nameof(InferenceCases))]
    public void inference_matches_the_oracle(string file, string name, Implementation implementation)
    {
        var c = InferenceCase(file, name);
        var what = $"{file}/{name} ({implementation})";
        var model = Model(c, implementation);
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var expected = c.GetProperty("expected");

        Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths), $"{what} Score");

        var (logProb, posteriors) = model.ScoreSamples(x, lengths);
        Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb, $"{what} ScoreSamples log probability");
        Close.Equal(expected.GetProperty("posteriors").AsMatrix(), posteriors, $"{what} ScoreSamples posteriors");

        var (viterbiLogProb, viterbiStates) = model.Decode(x, lengths, DecoderAlgorithm.Viterbi);
        Close.Equal(expected.GetProperty("viterbi").GetProperty("log_prob").AsDouble(), viterbiLogProb, $"{what} Viterbi log probability");
        Assert.Equal(expected.GetProperty("viterbi").GetProperty("states").AsInts(), viterbiStates);

        var (mapLogProb, mapStates) = model.Decode(x, lengths, DecoderAlgorithm.Map);
        Close.Equal(expected.GetProperty("map").GetProperty("log_prob").AsDouble(), mapLogProb, $"{what} MAP log probability");
        Assert.Equal(expected.GetProperty("map").GetProperty("states").AsInts(), mapStates);

        Assert.Equal(expected.GetProperty("viterbi").GetProperty("states").AsInts(), model.Predict(x, lengths));

        Close.Equal(expected.GetProperty("posteriors").AsMatrix(), model.PredictProba(x, lengths), $"{what} PredictProba");
    }
}
