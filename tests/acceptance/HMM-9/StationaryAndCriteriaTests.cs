using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm9;

/// <summary>
/// Acceptance tests for the stationary distribution and the information criteria of <see cref="BaseHmm"/> (ticket HMM-9,
/// criterion C4): GetStationaryDistribution reproduces every case of oracle/golden/stationary.json (the result sums to 1 and
/// is a fixed point of the transition matrix, i.e. transmat^T * stationary = stationary) and throws InvalidOperationException
/// when TransMat is null, with the start probabilities left unset; Aic and Bic equal -2 * Score + 2 * k and -2 * Score + k *
/// ln(rows) with k the sum of ALL the values of GetNFitScalarsPerParam (not only the letters of Params), with and without
/// lengths; they match the aic and bic of every fit case of oracle/golden/base.json with both implementations; and they throw
/// InvalidOperationException when the number of free scalar parameters per letter is unknown.
/// </summary>
public class StationaryAndCriteriaTests
{
    /// <summary>Every stationary case of the oracle.</summary>
    public static IEnumerable<object[]> StationaryCases =>
        Golden.Load("stationary").GetProperty("cases").Items()
            .Select(c => new object[] { c.GetProperty("name").AsString() });

    /// <summary>Every fit case of the oracle, once per implementation.</summary>
    public static IEnumerable<object[]> FitCases =>
        Golden.Load("base").GetProperty("fit_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(impl => new object[] { name, impl }));

    private static JsonElement GoldenCase(string file, string section, string name) =>
        Assert.Single(Golden.Load(file).GetProperty(section).Items(), c => c.GetProperty("name").AsString() == name);

    [Theory]
    [MemberData(nameof(StationaryCases))]
    public void get_stationary_distribution_reproduces_the_oracle(string name)
    {
        var c = GoldenCase("stationary", "cases", name);
        var transMat = c.GetProperty("transmat").AsMatrix();
        var expected = c.GetProperty("expected").AsVector();
        var nc = transMat.GetLength(0);

        var model = new StubBaseHmm(nc, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "", Implementation.Log)
        { TransMat = transMat }; // the stationary distribution only needs the transition matrix
        var stationary = model.GetStationaryDistribution();

        Close.Equal(expected, stationary, name + " stationary");
        var sum = 0.0;
        for (var i = 0; i < nc; i++)
            sum += stationary[i];
        Close.Equal(1.0, sum, name + " sum");
        for (var i = 0; i < nc; i++)
        {
            var product = 0.0;
            for (var j = 0; j < nc; j++)
                product += stationary[j] * transMat[j, i];
            Close.Equal(stationary[i], product, $"{name} invariant[{i}]");
        }
    }

    [Fact]
    public void get_stationary_distribution_throws_when_transmat_is_null()
    {
        var model = new StubBaseHmm(2, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "", Implementation.Log)
        { StartProb = new[] { 0.5, 0.5 } };
        Assert.Throws<InvalidOperationException>(() => model.GetStationaryDistribution());
    }

    /// <summary>The 2-component model of the wikipedia inference case of base.json, with the start probabilities and the
    /// transition matrix set explicitly (no fit, no initialization).</summary>
    private static StubBaseHmm WikipediaModel(string @params)
    {
        var c = GoldenCase("base", "inference_cases", "wikipedia");
        var model = new StubBaseHmm(2, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, @params, "", Implementation.Log);
        model.StartProb = c.GetProperty("startprob").AsVector();
        model.TransMat = c.GetProperty("transmat").AsMatrix();
        return model;
    }

    private static double[,] WikipediaX() => GoldenCase("base", "inference_cases", "wikipedia").GetProperty("X").AsMatrix();

    /// <summary>The sum of ALL the values of GetNFitScalarsPerParam of the stub: (nc - 1) for 's' + nc * (nc - 1) for 't'.</summary>
    private static int AllFreeScalars(int nComponents) => (nComponents - 1) + nComponents * (nComponents - 1);

    [Theory]
    [InlineData("st")]
    [InlineData("s")] // k counts all the values of the dictionary, not only the letters of Params
    public void aic_equals_minus_twice_the_score_plus_twice_k(string @params)
    {
        var x = WikipediaX();
        var model = WikipediaModel(@params);
        var k = AllFreeScalars(model.NComponents);
        var score = model.Score(x);

        Close.Equal(-2 * score + 2 * k, model.Aic(x), "aic");
    }

    [Theory]
    [InlineData("st")]
    [InlineData("s")]
    public void bic_equals_minus_twice_the_score_plus_k_times_ln_of_the_rows(string @params)
    {
        var x = WikipediaX();
        var model = WikipediaModel(@params);
        var k = AllFreeScalars(model.NComponents);
        var score = model.Score(x);

        Close.Equal(-2 * score + k * Math.Log(x.GetLength(0)), model.Bic(x), "bic");
    }

    [Fact]
    public void aic_and_bic_with_lengths_use_the_score_of_those_sequences()
    {
        var x = WikipediaX();
        var lengths = new[] { 3, 2 };
        var model = WikipediaModel("st");
        var k = AllFreeScalars(model.NComponents);
        var score = model.Score(x, lengths);
        Assert.NotEqual(score, model.Score(x)); // the scores of the two sequences and of one differ, so the test is meaningful

        Close.Equal(-2 * score + 2 * k, model.Aic(x, lengths), "aic");
        Close.Equal(-2 * score + k * Math.Log(x.GetLength(0)), model.Bic(x, lengths), "bic");
    }

    [Theory]
    [MemberData(nameof(FitCases))]
    public void aic_and_bic_match_the_aic_and_bic_of_the_fit_cases(string name, Implementation implementation)
    {
        var c = GoldenCase("base", "fit_cases", name);
        var ctor = c.GetProperty("constructor");
        var model = new StubBaseHmm(
            ctor.GetProperty("n_components").AsInt(),
            ctor.Has("startprob_prior") ? ctor.GetProperty("startprob_prior").AsDouble() : 1.0,
            ctor.Has("transmat_prior") ? ctor.GetProperty("transmat_prior").AsDouble() : 1.0,
            DecoderAlgorithm.Viterbi,
            null,
            ctor.GetProperty("n_iter").AsInt(),
            ctor.GetProperty("tol").AsDouble(),
            false,
            ctor.GetProperty("params").AsString(),
            "",
            implementation);
        model.StartProb = c.GetProperty("init").GetProperty("startprob").AsVector();
        model.TransMat = c.GetProperty("init").GetProperty("transmat").AsMatrix();
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        model.Fit(x, lengths);

        var expected = c.GetProperty("expected");
        Close.Equal(expected.GetProperty("aic").AsDouble(), model.Aic(x, lengths), name + " aic");
        Close.Equal(expected.GetProperty("bic").AsDouble(), model.Bic(x, lengths), name + " bic");
    }

    [Fact]
    public void aic_and_bic_throw_when_the_number_of_free_scalars_is_unknown()
    {
        var x = new double[,] { { -0.5, -0.7 }, { -0.1, -0.2 }, { -0.3, -0.4 } };
        var model = new NullScalarsStub
        {
            StartProb = new[] { 0.5, 0.5 },
            TransMat = new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } },
        };

        Assert.Throws<InvalidOperationException>(() => model.Aic(x));
        Assert.Throws<InvalidOperationException>(() => model.Bic(x));
    }
}

/// <summary>
/// A variant of <see cref="StubBaseHmm"/> that does not override GetNFitScalarsPerParam, so it returns null: the information
/// criteria must reject it with InvalidOperationException, and Init must not raise the degenerate-solution warning for it.
/// </summary>
internal sealed class NullScalarsStub : BaseHmm
{
    public NullScalarsStub()
        : base(2, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "", Implementation.Log)
    {
    }

    /// <summary>Runs the protected initialization hook.</summary>
    public void CallInit(double[,] x, int[] lengths) => Init(x, lengths);

    protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

    protected override double[] GenerateSampleFromState(int state, Random random) => new[] { (double)state };
}
