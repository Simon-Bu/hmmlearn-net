using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm11;

/// <summary>
/// Acceptance tests for the fitting of <see cref="MultinomialHmm"/> (ticket HMM-11, criterion C2): Fit with the oracle's
/// constructor and initial parameters reproduces startprob, transmat, emissionprob, n_features, the monitor history and
/// iteration count, converged, score, AIC and BIC of oracle/golden/multinomial.json with both implementations (default
/// parameters, a tight tolerance, emission-only parameters, and inferred trial counts that vary per sample). The M-step
/// row-normalizes the observed counts without a prior (a row of zeros stays zeros), updates only the letters in Params and
/// replaces the arrays in place; GetNFitScalarsPerParam counts nc - 1, nc * (nc - 1) and nc * (nf - 1) free scalars for the
/// letters s, t and e.
/// </summary>
public class FitParityTests
{
    /// <summary>Every fit case of the oracle, once per implementation.</summary>
    public static IEnumerable<object[]> FitCases =>
        Golden.Load("multinomial").GetProperty("fit_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(impl => new object[] { name, impl }));

    private static JsonElement FitCase(string name) =>
        Assert.Single(Golden.Load("multinomial").GetProperty("fit_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    /// <summary>Builds the model of the oracle case from its constructor (the golden constructors omit the Dirichlet priors,
    /// so they keep the default 1.0) and the given implementation, sets the init arrays (init_params "" : nothing is
    /// initialized) and fits.</summary>
    private static MultinomialHmm BuildAndFit(string name, Implementation implementation)
    {
        var c = FitCase(name);
        var ctor = c.GetProperty("constructor");
        var model = new MultinomialHmm(
            ctor.GetProperty("n_components").AsInt(),
            ctor.Has("n_trials") ? (int?)ctor.GetProperty("n_trials").AsInt() : null,
            ctor.Has("startprob_prior") ? ctor.GetProperty("startprob_prior").AsDouble() : 1.0,
            ctor.Has("transmat_prior") ? ctor.GetProperty("transmat_prior").AsDouble() : 1.0,
            DecoderAlgorithm.Viterbi, null,
            ctor.GetProperty("n_iter").AsInt(),
            ctor.GetProperty("tol").AsDouble(),
            false,
            ctor.GetProperty("params").AsString(),
            "",
            implementation);
        model.StartProb = c.GetProperty("init").GetProperty("startprob").AsVector();
        model.TransMat = c.GetProperty("init").GetProperty("transmat").AsMatrix();
        model.EmissionProb = c.GetProperty("init").GetProperty("emissionprob").AsMatrix();
        model.Fit(c.GetProperty("X").AsMatrix(), c.GetProperty("lengths").AsInts());
        return model;
    }

    /// <summary>Fit reproduces the expected parameters, n_features, monitor history and iteration count, converged, score,
    /// AIC and BIC of the oracle case, with both implementations.</summary>
    [Theory]
    [MemberData(nameof(FitCases))]
    public void fit_reproduces_the_oracle(string name, Implementation implementation)
    {
        var c = FitCase(name);
        var model = BuildAndFit(name, implementation);
        var expected = c.GetProperty("expected");
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();

        Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb!,
            $"{name}: startprob ({implementation})");
        Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat!,
            $"{name}: transmat ({implementation})");
        Close.Equal(expected.GetProperty("emissionprob").AsMatrix(), model.EmissionProb!,
            $"{name}: emissionprob ({implementation})");
        Assert.True(model.NFeatures == expected.GetProperty("n_features").AsInt(),
            $"{name}: n_features ({implementation}): expected {expected.GetProperty("n_features").AsInt()}, got {model.NFeatures}");
        Close.Equal(expected.GetProperty("history").AsVector(), model.Monitor.History.ToArray(),
            $"{name}: monitor history ({implementation})");
        Assert.True(model.Monitor.Iter == expected.GetProperty("n_iter_done").AsInt(),
            $"{name}: monitor iterations ({implementation}): expected {expected.GetProperty("n_iter_done").AsInt()}, got {model.Monitor.Iter}");
        Assert.True(model.Monitor.Converged == expected.GetProperty("converged").AsBool(),
            $"{name}: monitor converged ({implementation}): expected {expected.GetProperty("converged").AsBool()}, got {model.Monitor.Converged}");
        Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths),
            $"{name}: score ({implementation})");
        Close.Equal(expected.GetProperty("aic").AsDouble(), model.Aic(x, lengths),
            $"{name}: aic ({implementation})");
        Close.Equal(expected.GetProperty("bic").AsDouble(), model.Bic(x, lengths),
            $"{name}: bic ({implementation})");
    }

    /// <summary>Exposes the protected <see cref="MultinomialHmm.DoMStep"/> and
    /// <see cref="AbstractHmm.GetNFitScalarsPerParam"/> of the model under test.</summary>
    private sealed class MStepHarness : MultinomialHmm
    {
        public MStepHarness(string @params)
            : base(2, 5, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, @params, "", Implementation.Log)
        {
            StartProb = new[] { 0.4, 0.6 };
            TransMat = new double[,] { { 0.7, 0.3 }, { 0.5, 0.5 } };
            EmissionProb = new double[,] { { 0.2, 0.3, 0.5 }, { 0.1, 0.4, 0.5 } };
            NFeatures = 3;
        }

        public void MStep(SufficientStatistics stats) => DoMStep(stats);
        public IReadOnlyDictionary<char, int>? NFitScalars() => GetNFitScalarsPerParam();
    }

    /// <summary>With params "ste" and priors 1.0, the M-step row-normalizes the observed counts (no prior), normalizes the
    /// start and the transitions, and replaces the arrays (it does not reuse the statistics arrays nor the old parameters).</summary>
    [Fact]
    public void the_m_step_row_normalizes_the_statistics_without_a_prior()
    {
        var model = new MStepHarness("ste");
        var stats = new CategoricalStatistics(2, 3);
        stats.Obs[0, 0] = 2.0; stats.Obs[0, 1] = 3.0; stats.Obs[0, 2] = 5.0;
        stats.Obs[1, 0] = 1.0; stats.Obs[1, 1] = 1.0; stats.Obs[1, 2] = 2.0;
        stats.Start[0] = 1.0; stats.Start[1] = 3.0;
        stats.Trans[0, 0] = 1.0; stats.Trans[0, 1] = 1.0;
        stats.Trans[1, 0] = 2.0; stats.Trans[1, 1] = 2.0;
        var startProb = model.StartProb;

        model.MStep(stats);

        Close.Equal(RowNormalize(stats.Obs), model.EmissionProb!, "emissionprob after the M-step");
        Close.Equal(NormalizeCopy(stats.Start), model.StartProb!, "startprob after the M-step");
        Close.Equal(RowNormalize(stats.Trans), model.TransMat!, "transmat after the M-step");
        Assert.NotSame(startProb, model.StartProb);
        Assert.NotSame(stats.Obs, model.EmissionProb);
    }

    /// <summary>An observed row of zeros stays all zeros after the M-step (no prior to smooth it), and the other rows are
    /// row-normalized.</summary>
    [Fact]
    public void a_row_of_zero_observed_counts_stays_zero_in_the_m_step()
    {
        var model = new MStepHarness("e");
        var stats = new CategoricalStatistics(2, 3);
        stats.Obs[0, 0] = 1.0;
        stats.Obs[0, 1] = 1.0;
        // The second row of Obs stays zero.

        model.MStep(stats);

        var normalized = RowNormalize(stats.Obs);
        for (var j = 0; j < normalized.GetLength(1); j++)
            Close.Equal(normalized[0, j], model.EmissionProb![0, j], $"emissionprob[0,{j}] after the M-step");
        for (var j = 0; j < model.EmissionProb!.GetLength(1); j++)
            Assert.True(model.EmissionProb[1, j] == 0.0,
                $"the zero row of observed counts must stay zero in the M-step, got {model.EmissionProb[1, j]:R} at [1,{j}]");
    }

    /// <summary>The M-step updates only the letters in Params: with params "s" the transmat and the emissionprob keep their
    /// exact arrays, while the startprob is replaced by the normalized start.</summary>
    [Fact]
    public void the_m_step_updates_only_the_parameters_in_params()
    {
        var model = new MStepHarness("s");
        var transMat = model.TransMat;
        var emissionProb = model.EmissionProb;
        var stats = new CategoricalStatistics(2, 3);
        stats.Start[0] = 1.0;
        stats.Start[1] = 3.0;

        model.MStep(stats);

        Assert.Same(transMat, model.TransMat);
        Assert.Same(emissionProb, model.EmissionProb);
        Close.Equal(NormalizeCopy(stats.Start), model.StartProb!, "startprob after the M-step");
    }

    /// <summary>Exposes the protected <see cref="AbstractHmm.GetNFitScalarsPerParam"/> of the model under test.</summary>
    private sealed class ScalarHarness : MultinomialHmm
    {
        public ScalarHarness(int nComponents, int? nFeatures)
            : base(nComponents, null, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "ste", "", Implementation.Log)
        {
            NFeatures = nFeatures;
        }

        public IReadOnlyDictionary<char, int>? NFitScalars() => GetNFitScalarsPerParam();
    }

    /// <summary>GetNFitScalarsPerParam is null before the number of features is known, and then counts nc - 1, nc * (nc - 1)
    /// and nc * (nf - 1) free scalars for the letters s, t and e.</summary>
    [Fact]
    public void n_fit_scalars_per_param_counts_the_free_scalars_of_each_letter()
    {
        var model = new ScalarHarness(3, null);
        Assert.Null(model.NFitScalars());

        model.NFeatures = 5;
        var result = model.NFitScalars()!;
        Assert.Equal(3, result.Count);
        Assert.True(result['s'] == model.NComponents - 1,
            $"s: expected {model.NComponents - 1}, got {result['s']}");
        Assert.True(result['t'] == model.NComponents * (model.NComponents - 1),
            $"t: expected {model.NComponents * (model.NComponents - 1)}, got {result['t']}");
        Assert.True(result['e'] == model.NComponents * (model.NFeatures - 1),
            $"e: expected {model.NComponents * (model.NFeatures - 1)}, got {result['e']}");
    }

    /// <summary>Fit replaces the caller's startprob, transmat and emissionprob with new arrays and does not mutate the
    /// observations.</summary>
    [Fact]
    public void fit_replaces_the_callers_arrays_and_does_not_mutate_the_observations()
    {
        var c = FitCase("five-trials-defaults");
        var startProb = c.GetProperty("init").GetProperty("startprob").AsVector();
        var transMat = c.GetProperty("init").GetProperty("transmat").AsMatrix();
        var emissionProb = c.GetProperty("init").GetProperty("emissionprob").AsMatrix();
        var x = c.GetProperty("X").AsMatrix();
        var xBefore = (double[,])x.Clone();

        var model = new MultinomialHmm(2, 5, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "ste", "", Implementation.Log);
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.EmissionProb = emissionProb;
        model.Fit(x, c.GetProperty("lengths").AsInts());

        Assert.NotSame(startProb, model.StartProb);
        Assert.NotSame(transMat, model.TransMat);
        Assert.NotSame(emissionProb, model.EmissionProb);
        Close.Equal(xBefore, x, "the observations are not mutated by Fit");
    }

    /// <summary>Each row divided by its sum; a row of zeros stays zeros (the M-step contract for the emission counts).</summary>
    private static double[,] RowNormalize(double[,] a)
    {
        var result = new double[a.GetLength(0), a.GetLength(1)];
        for (var i = 0; i < a.GetLength(0); i++)
        {
            var sum = 0.0;
            for (var j = 0; j < a.GetLength(1); j++)
                sum += a[i, j];
            for (var j = 0; j < a.GetLength(1); j++)
                result[i, j] = sum > 0.0 ? a[i, j] / sum : 0.0;
        }
        return result;
    }

    private static double[] NormalizeCopy(double[] a)
    {
        var result = (double[])a.Clone();
        Utils.Normalize(result);
        return result;
    }
}
