using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm10;

/// <summary>
/// Acceptance tests for the fit of <see cref="CategoricalHmm"/> (ticket HMM-10, criterion C2): Fit reproduces the
/// parameters, the monitor history, the iteration count, the convergence flag, the score, aic and bic of every fit case of
/// oracle/golden/categorical.json with both implementations (priors above and below 1, parameter subsets, a left-to-right
/// model that keeps its zeros, sequences of length 1, an explicit n_features larger than the largest symbol). The M-step is
/// pinned on trivial inputs: EmissionProb = row-normalized max(EmissionProbPrior - 1 + Obs, 0), only the letters of Params
/// are updated, and the caller's arrays are replaced, not mutated.
/// </summary>
public class FitParityTests
{
    /// <summary>Every fit case of the oracle, once per implementation.</summary>
    public static IEnumerable<object[]> FitCases =>
        Golden.Load("categorical").GetProperty("fit_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(impl => new object[] { name, impl }));

    private static JsonElement FitCase(string name) =>
        Assert.Single(Golden.Load("categorical").GetProperty("fit_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    /// <summary>Builds the model with the constructor of the oracle case (init_params "" : the init arrays are the starting
    /// parameters, no initialization runs) and fits it on the oracle data.</summary>
    private static CategoricalHmm BuildAndFit(string name, Implementation implementation)
    {
        var c = FitCase(name);
        var ctor = c.GetProperty("constructor");
        var model = new CategoricalHmm(
            ctor.GetProperty("n_components").AsInt(),
            ctor.Has("startprob_prior") ? ctor.GetProperty("startprob_prior").AsDouble() : 1.0,
            ctor.Has("transmat_prior") ? ctor.GetProperty("transmat_prior").AsDouble() : 1.0,
            ctor.Has("emissionprob_prior") ? ctor.GetProperty("emissionprob_prior").AsDouble() : 1.0,
            ctor.Has("n_features") ? (int?)ctor.GetProperty("n_features").AsInt() : null,
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
        model.EmissionProb = c.GetProperty("init").GetProperty("emissionprob").AsMatrix();
        model.Fit(c.GetProperty("X").AsMatrix(), c.GetProperty("lengths").AsInts());
        return model;
    }

    /// <summary>Fit reproduces startprob, transmat, emissionprob, n_features, the monitor history, the iteration count, the
    /// convergence flag, the score, aic and bic of the oracle case, with both implementations.</summary>
    [Theory]
    [MemberData(nameof(FitCases))]
    public void fit_reproduces_the_oracle(string name, Implementation implementation)
    {
        var c = FitCase(name);
        var model = BuildAndFit(name, implementation);
        var expected = c.GetProperty("expected");
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();

        Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb!, $"{name}: startprob ({implementation})");
        Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat!, $"{name}: transmat ({implementation})");
        Close.Equal(expected.GetProperty("emissionprob").AsMatrix(), model.EmissionProb!, $"{name}: emissionprob ({implementation})");
        Assert.True(model.NFeatures!.Value == expected.GetProperty("n_features").AsInt(),
            $"{name}: n_features ({implementation}): expected {expected.GetProperty("n_features").AsInt()}, got {model.NFeatures!.Value}");
        Close.Equal(expected.GetProperty("history").AsVector(), model.Monitor.History.ToArray(), $"{name}: history ({implementation})");
        Assert.True(model.Monitor.Iter == expected.GetProperty("n_iter_done").AsInt(),
            $"{name}: n_iter_done ({implementation}): expected {expected.GetProperty("n_iter_done").AsInt()}, got {model.Monitor.Iter}");
        Assert.True(model.Monitor.Converged == expected.GetProperty("converged").AsBool(),
            $"{name}: converged ({implementation}): expected {expected.GetProperty("converged").AsBool()}, got {model.Monitor.Converged}");
        Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths), $"{name}: score ({implementation})");
        Close.Equal(expected.GetProperty("aic").AsDouble(), model.Aic(x, lengths), $"{name}: aic ({implementation})");
        Close.Equal(expected.GetProperty("bic").AsDouble(), model.Bic(x, lengths), $"{name}: bic ({implementation})");
    }

    /// <summary>Fit replaces the caller's parameter arrays with new ones and does not mutate the caller's X.</summary>
    [Fact]
    public void fit_replaces_the_callers_parameter_arrays_with_new_ones_and_does_not_mutate_x()
    {
        var c = FitCase("two-states-defaults");
        var ctor = c.GetProperty("constructor");
        var model = new CategoricalHmm(
            ctor.GetProperty("n_components").AsInt(),
            ctor.Has("startprob_prior") ? ctor.GetProperty("startprob_prior").AsDouble() : 1.0,
            ctor.Has("transmat_prior") ? ctor.GetProperty("transmat_prior").AsDouble() : 1.0,
            ctor.Has("emissionprob_prior") ? ctor.GetProperty("emissionprob_prior").AsDouble() : 1.0,
            ctor.Has("n_features") ? (int?)ctor.GetProperty("n_features").AsInt() : null,
            DecoderAlgorithm.Viterbi,
            null,
            ctor.GetProperty("n_iter").AsInt(),
            ctor.GetProperty("tol").AsDouble(),
            false,
            ctor.GetProperty("params").AsString(),
            "",
            Implementation.Log);
        var startProb = c.GetProperty("init").GetProperty("startprob").AsVector();
        var transMat = c.GetProperty("init").GetProperty("transmat").AsMatrix();
        var emissionProb = c.GetProperty("init").GetProperty("emissionprob").AsMatrix();
        var x = c.GetProperty("X").AsMatrix();
        var startProb0 = (double[])startProb.Clone();
        var transMat0 = (double[,])transMat.Clone();
        var emissionProb0 = (double[,])emissionProb.Clone();
        var x0 = (double[,])x.Clone();
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.EmissionProb = emissionProb;

        model.Fit(x, c.GetProperty("lengths").AsInts());

        Assert.NotSame(startProb, model.StartProb);
        Assert.NotSame(transMat, model.TransMat);
        Assert.NotSame(emissionProb, model.EmissionProb);
        Close.Equal(startProb0, startProb, "the caller's startprob is not mutated");
        Close.Equal(transMat0, transMat, "the caller's transmat is not mutated");
        Close.Equal(emissionProb0, emissionProb, "the caller's emissionprob is not mutated");
        Close.Equal(x0, x, "the caller's x is not mutated");
    }

    /// <summary>The M-step with a unit prior row-normalizes Obs: EmissionProb = row-normalized max(1 - 1 + Obs, 0), in a new
    /// array; the parameters not in Params keep their exact reference.</summary>
    [Fact]
    public void the_m_step_with_a_unit_prior_row_normalizes_the_expected_counts()
    {
        var model = new MStepHarness(1.0, "e");
        var startProb = new[] { 0.6, 0.4 };
        var transMat = new double[,] { { 0.7, 0.3 }, { 0.4, 0.6 } };
        var emissionProb = new double[,] { { 0.2, 0.3, 0.5 }, { 0.5, 0.25, 0.25 } };
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.EmissionProb = emissionProb;
        var obs = new double[,] { { 2.0, 3.0, 5.0 }, { 1.0, 1.0, 2.0 } };

        model.DoMStepDirect(new CategoricalStatistics(2, 3) { Obs = obs });

        Close.Equal(ExpectedEmissionProb(1.0, obs), model.EmissionProb!, "emissionprob after the m-step");
        Assert.NotSame(emissionProb, model.EmissionProb);
        Assert.Same(startProb, model.StartProb);
        Assert.Same(transMat, model.TransMat);
    }

    /// <summary>The M-step with a prior below one clips negative entries to zero; a row that clips to all zeros stays zero
    /// (the normalization of a zero sum is the one of the base start and transition updates).</summary>
    [Fact]
    public void the_m_step_with_a_prior_below_one_clips_to_zero_and_keeps_a_clipped_row_zero()
    {
        var model = new MStepHarness(0.5, "e");
        model.StartProb = new[] { 0.5, 0.5 };
        model.TransMat = new double[,] { { 0.5, 0.5 }, { 0.5, 0.5 } };
        model.EmissionProb = new double[,] { { 0.4, 0.3, 0.3 }, { 0.4, 0.3, 0.3 } };
        // Row 0 clips to all zeros (every entry is below 1 - prior); row 1 keeps its positive entries.
        var obs = new double[,] { { 0.2, 0.4, 0.4 }, { 1.8, 0.1, 0.6 } };

        model.DoMStepDirect(new CategoricalStatistics(2, 3) { Obs = obs });

        Close.Equal(ExpectedEmissionProb(0.5, obs), model.EmissionProb!, "emissionprob after the m-step");
    }

    /// <summary>The M-step does not touch the parameters not named by Params: with Params "s" the EmissionProb keeps its
    /// exact reference.</summary>
    [Fact]
    public void the_m_step_keeps_the_parameters_not_in_params_unchanged()
    {
        var model = new MStepHarness(1.0, "s");
        var emissionProb = new double[,] { { 0.2, 0.3, 0.5 }, { 0.5, 0.25, 0.25 } };
        model.StartProb = new[] { 0.6, 0.4 };
        model.TransMat = new double[,] { { 0.7, 0.3 }, { 0.4, 0.6 } };
        model.EmissionProb = emissionProb;
        var stats = new CategoricalStatistics(2, 3)
        {
            Obs = new double[,] { { 1.0, 2.0, 3.0 }, { 3.0, 2.0, 1.0 } },
            Start = new[] { 2.0, 8.0 },
        };

        model.DoMStepDirect(stats);

        Assert.Same(emissionProb, model.EmissionProb);
    }

    /// <summary>NFitScalars counts the free scalar parameters of each letter: 's' = n - 1, 't' = n (n - 1), 'e' = n (m - 1).</summary>
    [Fact]
    public void n_fit_scalars_counts_the_free_parameters_of_each_letter()
    {
        var nComponents = 3;
        var nFeatures = 5;

        var scalars = CategoricalEmission.NFitScalars(nComponents, nFeatures);

        Assert.Equal(3, scalars.Count);
        Assert.Equal(nComponents - 1, scalars['s']);
        Assert.Equal(nComponents * (nComponents - 1), scalars['t']);
        Assert.Equal(nComponents * (nFeatures - 1), scalars['e']);
    }

    /// <summary>AccumulateObs adds the posterior of each sample to the expected count of its symbol for each state (it
    /// accumulates, it does not assign).</summary>
    [Fact]
    public void accumulate_obs_adds_the_posteriors_to_the_expected_counts()
    {
        var obs = new double[,] { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 } };
        var x = new double[,] { { 0 }, { 2 }, { 1 } };
        var posteriors = new double[,] { { 0.7, 0.3 }, { 0.2, 0.8 }, { 0.9, 0.1 } };
        var expected = (double[,])obs.Clone();
        for (var i = 0; i < x.GetLength(0); i++)
        {
            var symbol = (int)x[i, 0];
            for (var c = 0; c < posteriors.GetLength(1); c++)
                expected[c, symbol] += posteriors[i, c];
        }

        CategoricalEmission.AccumulateObs(obs, x, posteriors);

        Close.Equal(expected, obs, "obs after the accumulation");
    }

    /// <summary>The expected EmissionProb after the M-step, computed from the test's own prior and Obs: max(prior - 1 + obs,
    /// 0), each row divided by its own sum, a row of sum 0 left at zero.</summary>
    private static double[,] ExpectedEmissionProb(double prior, double[,] obs)
    {
        var rows = obs.GetLength(0);
        var cols = obs.GetLength(1);
        var result = new double[rows, cols];
        for (var i = 0; i < rows; i++)
        {
            var sum = 0.0;
            for (var j = 0; j < cols; j++)
            {
                result[i, j] = Math.Max(prior - 1.0 + obs[i, j], 0.0);
                sum += result[i, j];
            }
            if (sum == 0.0)
                continue;
            for (var j = 0; j < cols; j++)
                result[i, j] /= sum;
        }
        return result;
    }

    /// <summary>Exposes the protected <see cref="CategoricalHmm.DoMStep"/> of the model under test.</summary>
    private sealed class MStepHarness : CategoricalHmm
    {
        public MStepHarness(double emissionProbPrior, string @params)
            : base(2, 1.0, 1.0, emissionProbPrior, 3, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, @params, "", Implementation.Log)
        {
        }

        public void DoMStepDirect(SufficientStatistics stats) => DoMStep(stats);
    }
}
