using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm9;

/// <summary>
/// Acceptance tests for the EM fit of <see cref="BaseHmm"/> (ticket HMM-9, criterion C2): every fit case of
/// oracle/golden/base.json (default and tight tolerances, the s, t and st parameter sets, priors below and above one,
/// structural zeros of a left-to-right model, sequences of length one, a single iteration) reproduces startprob, transmat,
/// the monitor history, iteration count and convergence flag, score, aic and bic with both implementations. The M-step itself
/// is pinned with direct DoMStep calls on trivial inputs whose exact results are computed from the test's own values: unit
/// priors normalize the statistics, a prior below one clips the negatives (a row that clips to all zeros stays zero), the
/// structural zeros of the current parameters are kept, and only the letters of Params are updated (the other parameter
/// keeps its exact reference); ComputeLowerBound is the log probability itself; and Fit replaces the caller's parameter
/// arrays with new ones instead of mutating them.
/// </summary>
public class FitParityTests
{
    /// <summary>Every fit case of the oracle, once per implementation.</summary>
    public static IEnumerable<object[]> FitCases =>
        Golden.Load("base").GetProperty("fit_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(impl => new object[] { name, impl }));

    private static JsonElement FitCase(string name) =>
        Assert.Single(Golden.Load("base").GetProperty("fit_cases").Items(), c => c.GetProperty("name").AsString() == name);

    /// <summary>Builds the stub with the constructor of the oracle case, initializes it with the init parameters and runs
    /// the fit on the oracle data.</summary>
    private static StubBaseHmm BuildAndFit(string name, Implementation implementation)
    {
        var c = FitCase(name);
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
        model.Fit(c.GetProperty("X").AsMatrix(), c.GetProperty("lengths").AsInts());
        return model;
    }

    /// <summary>A stub with the given parameters, ready for a direct DoMStep call.</summary>
    private static StubBaseHmm MStepModel(int nComponents, string @params, double startProbPrior, double transMatPrior,
        double[] startProb, double[,] transMat) =>
        new(nComponents, startProbPrior, transMatPrior, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, @params, "", Implementation.Log)
        { StartProb = startProb, TransMat = transMat };

    [Theory]
    [MemberData(nameof(FitCases))]
    public void fit_reproduces_the_oracle(string name, Implementation implementation)
    {
        var c = FitCase(name);
        var model = BuildAndFit(name, implementation);
        var expected = c.GetProperty("expected");
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();

        Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb!, name + " startprob");
        Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat!, name + " transmat");
        Close.Equal(expected.GetProperty("history").AsVector(), model.Monitor.History.ToArray(), name + " history");
        Assert.Equal(expected.GetProperty("n_iter_done").AsInt(), model.Monitor.Iter);
        Assert.Equal(expected.GetProperty("converged").AsBool(), model.Monitor.Converged);
        Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths), name + " score");
        Close.Equal(expected.GetProperty("aic").AsDouble(), model.Aic(x, lengths), name + " aic");
        Close.Equal(expected.GetProperty("bic").AsDouble(), model.Bic(x, lengths), name + " bic");
    }

    [Fact]
    public void fit_replaces_the_callers_parameters_with_new_arrays()
    {
        var c = FitCase("defaults");
        var ctor = c.GetProperty("constructor");
        var model = new StubBaseHmm(
            ctor.GetProperty("n_components").AsInt(), 1.0, 1.0, DecoderAlgorithm.Viterbi, null,
            ctor.GetProperty("n_iter").AsInt(), ctor.GetProperty("tol").AsDouble(), false, "st", "", Implementation.Log);
        var startProb = c.GetProperty("init").GetProperty("startprob").AsVector();
        var transMat = c.GetProperty("init").GetProperty("transmat").AsMatrix();
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var startProb0 = (double[])startProb.Clone();
        var transMat0 = (double[,])transMat.Clone();
        var x0 = (double[,])x.Clone();
        model.StartProb = startProb;
        model.TransMat = transMat;

        model.Fit(x, lengths);

        Assert.NotSame(startProb, model.StartProb);
        Assert.NotSame(transMat, model.TransMat);
        Close.Equal(startProb0, startProb, "caller's startprob");
        Close.Equal(transMat0, transMat, "caller's transmat");
        Close.Equal(x0, x, "caller's x");
    }

    [Fact]
    public void m_step_with_unit_priors_normalizes_the_statistics()
    {
        var model = MStepModel(2, "st", 1.0, 1.0, new[] { 0.25, 0.75 }, new double[,] { { 0.9, 0.1 }, { 0.1, 0.9 } });
        var stats = new SufficientStatistics(2)
        {
            NObs = 2,
            Start = new[] { 1.0, 3.0 },
            Trans = new double[,] { { 1.0, 3.0 }, { 3.0, 1.0 } },
        };

        model.CallDoMStep(stats);

        Close.Equal(new[] { 0.25, 0.75 }, model.StartProb!, "startprob");
        Close.Equal(new double[,] { { 0.25, 0.75 }, { 0.75, 0.25 } }, model.TransMat!, "transmat");
    }

    [Fact]
    public void m_step_clips_the_negatives_created_by_a_prior_below_one()
    {
        var model = MStepModel(2, "st", 0.2, 0.2, new[] { 0.5, 0.5 }, new double[,] { { 0.5, 0.5 }, { 0.5, 0.5 } });
        var stats = new SufficientStatistics(2)
        {
            NObs = 2,
            Start = new[] { 0.3, 4.0 },
            Trans = new double[,] { { 0.3, 0.3 }, { 0.3, 4.0 } },
        };

        model.CallDoMStep(stats);

        // prior - 1 + 0.3 < 0 clips to 0; the first trans row clips to all zeros and stays zero.
        Close.Equal(new[] { 0.0, 1.0 }, model.StartProb!, "startprob");
        Close.Equal(new double[,] { { 0.0, 0.0 }, { 0.0, 1.0 } }, model.TransMat!, "transmat");
    }

    [Fact]
    public void m_step_keeps_the_structural_zeros_of_a_left_to_right_model()
    {
        var model = MStepModel(3, "st", 1.0, 1.0, new[] { 1.0, 0.0, 0.0 },
            new double[,] { { 0.5, 0.5, 0.0 }, { 0.0, 0.6, 0.4 }, { 0.0, 0.0, 1.0 } });
        var stats = new SufficientStatistics(3)
        {
            NObs = 3,
            Start = new[] { 0.5, 0.25, 6.25 },
            Trans = new double[,] { { 2.0, 6.0, 9.0 }, { 3.0, 3.0, 2.0 }, { 4.0, 5.0, 6.0 } },
        };

        model.CallDoMStep(stats);

        // entries that are exactly zero in the current parameters stay zero and are not counted in the normalization.
        Close.Equal(new[] { 1.0, 0.0, 0.0 }, model.StartProb!, "startprob");
        Close.Equal(new double[,] { { 0.25, 0.75, 0.0 }, { 0.0, 0.6, 0.4 }, { 0.0, 0.0, 1.0 } }, model.TransMat!, "transmat");
    }

    [Fact]
    public void m_step_only_updates_the_parameters_named_by_params()
    {
        var startProb = new[] { 0.9, 0.1 };
        var model = MStepModel(2, "t", 1.0, 1.0, startProb, new double[,] { { 0.5, 0.5 }, { 0.5, 0.5 } });
        var stats = new SufficientStatistics(2)
        {
            NObs = 2,
            Start = new[] { 1.0, 3.0 },
            Trans = new double[,] { { 1.0, 3.0 }, { 3.0, 1.0 } },
        };

        model.CallDoMStep(stats);

        Assert.Same(startProb, model.StartProb);
        Close.Equal(new double[,] { { 0.25, 0.75 }, { 0.75, 0.25 } }, model.TransMat!, "transmat");
    }

    [Fact]
    public void compute_lower_bound_is_the_log_probability_itself()
    {
        var model = MStepModel(2, "st", 1.0, 1.0, new[] { 0.5, 0.5 }, new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } });
        Close.Equal(2.75, model.CallComputeLowerBound(2.75), "lower bound");
    }
}
