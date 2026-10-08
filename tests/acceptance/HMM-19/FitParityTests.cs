using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm19;

/// <summary>
/// Acceptance tests for the fit of <see cref="VariationalBaseHmm"/> (ticket HMM-19, criterion C2): Fit reproduces the
/// variational posteriors, startprob, transmat, monitor history, iteration count, convergence flag and score of every fit case
/// of oracle/golden/variational_base.json with both implementations (a long run to the iteration cap, a short one, sequences of
/// varying lengths, strong priors, a transitions-only fit that never touches startprob, and the oracle's stub likelihood,
/// which returns a copy of the log-frame-probability observations). The fit must not mutate the caller's init arrays: the
/// M-step and the initialization replace them with new arrays.
/// </summary>
public class FitParityTests
{
    /// <summary>Every fit case of the oracle, once per implementation.</summary>
    public static IEnumerable<object[]> FitCases =>
        Golden.Load("variational_base").GetProperty("fit_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(impl => new object[] { name, impl }));

    private static JsonElement FitCase(string name) =>
        Assert.Single(Golden.Load("variational_base").GetProperty("fit_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    /// <summary>Builds the model with the constructor of the oracle case (init_params "" : the variational arrays are set
    /// explicitly, so no initialization runs and the fit is deterministic) and fits it on the oracle data.</summary>
    private static StubVariationalHmm BuildAndFit(string name, Implementation implementation)
    {
        var c = FitCase(name);
        var ctor = c.GetProperty("constructor");
        var init = c.GetProperty("init");
        var model = new StubVariationalHmm(
            ctor.GetProperty("n_components").AsInt(),
            null,
            null,
            DecoderAlgorithm.Viterbi,
            null,
            ctor.GetProperty("n_iter").AsInt(),
            ctor.GetProperty("tol").AsDouble(),
            false,
            ctor.GetProperty("params").AsString(),
            "",
            implementation);
        model.StartProbPriorValues = init.GetProperty("startprob_prior").AsVector();
        model.StartProbPosterior = init.GetProperty("startprob_posterior").AsVector();
        model.TransMatPriorValues = init.GetProperty("transmat_prior").AsMatrix();
        model.TransMatPosterior = init.GetProperty("transmat_posterior").AsMatrix();
        model.Fit(c.GetProperty("X").AsMatrix(), c.GetProperty("lengths").AsInts());
        return model;
    }

    /// <summary>Fit reproduces the variational posteriors, startprob, transmat, the monitor history, the iteration count, the
    /// convergence flag and the score of the oracle case, with both implementations; an expected value that is null means the
    /// parameter was never set (a params "t" fit leaves startprob unset, and then the score is unavailable).</summary>
    [Theory]
    [MemberData(nameof(FitCases))]
    public void fit_reproduces_the_oracle(string name, Implementation implementation)
    {
        var c = FitCase(name);
        var model = BuildAndFit(name, implementation);
        var expected = c.GetProperty("expected");
        var what = $"{name}/{implementation}";

        Close.Equal(expected.GetProperty("startprob_posterior").AsVector(), model.StartProbPosterior, what + " startprob_posterior");
        Close.Equal(expected.GetProperty("transmat_posterior").AsMatrix(), model.TransMatPosterior, what + " transmat_posterior");
        if (expected.Has("startprob"))
            Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb, what + " startprob");
        else
            Assert.Null(model.StartProb);
        if (expected.Has("transmat"))
            Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat, what + " transmat");
        else
            Assert.Null(model.TransMat);
        Close.Equal(expected.GetProperty("history").AsVector(), model.Monitor.History.ToArray(), what + " history");
        Assert.Equal(expected.GetProperty("n_iter_done").AsInt(), model.Monitor.Iter);
        Assert.Equal(expected.GetProperty("converged").AsBool(), model.Monitor.Converged);
        if (expected.Has("score"))
            Close.Equal(expected.GetProperty("score").AsDouble(),
                model.Score(c.GetProperty("X").AsMatrix(), c.GetProperty("lengths").AsInts()), what + " score");
    }

    /// <summary>Fit does not mutate the caller's init arrays: the posteriors the model ends up with are new arrays, and every
    /// array the caller passed keeps its values.</summary>
    [Fact]
    public void fit_does_not_mutate_the_callers_init_arrays()
    {
        var c = FitCase("ten-iterations");
        var ctor = c.GetProperty("constructor");
        var init = c.GetProperty("init");
        var startProbPriorValues = init.GetProperty("startprob_prior").AsVector();
        var startProbPosterior = init.GetProperty("startprob_posterior").AsVector();
        var transMatPriorValues = init.GetProperty("transmat_prior").AsMatrix();
        var transMatPosterior = init.GetProperty("transmat_posterior").AsMatrix();
        var startProbPriorValues0 = (double[])startProbPriorValues.Clone();
        var startProbPosterior0 = (double[])startProbPosterior.Clone();
        var transMatPriorValues0 = (double[,])transMatPriorValues.Clone();
        var transMatPosterior0 = (double[,])transMatPosterior.Clone();

        var model = new StubVariationalHmm(
            ctor.GetProperty("n_components").AsInt(),
            null,
            null,
            DecoderAlgorithm.Viterbi,
            null,
            ctor.GetProperty("n_iter").AsInt(),
            ctor.GetProperty("tol").AsDouble(),
            false,
            ctor.GetProperty("params").AsString(),
            "",
            Implementation.Log);
        model.StartProbPriorValues = startProbPriorValues;
        model.StartProbPosterior = startProbPosterior;
        model.TransMatPriorValues = transMatPriorValues;
        model.TransMatPosterior = transMatPosterior;
        model.Fit(c.GetProperty("X").AsMatrix(), c.GetProperty("lengths").AsInts());

        Assert.NotSame(startProbPosterior, model.StartProbPosterior);
        Assert.NotSame(transMatPosterior, model.TransMatPosterior);
        Close.Equal(startProbPriorValues0, startProbPriorValues, "caller's startprob_prior_");
        Close.Equal(startProbPosterior0, startProbPosterior, "caller's startprob_posterior_");
        Close.Equal(transMatPriorValues0, transMatPriorValues, "caller's transmat_prior_");
        Close.Equal(transMatPosterior0, transMatPosterior, "caller's transmat_posterior_");
    }
}

/// <summary>
/// The stub of the ported <see cref="VariationalBaseHmm"/> (ticket HMM-19): it inherits the stubbed base class and provides the
/// members the tests need. ComputeLogLikelihood and ComputeSubnormLogLikelihood both return a copy of the observations, which
/// are the log frame probabilities (exactly like the oracle's stub); GenerateSampleFromState returns a one-frame sample; and
/// GetNFitScalarsPerParam reports nc - 1 free scalars for 's' and nc (nc - 1) for 't'.
/// </summary>
internal sealed class StubVariationalHmm : VariationalBaseHmm
{
    public StubVariationalHmm(int nComponents, double? startProbPrior, double? transMatPrior, DecoderAlgorithm algorithm,
        int? randomState, int nIter, double tol, bool verbose, string @params, string initParams, Implementation implementation)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams,
            implementation)
    {
    }

    public void RunInit(double[,] x, int[] lengths) => Init(x, lengths);

    public void RunCheck() => Check();

    public void RunEstepBegin() => EstepBegin();

    public void RunMStep(SufficientStatistics stats) => DoMStep(stats);

    public double Bound(double currLogProb) => ComputeLowerBound(currLogProb);

    public double[]? Sub => StartProbSubnorm;

    public double[,]? SubTrans => TransMatSubnorm;

    protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

    protected override double[,] ComputeSubnormLogLikelihood(double[,] x) => (double[,])x.Clone();

    protected override double[] GenerateSampleFromState(int state, Random random) => new[] { (double)state };

    protected override IReadOnlyDictionary<char, int> GetNFitScalarsPerParam()
    {
        var nc = NComponents;
        return new Dictionary<char, int> { ['s'] = nc - 1, ['t'] = nc * (nc - 1) };
    }
}
