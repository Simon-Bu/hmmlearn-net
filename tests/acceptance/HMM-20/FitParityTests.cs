using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm20;

/// <summary>
/// Acceptance tests for the fit of <see cref="VariationalCategoricalHmm"/> (ticket HMM-20, criterion C1): Fit with explicit initial
/// parameters reproduces the six prior and posterior arrays, startprob, transmat, emissionprob, the monitor history, the number
/// of iterations done, the convergence flag and the score of every fit case of oracle/golden/variational_categorical.json with both
/// the log and the scaling implementations (all of s, t and e; e with t; s with t; a few iterations; sequences of length 1). An
/// expected value that is null means the parameter was never set: a params without 's' leaves startprob unset and a params without
/// 'e' leaves emissionprob unset, in which cases the score is not defined and is not compared. The fit replaces the caller's
/// posterior arrays with new arrays and does not mutate the ones that were passed in.
/// </summary>
public class FitParityTests
{
    private static readonly JsonElement Golden = Support.Golden.Load("variational_categorical");

    /// <summary>Every fit case of the oracle, once per implementation.</summary>
    public static IEnumerable<object[]> FitCases =>
        Golden.GetProperty("fit_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(impl => new object[] { name, impl }));

    private static JsonElement FitCase(string name) =>
        Assert.Single(Golden.GetProperty("fit_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    /// <summary>Builds the model with the constructor of the oracle case (InitParams "": the variational arrays are set explicitly,
    /// so no initialization runs and the fit is deterministic) and fits it on the oracle data.</summary>
    private static ExposedModel BuildAndFit(string name, Implementation implementation)
    {
        var c = FitCase(name);
        var ctor = c.GetProperty("constructor");
        var init = c.GetProperty("init");
        var model = new ExposedModel(
            ctor.GetProperty("n_components").AsInt(),
            null,
            null,
            null,
            ctor.GetProperty("n_features").AsInt(),
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
        model.EmissionProbPriorValues = init.GetProperty("emissionprob_prior").AsMatrix();
        model.EmissionProbPosterior = init.GetProperty("emissionprob_posterior").AsMatrix();
        model.Fit(c.GetProperty("X").AsMatrix(), c.GetProperty("lengths").AsInts());
        return model;
    }

    /// <summary>Fit reproduces the six prior and posterior arrays, startprob, transmat, emissionprob, the monitor history, the
    /// number of iterations done, the convergence flag and the score of every oracle fit case, with both implementations.</summary>
    [Theory]
    [MemberData(nameof(FitCases))]
    public void fit_reproduces_the_oracle(string name, Implementation implementation)
    {
        var c = FitCase(name);
        var model = BuildAndFit(name, implementation);
        var expected = c.GetProperty("expected");
        var what = $"{name}/{implementation}";

        Assert.NotNull(model.StartProbPriorValues);
        Close.Equal(expected.GetProperty("startprob_prior").AsVector(), model.StartProbPriorValues!, what + " startprob_prior");
        Assert.NotNull(model.StartProbPosterior);
        Close.Equal(expected.GetProperty("startprob_posterior").AsVector(), model.StartProbPosterior!, what + " startprob_posterior");
        Assert.NotNull(model.TransMatPriorValues);
        Close.Equal(expected.GetProperty("transmat_prior").AsMatrix(), model.TransMatPriorValues!, what + " transmat_prior");
        Assert.NotNull(model.TransMatPosterior);
        Close.Equal(expected.GetProperty("transmat_posterior").AsMatrix(), model.TransMatPosterior!, what + " transmat_posterior");
        Assert.NotNull(model.EmissionProbPriorValues);
        Close.Equal(expected.GetProperty("emissionprob_prior").AsMatrix(), model.EmissionProbPriorValues!, what + " emissionprob_prior");
        Assert.NotNull(model.EmissionProbPosterior);
        Close.Equal(expected.GetProperty("emissionprob_posterior").AsMatrix(), model.EmissionProbPosterior!, what + " emissionprob_posterior");

        if (expected.Has("startprob"))
        {
            Assert.NotNull(model.StartProb);
            Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb!, what + " startprob");
        }
        else
        {
            Assert.Null(model.StartProb);
        }

        if (expected.Has("transmat"))
        {
            Assert.NotNull(model.TransMat);
            Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat!, what + " transmat");
        }
        else
        {
            Assert.Null(model.TransMat);
        }

        if (expected.Has("emissionprob"))
        {
            Assert.NotNull(model.EmissionProb);
            Close.Equal(expected.GetProperty("emissionprob").AsMatrix(), model.EmissionProb!, what + " emissionprob");
        }
        else
        {
            Assert.Null(model.EmissionProb);
        }

        Close.Equal(expected.GetProperty("history").AsVector(), model.Monitor.History.ToArray(), what + " monitor history");
        Assert.Equal(expected.GetProperty("n_iter_done").AsInt(), model.Monitor.Iter);
        Assert.Equal(expected.GetProperty("converged").AsBool(), model.Monitor.Converged);

        if (expected.Has("score"))
        {
            Close.Equal(expected.GetProperty("score").AsDouble(),
                model.Score(c.GetProperty("X").AsMatrix(), c.GetProperty("lengths").AsInts()), what + " score");
        }
    }

    /// <summary>The fit of a params "ste" case replaces the caller's posterior arrays with new arrays and does not mutate the ones
    /// that were passed in; the fitted emission probabilities are a new matrix whose rows sum to 1.</summary>
    [Fact]
    public void fit_replaces_the_callers_posteriors_with_new_arrays_without_mutating_them()
    {
        const string Name = "defaults-100-iterations-max"; // params "ste": every parameter is updated
        var init = FitCase(Name).GetProperty("init");
        var startPosterior = init.GetProperty("startprob_posterior").AsVector();
        var transPosterior = init.GetProperty("transmat_posterior").AsMatrix();
        var emissionPosterior = init.GetProperty("emissionprob_posterior").AsMatrix();
        var startBefore = (double[])startPosterior.Clone();
        var transBefore = (double[,])transPosterior.Clone();
        var emissionBefore = (double[,])emissionPosterior.Clone();

        var model = BuildAndFit(Name, Implementation.Log);

        Assert.NotSame(startPosterior, model.StartProbPosterior);
        Assert.NotSame(transPosterior, model.TransMatPosterior);
        Assert.NotSame(emissionPosterior, model.EmissionProbPosterior);
        Close.Equal(startBefore, startPosterior, "the caller's startprob posterior must not be mutated");
        Close.Equal(transBefore, transPosterior, "the caller's transmat posterior must not be mutated");
        Close.Equal(emissionBefore, emissionPosterior, "the caller's emission posterior must not be mutated");

        Assert.NotNull(model.EmissionProb);
        var emissionProb = model.EmissionProb!;
        for (var c = 0; c < emissionProb.GetLength(0); c++)
        {
            var sum = 0.0;
            for (var j = 0; j < emissionProb.GetLength(1); j++)
                sum += emissionProb[c, j];
            Close.Equal(1.0, sum, $"the row {c} of the fitted emissionprob must sum to 1");
        }
    }
}

/// <summary>
/// <see cref="VariationalCategoricalHmm"/> with public wrappers on the protected members, so the acceptance tests of ticket HMM-20
/// can call Init, Check, EstepBegin, DoMStep and the emission-specific computations. Defined once, used by all test classes of
/// this ticket (they share the namespace).
/// </summary>
internal sealed class ExposedModel : VariationalCategoricalHmm
{
    public ExposedModel(int nComponents = 1, double? startProbPrior = null, double? transMatPrior = null,
        double? emissionProbPrior = null, int? nFeatures = null, DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi,
        int? randomState = null, int nIter = 100, double tol = 1e-6, bool verbose = false, string @params = "ste",
        string initParams = "ste", Implementation implementation = Implementation.Log)
        : base(nComponents, startProbPrior, transMatPrior, emissionProbPrior, nFeatures, algorithm, randomState, nIter,
            tol, verbose, @params, initParams, implementation)
    {
    }

    public void RunInit(double[,] x, int[] lengths) => Init(x, lengths);

    public void RunCheck() => Check();

    public void RunEstepBegin() => EstepBegin();

    public void RunMStep(SufficientStatistics stats) => DoMStep(stats);

    public double[,]? LogSubnorm => EmissionProbLogSubnorm;

    public double[,] SubnormLogLikelihood(double[,] x) => ComputeSubnormLogLikelihood(x);

    public double Bound(double currLogProb) => ComputeLowerBound(currLogProb);

    public double[,] Likelihood(double[,] x) => ComputeLikelihood(x);

    public double[,] LogLikelihood(double[,] x) => ComputeLogLikelihood(x);

    public SufficientStatistics InitializeStats() => InitializeSufficientStatistics();

    public void Accumulate(SufficientStatistics stats, double[,] x, double[,] lattice, double[,] posteriors,
        double[,] fwdlattice, double[,] bwdlattice)
        => AccumulateSufficientStatistics(stats, x, lattice, posteriors, fwdlattice, bwdlattice);

    public double[] SampleFromState(int state, Random random) => GenerateSampleFromState(state, random);

    public void CheckAndSet(double[,] x) => CheckAndSetNFeatures(x);

    public IReadOnlyDictionary<char, int>? NFitScalars() => GetNFitScalarsPerParam();
}
