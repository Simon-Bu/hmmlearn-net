using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm15;

/// <summary>
/// Acceptance tests for the fit of <see cref="PoissonHmm"/> (ticket HMM-15, criterion C2): for every fit case of
/// oracle/golden/poisson.json and both implementations, Fit(x, lengths) must reproduce the oracle fit: the final startprob,
/// transmat and lambdas, the monitor history, the number of completed iterations, the convergence flag, and the score/Aic/Bic
/// of the result (defaults, tight tolerance, rates only, and the gamma prior that uses the total posterior mass of ALL states).
/// The model is built from the constructor of the case (snake_case keys map to the C# names) with InitParams = "" so that Fit
/// does not reinitialize. The M-step is pinned with a direct DoMStep call: n is the sum of ALL entries of Post (the total mass
/// of every state, not the mass of the state) and each rate is (LambdasPrior + n * Obs[c, f] / Post[c]) / (LambdasWeight + n),
/// the expected values computed in the test from the test's own statistics.
/// </summary>
public class FitParityTests
{
    public static IEnumerable<object[]> FitCases =>
        Golden.Load("poisson").GetProperty("fit_cases").Items()
            .Select(c => c.GetProperty("name").AsString())
            .SelectMany(name => new[] { Implementation.Log, Implementation.Scaling }
                .Select(implementation => new object[] { name, implementation }));

    private static JsonElement Case(string name) =>
        Assert.Single(Golden.Load("poisson").GetProperty("fit_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    /// <summary>The model of the case: its constructor settings, no init yet, and InitParams = "" (Fit must not reinit).</summary>
    private static PoissonHmm NewModel(string name, Implementation implementation)
    {
        var ctor = Case(name).GetProperty("constructor");
        return new PoissonHmm(
            nComponents: ctor.GetProperty("n_components").AsInt(),
            startProbPrior: ctor.Has("startprob_prior") ? ctor.GetProperty("startprob_prior").AsDouble() : 1.0,
            transMatPrior: ctor.Has("transmat_prior") ? ctor.GetProperty("transmat_prior").AsDouble() : 1.0,
            lambdasPrior: ctor.Has("lambdas_prior") ? ctor.GetProperty("lambdas_prior").AsDouble() : 0.0,
            lambdasWeight: ctor.Has("lambdas_weight") ? ctor.GetProperty("lambdas_weight").AsDouble() : 0.0,
            algorithm: DecoderAlgorithm.Viterbi,
            randomState: null,
            nIter: ctor.GetProperty("n_iter").AsInt(),
            tol: ctor.GetProperty("tol").AsDouble(),
            verbose: false,
            @params: ctor.GetProperty("params").AsString(),
            initParams: "",
            implementation);
    }

    /// <summary>Sets the explicit starting parameters of the case (they replace the initialization, InitParams is "").</summary>
    private static void ApplyInit(PoissonHmm model, JsonElement c)
    {
        var init = c.GetProperty("init");
        model.StartProb = init.GetProperty("startprob").AsVector();
        model.TransMat = init.GetProperty("transmat").AsMatrix();
        model.Lambdas = init.GetProperty("lambdas").AsMatrix();
    }

    [Theory]
    [MemberData(nameof(FitCases))]
    public void fit_reproduces_the_oracle(string name, Implementation implementation)
    {
        var c = Case(name);
        var what = $"{name} ({implementation})";
        var model = NewModel(name, implementation);
        ApplyInit(model, c);
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();

        model.Fit(x, lengths);

        var expected = c.GetProperty("expected");
        Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb!, $"{what} startprob");
        Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat!, $"{what} transmat");
        Close.Equal(expected.GetProperty("lambdas").AsMatrix(), model.Lambdas!, $"{what} lambdas");
        Close.Equal(expected.GetProperty("history").AsVector(), model.Monitor.History.ToArray(), $"{what} history");
        Assert.Equal(expected.GetProperty("n_iter_done").AsInt(), model.Monitor.Iter);
        Assert.Equal(expected.GetProperty("converged").AsBool(), model.Monitor.Converged);
        Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths), $"{what} score");
        Close.Equal(expected.GetProperty("aic").AsDouble(), model.Aic(x, lengths), $"{what} aic");
        Close.Equal(expected.GetProperty("bic").AsDouble(), model.Bic(x, lengths), $"{what} bic");
    }

    /// <summary>The params of the "defaults" case is "stl": every parameter is updated by the M-step, and every update
    /// replaces the caller's array with a new one (no in-place mutation, the ticket constraint on Fit and on the M-step).</summary>
    [Fact]
    public void fit_replaces_the_callers_arrays_instead_of_mutating_them()
    {
        const string name = "defaults";
        var c = Case(name);
        var model = NewModel(name, Implementation.Log);
        ApplyInit(model, c);
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var startProb = model.StartProb!;
        var transMat = model.TransMat!;
        var lambdas = model.Lambdas!;
        var startProb0 = (double[])startProb.Clone();
        var transMat0 = (double[,])transMat.Clone();
        var lambdas0 = (double[,])lambdas.Clone();

        model.Fit(x, lengths);

        Assert.NotSame(startProb, model.StartProb);
        Assert.NotSame(transMat, model.TransMat);
        Assert.NotSame(lambdas, model.Lambdas);
        Close.Equal(startProb0, startProb, "the caller's startprob");
        Close.Equal(transMat0, transMat, "the caller's transmat");
        Close.Equal(lambdas0, lambdas, "the caller's lambdas");
    }

    /// <summary>
    /// The M-step with a gamma prior: n is the sum of ALL entries of Post (the total posterior mass of every state, as hmmlearn
    /// does - not the mass of the state) and each rate is (LambdasPrior + n * Obs[c, f] / Post[c]) / (LambdasWeight + n); with
    /// the default prior 0 and 0 it is the weighted mean of the counts. The expected matrix is computed in the test from the
    /// test's own statistics; the update replaces Lambdas with a new array and leaves the caller's array unchanged.
    /// </summary>
    [Fact]
    public void the_m_step_uses_the_total_posterior_mass_of_all_states()
    {
        const double lambdasPrior = 1.5;
        const double lambdasWeight = 0.75;
        var model = new MStepProbe(lambdasPrior, lambdasWeight);
        model.StartProb = new[] { 0.6, 0.4 };
        model.TransMat = new double[,] { { 0.7, 0.3 }, { 0.4, 0.6 } };
        var lambdas = new double[2, 3] { { 1.0, 1.0, 1.0 }, { 2.0, 2.0, 2.0 } };
        model.Lambdas = lambdas;
        var stats = new PoissonStatistics(2, 3)
        {
            Post = new[] { 3.0, 1.0 },
            Obs = new double[2, 3] { { 9.0, 6.0, 3.0 }, { 4.0, 2.0, 0.5 } },
        };
        var lambdas0 = (double[,])lambdas.Clone();

        model.CallDoMStep(stats);

        var n = 0.0;
        for (var c = 0; c < stats.Post.Length; c++)
            n += stats.Post[c];
        var expected = new double[2, 3];
        for (var c = 0; c < 2; c++)
            for (var f = 0; f < 3; f++)
                expected[c, f] = (lambdasPrior + n * stats.Obs[c, f] / stats.Post[c]) / (lambdasWeight + n);
        Close.Equal(expected, model.Lambdas!, "the updated lambdas");
        Assert.NotSame(lambdas, model.Lambdas);
        Close.Equal(lambdas0, lambdas, "the caller's lambdas");
    }

    /// <summary>The number of free scalars per parameter letter is 's' = nc - 1, 't' = nc * (nc - 1), 'l' = nc * nf when
    /// NFeatures is known (the counts of Aic and Bic), and null while NFeatures is unknown.</summary>
    [Fact]
    public void the_number_of_fit_scalars_per_param_is_the_documented_count()
    {
        var model = new MStepProbe(0.0, 0.0);

        Assert.Null(model.CallNFitScalars());

        model.NFeatures = 3;
        var per = model.CallNFitScalars()!;
        var nc = model.NComponents;
        Assert.Equal(nc - 1, per['s']);
        Assert.Equal(nc * (nc - 1), per['t']);
        Assert.Equal(nc * 3, per['l']);
    }

    /// <summary>A Poisson model with the given M-step settings, ready for a direct DoMStep call (params "l": the base M-step
    /// is a no-op, the update is the one of the rates).</summary>
    private sealed class MStepProbe : PoissonHmm
    {
        public MStepProbe(double lambdasPrior, double lambdasWeight)
            : base(2, 1.0, 1.0, lambdasPrior, lambdasWeight, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "l", "", Implementation.Log)
        {
        }

        public void CallDoMStep(SufficientStatistics stats) => DoMStep(stats);

        public IReadOnlyDictionary<char, int>? CallNFitScalars() => GetNFitScalarsPerParam();
    }
}
