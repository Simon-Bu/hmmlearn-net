using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm14;

/// <summary>
/// Acceptance tests for learning a Gaussian HMM (ticket HMM-14, criterion C1): for every fit case of the four oracle
/// files (gaussian_spherical, gaussian_diag, gaussian_tied, gaussian_full) and both implementations, Fit(x, lengths)
/// must reproduce the oracle fit: the final startprob, transmat, means, covars, the monitor history, the number of
/// completed iterations, the convergence flag, and the score/Aic/Bic of the result. The model is built from the
/// constructor of the case (n_components, covariance_type, priors/weights, n_iter, tol, params) with the case's init
/// parameters and InitParams = "" so that Fit does not reinitialize. The M-step itself is pinned with direct DoMStep
/// calls on trivial inputs whose exact results are computed in the test from the test's own values: the means update,
/// the scalar CovarsPrior added to EVERY entry (off-diagonal ones included) of the Tied and Full covariance matrices,
/// the per-state denominators, the Spherical mean-over-features with its 1e-5 denominator floor, and the replacement
/// of Means by a new array instead of an in-place mutation.
/// </summary>
public class FitParityTests
{
    private static readonly string[] Files = ["gaussian_spherical", "gaussian_diag", "gaussian_tied", "gaussian_full"];

    public static IEnumerable<object[]> FitCases
    {
        get
        {
            foreach (var file in Files)
            {
                foreach (var c in Golden.Load(file).GetProperty("fit_cases").Items())
                    foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
                        yield return new object[] { file, c.GetProperty("name").AsString(), implementation };
            }
        }
    }

    private static JsonElement Case(string file, string name) =>
        Golden.Load(file).GetProperty("fit_cases")
            .Items()
            .Single(e => e.GetProperty("name").AsString() == name);

    /// <summary>The model of the case: its constructor settings, no init yet, and InitParams = "" (Fit must not reinit).</summary>
    private static GaussianHmm NewModel(string file, string name, Implementation implementation)
    {
        var ctor = Case(file, name).GetProperty("constructor");
        return new GaussianHmm(
            nComponents: ctor.GetProperty("n_components").AsInt(),
            covarianceType: Enum.Parse<CovarianceType>(ctor.GetProperty("covariance_type").AsString(), ignoreCase: true),
            minCovar: 1e-3,
            startProbPrior: ctor.Has("startprob_prior") ? ctor.GetProperty("startprob_prior").AsDouble() : 1.0,
            transMatPrior: ctor.Has("transmat_prior") ? ctor.GetProperty("transmat_prior").AsDouble() : 1.0,
            meansPrior: ctor.Has("means_prior") ? ctor.GetProperty("means_prior").AsDouble() : 0.0,
            meansWeight: ctor.Has("means_weight") ? ctor.GetProperty("means_weight").AsDouble() : 0.0,
            covarsPrior: ctor.Has("covars_prior") ? ctor.GetProperty("covars_prior").AsDouble() : 1e-2,
            covarsWeight: ctor.Has("covars_weight") ? ctor.GetProperty("covars_weight").AsDouble() : 1.0,
            algorithm: DecoderAlgorithm.Viterbi,
            randomState: null,
            nIter: ctor.GetProperty("n_iter").AsInt(),
            tol: ctor.GetProperty("tol").AsDouble(),
            verbose: false,
            @params: ctor.GetProperty("params").AsString(),
            initParams: "",
            implementation);
    }

    private static void ApplyInit(GaussianHmm model, JsonElement c)
    {
        var init = c.GetProperty("init");
        model.StartProb = init.GetProperty("startprob").AsVector();
        model.TransMat = init.GetProperty("transmat").AsMatrix();
        model.Means = init.GetProperty("means").AsMatrix();
        model.Covars = init.GetProperty("covars").AsTensor3();
    }

    [Theory]
    [MemberData(nameof(FitCases))]
    public void fit_reproduces_the_oracle(string file, string name, Implementation implementation)
    {
        var c = Case(file, name);
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var model = NewModel(file, name, implementation);
        ApplyInit(model, c);

        model.Fit(x, lengths);

        var expected = c.GetProperty("expected");
        var what = $"{file}/{name} ";
        Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb!, what + "startprob");
        Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat!, what + "transmat");
        Close.Equal(expected.GetProperty("means").AsMatrix(), model.Means!, what + "means");
        Close.Equal(expected.GetProperty("covars").AsTensor3(), model.Covars!, what + "covars");
        Close.Equal(expected.GetProperty("history").AsVector(), model.Monitor.History.ToArray(), what + "history");
        Assert.Equal(expected.GetProperty("n_iter_done").AsInt(), model.Monitor.Iter);
        Assert.Equal(expected.GetProperty("converged").AsBool(), model.Monitor.Converged);
        Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths), what + "score");
        Close.Equal(expected.GetProperty("aic").AsDouble(), model.Aic(x, lengths), what + "aic");
        Close.Equal(expected.GetProperty("bic").AsDouble(), model.Bic(x, lengths), what + "bic");
    }

    [Fact]
    public void fit_replaces_the_callers_arrays_instead_of_mutating_them()
    {
        var file = "gaussian_tied";
        var name = "priors";
        var c = Case(file, name);
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var init = c.GetProperty("init");
        var startProb = init.GetProperty("startprob").AsVector();
        var transMat = init.GetProperty("transmat").AsMatrix();
        var means = init.GetProperty("means").AsMatrix();
        var covars = init.GetProperty("covars").AsTensor3();
        var x0 = (double[,])x.Clone();
        var startProb0 = (double[])startProb.Clone();
        var transMat0 = (double[,])transMat.Clone();
        var means0 = (double[,])means.Clone();
        var covars0 = (double[,,])covars.Clone();

        var model = NewModel(file, name, Implementation.Log);
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.Means = means;
        model.Covars = covars;
        model.Fit(x, lengths);

        // params of this case is "stmc": every parameter is updated, each with a new array.
        Assert.NotSame(startProb, model.StartProb);
        Assert.NotSame(transMat, model.TransMat);
        Assert.NotSame(means, model.Means);
        Close.Equal(x0, x, "caller's x");
        Close.Equal(startProb0, startProb, "caller's startprob");
        Close.Equal(transMat0, transMat, "caller's transmat");
        Close.Equal(means0, means, "caller's means");
        Close.Equal(covars0, covars, "caller's covars");

        var expected = c.GetProperty("expected");
        Close.Equal(expected.GetProperty("means").AsMatrix(), model.Means!, "fitted means");
        Close.Equal(expected.GetProperty("covars").AsTensor3(), model.Covars!, "fitted covars");
    }

    [Theory]
    [InlineData(CovarianceType.Tied)]
    [InlineData(CovarianceType.Full)]
    public void m_step_adds_the_scalar_covars_prior_to_every_entry_of_tied_and_full_matrices(CovarianceType type)
    {
        var means = new double[2, 2]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
        };
        var model = new MStepProbe(type, "mc", 0.5, 2.0, 0.25, 3.0);
        model.StartProb = new[] { 0.5, 0.5 };
        model.TransMat = new double[2, 2]
        {
            { 0.5, 0.5 },
            { 0.5, 0.5 },
        };
        model.Means = means;
        model.Covars = type == CovarianceType.Tied
            ? new double[1, 2, 2] { { { 1.0, 0.2 }, { 0.2, 1.0 } } }
            : new double[2, 2, 2]
            {
                { { 1.0, 0.2 }, { 0.2, 1.0 } },
                { { 1.0, 0.1 }, { 0.1, 1.0 } },
            };
        model.NFeatures = 2;

        // Statistics that come from data (a hand-made Obs and ObsObsT that do not belong to any data set give negative variances,
        // which the validating Covars setter rightly rejects): four weighted observations of two states, accumulated by the helper.
        var x = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 }, { 2.0, 2.0 }, { 1.0, 3.0 } };
        var posteriors = new double[,] { { 0.9, 0.1 }, { 0.6, 0.4 }, { 0.2, 0.8 }, { 0.5, 0.5 } };
        var stats = new GaussianStatistics(2, 2, type);
        GaussianEmission.Accumulate(stats, x, posteriors, type, true, true);
        var means0 = (double[,])means.Clone();

        model.CallDoMStep(stats);

        var expectedCovars = ExpectedCovars(stats, type, 0.5, 2.0, 0.25, 3.0);
        Close.Equal(ExpectedMeans(stats, 0.5, 2.0), model.Means!, "means update");
        Close.Equal(means0, means, "caller's means");
        Assert.NotSame(means, model.Means);
        Close.Equal(expectedCovars, model.Covars!, "covars update");
        // the quirk: the prior is a scalar added to every entry, off-diagonal ones included.
        var offDiagonal = type == CovarianceType.Tied ? expectedCovars[0, 0, 1] : expectedCovars[1, 0, 1];
        Close.Equal(offDiagonal, model.Covars![type == CovarianceType.Tied ? 0 : 1, 0, 1],
            "off-diagonal entry of the covariance update");
    }

    [Fact]
    public void m_step_spherical_averages_the_features_and_floors_the_state_denominator_at_1e_5()
    {
        var means = new double[2, 2]
        {
            { 1.0, 1.0 },
            { 1.0, 1.0 },
        };
        var model = new MStepProbe(CovarianceType.Spherical, "mc", 0.0, 1.0, 0.5, 1.0);
        model.StartProb = new[] { 0.5, 0.5 };
        model.TransMat = new double[2, 2]
        {
            { 0.5, 0.5 },
            { 0.5, 0.5 },
        };
        model.Means = means;
        model.Covars = new double[2, 1, 1]
        {
            { { 1.0 } },
            { { 1.0 } },
        };
        model.NFeatures = 2;

        // state 0 was never observed (Post[0] = 0): its denominator hits the 1e-5 floor.
        var stats = new GaussianStatistics(2, 2, CovarianceType.Spherical)
        {
            NObs = 2,
            Start = new[] { 1.0, 2.0 },
            Trans = new double[2, 2]
            {
                { 0.5, 0.5 },
                { 0.5, 0.5 },
            },
            Post = new[] { 0.0, 3.0 },
            Obs = new double[2, 2]
            {
                { 0.0, 0.0 },
                { 1.0, 2.0 },
            },
            Obs2 = new double[2, 2]
            {
                { 0.0, 0.0 },
                { 3.0, 4.0 },
            },
        };

        model.CallDoMStep(stats);

        var expected = new double[2, 1, 1];
        var meansWeight = 1.0;
        var meansPrior = 0.0;
        var covarsPrior = 0.5;
        var covarsWeight = 1.0;
        for (var c = 0; c < 2; c++)
        {
            var mean = 0.0;
            for (var f = 0; f < 2; f++)
            {
                var m = (meansWeight * meansPrior + stats.Obs[c, f]) / (meansWeight + stats.Post[c]);
                var cn = meansWeight * (m - meansPrior) * (m - meansPrior)
                    + stats.Obs2[c, f]
                    - 2.0 * m * stats.Obs[c, f]
                    + m * m * stats.Post[c];
                mean += (covarsPrior + cn) / Math.Max(Math.Max(covarsWeight - 1.0, 0.0) + stats.Post[c], 1e-5);
            }
            expected[c, 0, 0] = mean / 2.0;
        }
        Close.Equal(expected, model.Covars!, "spherical covars update");
        Close.Equal((covarsPrior / 1e-5), model.Covars![0, 0, 0], "unobserved state: (prior + 0) over the 1e-5 floor");
    }

    /// <summary>The means update of the M-step, from the spec: (meansWeight * meansPrior + Obs[c, f]) / (meansWeight + Post[c]).</summary>
    private static double[,] ExpectedMeans(GaussianStatistics stats, double meansPrior, double meansWeight)
    {
        var result = new double[stats.Post.Length, stats.Obs.GetLength(1)];
        for (var c = 0; c < result.GetLength(0); c++)
            for (var f = 0; f < result.GetLength(1); f++)
                result[c, f] = (meansWeight * meansPrior + stats.Obs[c, f]) / (meansWeight + stats.Post[c]);
        return result;
    }

    /// <summary>
    /// The Tied and Full covariance updates of the M-step, from the spec: c_n[c] = meansWeight * outer(meandiff, meandiff)
    /// + ObsObsT[c] - outer(Obs[c], means[c]) - outer(means[c], Obs[c]) + outer(means[c], means[c]) * Post[c], the means
    /// just updated; the scalar covarsPrior added to EVERY entry; cvweight = max(covarsWeight - n_features, 0); Tied: one
    /// matrix over the sum of the c_n, Full: one matrix per state.
    /// </summary>
    private static double[,,] ExpectedCovars(GaussianStatistics stats, CovarianceType type,
        double meansPrior, double meansWeight, double covarsPrior, double covarsWeight)
    {
        var nc = stats.Post.Length;
        var nf = stats.Obs.GetLength(1);
        var means = ExpectedMeans(stats, meansPrior, meansWeight);
        var per = new double[nc, nf, nf];
        for (var c = 0; c < nc; c++)
            for (var i = 0; i < nf; i++)
                for (var j = 0; j < nf; j++)
                    per[c, i, j] = meansWeight * (means[c, i] - meansPrior) * (means[c, j] - meansPrior)
                        + stats.ObsObsT![c, i, j]
                        - stats.Obs[c, i] * means[c, j]
                        - stats.Obs[c, j] * means[c, i]
                        + means[c, i] * means[c, j] * stats.Post[c];

        if (type == CovarianceType.Tied)
        {
            var result = new double[1, nf, nf];
            for (var i = 0; i < nf; i++)
                for (var j = 0; j < nf; j++)
                {
                    var sum = 0.0;
                    var post = 0.0;
                    for (var c = 0; c < nc; c++)
                    {
                        sum += per[c, i, j];
                        post += stats.Post[c];
                    }
                    result[0, i, j] = (covarsPrior + sum) / (Math.Max(covarsWeight - nf, 0.0) + post);
                }
            return result;
        }

        var full = new double[nc, nf, nf];
        for (var c = 0; c < nc; c++)
            for (var i = 0; i < nf; i++)
                for (var j = 0; j < nf; j++)
                    full[c, i, j] = (covarsPrior + per[c, i, j]) / (Math.Max(covarsWeight - nf, 0.0) + stats.Post[c]);
        return full;
    }

    /// <summary>A Gaussian HMM with the given M-step settings, ready for a direct DoMStep call.</summary>
    private sealed class MStepProbe : GaussianHmm
    {
        public MStepProbe(CovarianceType type, string @params, double meansPrior, double meansWeight,
            double covarsPrior, double covarsWeight)
            : base(2, type, 1e-3, 1.0, 1.0, meansPrior, meansWeight, covarsPrior, covarsWeight,
                DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, @params, "", Implementation.Log)
        {
        }

        public void CallDoMStep(GaussianStatistics stats) => DoMStep(stats);
    }
}
