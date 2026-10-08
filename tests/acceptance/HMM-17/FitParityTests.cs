using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm17;

/// <summary>
/// Acceptance tests for learning the GMM-HMM (ticket HMM-17, criterion C1): for every fit case of the four oracle files
/// (gmm_spherical, gmm_diag, gmm_tied, gmm_full) and both implementations, Fit(x, lengths) with the case's constructor
/// settings, its explicit init (initParams = "") and its parameter letters must reproduce the oracle fit: the final
/// startprob, transmat, weights, means, covars (compact rank 4), the monitor's history, the number of completed
/// iterations, the convergence flag and the resulting score, Aic and Bic. The M-step itself is pinned by calling
/// DoMStep directly on a trivial hand-built GmmStatistics whose expected results the test computes from the test's own
/// inputs by the documented formulas: the weights update with a per-state denominator, the means update (including the
/// zero-weight component whose all-zero numerator gets mean 0 instead of NaN), the scalar CovarsPrior added to every
/// entry of full/tied matrices (off-diagonal entries included), and replacing Weights/Means/Covars with new arrays
/// instead of mutating them in place.
/// </summary>
public class FitParityTests
{
    private static readonly string[] Files = ["gmm_spherical", "gmm_diag", "gmm_tied", "gmm_full"];

    public static IEnumerable<object[]> FitCases
    {
        get
        {
            foreach (var file in Files)
                foreach (var c in Golden.Load(file).GetProperty("fit_cases").Items())
                    foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
                        yield return new object[] { file, c.GetProperty("name").AsString(), implementation };
        }
    }

    private static JsonElement Case(string file, string name) =>
        Golden.Load(file).GetProperty("fit_cases")
            .Items()
            .Single(e => e.GetProperty("name").AsString() == name);

    /// <summary>Case model: the constructor settings (hmmlearn's defaults for keys not present), no init yet, InitParams = "".</summary>
    private static GmmHmm NewModel(string file, string name, Implementation implementation)
    {
        var ctor = Case(file, name).GetProperty("constructor");
        return new GmmHmm(
            nComponents: ctor.GetProperty("n_components").AsInt(),
            nMix: ctor.GetProperty("n_mix").AsInt(),
            minCovar: 1e-3,
            startProbPrior: ctor.Has("startprob_prior") ? ctor.GetProperty("startprob_prior").AsDouble() : 1.0,
            transMatPrior: ctor.Has("transmat_prior") ? ctor.GetProperty("transmat_prior").AsDouble() : 1.0,
            weightsPrior: ctor.Has("weights_prior") ? ctor.GetProperty("weights_prior").AsDouble() : 1.0,
            meansPrior: ctor.Has("means_prior") ? ctor.GetProperty("means_prior").AsDouble() : 0.0,
            meansWeight: ctor.Has("means_weight") ? ctor.GetProperty("means_weight").AsDouble() : 0.0,
            covarsPrior: ctor.Has("covars_prior") ? (double?)ctor.GetProperty("covars_prior").AsDouble() : null,
            covarsWeight: ctor.Has("covars_weight") ? (double?)ctor.GetProperty("covars_weight").AsDouble() : null,
            algorithm: DecoderAlgorithm.Viterbi,
            covarianceType: Enum.Parse<CovarianceType>(ctor.GetProperty("covariance_type").AsString(), ignoreCase: true),
            randomState: null,
            nIter: ctor.GetProperty("n_iter").AsInt(),
            tol: ctor.GetProperty("tol").AsDouble(),
            verbose: false,
            @params: ctor.GetProperty("params").AsString(),
            initParams: "",
            implementation);
    }

    [Theory]
    [MemberData(nameof(FitCases))]
    public void fit_reproduces_the_oracle(string file, string name, Implementation implementation)
    {
        var c = Case(file, name);
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var model = NewModel(file, name, implementation);
        var init = c.GetProperty("init");
        model.StartProb = init.GetProperty("startprob").AsVector();
        model.TransMat = init.GetProperty("transmat").AsMatrix();
        model.Weights = init.GetProperty("weights").AsMatrix();
        model.Means = init.GetProperty("means").AsTensor3();
        model.Covars = init.GetProperty("covars").AsTensor4();
        model.NFeatures = x.GetLength(1);

        model.Fit(x, lengths);

        var expected = c.GetProperty("expected");
        var what = $"{file}/{name} ";
        Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb!, what + "startprob");
        Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat!, what + "transmat");
        Close.Equal(expected.GetProperty("weights").AsMatrix(), model.Weights!, what + "weights");
        Close.Equal(expected.GetProperty("means").AsTensor3(), model.Means!, what + "means");
        Close.Equal(expected.GetProperty("covars").AsTensor4(), model.Covars!, what + "covars");
        Close.Equal(expected.GetProperty("history").AsVector(), model.Monitor.History.ToArray(), what + "history");
        Assert.Equal(expected.GetProperty("n_iter_done").AsInt(), model.Monitor.Iter);
        Assert.Equal(expected.GetProperty("converged").AsBool(), model.Monitor.Converged);
        Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths), what + "score");
        Close.Equal(expected.GetProperty("aic").AsDouble(), model.Aic(x, lengths), what + "aic");
        Close.Equal(expected.GetProperty("bic").AsDouble(), model.Bic(x, lengths), what + "bic");
    }

    private sealed class MStepProbe : GmmHmm
    {
        public MStepProbe(CovarianceType type, string @params, double weightsPrior, double meansPrior, double meansWeight,
            double? covarsPrior, double? covarsWeight)
            : base(2, 2, 1e-3, 1.0, 1.0, weightsPrior, meansPrior, meansWeight, covarsPrior, covarsWeight,
                DecoderAlgorithm.Viterbi, type, null, 0, 1e-2, false, @params, "", Implementation.Log)
        {
        }

        public void CallDoMStep(GmmStatistics stats) => DoMStep(stats);
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Tied)]
    [InlineData(CovarianceType.Full)]
    public void m_step_updates_weights_means_and_covars_with_the_documented_formulas(CovarianceType type)
    {
        const int nc = 2, nm = 2, nf = 2;
        const double weightsPrior = 2.0, meansPrior = 0.5, meansWeight = 1.5, covarsPrior = 0.3, covarsWeight = 3.5;
        var (d1, d2, d3) = type switch
        {
            CovarianceType.Spherical => (nm, 1, 1),
            CovarianceType.Diag => (nm, 1, nf),
            CovarianceType.Full => (nm, nf, nf),
            _ => (1, nf, nf),
        };

        // Hand-built statistics with distinct non-trivial values (the MN block already holds meansWeight * meansPrior).
        var stats = new GmmStatistics(nc, nm, nf, meansWeight, meansPrior, new[] { nc, d1, d2, d3 });
        for (var c = 0; c < nc; c++)
        {
            for (var m = 0; m < nm; m++)
            {
                stats.PostMixSum[c, m] = 1.0 + 0.1 * (c * nm + m);
                for (var f = 0; f < nf; f++)
                    stats.MN[c, m, f] = 10.0 + 0.5 * (c * nm * nf + m * nf + f);
            }
            stats.PostSum[c] = 0.0;
            for (var m = 0; m < nm; m++)
                stats.PostSum[c] += stats.PostMixSum[c, m];
        }
        for (var c = 0; c < nc; c++)
            for (var m = 0; m < nm; m++)
                for (var a = 0; a < d2; a++)
                    for (var b = 0; b < d3; b++)
                        stats.CN[c, type == CovarianceType.Tied ? 0 : m, a, b] = 2.0 + 0.1 * c + 0.2 * m + 0.3 * a + 0.4 * b;

        var weights = new double[nc, nm] { { 0.5, 0.5 }, { 0.25, 0.75 } };
        var means = new double[nc, nm, nf];
        for (var c = 0; c < nc; c++)
            for (var m = 0; m < nm; m++)
                for (var f = 0; f < nf; f++)
                    means[c, m, f] = 4.0 + 0.1 * (c * nm * nf + m * nf + f);
        var covars = new double[nc, d1, d2, d3];
        var weights0 = (double[,])weights.Clone();
        var means0 = (double[,,])means.Clone();
        var covars0 = (double[,,,])covars.Clone();

        var model = new MStepProbe(type, "mcw", weightsPrior, meansPrior, meansWeight, covarsPrior, covarsWeight);
        model.StartProb = new[] { 0.5, 0.5 };
        model.TransMat = new double[nc, nc] { { 0.5, 0.5 }, { 0.5, 0.5 } };
        model.Weights = weights;
        model.Means = means;
        model.Covars = covars;
        model.NFeatures = nf;

        model.CallDoMStep(stats);

        // Expected values, computed from the test's own inputs by the documented formulas.
        var expectedWeights = new double[nc, nm];
        for (var c = 0; c < nc; c++)
        {
            var wD = stats.PostSum[c] + nm * (weightsPrior - 1.0);
            for (var m = 0; m < nm; m++)
                expectedWeights[c, m] = (stats.PostMixSum[c, m] + (weightsPrior - 1.0)) / wD;
        }

        var expectedMeans = new double[nc, nm, nf];
        for (var c = 0; c < nc; c++)
            for (var m = 0; m < nm; m++)
            {
                // No zero-weight component here, so the denominator is never forced to 1.
                var mD = stats.PostMixSum[c, m] + meansWeight;
                for (var f = 0; f < nf; f++)
                    expectedMeans[c, m, f] = stats.MN[c, m, f] / mD;
            }

        var centered = new double[nc, nm, nf];
        for (var c = 0; c < nc; c++)
            for (var m = 0; m < nm; m++)
                for (var f = 0; f < nf; f++)
                    centered[c, m, f] = expectedMeans[c, m, f] - meansPrior;
        var expectedCovars = new double[nc, d1, d2, d3];
        switch (type)
        {
            case CovarianceType.Spherical:
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                    {
                        var norm2 = 0.0;
                        for (var f = 0; f < nf; f++)
                            norm2 += centered[c, m, f] * centered[c, m, f];
                        expectedCovars[c, m, 0, 0] =
                            (meansWeight * norm2 + 2.0 * covarsWeight + stats.CN[c, m, 0, 0]) /
                            (nf * (stats.PostMixSum[c, m] + 1.0) + 2.0 * (covarsPrior + 1.0));
                    }
                break;
            case CovarianceType.Diag:
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                        for (var f = 0; f < nf; f++)
                            expectedCovars[c, m, 0, f] =
                                (meansWeight * centered[c, m, f] * centered[c, m, f] + 2.0 * covarsWeight + stats.CN[c, m, 0, f]) /
                                (stats.PostMixSum[c, m] + 1.0 + 2.0 * (covarsPrior + 1.0));
                break;
            case CovarianceType.Full:
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                        for (var i = 0; i < nf; i++)
                            for (var j = 0; j < nf; j++)
                                expectedCovars[c, m, i, j] =
                                    (covarsPrior + meansWeight * centered[c, m, i] * centered[c, m, j] + stats.CN[c, m, i, j]) /
                                    (stats.PostMixSum[c, m] + 1.0 + covarsWeight + nf + 1.0);
                break;
            default: // Tied: one matrix per state, the sum over its mixture components.
                for (var c = 0; c < nc; c++)
                {
                    var cD = stats.PostSum[c] + nm + covarsWeight + nf + 1.0;
                    for (var i = 0; i < nf; i++)
                        for (var j = 0; j < nf; j++)
                        {
                            var sum = covarsPrior + stats.CN[c, 0, i, j];
                            for (var m = 0; m < nm; m++)
                                sum += meansWeight * centered[c, m, i] * centered[c, m, j];
                            expectedCovars[c, 0, i, j] = sum / cD;
                        }
                }
                break;
        }

        Close.Equal(expectedWeights, model.Weights!, "weights update");
        Close.Equal(expectedMeans, model.Means!, "means update");
        Close.Equal(expectedCovars, model.Covars!, "covars update");
        // The result replaces the caller's arrays; the caller's arrays stay untouched.
        Assert.NotSame(weights, model.Weights);
        Assert.NotSame(means, model.Means);
        Assert.NotSame(covars, model.Covars);
        Close.Equal(weights0, weights, "caller's weights");
        Close.Equal(means0, means, "caller's means");
        Close.Equal(covars0, covars, "caller's covars");
    }

    [Fact]
    public void m_step_gives_an_unused_component_the_zero_mean_instead_of_a_nan()
    {
        // The hmmlearn defaults (weights_prior 1, means_prior 0, means_weight 0): the 'm' update on its own.
        var model = new MStepProbe(CovarianceType.Diag, "m", 1.0, 0.0, 0.0, null, null);
        model.StartProb = new[] { 0.5, 0.5 };
        model.TransMat = new double[2, 2] { { 0.5, 0.5 }, { 0.5, 0.5 } };
        model.Weights = new double[2, 2] { { 0.5, 0.5 }, { 0.0, 1.0 } }; // mixture (1, 0) has weight exactly 0
        model.Means = new double[2, 2, 2];
        model.Covars = new double[2, 2, 1, 2];
        model.NFeatures = 2;

        var stats = new GmmStatistics(2, 2, 2, 0.0, 0.0, new[] { 2, 2, 1, 2 }); // MN starts all zero
        stats.PostMixSum[0, 0] = 2.0;
        stats.PostMixSum[0, 1] = 3.0;
        stats.PostMixSum[1, 1] = 4.0; // (1, 0) gets no mass at all
        stats.PostSum[0] = 5.0;
        stats.PostSum[1] = 4.0;
        stats.MN[1, 1, 0] = 8.0;
        stats.MN[1, 1, 1] = 4.0;

        model.CallDoMStep(stats);

        // A zero-weight component whose numerator row is all zero must get mean 0 (the forced denominator), not NaN.
        Close.Equal(0.0, model.Means![1, 0, 0], "dead mixture mean, feature 0");
        Close.Equal(0.0, model.Means[1, 0, 1], "dead mixture mean, feature 1");
        Close.Equal(8.0 / stats.PostMixSum[1, 1], model.Means[1, 1, 0], "live mixture mean, feature 0");
        Close.Equal(4.0 / stats.PostMixSum[1, 1], model.Means[1, 1, 1], "live mixture mean, feature 1");
    }
}
