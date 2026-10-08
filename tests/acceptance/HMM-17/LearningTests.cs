using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm17;

/// <summary>
/// Acceptance tests for learning the GMM-HMM (ticket HMM-17, criterion C3): on data sampled from a well-separated
/// three-state mixture model, a short fit from the random initialization produces valid parameters (the rows of the weights
/// sum to 1, the covariances pass Check, the score is finite); a fit that starts from a perturbed copy of the true parameters
/// recovers the true parameters (EM from the random initialization can end in a poor local optimum, hmmlearn included, so
/// the recovery does not start there); a single-iteration Fit (NIter = 1, InitParams = "" after the first Fit) never
/// decreases the data log-likelihood; and Aic and Bic use the number of free scalar parameters documented for all the
/// parameter letters, not only the letters of Params: Aic = -2 * Score + 2 * k, Bic = -2 * Score + k * ln(n_samples).
/// </summary>
public class LearningTests
{
    private const int NComponents = 3;
    private const int NMix = 2;
    private const int NFeatures = 2;

    private sealed class LearningProbe : GmmHmm
    {
        public LearningProbe(CovarianceType type, int? randomState, int nIter, string initParams, string @params)
            : base(LearningTests.NComponents, LearningTests.NMix, 1e-3, 1.0, 1.0, 1.0, 0.0, 0.0, null, null, DecoderAlgorithm.Viterbi, type, randomState,
                nIter, 1e-2, false, @params, initParams, Implementation.Log)
        {
        }

        public void CallCheck() => Check();
    }

    /// <summary>The true model: 3 states 12 apart along feature 0, each with 2 mixtures at ±1 (feature 0), variance 1/4, weight 1/2.</summary>
    private static GmmHmm TrueModel(CovarianceType type)
    {
        var model = new GmmHmm(NComponents, NMix, 1e-3, 1.0, 1.0, 1.0, 0.0, 0.0, null, null, DecoderAlgorithm.Viterbi,
            type, null, 0, 1e-2, false, "stmcw", "", Implementation.Log);
        model.NFeatures = NFeatures;
        var startProb = new double[NComponents];
        for (var c = 0; c < NComponents; c++)
            startProb[c] = 1.0 / NComponents;
        model.StartProb = startProb;
        var transMat = new double[NComponents, NComponents];
        for (var c = 0; c < NComponents; c++)
            for (var d = 0; d < NComponents; d++)
                transMat[c, d] = c == d ? 0.9 : 0.1 / (NComponents - 1);
        model.TransMat = transMat;
        var weights = new double[NComponents, NMix];
        for (var c = 0; c < NComponents; c++)
            for (var m = 0; m < NMix; m++)
                weights[c, m] = 0.5;
        model.Weights = weights;
        model.Means = TrueMeans();
        model.Covars = TrueCovars(type);
        return model;
    }

    private static double[,,] TrueMeans()
    {
        var means = new double[NComponents, NMix, NFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var m = 0; m < NMix; m++)
            {
                means[c, m, 0] = 12.0 * c + (m == 0 ? -1.0 : 1.0);
                means[c, m, 1] = 0.0;
            }
        return means;
    }

    private static double[,,,] TrueCovars(CovarianceType type)
    {
        const double variance = 0.25;
        var (d1, d2, d3) = type switch
        {
            CovarianceType.Spherical => (NMix, 1, 1),
            CovarianceType.Diag => (NMix, 1, NFeatures),
            CovarianceType.Full => (NMix, NFeatures, NFeatures),
            _ => (1, NFeatures, NFeatures),
        };
        var result = new double[NComponents, d1, d2, d3];
        for (var c = 0; c < NComponents; c++)
        {
            switch (type)
            {
                case CovarianceType.Spherical:
                    for (var m = 0; m < NMix; m++)
                        result[c, m, 0, 0] = variance;
                    break;
                case CovarianceType.Diag:
                    for (var m = 0; m < NMix; m++)
                        for (var f = 0; f < NFeatures; f++)
                            result[c, m, 0, f] = variance;
                    break;
                case CovarianceType.Full:
                    for (var m = 0; m < NMix; m++)
                        for (var f = 0; f < NFeatures; f++)
                            result[c, m, f, f] = variance;
                    break;
                default: // Tied: one matrix per state.
                    for (var f = 0; f < NFeatures; f++)
                        result[c, 0, f, f] = variance;
                    break;
            }
        }
        return result;
    }

    private static int Index(CovarianceType type) => type switch
    {
        CovarianceType.Spherical => 0,
        CovarianceType.Diag => 1,
        CovarianceType.Full => 2,
        _ => 3,
    };

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Tied)]
    [InlineData(CovarianceType.Full)]
    public void a_short_fit_from_the_random_initialization_gives_valid_parameters(CovarianceType type)
    {
        const int samples = 1200;
        var (x, _) = TrueModel(type).Sample(samples, random: new Random(2017 + Index(type)));
        var lengths = new[] { samples };

        var model = new LearningProbe(type, 777 + Index(type), 5, "stmcw", "stmcw");
        model.Fit(x, lengths);

        Assert.Equal(NFeatures, model.NFeatures!.Value);
        var weights = model.Weights!;
        for (var c = 0; c < NComponents; c++)
        {
            var sum = 0.0;
            for (var m = 0; m < NMix; m++)
            {
                Assert.True(weights[c, m] > 0.0, $"weight [{c}, {m}] is {weights[c, m]}");
                sum += weights[c, m];
            }
            Close.Equal(1.0, sum, $"weights row {c} sum");
        }

        model.CallCheck();
        Assert.True(double.IsFinite(model.Score(x, lengths)), "the score of the fitted model is not finite");
    }

    /// <summary>The true structure with the mixtures pulled together (+-0.7 instead of +-1), the variances doubled, the weights unbalanced and the transitions blurred.</summary>
    private static LearningProbe PerturbedTruthModel(CovarianceType type, int nIter)
    {
        var model = new LearningProbe(type, null, nIter, "", "stmcw");
        model.NFeatures = NFeatures;
        var startProb = new double[NComponents];
        for (var c = 0; c < NComponents; c++)
            startProb[c] = 1.0 / NComponents;
        model.StartProb = startProb;
        var transMat = new double[NComponents, NComponents];
        for (var c = 0; c < NComponents; c++)
            for (var d = 0; d < NComponents; d++)
                transMat[c, d] = c == d ? 0.8 : 0.2 / (NComponents - 1);
        model.TransMat = transMat;
        var weights = new double[NComponents, NMix];
        for (var c = 0; c < NComponents; c++)
        {
            weights[c, 0] = 0.4;
            weights[c, 1] = 0.6;
        }
        model.Weights = weights;
        var means = new double[NComponents, NMix, NFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var m = 0; m < NMix; m++)
            {
                means[c, m, 0] = 12.0 * c + (m == 0 ? -0.7 : 0.7);
                means[c, m, 1] = 0.2;
            }
        model.Means = means;
        var covars = TrueCovars(type);
        for (var a = 0; a < covars.GetLength(0); a++)
            for (var b = 0; b < covars.GetLength(1); b++)
                for (var c = 0; c < covars.GetLength(2); c++)
                    for (var d = 0; d < covars.GetLength(3); d++)
                        covars[a, b, c, d] *= 2.0;
        model.Covars = covars;
        return model;
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Tied)]
    [InlineData(CovarianceType.Full)]
    public void fit_from_a_perturbed_truth_recovers_the_true_model(CovarianceType type)
    {
        const int samples = 1500;
        var (x, _) = TrueModel(type).Sample(samples, random: new Random(2017 + Index(type)));
        var lengths = new[] { samples };

        var model = PerturbedTruthModel(type, 300);
        model.Fit(x, lengths);

        model.CallCheck();
        var truth = TrueMeans();
        var means = model.Means!;
        var weights = model.Weights!;
        var covars = model.Covars!;
        var transMat = model.TransMat!;
        for (var c = 0; c < NComponents; c++)
        {
            Assert.True(transMat[c, c] >= 0.75, $"transmat[{c}, {c}] is {transMat[c, c]}, the true value is 0.9");
            for (var m = 0; m < NMix; m++)
            {
                Assert.InRange(weights[c, m], 0.35, 0.65);
                for (var f = 0; f < NFeatures; f++)
                    Assert.True(Math.Abs(means[c, m, f] - truth[c, m, f]) <= 0.3,
                        $"state {c}, mixture {m}, feature {f}: fitted {means[c, m, f]} is not within 0.3 of the true {truth[c, m, f]}");
            }
        }

        // Covariances recovered in a band around the true variance 0.25 (the true off-diagonals are 0).
        for (var c = 0; c < NComponents; c++)
        {
            switch (type)
            {
                case CovarianceType.Spherical:
                    for (var m = 0; m < NMix; m++)
                        Assert.InRange(covars[c, m, 0, 0], 0.15, 0.40);
                    break;
                case CovarianceType.Diag:
                    for (var m = 0; m < NMix; m++)
                        for (var f = 0; f < NFeatures; f++)
                            Assert.InRange(covars[c, m, 0, f], 0.15, 0.40);
                    break;
                case CovarianceType.Full:
                    for (var m = 0; m < NMix; m++)
                    {
                        for (var f = 0; f < NFeatures; f++)
                            Assert.InRange(covars[c, m, f, f], 0.15, 0.40);
                        Assert.InRange(Math.Abs(covars[c, m, 0, 1]), 0.0, 0.15);
                    }
                    break;
                default: // Tied.
                    Assert.InRange(covars[c, 0, 0, 0], 0.15, 0.40);
                    Assert.InRange(covars[c, 0, 1, 1], 0.15, 0.40);
                    Assert.InRange(Math.Abs(covars[c, 0, 0, 1]), 0.0, 0.15);
                    break;
            }
        }
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Tied)]
    [InlineData(CovarianceType.Full)]
    public void single_iteration_fits_never_decrease_the_log_likelihood(CovarianceType type)
    {
        const int samples = 800;
        var (x, _) = TrueModel(type).Sample(samples, random: new Random(31337 + Index(type)));
        var lengths = new[] { samples };

        var model = new LearningProbe(type, 4242 + Index(type), 0, "stmcw", "stmcw");
        model.Fit(x, lengths); // NIter = 0: the random initialization only
        var previous = model.Score(x, lengths);

        model.InitParams = "";
        model.NIter = 1;
        for (var i = 0; i < 40; i++)
        {
            model.Fit(x, lengths);
            var current = model.Score(x, lengths);
            var margin = 1e-9 * Math.Max(1.0, Math.Abs(previous));
            Assert.True(current >= previous - margin,
                $"iteration {i}: the log likelihood decreased from {previous} to {current}");
            previous = current;
        }
    }

    [Theory]
    [InlineData(CovarianceType.Spherical, "stmcw")]
    [InlineData(CovarianceType.Spherical, "c")]
    [InlineData(CovarianceType.Diag, "stmcw")]
    [InlineData(CovarianceType.Diag, "c")]
    [InlineData(CovarianceType.Tied, "stmcw")]
    [InlineData(CovarianceType.Tied, "c")]
    [InlineData(CovarianceType.Full, "stmcw")]
    [InlineData(CovarianceType.Full, "c")]
    public void aic_and_bic_use_the_documented_number_of_free_scalar_parameters(CovarianceType type, string @params)
    {
        const int samples = 900;
        var (x, _) = TrueModel(type).Sample(samples, random: new Random(555 + Index(type)));
        var lengths = new[] { samples };

        var model = new LearningProbe(type, 666 + Index(type), 5, "stmcw", @params);
        model.Fit(x, lengths);

        var k = (NComponents - 1)                          // 's'
            + NComponents * (NComponents - 1)              // 't'
            + NComponents * NMix * NFeatures               // 'm'
            + (type switch                                 // 'c'
            {
                CovarianceType.Spherical => NComponents * NMix,
                CovarianceType.Diag => NComponents * NMix * NFeatures,
                CovarianceType.Full => NComponents * NMix * NFeatures * (NFeatures + 1) / 2,
                _ => NComponents * NFeatures * (NFeatures + 1) / 2,
            })
            + (NMix - 1);                                  // 'w'
        var score = model.Score(x, lengths);
        Close.Equal(-2.0 * score + 2.0 * k, model.Aic(x, lengths), "Aic");
        Close.Equal(-2.0 * score + k * Math.Log(x.GetLength(0)), model.Bic(x, lengths), "Bic");
    }
}
