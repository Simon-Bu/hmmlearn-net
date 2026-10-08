using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm17;

/// <summary>
/// Acceptance tests for learning the GMM-HMM (ticket HMM-17, criterion C3): on data sampled from a well-separated
/// three-state mixture model, a fit from the random initialization produces valid parameters (the rows of the weights
/// sum to 1 with no collapsed mixture component, and the covariances pass Check) and recovers the true parameters up
/// to a permutation of the states and of the mixtures within each state; a single-iteration Fit (NIter = 1,
/// InitParams = "" after the first Fit) never decreases the data log-likelihood; and Aic and Bic use the number of
/// free scalar parameters documented for all the parameter letters, not only the letters of Params:
/// Aic = -2 * Score + 2 * k, Bic = -2 * Score + k * ln(n_samples).
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
    public void fit_from_a_random_initialization_recovers_the_true_model(CovarianceType type)
    {
        const int samples = 1200;
        var (x, _) = TrueModel(type).Sample(samples, random: new Random(2017));
        var lengths = new[] { samples };

        var model = new LearningProbe(type, 777 + Index(type), 200, "stmcw", "stmcw");
        model.Fit(x, lengths);

        Assert.Equal(NFeatures, model.NFeatures!.Value);

        // Valid parameters: the rows of the weights sum to 1 and no mixture component collapses.
        for (var c = 0; c < NComponents; c++)
        {
            var sum = 0.0;
            for (var m = 0; m < NMix; m++)
            {
                Assert.InRange(model.Weights![c, m], 0.1, 0.9);
                sum += model.Weights[c, m];
            }
            Close.Equal(1.0, sum, $"weights row {c} sum");
        }

        // The covariances are accepted by Check.
        model.CallCheck();

        // Means recovered up to a permutation of the states and of the mixtures within each state.
        var means = model.Means!;
        var covars = model.Covars!;
        var truth = TrueMeans();
        var permutation = BestPermutation(means, truth);
        for (var c = 0; c < NComponents; c++)
        {
            var t = permutation[c];
            var assignment = AssignMixtures(StateMeans(means, c), StateMeans(truth, t));
            for (var m = 0; m < NMix; m++)
            {
                var tm = assignment[m];
                for (var f = 0; f < NFeatures; f++)
                    Assert.True(Math.Abs(means[c, m, f] - truth[t, tm, f]) <= 1.0,
                        $"state {c}, mixture {m}, feature {f}: fitted {means[c, m, f]} is not within 1.0 of the true {truth[t, tm, f]} (state {t}, mixture {tm})");
            }
        }

        // Covariances recovered in a wide band around the true variance (the true off-diagonals are 0).
        for (var c = 0; c < NComponents; c++)
        {
            switch (type)
            {
                case CovarianceType.Spherical:
                    for (var m = 0; m < NMix; m++)
                        Assert.InRange(covars[c, m, 0, 0], 0.02, 0.9);
                    break;
                case CovarianceType.Diag:
                    for (var m = 0; m < NMix; m++)
                        for (var f = 0; f < NFeatures; f++)
                            Assert.InRange(covars[c, m, 0, f], 0.02, 0.9);
                    break;
                case CovarianceType.Full:
                    for (var m = 0; m < NMix; m++)
                    {
                        for (var f = 0; f < NFeatures; f++)
                            Assert.InRange(covars[c, m, f, f], 0.02, 0.9);
                        Assert.InRange(Math.Abs(covars[c, m, 0, 1]), 0.0, 0.3);
                    }
                    break;
                default: // Tied.
                    Assert.InRange(covars[c, 0, 0, 0], 0.02, 0.9);
                    Assert.InRange(covars[c, 0, 1, 1], 0.02, 0.9);
                    Assert.InRange(Math.Abs(covars[c, 0, 0, 1]), 0.0, 0.3);
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

    private static int[] BestPermutation(double[,,] fitted, double[,,] truth)
    {
        var permutations = new List<int[]>();
        var current = new int[NComponents];
        for (var i = 0; i < NComponents; i++)
            current[i] = i;
        DoPermutations(current, NComponents - 1, permutations);
        var best = new int[NComponents];
        var bestDistance = double.MaxValue;
        foreach (var p in permutations)
        {
            var distance = 0.0;
            for (var c = 0; c < NComponents; c++)
            {
                var assignment = AssignMixtures(StateMeans(fitted, c), StateMeans(truth, p[c]));
                for (var m = 0; m < NMix; m++)
                    for (var f = 0; f < NFeatures; f++)
                        distance += (fitted[c, m, f] - truth[p[c], assignment[m], f]) * (fitted[c, m, f] - truth[p[c], assignment[m], f]);
            }
            if (distance < bestDistance)
            {
                bestDistance = distance;
                for (var i = 0; i < NComponents; i++)
                    best[i] = p[i];
            }
        }
        return best;
    }

    private static void DoPermutations(int[] current, int last, List<int[]> into)
    {
        if (last == 0)
        {
            into.Add((int[])current.Clone());
            return;
        }
        for (var i = 0; i <= last; i++)
        {
            (current[last], current[i]) = (current[i], current[last]);
            DoPermutations(current, last - 1, into);
            (current[last], current[i]) = (current[i], current[last]);
        }
    }

    private static double[,] StateMeans(double[,,] means, int state)
    {
        var nm = means.GetLength(1);
        var nf = means.GetLength(2);
        var result = new double[nm, nf];
        for (var m = 0; m < nm; m++)
            for (var f = 0; f < nf; f++)
                result[m, f] = means[state, m, f];
        return result;
    }

    /// <summary>Which true mixture each fitted mixture of one state belongs to (the mixtures are at ±1 along feature 0).</summary>
    private static int[] AssignMixtures(double[,] fitted, double[,] truth)
    {
        var stay = SquaredDistance(fitted, 0, truth, 0) + SquaredDistance(fitted, 1, truth, 1);
        var swap = SquaredDistance(fitted, 0, truth, 1) + SquaredDistance(fitted, 1, truth, 0);
        return swap < stay ? new[] { 1, 0 } : new[] { 0, 1 };
    }

    private static double SquaredDistance(double[,] a, int ia, double[,] b, int ib)
    {
        var sum = 0.0;
        for (var j = 0; j < a.GetLength(1); j++)
            sum += (a[ia, j] - b[ib, j]) * (a[ia, j] - b[ib, j]);
        return sum;
    }
}
