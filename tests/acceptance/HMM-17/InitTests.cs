using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm17;

/// <summary>
/// Acceptance tests for the GMM-HMM initialization (ticket HMM-17, criterion C2): Init first runs the base init
/// (startprob, transmat, number of features), then computes cv, the data covariance (column sample covariance divided
/// by n - 1, zero for a single row) plus MinCovar on the diagonal, clusters x into NComponents clusters with
/// KMeans.Fit, re-clusters every cluster that has at least NMix samples into NMix clusters (whose centers are the
/// state's means) and gives a smaller cluster NMix draws from the Gaussian centered on the mean of the NComponents
/// centers with covariance cv; and only for the letters of InitParams that need it it sets Weights = 1 / NMix
/// everywhere, Means = the centers found and Covars = cv distributed to the covariance type (tied/full the matrix
/// itself, diag its diagonal, spherical the mean of all entries of cv, the same for every state and mixture, compact
/// rank 4). It is reproducible for a fixed RandomState (and not for another), leaves an explicit parameter (the same
/// reference) alone when InitParams does not name it, raises exactly the "will be overwritten" warning for a set
/// parameter that InitParams does name, and never mutates the caller's arrays.
/// </summary>
public class InitTests
{
    private sealed class InitProbe : GmmHmm
    {
        public InitProbe(int nComponents, int nMix, CovarianceType type, string initParams, int? randomState, double minCovar = 1e-3)
            : base(nComponents, nMix, minCovar, 1.0, 1.0, 1.0, 0.0, 0.0, null, null, DecoderAlgorithm.Viterbi, type, randomState,
                0, 1e-2, false, "stmcw", initParams, Implementation.Log)
        {
        }

        public void CallInit(double[,] x, int[] lengths) => Init(x, lengths);
        public void CallCheck() => Check();
    }

    /// <summary>16 observations in two well-separated clusters of 8 (each has at least NMix = 2 samples).</summary>
    private static double[,] Data() => new double[16, 2]
    {
        { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.5, 0.5 }, { 0.2, -0.3 }, { 0.3, 0.7 }, { -0.4, 0.1 }, { 1.1, 0.6 }, { 0.7, -0.5 },
        { 10.0, 0.0 }, { 11.0, 0.0 }, { 10.5, 0.5 }, { 10.2, 0.3 }, { 10.9, 0.8 }, { 9.6, -0.4 }, { 11.2, 0.2 }, { 10.4, 0.9 },
    };

    /// <summary>The data covariance per the spec: column sample covariance (divided by n - 1, zero for a single row) plus MinCovar on the diagonal.</summary>
    private static double[,] DataCovariance(double[,] x, double minCovar)
    {
        var n = x.GetLength(0);
        var nf = x.GetLength(1);
        var mean = new double[nf];
        for (var i = 0; i < n; i++)
            for (var f = 0; f < nf; f++)
                mean[f] += x[i, f] / n;
        var cov = new double[nf, nf];
        for (var f = 0; f < nf; f++)
            for (var g = 0; g < nf; g++)
            {
                var sum = 0.0;
                for (var i = 0; i < n; i++)
                    sum += (x[i, f] - mean[f]) * (x[i, g] - mean[g]);
                cov[f, g] = n > 1 ? sum / (n - 1) : 0.0;
                if (f == g)
                    cov[f, g] += minCovar;
            }
        return cov;
    }

    /// <summary>cv distributed to the compact rank 4 of the type: tied/full the matrix itself, diag its diagonal, spherical the mean of all entries of cv.</summary>
    private static double[,,,] Distributed(double[,] cv, CovarianceType type, int nc, int nm)
    {
        var nf = cv.GetLength(0);
        var (d1, d2, d3) = type switch
        {
            CovarianceType.Spherical => (nm, 1, 1),
            CovarianceType.Diag => (nm, 1, nf),
            CovarianceType.Full => (nm, nf, nf),
            _ => (1, nf, nf),
        };
        var result = new double[nc, d1, d2, d3];
        var meanOfAll = 0.0;
        for (var f = 0; f < nf; f++)
            for (var g = 0; g < nf; g++)
                meanOfAll += cv[f, g];
        meanOfAll /= nf * nf;
        for (var c = 0; c < nc; c++)
        {
            switch (type)
            {
                case CovarianceType.Spherical:
                    for (var m = 0; m < nm; m++)
                        result[c, m, 0, 0] = meanOfAll;
                    break;
                case CovarianceType.Diag:
                    for (var m = 0; m < nm; m++)
                        for (var f = 0; f < nf; f++)
                            result[c, m, 0, f] = cv[f, f];
                    break;
                case CovarianceType.Full:
                    for (var m = 0; m < nm; m++)
                        for (var f = 0; f < nf; f++)
                            for (var g = 0; g < nf; g++)
                                result[c, m, f, g] = cv[f, g];
                    break;
                default: // Tied: one matrix per state (no mixture axis).
                    for (var f = 0; f < nf; f++)
                        for (var g = 0; g < nf; g++)
                            result[c, 0, f, g] = cv[f, g];
                    break;
            }
        }
        return result;
    }

    private static void AssertRowSumsToOne(double[] row)
    {
        var sum = 0.0;
        foreach (var v in row)
        {
            Assert.True(v >= 0.0, $"negative entry {v}");
            sum += v;
        }
        Close.Equal(1.0, sum, "row sum");
    }

    private static void AssertRowsSumToOne(double[,] matrix)
    {
        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            var sum = 0.0;
            for (var j = 0; j < matrix.GetLength(1); j++)
            {
                Assert.True(matrix[i, j] >= 0.0, $"negative entry {matrix[i, j]}");
                sum += matrix[i, j];
            }
            Close.Equal(1.0, sum, $"row {i} sum");
        }
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Tied)]
    [InlineData(CovarianceType.Full)]
    public void init_sets_uniform_weights_the_kmeans_centers_and_the_data_covariance(CovarianceType type)
    {
        const int nc = 2, nm = 2;
        const double minCovar = 0.25;
        var x = Data();
        var x0 = (double[,])x.Clone();
        var model = new InitProbe(nc, nm, type, "stmcw", 42, minCovar);
        model.CallInit(x, new[] { 16 });

        Assert.Equal(x.GetLength(1), model.NFeatures!.Value);

        // Weights = 1 / NMix everywhere.
        var expectedWeights = new double[nc, nm];
        for (var c = 0; c < nc; c++)
            for (var m = 0; m < nm; m++)
                expectedWeights[c, m] = 1.0 / nm;
        Close.Equal(expectedWeights, model.Weights!, "weights: 1 / NMix everywhere");

        // Means: a second KMeans on each main cluster (the cluster's samples in the order they appear in x).
        var (mainCenters, labels) = KMeans.Fit(x, nc);
        var expectedMeans = new double[nc, nm, 2];
        for (var c = 0; c < nc; c++)
        {
            var rows = new List<double[]>();
            for (var i = 0; i < x.GetLength(0); i++)
                if (labels[i] == c)
                    rows.Add(new[] { x[i, 0], x[i, 1] });
            var cluster = new double[rows.Count, 2];
            for (var i = 0; i < rows.Count; i++)
            {
                cluster[i, 0] = rows[i][0];
                cluster[i, 1] = rows[i][1];
            }
            var centers = KMeans.Fit(cluster, nm).Centers;
            for (var m = 0; m < nm; m++)
                for (var f = 0; f < 2; f++)
                    expectedMeans[c, m, f] = centers[m, f];
        }
        Close.Equal(expectedMeans, model.Means!, "means: the two-stage k-means centers");

        // Covars: the data covariance plus MinCovar, distributed to the type.
        Close.Equal(Distributed(DataCovariance(x, minCovar), type, nc, nm), model.Covars!, "covars: the distributed data covariance");

        // Base init: startprob and transmat are set and valid (the letters of InitParams need them).
        AssertRowSumsToOne(model.StartProb!);
        AssertRowsSumToOne(model.TransMat!);
        Close.Equal(x0, x, "caller's x");
    }

    [Fact]
    public void init_is_reproducible_for_a_fixed_random_state_and_differs_for_another_seed()
    {
        var x = Data();
        var first = new InitProbe(2, 2, CovarianceType.Diag, "stmcw", 42);
        var second = new InitProbe(2, 2, CovarianceType.Diag, "stmcw", 42);
        var third = new InitProbe(2, 2, CovarianceType.Diag, "stmcw", 43);
        first.CallInit(x, new[] { 16 });
        second.CallInit(x, new[] { 16 });
        third.CallInit(x, new[] { 16 });

        Close.Equal(first.StartProb!, second.StartProb!, "startprob, same seed");
        Close.Equal(first.TransMat!, second.TransMat!, "transmat, same seed");
        Close.Equal(first.Weights!, second.Weights!, "weights, same seed");
        Close.Equal(first.Means!, second.Means!, "means, same seed");
        Close.Equal(first.Covars!, second.Covars!, "covars, same seed");

        // A different seed must move at least one of the base parameters (the Dirichlet draws).
        Assert.True(VectorsDiffer(first.StartProb!, third.StartProb!) || MatricesDiffer(first.TransMat!, third.TransMat!),
            "a different RandomState gave the same startprob and transmat");
    }

    private static bool VectorsDiffer(double[] a, double[] b)
    {
        for (var i = 0; i < a.Length; i++)
            if (a[i] != b[i])
                return true;
        return false;
    }

    private static bool MatricesDiffer(double[,] a, double[,] b)
    {
        for (var i = 0; i < a.GetLength(0); i++)
            for (var j = 0; j < a.GetLength(1); j++)
                if (a[i, j] != b[i, j])
                    return true;
        return false;
    }

    /// <summary>199 points in a tight ball around the origin plus one far outlier: the second main cluster has a single sample (&lt; NMix = 3).</summary>
    private static double[,] SmallClusterData()
    {
        var random = new Random(1234);
        var x = new double[200, 2];
        for (var i = 0; i < 199; i++)
        {
            x[i, 0] = 0.3 * (2.0 * random.NextDouble() - 1.0);
            x[i, 1] = 0.3 * (2.0 * random.NextDouble() - 1.0);
        }
        x[199, 0] = 50.0;
        x[199, 1] = 0.1;
        return x;
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

    [Fact]
    public void init_falls_back_to_gaussian_draws_for_a_cluster_with_fewer_samples_than_nmix()
    {
        const int nc = 2, nm = 3, nf = 2;
        const double minCovar = 1e-3;
        var x = SmallClusterData();
        var lengths = new[] { 200 };

        var (mainCenters, labels) = KMeans.Fit(x, nc);
        var counts = new int[nc];
        foreach (var label in labels)
            counts[label]++;
        var smallLabel = counts[0] < nm ? 0 : 1;
        var largeLabel = 1 - smallLabel;
        Assert.Equal(1, counts[smallLabel]);   // sanity: the outlier is alone in its main cluster
        Assert.Equal(199, counts[largeLabel]);

        // The cluster's samples in the order they appear in x, for the second k-means of the large cluster.
        var rowIndexes = new List<int>();
        for (var i = 0; i < x.GetLength(0); i++)
            if (labels[i] == largeLabel)
                rowIndexes.Add(i);
        var largeRows = new double[rowIndexes.Count, nf];
        for (var i = 0; i < rowIndexes.Count; i++)
            for (var f = 0; f < nf; f++)
                largeRows[i, f] = x[rowIndexes[i], f];
        var expectedLarge = KMeans.Fit(largeRows, nm).Centers;

        var cv = DataCovariance(x, minCovar);
        var mainCentroid = new double[nf];
        for (var f = 0; f < nf; f++)
        {
            mainCentroid[f] = 0.0;
            for (var c = 0; c < nc; c++)
                mainCentroid[f] += mainCenters[c, f] / nc;
        }
        var expectedCovars = new double[nc, nm, 1, nf];
        for (var c = 0; c < nc; c++)
            for (var m = 0; m < nm; m++)
                for (var f = 0; f < nf; f++)
                    expectedCovars[c, m, 0, f] = cv[f, f];

        var seeds = new[] { 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 };
        InitProbe? firstSeedModel = null;
        foreach (var seed in seeds)
        {
            var model = new InitProbe(nc, nm, CovarianceType.Diag, "stmcw", seed, minCovar);
            model.CallInit(x, lengths);

            // The large cluster: exactly the centers of a second k-means (deterministic, seed independent).
            Close.Equal(expectedLarge, StateMeans(model.Means!, largeLabel), "means of the large cluster");

            // Weights = 1 / NMix everywhere; covars = the distributed data covariance (independent of the fallback).
            for (var c = 0; c < nc; c++)
                for (var m = 0; m < nm; m++)
                    Close.Equal(1.0 / nm, model.Weights![c, m], $"weights[{c}, {m}]");
            Close.Equal(expectedCovars, model.Covars!, "covars: the distributed data covariance");

            // The small cluster: NMix draws from the Gaussian (mean of the NComponents centers, covariance cv).
            // Each draw is within 6 standard deviations of the center per feature; a wrong center would sit ~25 away
            // in feature 0 while 6 sigma there is only ~21.3.
            for (var m = 0; m < nm; m++)
                for (var f = 0; f < nf; f++)
                    Assert.True(Math.Abs(model.Means![smallLabel, m, f] - mainCentroid[f]) <= 6.0 * Math.Sqrt(cv[f, f]),
                        $"seed {seed}: mixture {m} of the small cluster at {model.Means[smallLabel, m, f]} is not within 6 sigma of the main centroid {mainCentroid[f]} (feature {f})");

            model.CallCheck(); // the initialized parameters are valid

            if (seed == seeds[0])
                firstSeedModel = model;
        }

        // Determinism: the same seed twice gives the same fallback draws (and the same base init).
        var again = new InitProbe(nc, nm, CovarianceType.Diag, "stmcw", seeds[0], minCovar);
        again.CallInit(x, lengths);
        Close.Equal(firstSeedModel!.StartProb!, again.StartProb!, "startprob, same seed");
        Close.Equal(firstSeedModel.TransMat!, again.TransMat!, "transmat, same seed");
        Close.Equal(firstSeedModel.Weights!, again.Weights!, "weights, same seed");
        Close.Equal(firstSeedModel.Means!, again.Means!, "means, same seed");
        Close.Equal(firstSeedModel.Covars!, again.Covars!, "covars, same seed");
    }

    [Fact]
    public void init_with_empty_init_params_leaves_every_explicit_parameter_untouched()
    {
        var x = Data();
        var (startProb, transMat, weights, means, covars) = ExplicitParameters();

        var model = new InitProbe(2, 2, CovarianceType.Diag, "", null);
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.Weights = weights;
        model.Means = means;
        model.Covars = covars;
        model.NFeatures = 2;
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.CallInit(x, new[] { 16 });

        Assert.Same(startProb, model.StartProb);
        Assert.Same(transMat, model.TransMat);
        Assert.Same(weights, model.Weights);
        Assert.Same(means, model.Means);
        Assert.Same(covars, model.Covars);
        Assert.Empty(warnings);
    }

    [Fact]
    public void init_replaces_only_the_parameters_named_by_init_params_and_warns_about_the_set_ones()
    {
        var x = Data();
        var (startProb, transMat, weights, means, covars) = ExplicitParameters();

        var model = new InitProbe(2, 2, CovarianceType.Diag, "mc", null);
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.Weights = weights;
        model.Means = means;
        model.Covars = covars;
        model.NFeatures = 2;
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.CallInit(x, new[] { 16 });

        Assert.Same(startProb, model.StartProb);
        Assert.Same(transMat, model.TransMat);
        Assert.Same(weights, model.Weights);
        Assert.NotSame(means, model.Means);
        Assert.NotSame(covars, model.Covars);

        Assert.Equal(2, warnings.Count);
        Assert.Contains(warnings, w => w.Contains("means_") && w.Contains("will be overwritten"));
        Assert.Contains(warnings, w => w.Contains("covars_") && w.Contains("will be overwritten"));

        // The replacements are exactly what the spec says.
        var (mainCenters, labels) = KMeans.Fit(x, 2);
        var expectedMeans = new double[2, 2, 2];
        for (var c = 0; c < 2; c++)
        {
            var rows = new List<double[]>();
            for (var i = 0; i < x.GetLength(0); i++)
                if (labels[i] == c)
                    rows.Add(new[] { x[i, 0], x[i, 1] });
            var cluster = new double[rows.Count, 2];
            for (var i = 0; i < rows.Count; i++)
            {
                cluster[i, 0] = rows[i][0];
                cluster[i, 1] = rows[i][1];
            }
            var centers = KMeans.Fit(cluster, 2).Centers;
            for (var m = 0; m < 2; m++)
                for (var f = 0; f < 2; f++)
                    expectedMeans[c, m, f] = centers[m, f];
        }
        Close.Equal(expectedMeans, model.Means!, "means replaced");
        Close.Equal(Distributed(DataCovariance(x, 1e-3), CovarianceType.Diag, 2, 2), model.Covars!, "covars replaced");
    }

    private static (double[] StartProb, double[,] TransMat, double[,] Weights, double[,,] Means, double[,,,] Covars) ExplicitParameters()
    {
        var startProb = new[] { 0.4, 0.6 };
        var transMat = new double[2, 2]
        {
            { 0.7, 0.3 },
            { 0.3, 0.7 },
        };
        var weights = new double[2, 2]
        {
            { 0.5, 0.5 },
            { 0.3, 0.7 },
        };
        var means = new double[2, 2, 2]
        {
            { { 1.0, 0.0 }, { 0.5, 0.2 } },
            { { 10.0, 0.1 }, { 10.5, 0.4 } },
        };
        var covars = new double[2, 2, 1, 2]
        {
            {
                { { 0.3, 0.1 } },
                { { 0.25, 0.05 } },
            },
            {
                { { 0.4, 0.08 } },
                { { 0.35, 0.06 } },
            },
        };
        return (startProb, transMat, weights, means, covars);
    }
}
