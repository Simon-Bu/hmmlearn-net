using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm14;

/// <summary>
/// Acceptance tests for the Gaussian initialization (ticket HMM-14, criterion C2): Init first runs the base
/// initialization (startprob, transmat, number of features) and then, only for the letters of InitParams that need it
/// (not set yet, or named by InitParams), sets Means to the k-means centers of x and Covars to the data covariance of
/// x (sample covariance of the columns, dividing by n - 1, zeros if x has a single row) plus MinCovar on the diagonal,
/// distributed to the covariance type; it is reproducible for a fixed RandomState (and not for another one), leaves the
/// explicit parameters alone (same reference) when InitParams does not name them, raises the exact "will be overwritten"
/// warning for every set parameter named by InitParams, and a random Init of a well-separated model starts close to the
/// true parameters with valid covariances.
/// </summary>
public class InitTests
{
    /// <summary>Eight observations in two well-separated clusters.</summary>
    private static double[,] Data() => new double[8, 2]
    {
        { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.5, 0.5 }, { 0.2, -0.3 },
        { 10.0, 0.0 }, { 11.0, 0.0 }, { 10.5, 0.5 }, { 10.2, 0.3 },
    };

    /// <summary>The data covariance from the spec: sample covariance of the columns (dividing by n - 1, zeros if one row) plus minCovar on the diagonal.</summary>
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

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void init_starts_from_the_kmeans_centers_and_the_data_covariance(CovarianceType type)
    {
        const double minCovar = 0.25;
        var x = Data();
        var model = new InitProbe(2, type, "stmc", 42, minCovar);
        model.CallInit(x, new[] { 8 });

        Close.Equal(KMeans.Fit(x, 2).Centers, model.Means!, "means: the k-means centers");
        var expected = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(DataCovariance(x, minCovar), type, 2);
        Close.Equal(expected, model.Covars!, "covars: the data covariance plus minCovar, distributed to the type");
    }

    [Fact]
    public void init_with_empty_init_params_leaves_every_explicit_parameter_untouched()
    {
        var x = Data();
        var startProb = new[] { 0.9, 0.1 };
        var transMat = new double[2, 2]
        {
            { 0.8, 0.2 },
            { 0.2, 0.8 },
        };
        var means = new double[2, 2]
        {
            { 1.0, 1.0 },
            { -1.0, -1.0 },
        };
        var covars = new double[2, 1, 1]
        {
            { { 0.5 } },
            { { 0.7 } },
        };

        var model = new InitProbe(2, CovarianceType.Spherical, "", null);
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.Means = means;
        model.Covars = covars;
        model.CallInit(x, new[] { 8 });

        Assert.Same(startProb, model.StartProb);
        Assert.Same(transMat, model.TransMat);
        Assert.Same(means, model.Means);
        Close.Equal(covars, model.Covars!, "covars untouched");
    }

    [Fact]
    public void init_with_mc_init_params_replaces_only_means_and_covars()
    {
        var x = Data();
        var startProb = new[] { 0.9, 0.1 };
        var transMat = new double[2, 2]
        {
            { 0.8, 0.2 },
            { 0.2, 0.8 },
        };
        var means = new double[2, 2]
        {
            { 1.0, 1.0 },
            { -1.0, -1.0 },
        };
        var covars = new double[2, 1, 1]
        {
            { { 0.5 } },
            { { 0.7 } },
        };

        var model = new InitProbe(2, CovarianceType.Spherical, "mc", null);
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.Means = means;
        model.Covars = covars;
        model.CallInit(x, new[] { 8 });

        Assert.Same(startProb, model.StartProb);
        Assert.Same(transMat, model.TransMat);
        Assert.NotSame(means, model.Means);
        Close.Equal(KMeans.Fit(x, 2).Centers, model.Means!, "means replaced by the k-means centers");
        var expected = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(DataCovariance(x, 1e-3), CovarianceType.Spherical, 2);
        Close.Equal(expected, model.Covars!, "covars replaced by the data covariance plus minCovar");
    }

    [Fact]
    public void init_is_reproducible_for_a_fixed_random_state_and_differs_for_another_seed()
    {
        var x = Data();
        var first = new InitProbe(2, CovarianceType.Diag, "stmc", 7);
        var second = new InitProbe(2, CovarianceType.Diag, "stmc", 7);
        var other = new InitProbe(2, CovarianceType.Diag, "stmc", 8);
        first.CallInit(x, new[] { 8 });
        second.CallInit(x, new[] { 8 });
        other.CallInit(x, new[] { 8 });

        Close.Equal(first.StartProb!, second.StartProb!, "startprob for the same seed");
        Close.Equal(first.TransMat!, second.TransMat!, "transmat for the same seed");
        Close.Equal(first.Means!, second.Means!, "means for the same seed");
        Close.Equal(first.Covars!, second.Covars!, "covars for the same seed");
        Assert.False(ArrayClose(first.StartProb!, other.StartProb!) && ArrayClose(first.TransMat!, other.TransMat!),
            "two different seeds drew the same startprob and transmat");
    }

    [Fact]
    public void init_warns_for_every_explicit_parameter_named_by_init_params_and_overwrites_it()
    {
        var x = Data();
        var startProb = new[] { 0.9, 0.1 };
        var transMat = new double[2, 2]
        {
            { 0.8, 0.2 },
            { 0.2, 0.8 },
        };
        var means = new double[2, 2]
        {
            { 1.0, 1.0 },
            { -1.0, -1.0 },
        };
        var covars = new double[2, 1, 1]
        {
            { { 0.5 } },
            { { 0.7 } },
        };

        var model = new InitProbe(2, CovarianceType.Spherical, "stmc", null);
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.Means = means;
        model.Covars = covars;
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.CallInit(x, new[] { 8 });

        Assert.Equal(4, warnings.Count);
        foreach (var name in new[] { "startprob_", "transmat_", "means_", "covars_" })
            Assert.True(warnings.Count(w => w.Contains(name) && w.Contains("will be overwritten")) == 1,
                $"expected exactly one warning naming {name} and saying it will be overwritten, got: [{string.Join(" | ", warnings)}]");
        Assert.NotSame(means, model.Means);
        Close.Equal(KMeans.Fit(x, 2).Centers, model.Means!, "means overwritten by the k-means centers");
    }

    [Fact]
    public void init_warns_only_for_the_set_parameters_named_by_init_params()
    {
        var x = Data();
        var means = new double[2, 2]
        {
            { 1.0, 1.0 },
            { -1.0, -1.0 },
        };

        var model = new InitProbe(2, CovarianceType.Spherical, "mc", null);
        model.Means = means;
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.CallInit(x, new[] { 8 });

        Assert.Equal(1, warnings.Count);
        Assert.True(warnings[0].Contains("means_") && warnings[0].Contains("will be overwritten"),
            $"unexpected warning: {warnings[0]}");
        // startprob and transmat were not set and are not named by init_params: NeedsInit returns true for an attribute that is
        // not set, so they are initialized (silently), like the covars below.
        Assert.NotNull(model.StartProb);
        Assert.NotNull(model.TransMat);
        Close.Equal(KMeans.Fit(x, 2).Centers, model.Means!, "means overwritten");
        Assert.NotNull(model.Covars);
        // covars was not set: initialized silently, without a warning.
        var expected = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(DataCovariance(x, 1e-3), CovarianceType.Spherical, 2);
        Close.Equal(expected, model.Covars!, "covars initialized silently");
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void random_init_of_a_well_separated_model_starts_close_to_the_true_parameters(CovarianceType type)
    {
        var (x, _) = TrueModel(type).Sample(600, random: new Random(1234));

        var model = new InitProbe(3, type, "stmc", 4242);
        model.CallInit(x, new[] { 600 });

        var trueMeans = TrueModel(type).Means!;
        var best = BestPermutation(model.Means!, trueMeans);
        for (var s = 0; s < 3; s++)
            for (var f = 0; f < 3; f++)
                Assert.True(Math.Abs(model.Means![s, f] - trueMeans[best[s], f]) <= 2.0,
                    $"fitted means {model.Means[s, f]} not close to true means {trueMeans[best[s], f]} (state {s} -> {best[s]}, feature {f})");

        var covars = model.Covars!;
        switch (type)
        {
            case CovarianceType.Spherical:
                for (var k = 0; k < 3; k++)
                    Assert.True(covars[k, 0, 0] > 0.0, $"the spherical variance of state {k} is not positive");
                break;
            case CovarianceType.Diag:
                for (var k = 0; k < 3; k++)
                    for (var f = 0; f < 3; f++)
                        Assert.True(covars[k, 0, f] > 0.0, $"the diagonal variance of state {k}, feature {f}, is not positive");
                break;
            default:
                for (var k = 0; k < covars.GetLength(0); k++)
                    AssertCovarianceMatrix(covars, k, $"state {k}");
                break;
        }
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void init_with_a_single_row_of_data_uses_the_zero_sample_covariance(CovarianceType type)
    {
        const double minCovar = 1e-3;
        var x = new double[1, 3] { { 1.0, 2.0, 3.0 } };

        var model = new InitProbe(1, type, "stmc", 42, minCovar);
        model.CallInit(x, new[] { 1 });

        Close.Equal(x, model.Means!, "means: the k-means center of the single row");
        var cov = new double[3, 3];
        for (var f = 0; f < 3; f++)
            cov[f, f] += minCovar;
        Close.Equal(InternalUtils.DistributeCovarMatrixToMatchCovarianceType(cov, type, 1), model.Covars!,
            "covars: the zero sample covariance plus minCovar, distributed to the type");
    }

    /// <summary>A true, well-separated 3-state, 3-feature model of the given covariance type.</summary>
    private static GaussianHmm TrueModel(CovarianceType type)
    {
        var model = new GaussianHmm(3, type, 1e-3, 1.0, 1.0, 0.0, 0.0, 1e-2, 1.0,
            DecoderAlgorithm.Viterbi, null, 0, 1e-2, false, "stmc", "", Implementation.Log);
        model.StartProb = new[] { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
        model.TransMat = new double[3, 3]
        {
            { 0.9, 0.05, 0.05 },
            { 0.05, 0.9, 0.05 },
            { 0.05, 0.05, 0.9 },
        };
        model.Means = new double[3, 3]
        {
            { 15.0, 0.0, 0.0 },
            { 0.0, 15.0, 0.0 },
            { 0.0, 0.0, 15.0 },
        };
        model.Covars = TrueCompactCovars(type);
        return model;
    }

    /// <summary>The full covariance matrix of every state of the true model (identity, or the correlated matrix for Full and Tied).</summary>
    private static double[,] TrueCovarianceTemplate(CovarianceType type) => type switch
    {
        CovarianceType.Full or CovarianceType.Tied => new double[3, 3]
        {
            { 1.0, 0.3, 0.0 },
            { 0.3, 1.0, 0.4 },
            { 0.0, 0.4, 1.0 },
        },
        _ => new double[3, 3]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 },
        },
    };

    private static double[,,] TrueCompactCovars(CovarianceType type)
    {
        var t = TrueCovarianceTemplate(type);
        var nc = type == CovarianceType.Tied ? 1 : 3;
        var covars = type switch
        {
            CovarianceType.Spherical => new double[3, 1, 1],
            CovarianceType.Diag => new double[3, 1, 3],
            _ => new double[nc, 3, 3],
        };
        for (var s = 0; s < nc; s++)
        {
            if (type == CovarianceType.Spherical)
                covars[s, 0, 0] = t[0, 0];
            else if (type == CovarianceType.Diag)
                for (var f = 0; f < 3; f++)
                    covars[s, 0, f] = t[f, f];
            else
                for (var i = 0; i < 3; i++)
                    for (var j = 0; j < 3; j++)
                        covars[s, i, j] = t[i, j];
        }
        return covars;
    }

    /// <summary>The true covariances as full matrices, one per state, for the comparison against FullCovars.</summary>
    private static double[,,] TrueFullCovars(CovarianceType type)
    {
        var t = TrueCovarianceTemplate(type);
        var result = new double[3, 3, 3];
        for (var s = 0; s < 3; s++)
            for (var i = 0; i < 3; i++)
                for (var j = 0; j < 3; j++)
                    result[s, i, j] = t[i, j];
        return result;
    }

    private static (double[,] X, int[] Lengths) Data(CovarianceType type, int seed)
    {
        var (x, _) = TrueModel(type).Sample(1200, random: new Random(seed));
        return (x, new[] { 1200 });
    }

    private static int Index(CovarianceType type) => type switch
    {
        CovarianceType.Spherical => 0,
        CovarianceType.Diag => 1,
        CovarianceType.Full => 2,
        _ => 3,
    };

    private static bool ArrayClose(Array a, Array b)
    {
        if (a.Length != b.Length || a.Rank != b.Rank)
            return false;
        var flatA = a.Cast<double>().ToArray();
        var flatB = b.Cast<double>().ToArray();
        for (var i = 0; i < flatA.Length; i++)
            if (!Close.IsClose(flatA[i], flatB[i]))
                return false;
        return true;
    }

    private static void AssertCovarianceMatrix(double[,,] covars, int k, string what)
    {
        var nf = covars.GetLength(2);
        var m = new double[nf, nf];
        for (var i = 0; i < nf; i++)
            for (var j = 0; j < nf; j++)
                m[i, j] = covars[k, i, j];
        for (var i = 0; i < nf; i++)
            for (var j = 0; j < nf; j++)
            {
                var a = m[i, j];
                var b = m[j, i];
                Assert.True(Math.Abs(a - b) <= 1e-8 + 1e-5 * Math.Abs(b),
                    $"{what}: the covariance is not symmetric, [{i}, {j}] = {a} vs [{j}, {i}] = {b}");
            }
        Assert.True(GaussianEmission.IsPositiveDefinite(m), $"{what}: the covariance is not positive definite");
    }

    private static int[] BestPermutation(double[,] fitted, double[,] truth)
    {
        var n = fitted.GetLength(0);
        var permutations = new List<int[]>();
        var current = new int[n];
        for (var i = 0; i < n; i++)
            current[i] = i;
        DoPermutations(current, n - 1, permutations);
        var best = current;
        var bestDistance = double.MaxValue;
        foreach (var p in permutations)
        {
            var distance = 0.0;
            for (var s = 0; s < n; s++)
                for (var f = 0; f < fitted.GetLength(1); f++)
                    distance += (fitted[s, f] - truth[p[s], f]) * (fitted[s, f] - truth[p[s], f]);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = p;
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

    /// <summary>A Gaussian HMM with the given initialization settings, ready for a direct Init call.</summary>
    private sealed class InitProbe : GaussianHmm
    {
        public InitProbe(int nComponents, CovarianceType type, string initParams, int? randomState,
            double minCovar = 1e-3, string @params = "stmc")
            : base(nComponents, type, minCovar, 1.0, 1.0, 0.0, 0.0, 1e-2, 1.0,
                DecoderAlgorithm.Viterbi, randomState, 0, 1e-2, false, @params, initParams, Implementation.Log)
        {
        }

        public void CallInit(double[,] x, int[] lengths) => Init(x, lengths);
    }
}
