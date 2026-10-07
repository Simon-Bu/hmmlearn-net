using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm13;

/// <summary>
/// Criterion C4 of ticket HMM-13: the E-step statistics of <see cref="GaussianHmm"/> — the allocation of
/// <see cref="GaussianStatistics"/>, the accumulation of <see cref="GaussianEmission.Accumulate"/>, the gating by the letters of
/// Params ('m' and 'c') of the statistics handed to the M-step during a one-iteration Fit (NIter = 1, InitParams ""), and
/// <see cref="GaussianEmission.NFitScalars"/>. Every expected statistic is computed in the test from its own inputs by plain loops
/// over the same model's posteriors (PredictProba); the fit-based tests capture the statistics in a DoMStep override, so no
/// fitting happens.
/// </summary>
public class StatisticsTests
{
    private static readonly double[] FitStartProb = { 0.4, 0.6 };
    private static readonly double[,] FitTransMat = { { 0.8, 0.2 }, { 0.3, 0.7 } };
    private static readonly double[,] FitMeans = { { 1.0, -2.0 }, { 3.0, 0.5 } };
    private static readonly double[,] X = { { 1.0, 0.0 }, { -1.0, 2.0 }, { 0.5, -1.5 }, { 2.0, 0.5 }, { -0.5, 1.0 } };
    private static readonly int[] Lengths = { 1, 4 };

    private static readonly double[,] AccumulateX = { { 0.0, 1.0 }, { -1.0, 2.0 }, { 3.0, 0.5 }, { 0.0, -0.5 } };
    private static readonly double[,] AccumulatePosteriors = { { 0.2, 0.8 }, { 0.5, 0.5 }, { 1.0, 0.0 }, { 0.25, 0.75 } };

    private static double[,,] Covars(CovarianceType type) => type switch
    {
        CovarianceType.Diag => new double[2, 1, 2] { { { 0.5, 0.25 } }, { { 1.0, 0.75 } } },
        CovarianceType.Full => new double[2, 2, 2] { { { 1.0, 0.3 }, { 0.3, 0.8 } }, { { 0.9, 0.2 }, { 0.2, 1.1 } } },
        CovarianceType.Tied => new double[1, 2, 2] { { { 1.2, 0.3 }, { 0.3, 0.9 } } },
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    /// <summary>
    /// A GaussianHmm that installs fixed parameters in Init (InitParams is ""), runs one iteration of Fit and captures the
    /// statistics handed to the M-step; nothing is fitted.
    /// </summary>
    private sealed class RecordingModel : GaussianHmm
    {
        private readonly CovarianceType _type;

        public GaussianStatistics? LastStats { get; private set; }

        public RecordingModel(CovarianceType type, string @params, Implementation implementation)
            : base(2, type, 1e-3, 1.0, 1.0, 0, 0, 1e-2, 1, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, @params, "", implementation)
        {
            _type = type;
        }

        protected override void Init(double[,] x, int[] lengths)
        {
            StartProb = (double[])StatisticsTests.FitStartProb.Clone();
            TransMat = (double[,])StatisticsTests.FitTransMat.Clone();
            Means = (double[,])StatisticsTests.FitMeans.Clone();
            Covars = (double[,,])Covars(_type).Clone();
            NFeatures = x.GetLength(1);
        }

        protected override double ComputeLowerBound(double currLogProb) => currLogProb;

        protected override void DoMStep(SufficientStatistics stats) => LastStats = (GaussianStatistics)stats;
    }

    private sealed class ScalarsModel : GaussianHmm
    {
        public ScalarsModel(CovarianceType type)
            : base(2, type, 1e-3, 1.0, 1.0, 0, 0, 1e-2, 1, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "stmc", "",
                Implementation.Log)
        {
        }

        public void RunCheck() => Check();

        public IReadOnlyDictionary<char, int>? NFitScalars => GetNFitScalarsPerParam();
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void gaussian_statistics_allocates_the_zeroed_fields(CovarianceType type)
    {
        const int nComponents = 3;
        const int nFeatures = 2;
        var stats = new GaussianStatistics(nComponents, nFeatures, type);
        Close.Equal(new double[nComponents], stats.Post, "Post is allocated and zero");
        Close.Equal(new double[nComponents, nFeatures], stats.Obs, "Obs is allocated and zero");
        Close.Equal(new double[nComponents, nFeatures], stats.Obs2, "Obs2 is allocated and zero");
        if (type is CovarianceType.Full or CovarianceType.Tied)
            Close.Equal(new double[nComponents, nFeatures, nFeatures], stats.ObsObsT!, "ObsObsT is allocated and zero");
        else
            Assert.Null(stats.ObsObsT);
    }

    [Theory]
    [InlineData(CovarianceType.Spherical, true, true)]
    [InlineData(CovarianceType.Spherical, true, false)]
    [InlineData(CovarianceType.Spherical, false, true)]
    [InlineData(CovarianceType.Spherical, false, false)]
    [InlineData(CovarianceType.Diag, true, true)]
    [InlineData(CovarianceType.Diag, false, true)]
    [InlineData(CovarianceType.Full, true, true)]
    [InlineData(CovarianceType.Full, true, false)]
    [InlineData(CovarianceType.Full, false, true)]
    [InlineData(CovarianceType.Tied, true, true)]
    [InlineData(CovarianceType.Tied, false, true)]
    public void accumulate_adds_the_posterior_weighted_sums(CovarianceType type, bool needsMean, bool needsCovars)
    {
        const int nComponents = 2;
        const int nFeatures = 2;
        var stats = new GaussianStatistics(nComponents, nFeatures, type);
        GaussianEmission.Accumulate(stats, AccumulateX, AccumulatePosteriors, type, needsMean, needsCovars);

        var n = AccumulateX.GetLength(0);
        var post = new double[nComponents];
        var obs = new double[nComponents, nFeatures];
        var obs2 = new double[nComponents, nFeatures];
        var obsObsT = new double[nComponents, nFeatures, nFeatures];
        for (var i = 0; i < n; i++)
            for (var c = 0; c < nComponents; c++)
            {
                if (needsMean)
                {
                    post[c] += AccumulatePosteriors[i, c];
                    for (var f = 0; f < nFeatures; f++)
                        obs[c, f] += AccumulatePosteriors[i, c] * AccumulateX[i, f];
                }
                if (needsCovars)
                {
                    if (type is CovarianceType.Spherical or CovarianceType.Diag)
                    {
                        for (var f = 0; f < nFeatures; f++)
                            obs2[c, f] += AccumulatePosteriors[i, c] * AccumulateX[i, f] * AccumulateX[i, f];
                    }
                    else
                    {
                        for (var j = 0; j < nFeatures; j++)
                            for (var k = 0; k < nFeatures; k++)
                                obsObsT[c, j, k] += AccumulatePosteriors[i, c] * AccumulateX[i, j] * AccumulateX[i, k];
                    }
                }
            }

        Close.Equal(post, stats.Post, "Post");
        Close.Equal(obs, stats.Obs, "Obs");
        if (type is CovarianceType.Spherical or CovarianceType.Diag)
        {
            Close.Equal(obs2, stats.Obs2, "Obs2");
            Assert.Null(stats.ObsObsT);
        }
        else
        {
            Close.Equal(new double[nComponents, nFeatures], stats.Obs2, "Obs2 must stay zero for full and tied");
            Close.Equal(obsObsT, stats.ObsObsT!, "ObsObsT");
        }
    }

    [Fact]
    public void accumulate_adds_to_the_existing_statistics()
    {
        const int nComponents = 2;
        const int nFeatures = 2;
        var stats = new GaussianStatistics(nComponents, nFeatures, CovarianceType.Spherical);
        GaussianEmission.Accumulate(stats, AccumulateX, AccumulatePosteriors, CovarianceType.Spherical, true, true);
        GaussianEmission.Accumulate(stats, AccumulateX, AccumulatePosteriors, CovarianceType.Spherical, true, true);

        var n = AccumulateX.GetLength(0);
        var post = new double[nComponents];
        var obs = new double[nComponents, nFeatures];
        var obs2 = new double[nComponents, nFeatures];
        for (var i = 0; i < n; i++)
            for (var c = 0; c < nComponents; c++)
            {
                var twice = 2.0 * AccumulatePosteriors[i, c];
                post[c] += twice;
                for (var f = 0; f < nFeatures; f++)
                {
                    obs[c, f] += twice * AccumulateX[i, f];
                    obs2[c, f] += twice * AccumulateX[i, f] * AccumulateX[i, f];
                }
            }

        Close.Equal(post, stats.Post, "Post must be the sum of both accumulate calls");
        Close.Equal(obs, stats.Obs, "Obs must be the sum of both accumulate calls");
        Close.Equal(obs2, stats.Obs2, "Obs2 must be the sum of both accumulate calls");
    }

    [Theory]
    [InlineData("mc", false)]
    [InlineData("m", false)]
    [InlineData("c", false)]
    [InlineData("stmc", false)]
    [InlineData("mc", true)]
    public void e_step_statistics_match_the_posterior_weighted_sums_of_the_diag_model(string @params, bool scaling)
    {
        var model = new RecordingModel(CovarianceType.Diag, @params, scaling ? Implementation.Scaling : Implementation.Log);
        model.Fit(X, Lengths);
        Assert.NotNull(model.LastStats);
        var stats = model.LastStats!;

        Assert.Equal(Lengths.Length, stats.NObs);

        var n = X.GetLength(0);
        const int nComponents = 2;
        const int nFeatures = 2;
        var posteriors = model.PredictProba(X, Lengths);
        var post = new double[nComponents];
        var obs = new double[nComponents, nFeatures];
        var obs2 = new double[nComponents, nFeatures];
        for (var i = 0; i < n; i++)
            for (var c = 0; c < nComponents; c++)
            {
                post[c] += posteriors[i, c];
                for (var f = 0; f < nFeatures; f++)
                {
                    obs[c, f] += posteriors[i, c] * X[i, f];
                    obs2[c, f] += posteriors[i, c] * X[i, f] * X[i, f];
                }
            }

        if (@params.Contains('m'))
        {
            Close.Equal(post, stats.Post, "Post must be the sum of the posteriors while 'm' is in Params");
            Close.Equal(obs, stats.Obs, "Obs must be the posteriors-weighted sum of the observations while 'm' is in Params");
        }
        else
        {
            Close.Equal(new double[nComponents], stats.Post, "Post must stay zero without 'm' in Params");
            Close.Equal(new double[nComponents, nFeatures], stats.Obs, "Obs must stay zero without 'm' in Params");
        }
        if (@params.Contains('c'))
            Close.Equal(obs2, stats.Obs2, "Obs2 must be the posteriors-weighted sum of the squared observations while 'c' is in Params");
        else
            Close.Equal(new double[nComponents, nFeatures], stats.Obs2, "Obs2 must stay zero without 'c' in Params");
        Assert.Null(stats.ObsObsT);
    }

    [Theory]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void e_step_statistics_of_full_and_tied_models_use_obs_obst(CovarianceType type)
    {
        var model = new RecordingModel(type, "mc", Implementation.Log);
        model.Fit(X, Lengths);
        Assert.NotNull(model.LastStats);
        var stats = model.LastStats!;

        Assert.Equal(Lengths.Length, stats.NObs);

        var n = X.GetLength(0);
        const int nComponents = 2;
        const int nFeatures = 2;
        var posteriors = model.PredictProba(X, Lengths);
        var post = new double[nComponents];
        var obs = new double[nComponents, nFeatures];
        var obsObsT = new double[nComponents, nFeatures, nFeatures];
        for (var i = 0; i < n; i++)
            for (var c = 0; c < nComponents; c++)
            {
                post[c] += posteriors[i, c];
                for (var f = 0; f < nFeatures; f++)
                    obs[c, f] += posteriors[i, c] * X[i, f];
                for (var j = 0; j < nFeatures; j++)
                    for (var k = 0; k < nFeatures; k++)
                        obsObsT[c, j, k] += posteriors[i, c] * X[i, j] * X[i, k];
            }

        Close.Equal(post, stats.Post, "Post");
        Close.Equal(obs, stats.Obs, "Obs");
        Close.Equal(new double[nComponents, nFeatures], stats.Obs2, "Obs2 must stay zero for full and tied covariances");
        Close.Equal(obsObsT, stats.ObsObsT, "ObsObsT must be the posteriors-weighted sum of the outer products");
    }

    [Theory]
    [InlineData(2, 3, CovarianceType.Spherical)]
    [InlineData(2, 3, CovarianceType.Diag)]
    [InlineData(3, 2, CovarianceType.Full)]
    [InlineData(3, 2, CovarianceType.Tied)]
    public void n_fit_scalars_gives_the_documented_counts(int nComponents, int nFeatures, CovarianceType type)
    {
        var expectedC = type switch
        {
            CovarianceType.Spherical => nComponents,
            CovarianceType.Diag => nComponents * nFeatures,
            CovarianceType.Full => nComponents * nFeatures * (nFeatures + 1) / 2,
            _ => nFeatures * (nFeatures + 1) / 2,
        };
        var actual = GaussianEmission.NFitScalars(nComponents, nFeatures, type);
        Assert.Equal(4, actual.Count);
        Assert.Equal(nComponents - 1, actual['s']);
        Assert.Equal(nComponents * (nComponents - 1), actual['t']);
        Assert.Equal(nComponents * nFeatures, actual['m']);
        Assert.Equal(expectedC, actual['c']);
    }

    [Fact]
    public void n_fit_scalars_per_param_is_null_without_features_and_matches_the_helper_after_check()
    {
        const int nComponents = 2;
        const int nFeatures = 3;
        var model = new ScalarsModel(CovarianceType.Full);
        Assert.Null(model.NFitScalars);

        var startProb = new double[nComponents];
        var transMat = new double[nComponents, nComponents];
        for (var i = 0; i < nComponents; i++)
        {
            startProb[i] = 1.0 / nComponents;
            for (var j = 0; j < nComponents; j++)
                transMat[i, j] = 1.0 / nComponents;
        }
        model.StartProb = startProb;
        model.TransMat = transMat;
        var means = new double[nComponents, nFeatures];
        for (var c = 0; c < nComponents; c++)
            for (var f = 0; f < nFeatures; f++)
                means[c, f] = c * nFeatures + f;
        model.Means = means;
        model.Covars = new double[nComponents, nFeatures, nFeatures]
        {
            { { 2.0, 0.1, 0.0 }, { 0.1, 1.0, 0.2 }, { 0.0, 0.2, 1.5 } },
            { { 1.5, 0.0, 0.0 }, { 0.0, 1.0, 0.1 }, { 0.0, 0.1, 1.0 } },
        };
        model.RunCheck();

        Assert.NotNull(model.NFitScalars);
        var perParam = model.NFitScalars!;
        var expected = GaussianEmission.NFitScalars(nComponents, nFeatures, CovarianceType.Full);
        Assert.Equal(expected.Count, perParam.Count);
        foreach (var entry in expected)
            Assert.Equal(entry.Value, perParam[entry.Key]);
    }
}
