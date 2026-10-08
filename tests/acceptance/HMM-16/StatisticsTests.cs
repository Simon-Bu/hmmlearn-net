using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm16;

/// <summary>
/// Criterion C4 of ticket HMM-16: the E-step statistics of <see cref="GmmHmm"/> — the allocation and initialization of
/// <see cref="GmmStatistics"/> (MN starts at MeansWeight * MeansPrior for every entry, the other statistics zero), the
/// accumulation of the E-step for every covariance type, with and without the 'm' and 'c' letters of Params (PostMixSum and
/// PostSum always), captured in a DoMStep override during a one-iteration Fit (NIter = 1, InitParams ""), and
/// <see cref="AbstractHmm.GetNFitScalarsPerParam"/> for the four types. Every expected statistic is computed in the test from
/// its own inputs: the state posteriors come from the same model's PredictProba, the component posteriors from
/// <see cref="Stats.LogMultivariateNormalDensity"/> and the plain weights, normalized with a plain log-domain loop.
/// </summary>
public class StatisticsTests
{
    private const int NComponents = 2;
    private const int NMix = 3;
    private const int NFeatures = 2;

    private static readonly double[,] X =
    {
        { 0.5, 1.2 }, { -1.0, 0.3 }, { 2.2, 0.8 }, { 0.1, -1.4 }, { -1.8, 0.2 },
        { 1.4, -0.9 }, { 0.3, 2.1 }, { -2.3, 0.6 }, { 0.9, 1.8 }, { -0.4, -0.7 },
    };
    private static readonly int[] Lengths = { 4, 6 };

    private static double[,,,] Covars(CovarianceType type) => type switch
    {
        CovarianceType.Spherical => new double[NComponents, NMix, 1, 1]
        {
            { { { 1.0 } }, { { 2.0 } }, { { 0.5 } } },
            { { { 1.5 } }, { { 0.8 } }, { { 1.2 } } },
        },
        CovarianceType.Diag => new double[NComponents, NMix, 1, NFeatures]
        {
            { { { 1.0, 2.0 } }, { { 2.0, 0.5 } }, { { 0.5, 1.0 } } },
            { { { 1.5, 0.8 } }, { { 0.8, 1.2 } }, { { 1.2, 0.6 } } },
        },
        CovarianceType.Full => new double[NComponents, NMix, NFeatures, NFeatures]
        {
            { { { 1.0, 0.3 }, { 0.3, 2.0 } }, { { 2.0, -0.5 }, { -0.5, 1.0 } }, { { 0.5, 0.2 }, { 0.2, 0.8 } } },
            { { { 1.5, 0.4 }, { 0.4, 0.8 } }, { { 0.8, 0.1 }, { 0.1, 1.2 } }, { { 1.2, -0.3 }, { -0.3, 0.6 } } },
        },
        CovarianceType.Tied => new double[NComponents, 1, NFeatures, NFeatures]
        {
            { { { 1.0, 0.3 }, { 0.3, 2.0 } } },
            { { { 1.5, 0.4 }, { 0.4, 0.8 } } },
        },
        _ => throw new ArgumentException($"Unexpected covariance type {type}"),
    };

    /// <summary>
    /// A GmmHmm that installs fixed parameters in Init (InitParams is ""), runs one iteration of Fit and captures the
    /// statistics handed to the M-step; nothing is fitted.
    /// </summary>
    private sealed class RecordingModel : GmmHmm
    {
        private readonly CovarianceType _type;

        public GmmStatistics? LastStats { get; private set; }

        public RecordingModel(CovarianceType type, string @params, Implementation implementation)
            : base(nComponents: StatisticsTests.NComponents, nMix: StatisticsTests.NMix, meansWeight: 2.5, meansPrior: 0.3,
                covarianceType: type, @params: @params, initParams: "", nIter: 1, implementation: implementation)
        {
            _type = type;
        }

        protected override void Init(double[,] x, int[] lengths)
        {
            StartProb = new[] { 0.7, 0.3 };
            TransMat = new double[StatisticsTests.NComponents, StatisticsTests.NComponents] { { 0.8, 0.2 }, { 0.3, 0.7 } };
            Weights = new double[StatisticsTests.NComponents, StatisticsTests.NMix] { { 0.5, 0.3, 0.2 }, { 0.2, 0.5, 0.3 } };
            Means = new double[StatisticsTests.NComponents, StatisticsTests.NMix, StatisticsTests.NFeatures]
            {
                { { 0.0, 1.0 }, { 2.0, 0.5 }, { -1.5, -0.5 } },
                { { 1.0, -1.0 }, { -2.0, 0.5 }, { 0.5, 2.0 } },
            };
            Covars = Covars(_type);
            NFeatures = x.GetLength(1);
        }

        protected override void DoMStep(SufficientStatistics stats) => LastStats = (GmmStatistics)stats;
    }

    private sealed class ScalarsModel : GmmHmm
    {
        public ScalarsModel(int nComponents, int nMix, CovarianceType type)
            : base(nComponents: nComponents, nMix: nMix, covarianceType: type, initParams: "")
        {
        }

        public void RunCheck() => Check();

        public IReadOnlyDictionary<char, int>? PerParam => GetNFitScalarsPerParam();
    }

    public static IEnumerable<object[]> EstepCases =>
        new[] { CovarianceType.Spherical, CovarianceType.Diag, CovarianceType.Full, CovarianceType.Tied }
            .SelectMany(type => new[] { "stmcw", "stwc", "stmw", "st", "stmc" }
                .SelectMany(@params => new[] { Implementation.Log, Implementation.Scaling }
                    .Select(implementation => new object[] { type, @params, implementation })));

    [Fact]
    public void the_gmm_statistics_constructor_allocates_and_initializes_its_statistics()
    {
        var stats = new GmmStatistics(2, 3, 2, 2.5, 0.3, new[] { 2, 3, 1, 2 });
        var prior = 2.5 * 0.3;

        Assert.Equal(0, stats.NObs);
        Close.Equal(new double[2], stats.Start, "Start");
        Close.Equal(new double[2, 2], stats.Trans, "Trans");
        Close.Equal(new double[2, 3], stats.PostMixSum, "PostMixSum");
        Close.Equal(new double[2], stats.PostSum, "PostSum");

        var mn = new double[2, 3, 2];
        for (var c = 0; c < 2; c++)
            for (var m = 0; m < 3; m++)
                for (var f = 0; f < 2; f++)
                    mn[c, m, f] = prior;
        Close.Equal(mn, stats.MN, "MN");
        Close.Equal(new double[2, 3, 1, 2], stats.CN, "CN");
    }

    [Theory]
    [MemberData(nameof(EstepCases))]
    public void e_step_statistics_match_the_posterior_weighted_sums(CovarianceType type, string @params, Implementation implementation)
    {
        var model = new RecordingModel(type, @params, implementation);
        model.Fit(X, Lengths);
        var stats = model.LastStats!;
        var what = $"{type} params=\"{@params}\" ({implementation})";

        var n = X.GetLength(0);
        Assert.Equal(Lengths.Length, stats.NObs);

        var posteriors = model.PredictProba(X, Lengths); // (n, nc), from the same model
        var componentPost = new double[n, NComponents, NMix];
        for (var c = 0; c < NComponents; c++)
        {
            var densities = Stats.LogMultivariateNormalDensity(X, StateMeans(model.Means, c),
                ComponentCovars(model.Covars, c, type), type); // (n, nm)
            for (var t = 0; t < n; t++)
            {
                var logProbs = new double[NMix];
                var rowMax = double.MinValue;
                for (var m = 0; m < NMix; m++)
                {
                    logProbs[m] = Math.Log(model.Weights[c, m]) + densities[t, m];
                    if (logProbs[m] > rowMax) rowMax = logProbs[m];
                }
                double sum = 0.0;
                for (var m = 0; m < NMix; m++)
                    sum += Math.Exp(logProbs[m] - rowMax);
                for (var m = 0; m < NMix; m++)
                    componentPost[t, c, m] = Math.Exp(logProbs[m] - rowMax) / sum;
            }
        }

        var expectedPostMixSum = new double[NComponents, NMix];
        var expectedPostSum = new double[NComponents];
        var expectedMn = new double[NComponents, NMix, NFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var m = 0; m < NMix; m++)
                for (var f = 0; f < NFeatures; f++)
                    expectedMn[c, m, f] = model.MeansWeight * model.MeansPrior; // the constructor's start value

        for (var t = 0; t < n; t++)
        {
            for (var c = 0; c < NComponents; c++)
            {
                expectedPostSum[c] += posteriors[t, c];
                for (var m = 0; m < NMix; m++)
                    expectedPostMixSum[c, m] += posteriors[t, c] * componentPost[t, c, m];
            }
            if (@params.Contains('m'))
                for (var c = 0; c < NComponents; c++)
                    for (var m = 0; m < NMix; m++)
                        for (var f = 0; f < NFeatures; f++)
                            expectedMn[c, m, f] += posteriors[t, c] * componentPost[t, c, m] * X[t, f];
        }

        var covarsShape = CovarsShape(type);
        var expectedCn = (double[,,,])Array.CreateInstance(typeof(double), covarsShape);
        if (@params.Contains('c'))
        {
            for (var t = 0; t < n; t++)
            {
                for (var c = 0; c < NComponents; c++)
                {
                    var stateMeans = StateMeans(model.Means, c);
                    for (var m = 0; m < NMix; m++)
                    {
                        var weight = posteriors[t, c] * componentPost[t, c, m];
                        if (weight == 0.0)
                            continue;
                        var d = new double[NFeatures];
                        for (var f = 0; f < NFeatures; f++)
                            d[f] = X[t, f] - stateMeans[m, f];
                        switch (type)
                        {
                            case CovarianceType.Spherical:
                                expectedCn[c, m, 0, 0] += weight * (d[0] * d[0] + d[1] * d[1]);
                                break;
                            case CovarianceType.Diag:
                                for (var f = 0; f < NFeatures; f++)
                                    expectedCn[c, m, 0, f] += weight * d[f] * d[f];
                                break;
                            default: // Full: per mixture; Tied: the one matrix of the state, summed over the mixtures
                            {
                                var mixture = type == CovarianceType.Tied ? 0 : m;
                                for (var j = 0; j < NFeatures; j++)
                                    for (var k = 0; k < NFeatures; k++)
                                        expectedCn[c, mixture, j, k] += weight * d[j] * d[k];
                                break;
                            }
                        }
                    }
                }
            }
        }

        Close.Equal(expectedPostMixSum, stats.PostMixSum, $"{what} PostMixSum");
        Close.Equal(expectedPostSum, stats.PostSum, $"{what} PostSum");
        Close.Equal(expectedMn, stats.MN, $"{what} MN");
        Close.Equal(expectedCn, stats.CN, $"{what} CN");

        var postSumTotal = 0.0;
        for (var c = 0; c < NComponents; c++)
            postSumTotal += stats.PostSum[c];
        Close.Equal((double)n, postSumTotal, $"{what} total post sum");
        for (var c = 0; c < NComponents; c++)
        {
            var rowSum = 0.0;
            for (var m = 0; m < NMix; m++)
                rowSum += stats.PostMixSum[c, m];
            Close.Equal(stats.PostSum[c], rowSum, $"{what} PostMixSum row {c} sums to PostSum");
        }
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void the_number_of_fit_scalars_per_param_matches_the_ticket(CovarianceType type)
    {
        const int nComponents = 2;
        const int nMix = 3;
        const int nFeatures = 3;
        var model = new ScalarsModel(nComponents, nMix, type);
        var means = new double[nComponents, nMix, nFeatures];
        for (var c = 0; c < nComponents; c++)
            for (var m = 0; m < nMix; m++)
                for (var f = 0; f < nFeatures; f++)
                    means[c, m, f] = 1.0;
        model.Means = means;
        model.Covars = type switch
        {
            CovarianceType.Spherical => new double[nComponents, nMix, 1, 1]
            {
                { { { 1.0 } }, { { 2.0 } }, { { 1.5 } } },
                { { { 1.5 } }, { { 1.0 } }, { { 2.0 } } },
            },
            CovarianceType.Diag => new double[nComponents, nMix, 1, nFeatures]
            {
                { { { 1.0, 2.0, 1.5 } }, { { 2.0, 1.0, 1.0 } }, { { 1.0, 1.5, 2.5 } } },
                { { { 1.5, 1.0, 1.0 } }, { { 1.0, 1.5, 2.0 } }, { { 2.0, 1.0, 1.0 } } },
            },
            CovarianceType.Full => new double[nComponents, nMix, nFeatures, nFeatures]
            {
                {
                    { { 2.0, 0.3, 0.1 }, { 0.3, 1.5, 0.2 }, { 0.1, 0.2, 1.2 } },
                    { { 1.5, 0.2, -0.3 }, { 0.2, 2.0, 0.4 }, { -0.3, 0.4, 1.8 } },
                    { { 1.2, 0.4, 0.2 }, { 0.4, 1.0, 0.3 }, { 0.2, 0.3, 1.6 } },
                },
                {
                    { { 1.8, 0.2, 0.3 }, { 0.2, 1.2, -0.2 }, { 0.3, -0.2, 1.4 } },
                    { { 1.4, 0.3, 0.1 }, { 0.3, 1.6, 0.2 }, { 0.1, 0.2, 1.1 } },
                    { { 1.6, 0.1, 0.4 }, { 0.1, 1.3, 0.2 }, { 0.4, 0.2, 1.7 } },
                },
            },
            _ => new double[nComponents, 1, nFeatures, nFeatures]
            {
                { { { 2.0, 0.3, 0.1 }, { 0.3, 1.5, 0.2 }, { 0.1, 0.2, 1.2 } } },
                { { { 1.8, 0.2, 0.3 }, { 0.2, 1.2, -0.2 }, { 0.3, -0.2, 1.4 } } },
            },
        };
        model.NFeatures = nFeatures;
        model.RunCheck();

        var expected = new Dictionary<char, int>
        {
            ['s'] = nComponents - 1,
            ['t'] = nComponents * (nComponents - 1),
            ['m'] = nComponents * nMix * nFeatures,
            ['c'] = type switch
            {
                CovarianceType.Spherical => nComponents * nMix,
                CovarianceType.Diag => nComponents * nMix * nFeatures,
                CovarianceType.Full => nComponents * nMix * nFeatures * (nFeatures + 1) / 2,
                _ => nComponents * nFeatures * (nFeatures + 1) / 2,
            },
            ['w'] = nMix - 1,
        };

        Assert.NotNull(model.PerParam);
        var perParam = model.PerParam!;
        Assert.Equal(expected.Count, perParam.Count);
        foreach (var entry in expected)
            Assert.Equal(entry.Value, perParam[entry.Key]);
    }

    private static double[,] StateMeans(double[,,] means, int c)
    {
        var result = new double[NMix, NFeatures];
        for (var m = 0; m < NMix; m++)
            for (var f = 0; f < NFeatures; f++)
                result[m, f] = means[c, m, f];
        return result;
    }

    private static double[,,] ComponentCovars(double[,,,] covars, int c, CovarianceType type)
    {
        var result = type switch
        {
            CovarianceType.Spherical => new double[NMix, 1, 1],
            CovarianceType.Diag => new double[NMix, 1, NFeatures],
            CovarianceType.Full => new double[NMix, NFeatures, NFeatures],
            _ => new double[1, NFeatures, NFeatures],
        };
        for (var m = 0; m < NMix; m++)
        {
            if (type == CovarianceType.Spherical)
                result[m, 0, 0] = covars[c, m, 0, 0];
            else if (type == CovarianceType.Diag)
                for (var f = 0; f < NFeatures; f++)
                    result[m, 0, f] = covars[c, m, 0, f];
            else if (type == CovarianceType.Full)
                for (var j = 0; j < NFeatures; j++)
                    for (var k = 0; k < NFeatures; k++)
                        result[m, j, k] = covars[c, m, j, k];
        }
        if (type == CovarianceType.Tied)
            for (var j = 0; j < NFeatures; j++)
                for (var k = 0; k < NFeatures; k++)
                    result[0, j, k] = covars[c, 0, j, k];
        return result;
    }

    private static int[] CovarsShape(CovarianceType type) => type switch
    {
        CovarianceType.Spherical => new[] { NComponents, NMix, 1, 1 },
        CovarianceType.Diag => new[] { NComponents, NMix, 1, NFeatures },
        CovarianceType.Full => new[] { NComponents, NMix, NFeatures, NFeatures },
        _ => new[] { NComponents, 1, NFeatures, NFeatures },
    };
}
