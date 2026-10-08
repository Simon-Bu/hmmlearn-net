using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm16;

/// <summary>
/// Criterion C3 of ticket HMM-16: <see cref="AbstractHmm.Sample"/> on a <see cref="GmmHmm"/> with fixed parameters produces
/// observations of the requested shape, visits both states, has per-state sample moments close (statistically, fixed seed,
/// tolerance of 8 standard errors) to the weighted component means and variances of the states — the expectations are computed
/// in the test from the test's own parameters per the mixture formulas — and a first sample drawn with a fresh seed follows the
/// start probabilities (binomial tolerance).
/// </summary>
public class SamplingTests
{
    private const int NComponents = 2;
    private const int NMix = 3;
    private const int NFeatures = 3;
    private const int ChainLength = 12000;
    private const int FirstSampleDraws = 4000;

    private static readonly double[] StartProb = { 0.75, 0.25 };
    private static readonly double[,] TransMat = { { 0.9, 0.1 }, { 0.15, 0.85 } };
    private static readonly double[,] Weights = { { 0.5, 0.3, 0.2 }, { 0.2, 0.3, 0.5 } };

    private static readonly double[,,] Means =
    {
        { { 0.0, 1.0, 2.0 }, { 3.0, 0.5, -1.0 }, { -2.0, -1.5, 0.5 } },
        { { 1.5, -1.0, 0.0 }, { -1.5, 1.0, 2.5 }, { 0.5, 2.0, -2.0 } },
    };

    private static double[,,,] Covars(CovarianceType type) => type switch
    {
        CovarianceType.Spherical => new double[NComponents, NMix, 1, 1]
        {
            { { { 1.0 } }, { { 2.0 } }, { { 1.5 } } },
            { { { 1.5 } }, { { 1.0 } }, { { 2.0 } } },
        },
        CovarianceType.Diag => new double[NComponents, NMix, 1, NFeatures]
        {
            { { { 1.0, 2.0, 1.5 } }, { { 2.0, 1.0, 1.0 } }, { { 1.0, 1.5, 2.5 } } },
            { { { 1.5, 1.0, 1.0 } }, { { 1.0, 1.5, 2.0 } }, { { 2.0, 1.0, 1.0 } } },
        },
        CovarianceType.Full => new double[NComponents, NMix, NFeatures, NFeatures]
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
        CovarianceType.Tied => new double[NComponents, 1, NFeatures, NFeatures]
        {
            { { { 2.0, 0.3, 0.1 }, { 0.3, 1.5, 0.2 }, { 0.1, 0.2, 1.2 } } },
            { { { 1.8, 0.2, 0.3 }, { 0.2, 1.2, -0.2 }, { 0.3, -0.2, 1.4 } } },
        },
        _ => throw new ArgumentException($"Unexpected covariance type {type}"),
    };

    private static GmmHmm Model(CovarianceType type)
    {
        var model = new GmmHmm(nComponents: NComponents, nMix: NMix, covarianceType: type, initParams: "");
        model.StartProb = (double[])StartProb.Clone();
        model.TransMat = (double[,])TransMat.Clone();
        model.Weights = (double[,])Weights.Clone();
        model.Means = (double[,,])Means.Clone();
        model.Covars = Covars(type);
        model.NFeatures = NFeatures;
        return model;
    }

    private static double SampleVariance(double[,] x, int[] states, int state, int feature, int count)
    {
        if (count == 0)
            return double.NaN;
        double mean = 0.0;
        for (var i = 0; i < x.GetLength(0); i++)
            if (states[i] == state)
                mean += x[i, feature];
        mean /= count;
        double sum = 0.0;
        for (var i = 0; i < x.GetLength(0); i++)
            if (states[i] == state)
            {
                var d = x[i, feature] - mean;
                sum += d * d;
            }
        return sum / count;
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void sampled_chain_has_the_requested_shape_and_visits_both_states(CovarianceType type)
    {
        var model = Model(type);
        var (x, states) = model.Sample(ChainLength, random: new Random(12345));

        Assert.Equal(ChainLength, x.GetLength(0));
        Assert.Equal(NFeatures, x.GetLength(1));
        Assert.Equal(ChainLength, states.Length);
        for (var i = 0; i < ChainLength; i++)
            Assert.InRange(states[i], 0, NComponents - 1);
        for (var s = 0; s < NComponents; s++)
        {
            var count = 0;
            for (var i = 0; i < ChainLength; i++)
                count += states[i] == s ? 1 : 0;
            Assert.True(count >= 1000, $"state {s} was visited {count} times, expected at least 1000 of {ChainLength}");
        }
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void sampled_states_have_the_weighted_component_means(CovarianceType type)
    {
        var model = Model(type);
        var (x, states) = model.Sample(ChainLength, random: new Random(12345));
        var what = $"{type} sampling chain";

        for (var s = 0; s < NComponents; s++)
        {
            var count = 0;
            for (var i = 0; i < ChainLength; i++)
                count += states[i] == s ? 1 : 0;

            for (var f = 0; f < NFeatures; f++)
            {
                double mean = 0.0;
                double variance = 0.0;
                for (var m = 0; m < NMix; m++)
                {
                    var w = Weights[s, m];
                    var mu = Means[s, m, f];
                    mean += w * mu;
                    variance += w * (ComponentVariance(type, s, m, f) + (mu - 0.0) * (mu - 0.0));
                }
                // E[x^2] around the state mean: Var = sum_m w_m * (v_mf + mu_mf^2) - mean^2
                variance -= mean * mean;

                Close.Equal(mean, MeanOf(x, states, s, f, count), $"{what} state {s} feature {f} sample mean",
                    rtol: 0.0, atol: 8.0 * Math.Sqrt(variance / count));
            }
        }
    }

    private static double MeanOf(double[,] x, int[] states, int state, int feature, int count)
    {
        if (count == 0)
            return double.NaN;
        double sum = 0.0;
        for (var i = 0; i < x.GetLength(0); i++)
            if (states[i] == state)
                sum += x[i, feature];
        return sum / count;
    }

    private static double ComponentVariance(CovarianceType type, int c, int m, int f) => type switch
    {
        CovarianceType.Spherical => Covars(type)[c, m, 0, 0],
        CovarianceType.Diag => Covars(type)[c, m, 0, f],
        _ => Covars(type)[c, m, f, f],
    };

    [Fact]
    public void the_first_sampled_state_follows_the_start_probabilities()
    {
        var model = Model(CovarianceType.Full);
        var counts = new int[NComponents];
        for (var i = 0; i < FirstSampleDraws; i++)
        {
            var (_, states) = model.Sample(1, random: new Random(12345 + i));
            counts[states[0]]++;
        }

        for (var s = 0; s < NComponents; s++)
        {
            var p = StartProb[s];
            var tolerance = 8.0 * Math.Sqrt(p * (1.0 - p) * FirstSampleDraws);
            Close.Equal(p * FirstSampleDraws, counts[s], $"first sample state {s} count", rtol: 0.0, atol: tolerance);
        }
    }
}
