using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm13;

/// <summary>
/// Criterion C3 of ticket HMM-13: the sampling of <see cref="GaussianHmm"/> (Sample and, behind it,
/// GenerateSampleFromState) for the four covariance types. With a fixed seed and enough samples (in the thousands), the per-state
/// sample mean and sample covariance must match the model's parameters and FullCovars within 6 standard errors, the shapes must be
/// (n_samples, n_features), the states must be in range, and the first state of a sample must follow StartProb. All parameters are
/// the test's own hand-made inputs; every expected value is computed in the test from them (Model.FullCovars for the sample
/// covariance target), never retyped from an oracle.
/// </summary>
public class SamplingTests
{
    private static readonly double[] StartProb = { 0.7, 0.3 };
    private static readonly double[,] TransMat = { { 0.95, 0.05 }, { 0.05, 0.95 } };
    private static readonly double[,] Means = { { 1.0, -2.0, 0.5 }, { -1.0, 3.0, -0.5 } };

    private static double[,,] Covars(CovarianceType type) => type switch
    {
        CovarianceType.Spherical => new double[2, 1, 1] { { { 1.0 } }, { { 0.8 } } },
        CovarianceType.Diag => new double[2, 1, 3] { { { 1.0, 2.0, 0.5 } }, { { 0.8, 1.5, 1.2 } } },
        CovarianceType.Full => new double[2, 3, 3]
        {
            { { 1.0, 0.3, 0.1 }, { 0.3, 2.0, 0.2 }, { 0.1, 0.2, 0.5 } },
            { { 0.8, -0.2, 0.05 }, { -0.2, 1.5, 0.1 }, { 0.05, 0.1, 1.2 } },
        },
        CovarianceType.Tied => new double[1, 3, 3] { { { 1.0, 0.2, 0.1 }, { 0.2, 1.5, 0.1 }, { 0.1, 0.1, 0.9 } } },
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    private static GaussianHmm Model(CovarianceType type)
    {
        var model = new GaussianHmm(2, type, 1e-3, 1.0, 1.0, 0, 0, 1e-2, 1, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false,
            "stmc", "", Implementation.Log);
        model.StartProb = StartProb;
        model.TransMat = TransMat;
        model.Means = Means;
        model.Covars = Covars(type);
        return model;
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void sample_per_state_mean_and_covariance_match_the_model(CovarianceType type)
    {
        const int nSamples = 10000;
        const int nFeatures = 3;
        var model = Model(type);
        var (x, states) = model.Sample(nSamples, random: new Random(12345));

        Assert.Equal(nSamples, x.GetLength(0));
        Assert.Equal(nFeatures, x.GetLength(1));
        Assert.Equal(nSamples, states.Length);

        var full = model.FullCovars;
        var counts = new int[2];
        var sums = new double[2, nFeatures];
        var pairSums = new double[2, nFeatures, nFeatures];
        for (var i = 0; i < nSamples; i++)
        {
            var s = states[i];
            Assert.InRange(s, 0, 1);
            counts[s]++;
            for (var f = 0; f < nFeatures; f++)
                sums[s, f] += x[i, f];
            for (var j = 0; j < nFeatures; j++)
                for (var k = 0; k < nFeatures; k++)
                    pairSums[s, j, k] += x[i, j] * x[i, k];
        }

        for (var s = 0; s < 2; s++)
        {
            // With a 0.95 self-transition probability, both states must be visited thousands of times out of 10000.
            Assert.True(counts[s] >= 1000, $"state {s} was sampled {counts[s]} times, expected at least 1000");
            var n = counts[s];
            for (var f = 0; f < nFeatures; f++)
            {
                var mean = sums[s, f] / n;
                var sigma = Math.Sqrt(full[s, f, f]);
                var tolerance = 6.0 * sigma / Math.Sqrt(n);
                Assert.InRange(mean, Means[s, f] - tolerance, Means[s, f] + tolerance);
            }
            for (var j = 0; j < nFeatures; j++)
                for (var k = 0; k < nFeatures; k++)
                {
                    var sampleCov = pairSums[s, j, k] / n - (sums[s, j] / n) * (sums[s, k] / n);
                    var expected = full[s, j, k];
                    // Standard error of a sample covariance entry of a multivariate normal: sqrt((sigma_jk^2 + sigma_jj sigma_kk) / n).
                    var standardError = Math.Sqrt((expected * expected + full[s, j, j] * full[s, k, k]) / n);
                    Assert.InRange(sampleCov, expected - 6.0 * standardError, expected + 6.0 * standardError);
                }
        }
    }

    [Fact]
    public void the_first_state_of_a_sample_follows_startprob()
    {
        const int draws = 4000;
        var counts = new int[2];
        for (var i = 0; i < draws; i++)
        {
            var model = Model(CovarianceType.Diag);
            var (_, states) = model.Sample(1, random: new Random(777 + i));
            Assert.Single(states);
            counts[states[0]]++;
        }
        for (var s = 0; s < 2; s++)
        {
            var expected = StartProb[s] * draws;
            var standardError = Math.Sqrt(StartProb[s] * (1.0 - StartProb[s]) * draws);
            Assert.InRange(counts[s], expected - 6.0 * standardError, expected + 6.0 * standardError);
        }
    }
}
