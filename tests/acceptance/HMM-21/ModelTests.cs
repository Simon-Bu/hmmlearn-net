using System;
using System.Collections.Generic;
using HmmLearn;
using HmmLearn.Tests.Support;
using Xunit;

namespace HmmLearn.Tests.Acceptance.Hmm21;

/// <summary>
/// GetNFitScalarsPerParam counts (null until the features are known, rejection of an undefined covariance type), score (sum of the state log
/// densities for a one-state model, split over sequences), decode/predict/predict_proba consistency, sampling from explicit parameters, the
/// covars setter, FullCovars and Means.
/// </summary>
public class ModelTests
{
    public static IEnumerable<object[]> AllTypes()
    {
        yield return new object[] { CovarianceType.Full };
        yield return new object[] { CovarianceType.Diag };
        yield return new object[] { CovarianceType.Tied };
        yield return new object[] { CovarianceType.Spherical };
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void GetNFitScalarsPerParamCountsEveryLetter(CovarianceType type)
    {
        var model = new ExposedModel(2, type, randomState: 1);
        model.NFeatures = 2;

        var scalars = model.Scalars();
        Assert.NotNull(scalars);
        Assert.Equal(1, scalars['s']);
        Assert.Equal(2, scalars['t']);
        Assert.Equal(6, scalars['m']);
        var c = type switch
        {
            CovarianceType.Full => 8,
            CovarianceType.Tied => 4,
            CovarianceType.Diag => 6,
            CovarianceType.Spherical => 4,
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };
        Assert.Equal(c, scalars['c']);
    }

    [Fact]
    public void GetNFitScalarsPerParamCountsForThreeStatesAndThreeFeatures()
    {
        var model = new ExposedModel(3, CovarianceType.Full, randomState: 1);
        model.NFeatures = 3;

        var scalars = model.Scalars();
        Assert.NotNull(scalars);
        Assert.Equal(2, scalars['s']);
        Assert.Equal(6, scalars['t']);
        Assert.Equal(12, scalars['m']);
        Assert.Equal(21, scalars['c']);
    }

    [Fact]
    public void GetNFitScalarsPerParamIsNullUntilTheFeaturesAreKnown()
    {
        var model = new ExposedModel(2, CovarianceType.Full, randomState: 1);
        Assert.Null(model.Scalars());
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void GetNFitScalarsPerParamRejectsAnUndefinedCovarianceType(CovarianceType type)
    {
        var model = new ExposedModel(2, type, randomState: 1);
        model.NFeatures = 2;
        model.CovarianceType = (CovarianceType)99;

        Assert.Throws<ArgumentException>(() => model.Scalars());
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void TheScoreOfOneStateModelIsTheSumOfTheStateLogDensities(CovarianceType type)
    {
        var model = OneState(type);
        var x = new double[,] { { 0, 0 }, { 1, 1 }, { 2, -1 }, { -1, 2 }, { 3, 3 }, { 0.5, -0.5 } };

        var expected = Stats.LogMultivariateNormalDensity(x, model.Means!, model.Covars!, type);
        var sum = 0.0;
        for (var i = 0; i < x.GetLength(0); i++)
            sum += expected[i, 0];
        Close.Equal(sum, model.Score(x), "score of a one-state model");
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void TheScoreSplitsOverSequences(CovarianceType type)
    {
        var model = Ready(type);
        var x1 = new double[,] { { 0, 0 }, { 1, 1 }, { -1, 0.5 } };
        var x2 = new double[,] { { 2, 2 }, { -2, -1 }, { 0.5, 1 }, { 1.5, -1.5 } };
        var total = new double[7, 2];
        for (var i = 0; i < 3; i++)
            for (var f = 0; f < 2; f++)
                total[i, f] = x1[i, f];
        for (var i = 0; i < 4; i++)
            for (var f = 0; f < 2; f++)
                total[3 + i, f] = x2[i, f];

        Close.Equal(model.Score(x1, new[] { 3 }) + model.Score(x2, new[] { 4 }), model.Score(total, new[] { 7 }),
            "score of the concatenated sequences");
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void DecodePredictAndProbaAreConsistent(CovarianceType type)
    {
        var model = Ready(type);
        var x = new double[,] { { 0, 0 }, { 1, 1 }, { -1, 0.5 }, { 2, 2 }, { -2, -1 } };
        var lengths = new[] { 2, 3 };

        var (logProb, states) = model.Decode(x, lengths);
        Assert.Equal(5, states.Length);
        Assert.True(double.IsFinite(logProb));
        for (var i = 0; i < states.Length; i++)
            Assert.True(states[i] >= 0 && states[i] < 2, $"state {states[i]} in range");

        // The Viterbi log probability never exceeds the forward log probability.
        Assert.True(logProb <= model.Score(x, lengths) + 1e-9, $"viterbi log probability {logProb} <= forward {model.Score(x, lengths)}");

        var predicted = model.Predict(x, lengths);
        Assert.Equal(states, predicted);

        var proba = model.PredictProba(x, lengths);
        Assert.Equal(5, proba.GetLength(0));
        Assert.Equal(2, proba.GetLength(1));
        for (var i = 0; i < 5; i++)
        {
            var rowSum = 0.0;
            for (var c = 0; c < 2; c++)
            {
                Assert.True(double.IsFinite(proba[i, c]) && proba[i, c] >= 0, $"posterior ({i}, {c}) = {proba[i, c]} in [0, 1)");
                rowSum += proba[i, c];
            }
            Close.Equal(1.0, rowSum, $"posterior row {i} sums to 1");
        }
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void TheSampledDataMatchesThePointEstimate(CovarianceType type)
    {
        // One state: N((1, -2), 0.25 I); wide tolerances - only the distribution must match.
        var model = OneState(type);
        var (x, states) = model.Sample(4000, random: new Random(123), currState: 0);

        Assert.Equal(4000, x.GetLength(0));
        Assert.Equal(2, x.GetLength(1));
        Assert.Equal(4000, states.Length);
        for (var i = 0; i < x.GetLength(0); i++)
        {
            Assert.Equal(0, states[i]);
            for (var f = 0; f < 2; f++)
                Assert.True(double.IsFinite(x[i, f]), $"sample ({i}, {f}) = {x[i, f]} is finite");
        }

        var means = model.Means!;
        for (var f = 0; f < 2; f++)
        {
            var mean = 0.0;
            for (var i = 0; i < 4000; i++)
                mean += x[i, f];
            mean /= 4000;
            Close.Equal(means[0, f], mean, $"sample mean of feature {f}", rtol: 0.1, atol: 0.05);
        }

        var variances = new double[2];
        var covariance = 0.0;
        var mean0 = 0.0;
        var mean1 = 0.0;
        for (var i = 0; i < 4000; i++)
        {
            mean0 += x[i, 0];
            mean1 += x[i, 1];
        }
        mean0 /= 4000;
        mean1 /= 4000;
        for (var i = 0; i < 4000; i++)
        {
            var d0 = x[i, 0] - mean0;
            var d1 = x[i, 1] - mean1;
            variances[0] += d0 * d0;
            variances[1] += d1 * d1;
            covariance += d0 * d1;
        }
        variances[0] /= 4000;
        variances[1] /= 4000;
        covariance /= 4000;
        Close.Equal(0.25, variances[0], "sample variance of feature 0", rtol: 0.2, atol: 0.02);
        Close.Equal(0.25, variances[1], "sample variance of feature 1", rtol: 0.2, atol: 0.02);
        Assert.True(Math.Abs(covariance) < 0.05, $"sample covariance {covariance} is near 0");
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void TheCovarsSetterValidatesAndCopies(CovarianceType type)
    {
        var model = new ExposedModel(1, type, randomState: 1);
        var valid = Compact(type, 1, 2, 0.5);
        var before = Clone(valid);

        model.Covars = valid;
        Close.Equal(before, model.Covars, "covars after setting");

        valid[0, 0, 0] = 99.0;
        Close.Equal(before, model.Covars, "the setter stores a copy");

        Assert.Throws<ArgumentException>(() => model.Covars = WrongShape(type));
        Assert.Throws<ArgumentException>(() => model.Covars = NotPositiveDefinite(type));
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void FullCovarsExpandsEveryType(CovarianceType type)
    {
        var model = new ExposedModel(2, type, randomState: 1);
        var covars = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 1.5 } : new[] { 0.5, 2.0 });
        model.Covars = covars;
        model.NFeatures = 2;

        var full = model.FullCovars;
        Assert.Equal(2, full.GetLength(0));
        Assert.Equal(2, full.GetLength(1));
        Assert.Equal(2, full.GetLength(2));
        Close.Equal(Utils.FillCovars(covars, type, 2, 2), full, "full covars of the " + type + " type");

        var withoutCovars = new ExposedModel(1, CovarianceType.Full, randomState: 1);
        Assert.Throws<InvalidOperationException>(() => _ = withoutCovars.FullCovars);
    }

    [Fact]
    public void MeansIsTheMeansPosterior()
    {
        var model = new ExposedModel(2, CovarianceType.Full, randomState: 1);
        Assert.Null(model.Means);

        var means = new double[,] { { 0.5, -1 }, { 2, 0.25 } };
        model.MeansPosterior = means;
        Close.Equal(means, model.Means, "means (the posterior mean)");
    }

    /// <summary>One state with every array set: mean (1, -2), covariances 0.25 I.</summary>
    private static ExposedModel OneState(CovarianceType type)
    {
        var model = new ExposedModel(1, type, randomState: 1, initParams: "");
        model.NFeatures = 2;
        model.StartProbPriorValues = new double[] { 1 };
        model.StartProbPosterior = new double[] { 5 };
        model.TransMatPriorValues = new double[,] { { 1 } };
        model.TransMatPosterior = new double[,] { { 5 } };
        model.MeansPriorValues = new double[,] { { 0, 0 } };
        model.MeansPosterior = new double[,] { { 1, -2 } };
        model.BetaPriorValues = new double[] { 1 };
        model.BetaPosterior = new double[] { 2 };
        model.DofPriorValues = new double[] { 3 };
        model.DofPosterior = new double[] { 4 };
        model.ScalePriorValues = Compact(type, 1, 2, 1.0);
        model.ScalePosterior = Compact(type, 1, 2, 2.0);
        model.Covars = Compact(type, 1, 2, 0.25);
        model.StartProb = new double[] { 1 };
        model.TransMat = new double[,] { { 1 } };
        return model;
    }

    /// <summary>Two states with every array set, plus point estimates for the forward pass.</summary>
    private static ExposedModel Ready(CovarianceType type)
    {
        var model = new ExposedModel(2, type, randomState: 1, initParams: "");
        model.NFeatures = 2;
        model.StartProbPriorValues = new double[] { 1, 1 };
        model.StartProbPosterior = new double[] { 2, 3 };
        model.TransMatPriorValues = new double[,] { { 1, 1 }, { 1, 1 } };
        model.TransMatPosterior = new double[,] { { 2, 3 }, { 3, 2 } };
        model.MeansPriorValues = new double[,] { { 0, 1 }, { 2, -1 } };
        model.MeansPosterior = new double[,] { { 1, -0.5 }, { -1, 2 } };
        model.BetaPriorValues = new double[] { 1, 2 };
        model.BetaPosterior = new double[] { 2, 3 };
        model.DofPriorValues = type == CovarianceType.Tied ? new double[] { 4 } : new double[] { 3, 5 };
        model.DofPosterior = type == CovarianceType.Tied ? new double[] { 5 } : new double[] { 4, 6 };
        model.ScalePriorValues = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 2.0 } : new[] { 2.0, 3.0 });
        model.ScalePosterior = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 3.0 } : new[] { 3.0, 4.0 });
        model.Covars = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 0.5 } : new[] { 0.5, 0.7 });
        model.StartProb = new double[] { 0.4, 0.6 };
        model.TransMat = new double[,] { { 0.3, 0.7 }, { 0.6, 0.4 } };
        return model;
    }

    /// <summary>A shape that violates the documented compact shapes: full must be (nc, nf, nf), tied (1, nf, nf), diag (nc, 1, nf), spherical (nc, 1, 1).</summary>
    private static double[,,] WrongShape(CovarianceType type) => type switch
    {
        CovarianceType.Full => new double[1, 2, 3],
        CovarianceType.Tied => new double[2, 2, 2],
        CovarianceType.Diag => new double[1, 2, 2],
        CovarianceType.Spherical => new double[1, 1, 2],
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    /// <summary>A compact covariance with a negative variance (or a non positive-definite matrix).</summary>
    private static double[,,] NotPositiveDefinite(CovarianceType type)
    {
        switch (type)
        {
            case CovarianceType.Full:
            case CovarianceType.Tied:
                // Eigenvalues 3 and -1: symmetric but not positive definite.
                var result = new double[1, 2, 2];
                result[0, 0, 0] = 1;
                result[0, 0, 1] = 2;
                result[0, 1, 0] = 2;
                result[0, 1, 1] = 1;
                return result;
            case CovarianceType.Diag:
                var diag = new double[1, 1, 2];
                diag[0, 0, 0] = 0.5;
                diag[0, 0, 1] = -0.5;
                return diag;
            case CovarianceType.Spherical:
                var spherical = new double[1, 1, 1];
                spherical[0, 0, 0] = -0.5;
                return spherical;
            default:
                throw new ArgumentOutOfRangeException(nameof(type));
        }
    }

    private static double[,,] Clone(double[,,] array)
    {
        var result = new double[array.GetLength(0), array.GetLength(1), array.GetLength(2)];
        array.CopyTo(result, 0);
        return result;
    }

    /// <summary>An isotropic compact covariance: perState[c] on the diagonal (or the single value), 0 off the diagonal.</summary>
    private static double[,,] Compact(CovarianceType type, int nc, int nf, double[] perState) => type switch
    {
        CovarianceType.Full => BuildFull(nc, nf, perState),
        CovarianceType.Tied => BuildTied(nf, perState[0]),
        CovarianceType.Diag => BuildDiag(nc, nf, perState),
        CovarianceType.Spherical => BuildSpherical(nc, perState),
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    private static double[,,] Compact(CovarianceType type, int nc, int nf, double value) =>
        Compact(type, nc, nf, PerState(nc, value));

    private static double[] PerState(int nc, double value)
    {
        var result = new double[nc];
        for (var i = 0; i < nc; i++)
            result[i] = value;
        return result;
    }

    private static double[,,] BuildFull(int nc, int nf, double[] perState)
    {
        var result = new double[nc, nf, nf];
        for (var c = 0; c < nc; c++)
            for (var f = 0; f < nf; f++)
                result[c, f, f] = perState[c];
        return result;
    }

    private static double[,,] BuildTied(int nf, double value)
    {
        var result = new double[1, nf, nf];
        for (var f = 0; f < nf; f++)
            result[0, f, f] = value;
        return result;
    }

    private static double[,,] BuildDiag(int nc, int nf, double[] perState)
    {
        var result = new double[nc, 1, nf];
        for (var c = 0; c < nc; c++)
            for (var f = 0; f < nf; f++)
                result[c, 0, f] = perState[c];
        return result;
    }

    private static double[,,] BuildSpherical(int nc, double[] perState)
    {
        var result = new double[nc, 1, 1];
        for (var c = 0; c < nc; c++)
            result[c, 0, 0] = perState[c];
        return result;
    }
}
