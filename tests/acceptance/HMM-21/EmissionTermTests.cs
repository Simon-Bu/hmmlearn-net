using System;
using System.Collections.Generic;
using HmmLearn;
using HmmLearn.Tests.Support;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Xunit;

namespace HmmLearn.Tests.Acceptance.Hmm21;

/// <summary>
/// The subnormalized emission log likelihood (shape, closed form, invariance under the covariance parameterization, monotonicity in
/// the distance to the posterior mean) and the variational lower bound (the base bound minus the per-state mean and Wishart KL terms;
/// the base bound itself when the posterior equals the prior).
/// </summary>
public class EmissionTermTests
{
    public static IEnumerable<object[]> AllTypes()
    {
        yield return new object[] { CovarianceType.Full };
        yield return new object[] { CovarianceType.Diag };
        yield return new object[] { CovarianceType.Tied };
        yield return new object[] { CovarianceType.Spherical };
    }

    /// <summary>A model with every array set explicitly (2 states, 2 features) plus the arrays it was built with.</summary>
    private sealed class ReadyModel
    {
        public ExposedModel Model;
        public double[] StartProbPrior;
        public double[] StartProbPosterior;
        public double[,] TransMatPrior;
        public double[,] TransMatPosterior;
        public double[,] MeansPrior;
        public double[,] MeansPosterior;
        public double[] BetaPrior;
        public double[] BetaPosterior;
        public double[] DofPrior;
        public double[] DofPosterior;
        public double[,,] ScalePrior;
        public double[,,] ScalePosterior;
    }

    private static ReadyModel Ready(CovarianceType type)
    {
        var model = new ExposedModel(2, type, randomState: 1, initParams: "");
        model.NFeatures = 2;

        var startProbPrior = new double[] { 1, 1 };
        var startProbPosterior = new double[] { 2, 3 };
        var transMatPrior = new double[,] { { 1, 1 }, { 1, 1 } };
        var transMatPosterior = new double[,] { { 2, 2 }, { 2, 2 } };
        var meansPrior = new double[,] { { 0, 1 }, { 2, -1 } };
        var meansPosterior = new double[,] { { 1, -0.5 }, { -1, 2 } };
        var betaPrior = new double[] { 1, 2 };
        var betaPosterior = new double[] { 2, 3 };
        var dofPrior = type == CovarianceType.Tied ? new double[] { 4 } : new double[] { 3, 5 };
        var dofPosterior = type == CovarianceType.Tied ? new double[] { 5 } : new double[] { 4, 6 };
        var scalePrior = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 2.0 } : new[] { 2.0, 3.0 });
        var scalePosterior = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 3.0 } : new[] { 3.0, 4.0 });

        model.StartProbPriorValues = startProbPrior;
        model.StartProbPosterior = startProbPosterior;
        model.TransMatPriorValues = transMatPrior;
        model.TransMatPosterior = transMatPosterior;
        model.MeansPriorValues = meansPrior;
        model.MeansPosterior = meansPosterior;
        model.BetaPriorValues = betaPrior;
        model.BetaPosterior = betaPosterior;
        model.DofPriorValues = dofPrior;
        model.DofPosterior = dofPosterior;
        model.ScalePriorValues = scalePrior;
        model.ScalePosterior = scalePosterior;
        model.Covars = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 0.5 } : new[] { 0.5, 0.7 });

        return new ReadyModel
        {
            Model = model,
            StartProbPrior = startProbPrior,
            StartProbPosterior = startProbPosterior,
            TransMatPrior = transMatPrior,
            TransMatPosterior = transMatPosterior,
            MeansPrior = meansPrior,
            MeansPosterior = meansPosterior,
            BetaPrior = betaPrior,
            BetaPosterior = betaPosterior,
            DofPrior = dofPrior,
            DofPosterior = dofPosterior,
            ScalePrior = scalePrior,
            ScalePosterior = scalePosterior,
        };
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void SubnormLogLikelihoodHasTheDocumentedShapeAndFiniteValues(CovarianceType type)
    {
        var x = new double[,] { { 0, 0 }, { 1, -1 }, { -2, 0.5 }, { 0.5, 0.5 }, { 3, 2 } };
        var result = Ready(type).Model.SubnormLogLikelihood(x);

        Assert.Equal(5, result.GetLength(0));
        Assert.Equal(2, result.GetLength(1));
        for (var i = 0; i < 5; i++)
            for (var c = 0; c < 2; c++)
                Assert.True(double.IsFinite(result[i, c]), $"log likelihood ({i}, {c}) = {result[i, c]} is finite");
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void OneFeatureMatchesTheClosedForm(CovarianceType type)
    {
        // One feature: the documented formula becomes an elementary expression with W = 1 / scale.
        var nc = 2;
        var meansPosterior = new double[,] { { 0.5 }, { -1.2 } };
        var betaPosterior = new double[] { 2, 3 };
        var dofPosterior = type == CovarianceType.Tied ? new double[] { 6 } : new double[] { 5, 7 };
        var scalePosterior = Compact(type, nc, 1, type == CovarianceType.Tied ? new[] { 1.8 } : new[] { 1.5, 2.25 });

        var model = new ExposedModel(nc, type, randomState: 1, initParams: "");
        model.NFeatures = 1;
        model.MeansPosterior = meansPosterior;
        model.BetaPosterior = betaPosterior;
        model.DofPosterior = dofPosterior;
        model.ScalePosterior = scalePosterior;
        model.Covars = Compact(type, nc, 1, type == CovarianceType.Tied ? new[] { 0.5 } : new[] { 0.5, 0.7 });

        var x = new double[,] { { 0.75 }, { -2.0 }, { 0.0 } };
        var result = model.SubnormLogLikelihood(x);
        Assert.Equal(3, result.GetLength(0));
        Assert.Equal(nc, result.GetLength(1));

        for (var c = 0; c < nc; c++)
        {
            var scaleValue = type == CovarianceType.Tied ? scalePosterior[0, 0, 0] : scalePosterior[c, 0, 0];
            var dofValue = type == CovarianceType.Tied ? dofPosterior[0] : dofPosterior[c];
            var precision = 1.0 / scaleValue;
            for (var i = 0; i < 3; i++)
            {
                var term1 = (SpecialFunctions.DiGamma(dofValue / 2.0) + Math.Log(2.0) + Math.Log(precision)) / 2.0;
                var delta = x[i, 0] - meansPosterior[c, 0];
                var dots = dofValue * delta * delta * precision;
                var term3 = 1.0 / betaPosterior[c];
                Close.Equal(term1 - (dots + term3) / 2.0, result[i, c], $"closed form at ({i}, {c})");
            }
        }
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void TheSameIsotropicModelGivesTheSameValuesInEveryParameterization(CovarianceType type)
    {
        // 0.7 I for every state (5 dof, one number for Tied): full, diag, spherical and tied are the same model, so the subnormalized
        // log likelihoods must coincide.
        var x = new double[,] { { 0, 0 }, { 1, -1 }, { -2, 0.5 }, { 0.5, 0.5 } };
        var reference = IsotropicModel(CovarianceType.Full).Model.SubnormLogLikelihood(x);
        var actual = IsotropicModel(type).Model.SubnormLogLikelihood(x);
        Close.Equal(reference, actual, "isotropic model across parameterizations");
    }

    private static ReadyModel IsotropicModel(CovarianceType type)
    {
        var model = new ExposedModel(2, type, randomState: 1, initParams: "");
        model.NFeatures = 2;
        model.MeansPosterior = new double[,] { { 1, -0.5 }, { -2, 0.25 } };
        model.BetaPosterior = new double[] { 1.5, 2.5 };
        model.DofPosterior = type == CovarianceType.Tied ? new double[] { 5 } : new double[] { 5, 5 };
        model.ScalePosterior = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 0.7 } : new[] { 0.7, 0.7 });
        model.Covars = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 0.5 } : new[] { 0.5, 0.7 });
        return new ReadyModel { Model = model };
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void TheSubnormLogLikelihoodIncreasesTowardThePosteriorMean(CovarianceType type)
    {
        var ready = Ready(type);
        var far = new double[] { 3, -2 };
        for (var c = 0; c < 2; c++)
        {
            var near = new double[2];
            for (var f = 0; f < 2; f++)
                near[f] = 0.5 * (far[f] + ready.MeansPosterior[c, f]);
            var x = new double[2, 2];
            for (var f = 0; f < 2; f++)
            {
                x[0, f] = far[f];
                x[1, f] = near[f];
            }
            var result = ready.Model.SubnormLogLikelihood(x);
            Assert.True(result[1, c] > result[0, c],
                $"state {c}: approaching the posterior mean must increase the log likelihood ({result[0, c]} -> {result[1, c]})");
        }
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void TheLowerBoundIsTheBaseBoundMinusTheEmissionKlTerms(CovarianceType type)
    {
        var ready = Ready(type);
        var model = ready.Model;
        const double logProb = -12.5;

        var expected = logProb - KlDivergence.KlDirichlet(ready.StartProbPosterior, ready.StartProbPrior);
        for (var i = 0; i < 2; i++)
            expected -= KlDivergence.KlDirichlet(Row(ready.TransMatPosterior, i), Row(ready.TransMatPrior, i));

        var scalePosteriorFull = Utils.FillCovars(ready.ScalePosterior, type, 2, 2);
        var scalePriorFull = Utils.FillCovars(ready.ScalePrior, type, 2, 2);
        for (var c = 0; c < 2; c++)
        {
            var dofPosterior = type == CovarianceType.Tied ? ready.DofPosterior[0] : ready.DofPosterior[c];
            var dofPrior = type == CovarianceType.Tied ? ready.DofPrior[0] : ready.DofPrior[c];
            var precision = Scale(Inverse(StateMatrix(scalePosteriorFull, c)), dofPosterior);
            var covarianceQ = Inverse(Scale(precision, ready.BetaPosterior[c]));
            var covarianceP = Inverse(Scale(precision, ready.BetaPrior[c]));
            expected -= KlDivergence.KlMultivariateNormalDistribution(Row(ready.MeansPosterior, c), covarianceQ,
                Row(ready.MeansPrior, c), covarianceP);
            if (type != CovarianceType.Tied || c == 0)
                expected -= KlDivergence.KlWishartDistribution(dofPosterior, StateMatrix(scalePosteriorFull, c), dofPrior, StateMatrix(scalePriorFull, c));
        }

        var bound = model.Bound(logProb);
        Close.Equal(expected, bound, "lower bound (base bound minus the emission KL terms)");

        // With different priors and posteriors every KL term is positive, so the bound sits below the base bound and the log probability.
        Assert.True(bound < logProb, $"bound {bound} is below the log probability {logProb}");
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void PosteriorEqualPriorLeavesTheBaseBound(CovarianceType type)
    {
        var model = new ExposedModel(2, type, randomState: 1, initParams: "");
        model.NFeatures = 2;
        model.StartProbPriorValues = new double[] { 2, 3 };
        model.StartProbPosterior = new double[] { 2, 3 };
        model.TransMatPriorValues = new double[,] { { 2, 2 }, { 2, 2 } };
        model.TransMatPosterior = new double[,] { { 2, 2 }, { 2, 2 } };
        model.MeansPriorValues = new double[,] { { 1, -0.5 }, { -1, 2 } };
        model.MeansPosterior = new double[,] { { 1, -0.5 }, { -1, 2 } };
        model.BetaPriorValues = new double[] { 2, 3 };
        model.BetaPosterior = new double[] { 2, 3 };
        model.DofPriorValues = type == CovarianceType.Tied ? new double[] { 5 } : new double[] { 4, 6 };
        model.DofPosterior = type == CovarianceType.Tied ? new double[] { 5 } : new double[] { 4, 6 };
        model.ScalePriorValues = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 3.0 } : new[] { 3.0, 4.0 });
        model.ScalePosterior = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 3.0 } : new[] { 3.0, 4.0 });

        // When the posterior equals the prior, every KL term vanishes and the bound is the log probability itself.
        Close.Equal(-7.25, model.Bound(-7.25), "bound with posterior equal to the prior");
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

    private static double[] Row(double[,] matrix, int i)
    {
        var result = new double[matrix.GetLength(1)];
        for (var j = 0; j < result.Length; j++)
            result[j] = matrix[i, j];
        return result;
    }

    private static double[,] StateMatrix(double[,,] tensor, int c)
    {
        var nf = tensor.GetLength(1);
        var result = new double[nf, nf];
        for (var f = 0; f < nf; f++)
            for (var g = 0; g < nf; g++)
                result[f, g] = tensor[c, f, g];
        return result;
    }

    private static double[,] Inverse(double[,] matrix) => Matrix<double>.Build.DenseOfArray(matrix).Inverse().ToArray();

    private static double[,] Scale(double[,] matrix, double value)
    {
        var result = new double[matrix.GetLength(0), matrix.GetLength(1)];
        for (var i = 0; i < matrix.GetLength(0); i++)
            for (var j = 0; j < matrix.GetLength(1); j++)
                result[i, j] = matrix[i, j] * value;
        return result;
    }
}
