using HmmLearn;
using HmmLearn.Tests.Support;
using Xunit;

namespace HmmLearn.Tests.Acceptance.Hmm22;

/// <summary>
/// A VariationalGaussianHmm that exposes the protected M-step, so the tests can run DoMStep alone on hand-made
/// statistics without a full fit.
/// </summary>
public class ExposedModel : VariationalGaussianHmm
{
    public ExposedModel(int nComponents, CovarianceType covarianceType, string @params)
        : base(nComponents, covarianceType, @params: @params, initParams: "")
    {
    }

    public void RunMStep(SufficientStatistics statistics) => DoMStep(statistics);
}

/// <summary>
/// HMM-22: the M-step alone, run on hand-made sufficient statistics, per covariance type and per letter of the params
/// string. Every expectation is computed in the test from the formulas documented on the (stubbed) DoMStep; no oracle
/// numbers are copied. The updates must produce new arrays (the caller's arrays are never mutated in place), and a
/// degenerate full covariance result must be rejected by the validating Covars setter.
/// </summary>
public class MStepTests
{
    private const int NComponents = 2;
    private const int NFeatures = 2;

    /// <summary>Posterior mass of each state in the hand-made statistics.</summary>
    private static readonly double[] Post = { 2.5, 3.5 };

    /// <summary>Mean of the weighted sample that generated the statistics, per state.</summary>
    private static readonly double[,] DataMeans = { { 1.5, 1.5 }, { 3, 4 } };

    /// <summary>Per-feature variance of the weighted sample (independent features).</summary>
    private const double FeatureVariance = 0.1;

    public static IEnumerable<object[]> CovarianceTypes()
    {
        yield return new object[] { CovarianceType.Spherical };
        yield return new object[] { CovarianceType.Diag };
        yield return new object[] { CovarianceType.Full };
        yield return new object[] { CovarianceType.Tied };
    }

    [Fact]
    public void UpdateOnlyTheMeans()
    {
        var model = new ExposedModel(NComponents, CovarianceType.Full, "m");
        var betaPrior = new double[] { 2, 4 };
        var meansPrior = new double[,] { { 1, 2 }, { 3, 4 } };
        var meansPosterior = new double[,] { { 0.5, 0.25 }, { 1.25, 0.75 } };
        var betaPosterior = new double[] { 1, 1 };
        var (dofPrior, scalePrior, covars) = PriorFor(CovarianceType.Full);
        Apply(model, CovarianceType.Full, meansPrior, betaPrior, meansPosterior, betaPosterior,
            dofPrior, scalePrior, covars, (double[])dofPrior.Clone(), (double[,,])scalePrior.Clone());

        var stats = MakeStatistics(CovarianceType.Full);
        var betaBefore = Clone(betaPosterior);
        var meansBefore = Clone(meansPosterior);
        var betaReference = model.BetaPosterior;
        var meansReference = model.MeansPosterior;
        var dofBefore = Clone(model.DofPosterior);
        var scaleBefore = Clone(model.ScalePosterior);
        var covarsBefore = Clone(model.Covars);
        var startPosteriorBefore = Clone(model.StartProbPosterior);
        var transPosteriorBefore = Clone(model.TransMatPosterior);

        model.RunMStep(stats);

        // The normal posterior of the means: beta_new = beta_prior + post,
        // means_new = (beta_prior * means_prior + obs) / beta_new.
        var expectedBeta = new double[NComponents];
        for (var c = 0; c < NComponents; c++)
            expectedBeta[c] = betaPrior[c] + stats.Post[c];
        Close.Equal(expectedBeta, model.BetaPosterior, "beta_posterior");

        var expectedMeans = new double[NComponents, NFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var f = 0; f < NFeatures; f++)
                expectedMeans[c, f] = (betaPrior[c] * meansPrior[c, f] + stats.Obs[c, f]) / expectedBeta[c];
        Close.Equal(expectedMeans, model.MeansPosterior, "means_posterior");

        // The updated arrays are new objects; the caller's arrays are untouched.
        Assert.NotSame(betaReference, model.BetaPosterior);
        Assert.NotSame(meansReference, model.MeansPosterior);
        Close.Equal(betaBefore, betaPosterior, "caller's beta_posterior");
        Close.Equal(meansBefore, meansPosterior, "caller's means_posterior");

        // Without 'c' the covariance parameters stay as they were.
        Close.Equal(dofBefore, model.DofPosterior, "dof_posterior unchanged");
        Close.Equal(scaleBefore, model.ScalePosterior, "scale_posterior unchanged");
        Close.Equal(covarsBefore, model.Covars, "covars unchanged");
        Close.Equal(startPosteriorBefore, model.StartProbPosterior, "startprob_posterior unchanged");
        Close.Equal(transPosteriorBefore, model.TransMatPosterior, "transmat_posterior unchanged");
    }

    [Fact]
    public void ZeroStatisticsKeepTheMeansAtTheirPriorValues()
    {
        var model = new ExposedModel(NComponents, CovarianceType.Full, "m");
        var betaPrior = new double[] { 2, 4 };
        var meansPrior = new double[,] { { 1.5, 0.5 }, { 3, 0.25 } };
        var meansPosterior = new double[,] { { 9, 9 }, { 9, 9 } };
        var betaPosterior = new double[] { 9, 9 };
        var (dofPrior, scalePrior, covars) = PriorFor(CovarianceType.Full);
        Apply(model, CovarianceType.Full, meansPrior, betaPrior, meansPosterior, betaPosterior,
            dofPrior, scalePrior, covars, (double[])dofPrior.Clone(), (double[,,])scalePrior.Clone());

        var zeroPost = new double[] { 0, 0 };
        var stats = new GaussianStatistics(NComponents, NFeatures, CovarianceType.Full);
        stats.NObs = 0;
        for (var c = 0; c < NComponents; c++)
        {
            stats.Post[c] = zeroPost[c];
            for (var f = 0; f < NFeatures; f++)
            {
                stats.Obs[c, f] = 0;
                stats.Obs2[c, f] = 0;
            }
        }

        model.RunMStep(stats);

        // With zero posterior mass the update (beta * means) / beta reproduces the prior values.
        Close.Equal(betaPrior, model.BetaPosterior, "beta_posterior");
        Close.Equal(meansPrior, model.MeansPosterior, "means_posterior");

        // All values here are dyadic, so (b * m) / b is exact: a few exact checks.
        Assert.Equal(4.0, model.BetaPosterior![1]);
        Assert.Equal(1.5, model.MeansPosterior![0, 0]);
        Assert.Equal(3.0, model.MeansPosterior![1, 0]);
    }

    [Theory]
    [MemberData(nameof(CovarianceTypes))]
    public void UpdateOnlyTheCovariances(CovarianceType type)
    {
        var model = new ExposedModel(NComponents, type, "c");
        var betaPrior = new double[] { 2, 4 };
        var meansPrior = Clone(DataMeans);
        var meansPosterior = Clone(DataMeans);
        var betaPosterior = new double[] { 3, 5 };
        var (dofPrior, scalePrior, covars) = PriorFor(type);
        var dofPosterior = (double[])dofPrior.Clone();
        var scalePosterior = (double[,,])scalePrior.Clone();
        Apply(model, type, meansPrior, betaPrior, meansPosterior, betaPosterior,
            dofPrior, scalePrior, covars, dofPosterior, scalePosterior);

        var stats = MakeStatistics(type);
        var meansBefore = Clone(meansPosterior);
        var betaBefore = Clone(betaPosterior);
        var dofBefore = Clone(dofPosterior);
        var scaleBefore = Clone(scalePosterior);
        var covarsBefore = Clone(covars);
        var dofReference = model.DofPosterior;
        var scaleReference = model.ScalePosterior;

        model.RunMStep(stats);

        // The stored means/beta are used (no 'm' in params), per the formulas documented on DoMStep.
        var (expectedDof, expectedScale, expectedCovars) = ExpectedCovars(type, stats, betaPrior, meansPrior,
            betaPosterior, meansPosterior, dofPrior, scalePrior);
        Close.Equal(expectedDof, model.DofPosterior, "dof_posterior");
        Close.Equal(expectedScale, model.ScalePosterior, "scale_posterior");
        Close.Equal(expectedCovars, model.Covars, "covars");

        // The updated arrays are new objects; the caller's arrays are untouched.
        Assert.NotSame(dofReference, model.DofPosterior);
        Assert.NotSame(scaleReference, model.ScalePosterior);
        Close.Equal(dofBefore, dofPosterior, "caller's dof_posterior");
        Close.Equal(scaleBefore, scalePosterior, "caller's scale_posterior");
        Close.Equal(covarsBefore, covars, "caller's covars");

        // Without 'm' the mean parameters stay as they were.
        Close.Equal(meansBefore, model.MeansPosterior, "means_posterior unchanged");
        Close.Equal(betaBefore, model.BetaPosterior, "beta_posterior unchanged");
    }

    [Theory]
    [MemberData(nameof(CovarianceTypes))]
    public void UpdateMeansAndCovariancesTogether(CovarianceType type)
    {
        var model = new ExposedModel(NComponents, type, "mc");
        var betaPrior = new double[] { 2, 4 };
        var meansPrior = Clone(DataMeans);
        var meansPosterior = new double[,] { { 2, 1 }, { 3, 5 } };
        var betaPosterior = new double[] { 2, 4 };
        var (dofPrior, scalePrior, covars) = PriorFor(type);
        var dofPosterior = (double[])dofPrior.Clone();
        var scalePosterior = (double[,,])scalePrior.Clone();
        Apply(model, type, meansPrior, betaPrior, meansPosterior, betaPosterior,
            dofPrior, scalePrior, covars, dofPosterior, scalePosterior);

        var stats = MakeStatistics(type);
        var meansBefore = Clone(meansPosterior);
        var betaBefore = Clone(betaPosterior);
        var dofBefore = Clone(dofPosterior);
        var scaleBefore = Clone(scalePosterior);
        var covarsBefore = Clone(covars);
        var meansReference = model.MeansPosterior;
        var betaReference = model.BetaPosterior;
        var dofReference = model.DofPosterior;
        var scaleReference = model.ScalePosterior;

        model.RunMStep(stats);

        // 'm' first: beta_new = beta_prior + post, means_new = (beta_prior * means_prior + obs) / beta_new.
        var expectedBeta = new double[NComponents];
        for (var c = 0; c < NComponents; c++)
            expectedBeta[c] = betaPrior[c] + stats.Post[c];
        var expectedMeans = new double[NComponents, NFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var f = 0; f < NFeatures; f++)
                expectedMeans[c, f] = (betaPrior[c] * meansPrior[c, f] + stats.Obs[c, f]) / expectedBeta[c];
        Close.Equal(expectedBeta, model.BetaPosterior, "beta_posterior");
        Close.Equal(expectedMeans, model.MeansPosterior, "means_posterior");

        // 'c' with the just-written means and beta (not the stored ones).
        var (expectedDof, expectedScale, expectedCovars) = ExpectedCovars(type, stats, betaPrior, meansPrior,
            expectedBeta, expectedMeans, dofPrior, scalePrior);
        Close.Equal(expectedDof, model.DofPosterior, "dof_posterior");
        Close.Equal(expectedScale, model.ScalePosterior, "scale_posterior");
        Close.Equal(expectedCovars, model.Covars, "covars");

        // The updated arrays are new objects; the caller's arrays are untouched.
        Assert.NotSame(meansReference, model.MeansPosterior);
        Assert.NotSame(betaReference, model.BetaPosterior);
        Assert.NotSame(dofReference, model.DofPosterior);
        Assert.NotSame(scaleReference, model.ScalePosterior);
        Close.Equal(meansBefore, meansPosterior, "caller's means_posterior");
        Close.Equal(betaBefore, betaPosterior, "caller's beta_posterior");
        Close.Equal(dofBefore, dofPosterior, "caller's dof_posterior");
        Close.Equal(scaleBefore, scalePosterior, "caller's scale_posterior");
        Close.Equal(covarsBefore, covars, "caller's covars");
    }

    [Fact]
    public void UpdateStartTransitionsMeansAndCovariances()
    {
        var model = new ExposedModel(NComponents, CovarianceType.Full, "stmc");
        var betaPrior = new double[] { 2, 4 };
        var meansPrior = Clone(DataMeans);
        var meansPosterior = new double[,] { { 2, 1 }, { 3, 5 } };
        var betaPosterior = new double[] { 2, 4 };
        var (dofPrior, scalePrior, covars) = PriorFor(CovarianceType.Full);
        var dofPosterior = (double[])dofPrior.Clone();
        var scalePosterior = (double[,,])scalePrior.Clone();
        Apply(model, CovarianceType.Full, meansPrior, betaPrior, meansPosterior, betaPosterior,
            dofPrior, scalePrior, covars, dofPosterior, scalePosterior);
        var startProbPosterior = new double[] { 2, 2 };
        var transMatPosterior = new double[,] { { 2, 2 }, { 2, 2 } };
        model.StartProbPriorValues = new double[] { 1, 1 };
        model.StartProbPosterior = startProbPosterior;
        model.TransMatPriorValues = new double[,] { { 1, 1 }, { 1, 1 } };
        model.TransMatPosterior = transMatPosterior;
        model.StartProb = new double[] { 0.5, 0.5 };
        model.TransMat = new double[,] { { 0.5, 0.5 }, { 0.5, 0.5 } };

        var stats = MakeStatistics(CovarianceType.Full);
        stats.Start = new double[] { 0.3, 0.7 };
        stats.Trans = new double[,] { { 1, 2 }, { 3, 4 } };
        var startPosteriorBefore = Clone(startProbPosterior);
        var transPosteriorBefore = Clone(transMatPosterior);
        var startReference = model.StartProbPosterior;
        var transReference = model.TransMatPosterior;

        model.RunMStep(stats);

        // The base M-step: posteriors = priors + statistics; point estimates = normalized posteriors.
        var expectedStartPosterior = new double[NComponents];
        for (var c = 0; c < NComponents; c++)
            expectedStartPosterior[c] = model.StartProbPriorValues![c] + stats.Start[c];
        Close.Equal(expectedStartPosterior, model.StartProbPosterior, "startprob_posterior");
        Assert.NotSame(startReference, model.StartProbPosterior);
        Close.Equal(startPosteriorBefore, startProbPosterior, "caller's startprob_posterior");

        var expectedStartProb = new double[NComponents];
        var startSum = 0.0;
        for (var c = 0; c < NComponents; c++)
            startSum += expectedStartPosterior[c];
        for (var c = 0; c < NComponents; c++)
            expectedStartProb[c] = expectedStartPosterior[c] / startSum;
        Close.Equal(expectedStartProb, model.StartProb, "startprob");

        var expectedTransPosterior = new double[NComponents, NComponents];
        for (var i = 0; i < NComponents; i++)
            for (var j = 0; j < NComponents; j++)
                expectedTransPosterior[i, j] = model.TransMatPriorValues![i, j] + stats.Trans[i, j];
        Close.Equal(expectedTransPosterior, model.TransMatPosterior, "transmat_posterior");
        Assert.NotSame(transReference, model.TransMatPosterior);
        Close.Equal(transPosteriorBefore, transMatPosterior, "caller's transmat_posterior");

        var expectedTransMat = new double[NComponents, NComponents];
        for (var i = 0; i < NComponents; i++)
        {
            var rowSum = 0.0;
            for (var j = 0; j < NComponents; j++)
                rowSum += expectedTransPosterior[i, j];
            for (var j = 0; j < NComponents; j++)
                expectedTransMat[i, j] = expectedTransPosterior[i, j] / rowSum;
        }
        Close.Equal(expectedTransMat, model.TransMat, "transmat");

        // The Gaussian part is the "mc" update (see UpdateMeansAndCovariancesTogether).
        var expectedBeta = new double[NComponents];
        for (var c = 0; c < NComponents; c++)
            expectedBeta[c] = betaPrior[c] + stats.Post[c];
        var expectedMeans = new double[NComponents, NFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var f = 0; f < NFeatures; f++)
                expectedMeans[c, f] = (betaPrior[c] * meansPrior[c, f] + stats.Obs[c, f]) / expectedBeta[c];
        Close.Equal(expectedBeta, model.BetaPosterior, "beta_posterior");
        Close.Equal(expectedMeans, model.MeansPosterior, "means_posterior");
        var (expectedDof, expectedScale, expectedCovars) = ExpectedCovars(CovarianceType.Full, stats, betaPrior,
            meansPrior, expectedBeta, expectedMeans, dofPrior, scalePrior);
        Close.Equal(expectedDof, model.DofPosterior, "dof_posterior");
        Close.Equal(expectedScale, model.ScalePosterior, "scale_posterior");
        Close.Equal(expectedCovars, model.Covars, "covars");
    }

    [Fact]
    public void DegenerateFullCovarianceResultThrows()
    {
        // Stored means far from the data means and a large stored beta make the posterior scale of state 0
        // indefinite (its determinant is negative), so the Covars assignment must be rejected.
        var model = new ExposedModel(NComponents, CovarianceType.Full, "c");
        var betaPrior = new double[] { 2, 4 };
        var meansPrior = new double[,] { { 1, 2 }, { 3, 4 } };
        var meansPosterior = new double[,] { { 0.5, 1.5 }, { 2.5, 3.5 } };
        var betaPosterior = new double[] { 6, 10 };
        var (dofPrior, scalePrior, covars) = PriorFor(CovarianceType.Full);
        Apply(model, CovarianceType.Full, meansPrior, betaPrior, meansPosterior, betaPosterior,
            dofPrior, scalePrior, covars, (double[])dofPrior.Clone(), (double[,,])scalePrior.Clone());

        var stats = MakeStatistics(CovarianceType.Full);
        Assert.Throws<ArgumentException>(() => model.RunMStep(stats));
    }

    /// <summary>
    /// Hand-made statistics of a weighted Gaussian sample: state c has posterior mass Post[c], mean DataMeans[c] and
    /// per-feature variance FeatureVariance (independent features), so Obs[c] = Post[c] * DataMeans[c],
    /// Obs2[c,f] = Post[c] * (DataMeans[c,f]^2 + FeatureVariance) and
    /// ObsObsT[c,i,j] = Post[c] * (DataMeans[c,i] * DataMeans[c,j] + (i == j ? FeatureVariance : 0)).
    /// </summary>
    private static GaussianStatistics MakeStatistics(CovarianceType type)
    {
        var stats = new GaussianStatistics(NComponents, NFeatures, type);
        stats.NObs = 100;
        for (var c = 0; c < NComponents; c++)
        {
            stats.Post[c] = Post[c];
            for (var i = 0; i < NFeatures; i++)
            {
                stats.Obs[c, i] = Post[c] * DataMeans[c, i];
                stats.Obs2[c, i] = Post[c] * (DataMeans[c, i] * DataMeans[c, i] + FeatureVariance);
                if (stats.ObsObsT != null)
                    for (var j = 0; j < NFeatures; j++)
                        stats.ObsObsT[c, i, j] = Post[c] * (DataMeans[c, i] * DataMeans[c, j] + (i == j ? FeatureVariance : 0.0));
            }
        }
        return stats;
    }

    /// <summary>
    /// The prior (and initial posterior) covariance parameters for every type: state 0 ~ I, state 1 ~ 2I (tied: I),
    /// dof 4 per state (8 tied) and initial covars 0.5 * I.
    /// </summary>
    private static (double[] DofPrior, double[,,] ScalePrior, double[,,] Covars) PriorFor(CovarianceType type) => type switch
    {
        CovarianceType.Spherical => (new double[] { 4, 4 },
                                     new double[NComponents, 1, 1] { { { 1 } }, { { 2 } } },
                                     new double[NComponents, 1, 1] { { { 0.5 } }, { { 0.5 } } }),
        CovarianceType.Diag => (new double[] { 4, 4 },
                                new double[NComponents, 1, NFeatures] { { { 1, 1 } }, { { 2, 2 } } },
                                new double[NComponents, 1, NFeatures] { { { 0.5, 0.5 } }, { { 0.5, 0.5 } } }),
        CovarianceType.Tied => (new double[] { 8 },
                                new double[1, NFeatures, NFeatures] { { { 1, 0 }, { 0, 1 } } },
                                new double[1, NFeatures, NFeatures] { { { 0.5, 0 }, { 0, 0.5 } } }),
        _ => (new double[] { 4, 4 },
              new double[NComponents, NFeatures, NFeatures] { { { 1, 0 }, { 0, 1 } }, { { 2, 0 }, { 0, 2 } } },
              new double[NComponents, NFeatures, NFeatures] { { { 0.5, 0 }, { 0, 0.5 } }, { { 0.5, 0 }, { 0, 0.5 } } }),
    };

    /// <summary>
    /// Sets the start/transition and emission parameters (and the validating Covars initial value) of a model.
    /// </summary>
    private static void Apply(ExposedModel model, CovarianceType type,
        double[,] meansPrior, double[] betaPrior,
        double[,] meansPosterior, double[] betaPosterior,
        double[] dofPrior, double[,,] scalePrior, double[,,] covars,
        double[] dofPosterior, double[,,] scalePosterior)
    {
        model.StartProbPriorValues = new double[] { 1, 1 };
        model.StartProbPosterior = new double[] { 2, 2 };
        model.TransMatPriorValues = new double[,] { { 1, 1 }, { 1, 1 } };
        model.TransMatPosterior = new double[,] { { 2, 2 }, { 2, 2 } };
        model.StartProb = new double[] { 0.5, 0.5 };
        model.TransMat = new double[,] { { 0.5, 0.5 }, { 0.5, 0.5 } };
        model.MeansPriorValues = meansPrior;
        model.BetaPriorValues = betaPrior;
        model.MeansPosterior = meansPosterior;
        model.BetaPosterior = betaPosterior;
        model.DofPriorValues = dofPrior;
        model.DofPosterior = dofPosterior;
        model.ScalePriorValues = scalePrior;
        model.ScalePosterior = scalePosterior;
        model.Covars = covars;
    }

    /// <summary>
    /// The expected 'c' update: posterior dof = prior dof + post (summed over states for tied), posterior scale = prior
    /// scale + obs*obs.T (or its diag/spherical reductions) + beta_prior * means_prior^2 - beta * means^2 (the stored
    /// means/beta, or the just-written ones when 'm' was updated too), and covars = scale / dof.
    /// </summary>
    private static (double[] Dof, double[,,] Scale, double[,,] Covars) ExpectedCovars(
        CovarianceType type, GaussianStatistics stats,
        double[] betaPrior, double[,] meansPrior, double[] beta, double[,] means,
        double[] dofPrior, double[,,] scalePrior)
    {
        var scale = type switch
        {
            CovarianceType.Spherical => new double[NComponents, 1, 1],
            CovarianceType.Diag => new double[NComponents, 1, NFeatures],
            CovarianceType.Tied => new double[1, NFeatures, NFeatures],
            _ => new double[NComponents, NFeatures, NFeatures],
        };
        var dof = new double[scale.GetLength(0)];

        if (type == CovarianceType.Tied)
        {
            for (var c = 0; c < dof.Length; c++)
            {
                var mass = 0.0;
                for (var s = 0; s < NComponents; s++)
                {
                    mass += stats.Post[s];
                    for (var i = 0; i < NFeatures; i++)
                        for (var j = 0; j < NFeatures; j++)
                            scale[c, i, j] += stats.ObsObsT![s, i, j]
                                + betaPrior[s] * meansPrior[s, i] * meansPrior[s, j]
                                - beta[s] * means[s, i] * means[s, j];
                }
                for (var i = 0; i < NFeatures; i++)
                    for (var j = 0; j < NFeatures; j++)
                        scale[c, i, j] += scalePrior[c, i, j];
                dof[c] = dofPrior[c] + mass;
            }
        }
        else if (type == CovarianceType.Full)
        {
            // One matrix and one dof per state: the statistics of the state only.
            for (var c = 0; c < NComponents; c++)
            {
                for (var i = 0; i < NFeatures; i++)
                    for (var j = 0; j < NFeatures; j++)
                        scale[c, i, j] = scalePrior[c, i, j] + stats.ObsObsT![c, i, j]
                            + betaPrior[c] * meansPrior[c, i] * meansPrior[c, j]
                            - beta[c] * means[c, i] * means[c, j];
                dof[c] = dofPrior[c] + stats.Post[c];
            }
        }
        else if (type == CovarianceType.Diag)
        {
            for (var c = 0; c < NComponents; c++)
            {
                for (var f = 0; f < NFeatures; f++)
                    scale[c, 0, f] = scalePrior[c, 0, f] + stats.Obs2[c, f]
                        + betaPrior[c] * meansPrior[c, f] * meansPrior[c, f]
                        - beta[c] * means[c, f] * means[c, f];
                dof[c] = dofPrior[c] + stats.Post[c];
            }
        }
        else // Spherical: the mean over the features.
        {
            for (var c = 0; c < NComponents; c++)
            {
                var sum = 0.0;
                for (var f = 0; f < NFeatures; f++)
                    sum += stats.Obs2[c, f]
                        + betaPrior[c] * meansPrior[c, f] * meansPrior[c, f]
                        - beta[c] * means[c, f] * means[c, f];
                scale[c, 0, 0] = scalePrior[c, 0, 0] + sum / NFeatures;
                dof[c] = dofPrior[c] + stats.Post[c];
            }
        }

        var covars = new double[scale.GetLength(0), scale.GetLength(1), scale.GetLength(2)];
        for (var c = 0; c < covars.GetLength(0); c++)
            for (var i = 0; i < covars.GetLength(1); i++)
                for (var j = 0; j < covars.GetLength(2); j++)
                    covars[c, i, j] = scale[c, i, j] / dof[c];
        return (dof, scale, covars);
    }

    private static double[] Clone(double[] array)
    {
        var result = new double[array.Length];
        array.CopyTo(result, 0);
        return result;
    }

    private static double[,] Clone(double[,] array) => (double[,])array.Clone();

    private static double[,,] Clone(double[,,] array) => (double[,,])array.Clone();
}
