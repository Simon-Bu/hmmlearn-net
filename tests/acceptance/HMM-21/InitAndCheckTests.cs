using System;
using System.Linq;
using HmmLearn;
using HmmLearn.Tests.Support;
using Xunit;

namespace HmmLearn.Tests.Acceptance.Hmm21;

/// <summary>
/// Public wrappers around the protected members of <see cref="VariationalGaussianHmm"/> under test (Init, Check,
/// GetNFitScalarsPerParam, ComputeSubnormLogLikelihood, ComputeLowerBound); used by every test class of this ticket.
/// </summary>
internal sealed class ExposedModel : VariationalGaussianHmm
{
    public ExposedModel(int nComponents = 1, CovarianceType covarianceType = CovarianceType.Full, double? startProbPrior = null,
        double? transMatPrior = null, double[,]? meansPrior = null, double[]? betaPrior = null, double[]? dofPrior = null,
        double[,,]? scalePrior = null, DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi, int? randomState = null, int nIter = 100,
        double tol = 1e-6, bool verbose = false, string @params = "stmc", string initParams = "stmc",
        Implementation implementation = Implementation.Log)
        : base(nComponents, covarianceType, startProbPrior, transMatPrior, meansPrior, betaPrior, dofPrior, scalePrior, algorithm,
            randomState, nIter, tol, verbose, @params, initParams, implementation)
    {
    }

    public void RunInit(double[,] x, int[] lengths) => Init(x, lengths);

    public void RunCheck() => Check();

    public IReadOnlyDictionary<char, int>? Scalars() => GetNFitScalarsPerParam();

    public double[,] SubnormLogLikelihood(double[,] x) => ComputeSubnormLogLikelihood(x);

    public double Bound(double currLogProb) => ComputeLowerBound(currLogProb);
}

/// <summary>
/// Init of the four covariance types (defaults and constructor priors), Check (null / wrong-shaped / wrong-length arrays and an
/// undefined covariance type), and NFeatures.
/// </summary>
public class InitAndCheckTests
{
    public static IEnumerable<object[]> AllTypes()
    {
        yield return new object[] { CovarianceType.Full };
        yield return new object[] { CovarianceType.Diag };
        yield return new object[] { CovarianceType.Tied };
        yield return new object[] { CovarianceType.Spherical };
    }

    private static ExposedModel NewModel(CovarianceType type, string initParams = "stmc", int nComponents = 2, int? randomState = 42)
        => new(nComponents, type, randomState: randomState, initParams: initParams);

    /// <summary>Two well-separated 2-feature clusters, 20 samples each, Box-Muller with a fixed seed.</summary>
    private static double[,] TestData(int seed = 42)
    {
        var random = new Random(seed);
        var x = new double[40, 2];
        for (var i = 0; i < 40; i++)
        {
            var (cx, cy) = i < 20 ? (-3.0, 4.0) : (3.0, -4.0);
            x[i, 0] = cx + 0.3 * Normal(random);
            x[i, 1] = cy + 0.3 * Normal(random);
        }
        return x;
    }

    private static double Normal(Random random)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>Sample covariance of the columns of x, dividing by n - 1 (numpy's cov).</summary>
    private static double[,] SampleCovariance(double[,] x)
    {
        var n = x.GetLength(0);
        var f = x.GetLength(1);
        var mean = new double[f];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < f; j++)
                mean[j] += x[i, j] / n;
        var cv = new double[f, f];
        for (var i = 0; i < n; i++)
            for (var a = 0; a < f; a++)
                for (var b = 0; b < f; b++)
                    cv[a, b] += (x[i, a] - mean[a]) * (x[i, b] - mean[b]);
        for (var a = 0; a < f; a++)
            for (var b = 0; b < f; b++)
                cv[a, b] /= n - 1;
        return cv;
    }

    private static ExposedModel Initiated(CovarianceType type)
    {
        var model = NewModel(type);
        model.RunInit(TestData(), new int[] { 40 });
        return model;
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void InitFillsTheDefaults(CovarianceType type)
    {
        var x = TestData();
        var model = NewModel(type);
        model.RunInit(x, new int[] { x.GetLength(0) });

        Assert.Equal(2, model.NFeatures);

        // The k-means centers and cluster sizes seed the posteriors.
        var (centers, labels) = KMeans.Fit(x, model.NComponents);
        var sizes = new double[model.NComponents];
        for (var i = 0; i < labels.Length; i++)
            sizes[labels[i]] += 1.0;

        Close.Equal(centers, model.MeansPosterior, "means_posterior (k-means centers)");
        Close.Equal(sizes, model.BetaPosterior, "beta_posterior (cluster sizes)");

        // Centers are inside the data range of every feature.
        for (var c = 0; c < model.NComponents; c++)
            for (var f = 0; f < 2; f++)
            {
                var min = double.MaxValue;
                var max = double.MinValue;
                for (var i = 0; i < x.GetLength(0); i++)
                {
                    min = Math.Min(min, x[i, f]);
                    max = Math.Max(max, x[i, f]);
                }
                var center = model.MeansPosterior[c, f];
                Assert.True(center >= min && center <= max, $"center {center} outside [{min}, {max}]");
            }

        if (type == CovarianceType.Tied)
        {
            Close.Equal(new double[] { x.GetLength(0) }, model.DofPosterior, "dof_posterior (tied: the number of samples)");
            Close.Equal(new double[] { 2 }, model.DofPriorValues, "dof_prior (tied: n_features)");
        }
        else
        {
            Close.Equal(sizes, model.DofPosterior, "dof_posterior (cluster sizes)");
            Close.Equal(new double[] { 2, 2 }, model.DofPriorValues, "dof_prior (n_features for every state)");
        }

        Close.Equal(new double[] { 1, 1 }, model.BetaPriorValues, "beta_prior (ones)");

        // The means prior is the column mean of the data for every state.
        var columnMean = new double[2];
        for (var i = 0; i < x.GetLength(0); i++)
            for (var f = 0; f < 2; f++)
                columnMean[f] += x[i, f] / x.GetLength(0);
        var meansPrior = new double[model.NComponents, 2];
        for (var c = 0; c < model.NComponents; c++)
            for (var f = 0; f < 2; f++)
                meansPrior[c, f] = columnMean[f];
        Close.Equal(meansPrior, model.MeansPriorValues, "means_prior (column means of the data)");

        // The covars are the sample covariance (dividing by n - 1) plus 1e-3 on the diagonal, distributed to the type.
        var cv = SampleCovariance(x);
        for (var f = 0; f < 2; f++)
            cv[f, f] += 1e-3;
        Close.Equal(InternalUtils.DistributeCovarMatrixToMatchCovarianceType(cv, type, model.NComponents), model.Covars,
            "covars (distributed data covariance)");

        // The scale prior is the 1e-3 default of the type: 1e-3 times the identity for Full and Tied, 1e-3 everywhere for Diag and Spherical.
        var scaleShape = CompactShape(type, model.NComponents, 2);
        var scalePrior = new double[scaleShape[0], scaleShape[1], scaleShape[2]];
        var isIdentityDefault = type is CovarianceType.Full or CovarianceType.Tied;
        for (var i = 0; i < scalePrior.GetLength(0); i++)
            for (var a = 0; a < scalePrior.GetLength(1); a++)
                for (var b = 0; b < scalePrior.GetLength(2); b++)
                    scalePrior[i, a, b] = (isIdentityDefault && a != b) ? 0.0 : 1e-3;
        Close.Equal(scalePrior, model.ScalePriorValues, "scale_prior (1e-3 default of the type)");

        // The scale posterior is the covariance times the dof of the state.
        var covars = model.Covars!;
        var dof = model.DofPosterior!;
        var scalePosterior = type switch
        {
            CovarianceType.Full => FullTimesDof(covars, dof),
            CovarianceType.Diag => DiagTimesDof(covars, dof),
            CovarianceType.Spherical => SphericalTimesDof(covars, dof),
            CovarianceType.Tied => TiedTimesDof(covars, dof),
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };
        Close.Equal(scalePosterior, model.ScalePosterior, "scale_posterior (covars times dof)");

        // Base initialization: strictly positive Dirichlet posteriors with the documented sums.
        var startProbPosterior = model.StartProbPosterior!;
        Assert.Equal(model.NComponents, startProbPosterior.Length);
        Assert.True(startProbPosterior.All(v => v > 0), "startprob_posterior must be strictly positive");
        Close.Equal(1.0, startProbPosterior.Sum(), "startprob_posterior (one sequence: sums to 1)");
        var transMatPosterior = model.TransMatPosterior!;
        Assert.Equal(model.NComponents, transMatPosterior.GetLength(0));
        Assert.Equal(model.NComponents, transMatPosterior.GetLength(1));
        for (var r = 0; r < model.NComponents; r++)
        {
            var row = transMatPosterior[r, 0] + transMatPosterior[r, 1];
            Assert.True(transMatPosterior[r, 0] > 0 && transMatPosterior[r, 1] > 0, "transmat_posterior must be strictly positive");
            Close.Equal(40.0 / model.NComponents, row, $"transmat_posterior row {r} (sums to total length / n_components)");
        }
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void InitKeepsTheConstructorPriors(CovarianceType type)
    {
        var x = TestData();
        var meansPrior = new double[,] { { 1, 2 }, { 3, 4 } };
        var betaPrior = new double[] { 1.5, 2.5 };
        var dofPrior = type == CovarianceType.Tied ? new double[] { 3 } : new double[] { 3, 4 };
        var scalePrior = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 2.0 } : new[] { 2.0, 2.0 });

        var model = new ExposedModel(2, type, meansPrior: meansPrior, betaPrior: betaPrior, dofPrior: dofPrior, scalePrior: scalePrior,
            randomState: 42);
        model.RunInit(x, new int[] { x.GetLength(0) });

        Close.Equal(meansPrior, model.MeansPriorValues, "constructor means_prior kept");
        Close.Equal(betaPrior, model.BetaPriorValues, "constructor beta_prior kept");
        Close.Equal(dofPrior, model.DofPriorValues, "constructor dof_prior kept");
        Close.Equal(scalePrior, model.ScalePriorValues, "constructor scale_prior kept");

        // The posteriors still come from k-means.
        var (centers, labels) = KMeans.Fit(x, model.NComponents);
        var sizes = new double[model.NComponents];
        for (var i = 0; i < labels.Length; i++)
            sizes[labels[i]] += 1.0;
        Close.Equal(centers, model.MeansPosterior, "means_posterior (k-means centers)");
        Close.Equal(sizes, model.BetaPosterior, "beta_posterior (cluster sizes)");
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void InitParamsWithoutMeansOrCovarsLeaveTheExplicitArrays(CovarianceType type)
    {
        var x = TestData();
        var model = NewModel(type, initParams: "st");

        var meansPrior = new double[,] { { 0.5, -0.5 }, { 1.5, 2.5 } };
        var meansPosterior = new double[,] { { 0.7, -0.3 }, { 1.2, 3.1 } };
        var betaPrior = new double[] { 2, 4 };
        var betaPosterior = new double[] { 2.5, 4.5 };
        var dofPrior = type == CovarianceType.Tied ? new double[] { 5 } : new double[] { 5, 6 };
        var dofPosterior = type == CovarianceType.Tied ? new double[] { 9 } : new double[] { 7, 8 };
        var scalePrior = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 2.0 } : new[] { 2.0, 3.0 });
        var scalePosterior = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 1.0 } : new[] { 1.0, 2.0 });
        var covars = Compact(type, 2, 2, type == CovarianceType.Tied ? new[] { 0.5 } : new[] { 0.5, 0.7 });

        model.MeansPriorValues = meansPrior;
        model.MeansPosterior = meansPosterior;
        model.BetaPriorValues = betaPrior;
        model.BetaPosterior = betaPosterior;
        model.DofPriorValues = dofPrior;
        model.DofPosterior = dofPosterior;
        model.ScalePriorValues = scalePrior;
        model.ScalePosterior = scalePosterior;
        model.Covars = covars;

        model.RunInit(x, new int[] { x.GetLength(0) });

        Close.Equal(meansPrior, model.MeansPriorValues, "means_prior kept");
        Close.Equal(meansPosterior, model.MeansPosterior, "means_posterior kept");
        Close.Equal(betaPrior, model.BetaPriorValues, "beta_prior kept");
        Close.Equal(betaPosterior, model.BetaPosterior, "beta_posterior kept");
        Close.Equal(dofPrior, model.DofPriorValues, "dof_prior kept");
        Close.Equal(dofPosterior, model.DofPosterior, "dof_posterior kept");
        Close.Equal(scalePrior, model.ScalePriorValues, "scale_prior kept");
        Close.Equal(scalePosterior, model.ScalePosterior, "scale_posterior kept");
        Close.Equal(covars, model.Covars, "covars kept");
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void TheSameRandomStateReproducesTheInitialization(CovarianceType type)
    {
        var x = TestData();
        var first = NewModel(type, randomState: 7);
        var second = NewModel(type, randomState: 7);
        first.RunInit(x, new int[] { 40 });
        second.RunInit(x, new int[] { 40 });

        Close.Equal(first.StartProbPosterior!, second.StartProbPosterior!, "startprob_posterior with the same random state");
        Close.Equal(first.TransMatPosterior!, second.TransMatPosterior!, "transmat_posterior with the same random state");
        Close.Equal(first.MeansPosterior!, second.MeansPosterior!, "means_posterior with the same random state");
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void CheckPassesAfterInit(CovarianceType type)
    {
        Initiated(type).RunCheck();
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void CheckRejectsNullAndWrongShapedArrays(CovarianceType type)
    {
        // Every array the Check contract names must be non-null and the right shape: each violation, and only violations, throw.
        void ThrowsFor(Action<ExposedModel> mutation)
        {
            var model = Initiated(type);
            mutation(model);
            Assert.Throws<ArgumentException>(() => model.RunCheck());
        }

        ThrowsFor(m => m.MeansPriorValues = null);
        ThrowsFor(m => m.MeansPosterior = null);
        ThrowsFor(m => m.MeansPosterior = new double[2, 3]);
        ThrowsFor(m => m.MeansPriorValues = new double[2, 3]);

        ThrowsFor(m => m.BetaPriorValues = null);
        ThrowsFor(m => m.BetaPosterior = null);
        ThrowsFor(m => m.BetaPosterior = new double[3]);
        ThrowsFor(m => m.BetaPriorValues = new double[3]);

        ThrowsFor(m => m.DofPriorValues = null);
        ThrowsFor(m => m.DofPosterior = null);
        ThrowsFor(m => m.DofPosterior = type == CovarianceType.Tied ? new double[2] : new double[1]);
        ThrowsFor(m => m.DofPriorValues = type == CovarianceType.Tied ? new double[2] : new double[1]);

        ThrowsFor(m => m.ScalePriorValues = null);
        ThrowsFor(m => m.ScalePosterior = WrongScale(type));
        ThrowsFor(m => m.ScalePriorValues = WrongScale(type));

        ThrowsFor(m => m.CovarianceType = (CovarianceType)99);
    }

    /// <summary>A compact scale shape that is wrong for the type (NFeatures is 2): full (2, 1, 1), tied (2, 2, 2), diag (2, 1, 1), spherical (2, 1, 2).</summary>
    private static double[,,] WrongScale(CovarianceType type) => type switch
    {
        CovarianceType.Full => new double[2, 1, 1],
        CovarianceType.Tied => new double[2, 2, 2],
        CovarianceType.Diag => new double[2, 1, 1],
        CovarianceType.Spherical => new double[2, 1, 2],
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    /// <summary>The compact uniform shape of the type for nc states and nf features.</summary>
    private static int[] CompactShape(CovarianceType type, int nc, int nf) => type switch
    {
        CovarianceType.Full => new[] { nc, nf, nf },
        CovarianceType.Tied => new[] { 1, nf, nf },
        CovarianceType.Diag => new[] { nc, 1, nf },
        CovarianceType.Spherical => new[] { nc, 1, 1 },
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

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

    private static double[,,] BuildTied(int nf, double v)
    {
        var result = new double[1, nf, nf];
        for (var f = 0; f < nf; f++)
            result[0, f, f] = v;
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

    private static double[,,] FullTimesDof(double[,,] covars, double[] dof)
    {
        var nc = covars.GetLength(0);
        var nf = covars.GetLength(1);
        var result = new double[nc, nf, nf];
        for (var c = 0; c < nc; c++)
            for (var f = 0; f < nf; f++)
                for (var g = 0; g < nf; g++)
                    result[c, f, g] = covars[c, f, g] * dof[c];
        return result;
    }

    private static double[,,] DiagTimesDof(double[,,] covars, double[] dof)
    {
        var nc = covars.GetLength(0);
        var nf = covars.GetLength(2);
        var result = new double[nc, 1, nf];
        for (var c = 0; c < nc; c++)
            for (var f = 0; f < nf; f++)
                result[c, 0, f] = covars[c, 0, f] * dof[c];
        return result;
    }

    private static double[,,] SphericalTimesDof(double[,,] covars, double[] dof)
    {
        var nc = covars.GetLength(0);
        var result = new double[nc, 1, 1];
        for (var c = 0; c < nc; c++)
            result[c, 0, 0] = covars[c, 0, 0] * dof[c];
        return result;
    }

    private static double[,,] TiedTimesDof(double[,,] covars, double[] dof)
    {
        var nf = covars.GetLength(1);
        var result = new double[1, nf, nf];
        for (var f = 0; f < nf; f++)
            for (var g = 0; g < nf; g++)
                result[0, f, g] = covars[0, f, g] * dof[0];
        return result;
    }
}
