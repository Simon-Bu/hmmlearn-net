using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm13;

/// <summary>
/// Criterion C2 of ticket HMM-13: parameter validation of <see cref="GaussianHmm"/> — the Covars setter (shape per covariance
/// type, positive variances, symmetry and positive definiteness, copy semantics), Check (missing or mis-shaped Means or Covars,
/// a feature count mismatch, an undefined covariance type), and FullCovars (shape, expansion for the four covariance types
/// including spherical with several features, and the InvalidOperationException preconditions). All inputs are small hand-made
/// arrays; every expected value is computed in the test from its own inputs.
/// </summary>
public class ValidationTests
{
    /// <summary>Exposes the protected <see cref="GaussianHmm.Check"/>.</summary>
    private sealed class ExposedModel : GaussianHmm
    {
        public ExposedModel(int nComponents, CovarianceType covarianceType)
            : base(nComponents, covarianceType, 1e-3, 1.0, 1.0, 0, 0, 1e-2, 1, DecoderAlgorithm.Viterbi, null, 10, 1e-2,
                false, "stmc", "", Implementation.Log)
        {
        }

        public void RunCheck() => Check();
    }

    private static GaussianHmm Model(CovarianceType type, int nComponents = 3) =>
        new(nComponents, type, 1e-3, 1.0, 1.0, 0, 0, 1e-2, 1, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "stmc", "",
            Implementation.Log);

    private static void ValidStartAndTrans(GaussianHmm model, int nComponents = 2)
    {
        var startProb = new double[nComponents];
        for (var i = 0; i < nComponents; i++)
            startProb[i] = 1.0 / nComponents;
        var transMat = new double[nComponents, nComponents];
        for (var i = 0; i < nComponents; i++)
            for (var j = 0; j < nComponents; j++)
                transMat[i, j] = 1.0 / nComponents;
        model.StartProb = startProb;
        model.TransMat = transMat;
    }

    private static double[,,] Fill(int d0, int d1, int d2, double value = 1.0)
    {
        var tensor = new double[d0, d1, d2];
        for (var i = 0; i < d0; i++)
            for (var j = 0; j < d1; j++)
                for (var k = 0; k < d2; k++)
                    tensor[i, j, k] = value;
        return tensor;
    }

    [Theory]
    [InlineData(CovarianceType.Spherical, 2, 1, 1)]
    [InlineData(CovarianceType.Spherical, 3, 1, 2)]
    [InlineData(CovarianceType.Spherical, 3, 2, 1)]
    [InlineData(CovarianceType.Diag, 2, 1, 2)]
    [InlineData(CovarianceType.Diag, 3, 2, 2)]
    [InlineData(CovarianceType.Full, 1, 2, 2)]
    [InlineData(CovarianceType.Full, 2, 2, 2)]
    [InlineData(CovarianceType.Full, 3, 2, 3)]
    [InlineData(CovarianceType.Tied, 2, 2, 2)]
    [InlineData(CovarianceType.Tied, 1, 2, 3)]
    public void the_covars_setter_rejects_the_wrong_shape_for_the_covariance_type(CovarianceType type, int d0, int d1, int d2)
    {
        var model = Model(type);
        Assert.Throws<ArgumentException>(() => model.Covars = Fill(d0, d1, d2));
    }

    [Theory]
    [InlineData(CovarianceType.Spherical, 0.0)]
    [InlineData(CovarianceType.Spherical, -1.0)]
    [InlineData(CovarianceType.Diag, 0.0)]
    [InlineData(CovarianceType.Diag, -1.0)]
    public void the_covars_setter_rejects_non_positive_variances(CovarianceType type, double badVariance)
    {
        var model = Model(type, 2);
        var covars = type == CovarianceType.Spherical
            ? new double[2, 1, 1] { { { 2.0 } }, { { badVariance } } }
            : new double[2, 1, 2] { { { 1.0, badVariance } }, { { 1.0, 1.0 } } };
        Assert.Throws<ArgumentException>(() => model.Covars = covars);
    }

    [Theory]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void the_covars_setter_rejects_non_symmetric_full_and_tied_covars(CovarianceType type)
    {
        var model = Model(type, type == CovarianceType.Full ? 2 : 1);
        var covars = type == CovarianceType.Full
            ? new double[2, 2, 2]
            {
                { { 2.0, 0.0 }, { 0.0, 1.0 } },
                { { 1.0, 0.5 }, { 0.2, 1.0 } },
            }
            : new double[1, 2, 2] { { { 1.0, 0.5 }, { 0.2, 1.0 } } };
        Assert.Throws<ArgumentException>(() => model.Covars = covars);
    }

    [Theory]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void the_covars_setter_rejects_symmetric_but_not_positive_definite_covars(CovarianceType type)
    {
        var model = Model(type, type == CovarianceType.Full ? 2 : 1);
        // The bad component has eigenvalues 3 and -1: symmetric, indefinite.
        var covars = type == CovarianceType.Full
            ? new double[2, 2, 2]
            {
                { { 2.0, 0.0 }, { 0.0, 1.0 } },
                { { 1.0, 2.0 }, { 2.0, 1.0 } },
            }
            : new double[1, 2, 2] { { { 1.0, 2.0 }, { 2.0, 1.0 } } };
        Assert.Throws<ArgumentException>(() => model.Covars = covars);
    }

    [Fact]
    public void the_covars_setter_rejects_a_positive_semidefinite_tied_covar()
    {
        var model = Model(CovarianceType.Tied, 1);
        // Eigenvalues 1 and 0: PSD, not PD.
        var covars = new double[1, 2, 2] { { { 0.5, 0.5 }, { 0.5, 0.5 } } };
        Assert.Throws<ArgumentException>(() => model.Covars = covars);
    }

    [Fact]
    public void the_covars_setter_stores_a_copy_and_the_getter_returns_a_copy()
    {
        var model = Model(CovarianceType.Diag, 2);
        var input = new double[2, 1, 2] { { { 1.0, 2.0 } }, { { 3.0, 4.0 } } };
        var original = new double[2, 1, 2] { { { 1.0, 2.0 } }, { { 3.0, 4.0 } } };

        model.Covars = input;
        input[0, 0, 0] = 999.0;
        Close.Equal(original, model.Covars, "the setter must store a copy of the given tensor");

        var got = model.Covars;
        Close.Equal(original, got, "the getter must return the stored values");
        got[1, 0, 1] = 999.0;
        Close.Equal(original, model.Covars, "the getter must return a copy, not the stored tensor");
    }

    [Theory]
    [InlineData(0)] // Means null (given Covars are valid)
    [InlineData(1)] // Means with one row instead of two
    [InlineData(2)] // Means with zero columns
    public void check_rejects_missing_or_misshaped_means(int meansCase)
    {
        var model = new ExposedModel(2, CovarianceType.Diag);
        ValidStartAndTrans(model);
        model.Covars = new double[2, 1, 2] { { { 1.0, 2.0 } }, { { 3.0, 4.0 } } };
        if (meansCase == 0)
        {
        }
        else if (meansCase == 1)
            model.Means = new double[1, 2] { { 1.0, 2.0 } };
        else
            model.Means = new double[2, 0];
        Assert.Throws<ArgumentException>(() => model.RunCheck());
    }

    [Theory]
    [InlineData(0)] // Covars null (given Means are valid)
    [InlineData(1)] // diag covars (2, 1, 3): valid shape for the setter, but 3 features vs 2 in Means
    [InlineData(2)] // full covars (2, 2, 2) vs Means (2, 3): 2 features vs 3 in Means
    public void check_rejects_missing_covars_or_a_feature_count_mismatch(int covarsCase)
    {
        var model = new ExposedModel(2, covarsCase == 2 ? CovarianceType.Full : CovarianceType.Diag);
        ValidStartAndTrans(model);
        model.Means = covarsCase == 2 ? new double[2, 3] { { 0.0, 1.0, 2.0 }, { 3.0, 4.0, 5.0 } }
                                      : new double[2, 2] { { 1.0, 2.0 }, { 3.0, 4.0 } };
        if (covarsCase == 1)
            model.Covars = new double[2, 1, 3] { { { 1.0, 1.0, 1.0 } }, { { 2.0, 2.0, 2.0 } } };
        else if (covarsCase == 2)
            model.Covars = new double[2, 2, 2] { { { 2.0, 0.0 }, { 0.0, 1.0 } }, { { 1.5, 0.0 }, { 0.0, 1.0 } } };
        Assert.Throws<ArgumentException>(() => model.RunCheck());
    }

    [Fact]
    public void check_rejects_an_undefined_covariance_type()
    {
        var model = new ExposedModel(2, (CovarianceType)99);
        ValidStartAndTrans(model);
        model.Means = new double[2, 2] { { 1.0, 2.0 }, { 3.0, 4.0 } };
        model.Covars = new double[2, 1, 1] { { { 1.0 } }, { { 2.0 } } };
        Assert.Throws<ArgumentException>(() => model.RunCheck());
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void full_covars_expands_the_compact_covars(CovarianceType type)
    {
        const int nComponents = 2;
        const int nFeatures = 3;

        double[,,] covars;
        double[,,] expected = new double[nComponents, nFeatures, nFeatures];
        switch (type)
        {
            case CovarianceType.Spherical:
            {
                var variances = new[] { 1.5, 2.5 };
                covars = new double[nComponents, 1, 1] { { { variances[0] } }, { { variances[1] } } };
                for (var c = 0; c < nComponents; c++)
                    for (var f = 0; f < nFeatures; f++)
                        expected[c, f, f] = variances[c];
                break;
            }
            case CovarianceType.Diag:
            {
                covars = new double[nComponents, 1, nFeatures] { { { 1.5, 2.5, 0.5 } }, { { 2.0, 1.0, 3.0 } } };
                for (var c = 0; c < nComponents; c++)
                    for (var f = 0; f < nFeatures; f++)
                        expected[c, f, f] = covars[c, 0, f];
                break;
            }
            case CovarianceType.Full:
            {
                covars = new double[nComponents, nFeatures, nFeatures]
                {
                    { { 1.5, 0.3, 0.1 }, { 0.3, 2.5, 0.2 }, { 0.1, 0.2, 3.0 } },
                    { { 2.0, -0.1, 0.05 }, { -0.1, 1.0, 0.2 }, { 0.05, 0.2, 3.0 } },
                };
                for (var c = 0; c < nComponents; c++)
                    for (var j = 0; j < nFeatures; j++)
                        for (var k = 0; k < nFeatures; k++)
                            expected[c, j, k] = covars[c, j, k];
                break;
            }
            case CovarianceType.Tied:
            {
                var shared = new double[nFeatures, nFeatures] { { 1.2, 0.3, 0.1 }, { 0.3, 1.5, 0.2 }, { 0.1, 0.2, 0.9 } };
                covars = new double[1, nFeatures, nFeatures]
                {
                    {
                        { shared[0, 0], shared[0, 1], shared[0, 2] },
                        { shared[1, 0], shared[1, 1], shared[1, 2] },
                        { shared[2, 0], shared[2, 1], shared[2, 2] },
                    }
                };
                for (var c = 0; c < nComponents; c++)
                    for (var j = 0; j < nFeatures; j++)
                        for (var k = 0; k < nFeatures; k++)
                            expected[c, j, k] = shared[j, k];
                break;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(type));
        }

        var model = new ExposedModel(nComponents, type);
        ValidStartAndTrans(model, nComponents);
        var means = new double[nComponents, nFeatures];
        for (var c = 0; c < nComponents; c++)
            for (var f = 0; f < nFeatures; f++)
                means[c, f] = c * nFeatures + f;
        model.Means = means;
        model.Covars = covars;
        model.RunCheck();

        var full = model.FullCovars;
        Assert.Equal(nComponents, full.GetLength(0));
        Assert.Equal(nFeatures, full.GetLength(1));
        Assert.Equal(nFeatures, full.GetLength(2));
        Close.Equal(expected, full, $"FullCovars expansion for {type}");
    }

    [Fact]
    public void full_covars_requires_covars_and_a_known_number_of_features()
    {
        var model = Model(CovarianceType.Diag, 2);
        Assert.Throws<InvalidOperationException>(() => _ = model.FullCovars);

        model.Covars = new double[2, 1, 2] { { { 1.0, 2.0 } }, { { 3.0, 4.0 } } };
        // Means is still null, so the number of features is unknown.
        Assert.Throws<InvalidOperationException>(() => _ = model.FullCovars);
    }
}
