using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm16;

/// <summary>
/// Criterion C2 of ticket HMM-16: <see cref="GmmHmm.Check"/> rejects missing or mis-shaped Weights, Means or Covars, the
/// covariance shape per type, weights rows that do not sum to 1 and negative spherical or diag variances, raises the degenerate
/// (zero) variance warnings instead of throwing, checks the tied and full covariance matrices for symmetry and positive
/// semi-definiteness per matrix (warning on a null eigenvalue, exception when not symmetric or not positive definite), and
/// <see cref="GmmHmm.EffectiveCovarsPrior"/>/<see cref="GmmHmm.EffectiveCovarsWeight"/> give the documented defaults per
/// covariance type and the user's value when set. All cases are hand-made; only the exception type is asserted (never the
/// message), except that the warnings are matched on the substrings documented in the stub.
/// </summary>
public class ValidationTests
{
    private const int NComponents = 2;
    private const int NMix = 3;
    private const int NFeatures = 2;

    private sealed class ExposedModel : GmmHmm
    {
        public ExposedModel(int nComponents, int nMix, CovarianceType covarianceType,
            double meansWeight = 0.0, double meansPrior = 0.0, double? covarsPrior = null, double? covarsWeight = null)
            : base(nComponents: nComponents, nMix: nMix, meansWeight: meansWeight, meansPrior: meansPrior,
                covarsPrior: covarsPrior, covarsWeight: covarsWeight, covarianceType: covarianceType,
                initParams: "")
        {
        }

        public void RunCheck() => Check();
    }

    private static ExposedModel ValidModel(CovarianceType type)
    {
        var model = new ExposedModel(NComponents, NMix, type);
        model.StartProb = new[] { 0.7, 0.3 };
        model.TransMat = new double[NComponents, NComponents] { { 0.8, 0.2 }, { 0.3, 0.7 } };
        model.Weights = new double[NComponents, NMix] { { 0.5, 0.3, 0.2 }, { 0.2, 0.5, 0.3 } };
        model.Means = new double[NComponents, NMix, NFeatures]
        {
            { { 0.0, 1.0 }, { 2.0, 0.5 }, { -1.5, -0.5 } },
            { { 1.0, -1.0 }, { -2.0, 0.5 }, { 0.5, 2.0 } },
        };
        model.Covars = ValidCovars(type);
        model.NFeatures = NFeatures;
        return model;
    }

    private static double[,,,] ValidCovars(CovarianceType type) => type switch
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

    private static double[,,,] Ones4(int d0, int d1, int d2, int d3)
    {
        var tensor = new double[d0, d1, d2, d3];
        for (var i = 0; i < d0; i++)
            for (var j = 0; j < d1; j++)
                for (var k = 0; k < d2; k++)
                    for (var l = 0; l < d3; l++)
                        tensor[i, j, k, l] = 1.0;
        return tensor;
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void check_accepts_valid_parameters_for_each_covariance_type(CovarianceType type)
    {
        var model = ValidModel(type);
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.RunCheck();
        Assert.Empty(warnings);
    }

    [Theory]
    [InlineData(0)] // Weights null
    [InlineData(1)] // Means null
    [InlineData(2)] // Covars null
    public void check_rejects_missing_emission_parameters(int missing)
    {
        var model = ValidModel(CovarianceType.Diag);
        Assert.Throws<ArgumentException>(() =>
        {
            if (missing == 0)
                model.Weights = null;
            else if (missing == 1)
                model.Means = null;
            else
                model.Covars = null;
            model.RunCheck();
        });
    }

    [Theory]
    [InlineData(0)] // (3, 3): three components instead of two
    [InlineData(1)] // (2, 2): two mixtures instead of three
    public void check_rejects_misshaped_weights(int caseIndex)
    {
        var model = ValidModel(CovarianceType.Diag);
        var weights = caseIndex == 0
            ? new double[3, 3] { { 0.5, 0.3, 0.2 }, { 0.2, 0.5, 0.3 }, { 0.4, 0.2, 0.4 } }
            : new double[NComponents, 2] { { 0.6, 0.4 }, { 0.7, 0.3 } };
        Assert.Throws<ArgumentException>(() =>
        {
            model.Weights = weights;
            model.RunCheck();
        });
    }

    [Fact]
    public void check_rejects_weights_rows_that_do_not_sum_to_one()
    {
        var model = ValidModel(CovarianceType.Diag);
        var weights = new double[NComponents, NMix] { { 0.5, 0.3, 0.2 }, { 0.4, 0.3, 0.2 } };
        Assert.Throws<ArgumentException>(() =>
        {
            model.Weights = weights;
            model.RunCheck();
        });
    }

    [Theory]
    [InlineData(0)] // (2, 2, 2): two mixtures instead of three
    [InlineData(1)] // (3, 3, 2): three components instead of two
    public void check_rejects_misshaped_means(int caseIndex)
    {
        var model = ValidModel(CovarianceType.Diag);
        var means = caseIndex == 0
            ? new double[NComponents, 2, NFeatures]
            {
                { { 0.0, 1.0 }, { 2.0, 0.5 } },
                { { 1.0, -1.0 }, { -2.0, 0.5 } },
            }
            : new double[3, NMix, NFeatures]
            {
                { { 0.0, 1.0 }, { 2.0, 0.5 }, { -1.5, -0.5 } },
                { { 1.0, -1.0 }, { -2.0, 0.5 }, { 0.5, 2.0 } },
                { { -0.5, 0.2 }, { 0.3, -1.0 }, { 1.2, 0.4 } },
            };
        Assert.Throws<ArgumentException>(() =>
        {
            model.Means = means;
            model.RunCheck();
        });
    }

    [Fact]
    public void check_rejects_a_feature_count_mismatch()
    {
        var model = ValidModel(CovarianceType.Diag);
        Assert.Throws<ArgumentException>(() =>
        {
            model.NFeatures = 3; // known feature count, different from the 2 in Means
            model.RunCheck();
        });
    }

    [Theory]
    [InlineData(CovarianceType.Spherical, 0)] // (2, 2, 1, 1): two mixtures instead of three
    [InlineData(CovarianceType.Spherical, 1)] // (2, 3, 1, 2): the diag shape, not the spherical one
    [InlineData(CovarianceType.Diag, 0)] // (2, 3, 1, 1): the spherical shape, not the diag one
    [InlineData(CovarianceType.Diag, 1)] // (2, 3, 1, 3): three features instead of two
    [InlineData(CovarianceType.Full, 0)] // (2, 3, 1, 2): the diag shape, not the full one
    [InlineData(CovarianceType.Full, 1)] // (2, 2, 2, 2): two mixtures instead of three
    [InlineData(CovarianceType.Tied, 0)] // (2, 3, 2, 2): the full shape, not the tied one
    [InlineData(CovarianceType.Tied, 1)] // (2, 1, 2, 3): three features instead of two
    public void check_rejects_misshaped_covars(CovarianceType type, int caseIndex)
    {
        var model = ValidModel(type);
        var covars = type switch
        {
            CovarianceType.Spherical => caseIndex == 0 ? Ones4(NComponents, 2, 1, 1) : Ones4(NComponents, NMix, 1, 2),
            CovarianceType.Diag => caseIndex == 0 ? Ones4(NComponents, NMix, 1, 1) : Ones4(NComponents, NMix, 1, 3),
            CovarianceType.Full => caseIndex == 0 ? Ones4(NComponents, NMix, 1, 2) : Ones4(NComponents, 2, 2, 2),
            _ => caseIndex == 0 ? Ones4(NComponents, NMix, NFeatures, NFeatures) : Ones4(NComponents, 1, NFeatures, 3),
        };
        Assert.Throws<ArgumentException>(() =>
        {
            model.Covars = covars;
            model.RunCheck();
        });
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    public void check_rejects_negative_variances(CovarianceType type)
    {
        var model = ValidModel(type);
        var covars = (double[,,,])model.Covars!.Clone();
        if (type == CovarianceType.Spherical)
            covars[0, 1, 0, 0] = -0.5;
        else
            covars[1, 2, 0, 1] = -0.5;
        Assert.Throws<ArgumentException>(() =>
        {
            model.Covars = covars;
            model.RunCheck();
        });
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    public void a_zero_variance_raises_a_degenerate_mixture_covariance_warning(CovarianceType type)
    {
        var model = ValidModel(type);
        var covars = (double[,,,])model.Covars!.Clone();
        if (type == CovarianceType.Spherical)
            covars[0, 1, 0, 0] = 0.0;
        else
            covars[1, 2, 0, 0] = 0.0;
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.Covars = covars;
        model.RunCheck();
        Assert.Contains(warnings, w => w.Contains("Degenerate mixture covariance"));
    }

    [Fact]
    public void check_rejects_a_non_symmetric_tied_covar()
    {
        var model = ValidModel(CovarianceType.Tied);
        var covars = (double[,,,])model.Covars!.Clone();
        covars[0, 0, 1, 0] = 0.5; // was 0.3: matrix {1, 0.3; 0.5, 2} is not symmetric
        Assert.Throws<ArgumentException>(() =>
        {
            model.Covars = covars;
            model.RunCheck();
        });
    }

    [Fact]
    public void check_rejects_a_symmetric_but_indefinite_tied_covar()
    {
        var model = ValidModel(CovarianceType.Tied);
        var covars = (double[,,,])model.Covars!.Clone();
        covars[0, 0, 0, 1] = 2.0;
        covars[0, 0, 1, 0] = 2.0; // eigenvalues 3 and -1
        Assert.Throws<ArgumentException>(() =>
        {
            model.Covars = covars;
            model.RunCheck();
        });
    }

    [Fact]
    public void a_null_eigenvalue_in_a_tied_covar_raises_a_warning_not_an_exception()
    {
        var model = ValidModel(CovarianceType.Tied);
        var covars = (double[,,,])model.Covars!.Clone();
        covars[0, 0, 0, 1] = 1.0;
        covars[0, 0, 1, 0] = 1.0;
        covars[0, 0, 1, 1] = 1.0; // {1, 1; 1, 1}: eigenvalues 2 and 0
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.Covars = covars;
        model.RunCheck();
        Assert.Contains(warnings, w => w.Contains("null eigenvalue"));
    }

    [Fact]
    public void check_rejects_a_non_symmetric_full_covar()
    {
        var model = ValidModel(CovarianceType.Full);
        var covars = (double[,,,])model.Covars!.Clone();
        covars[0, 1, 0, 1] = -0.3; // was -0.5: matrix {2, -0.3; -0.5, 1} is not symmetric
        Assert.Throws<ArgumentException>(() =>
        {
            model.Covars = covars;
            model.RunCheck();
        });
    }

    [Fact]
    public void check_rejects_a_symmetric_but_indefinite_full_covar()
    {
        var model = ValidModel(CovarianceType.Full);
        var covars = (double[,,,])model.Covars!.Clone();
        covars[1, 2, 0, 0] = 1.0;
        covars[1, 2, 0, 1] = 2.0;
        covars[1, 2, 1, 0] = 2.0;
        covars[1, 2, 1, 1] = 1.0; // eigenvalues 3 and -1
        Assert.Throws<ArgumentException>(() =>
        {
            model.Covars = covars;
            model.RunCheck();
        });
    }

    [Fact]
    public void a_null_eigenvalue_in_a_full_covar_raises_a_warning_not_an_exception()
    {
        var model = ValidModel(CovarianceType.Full);
        var covars = (double[,,,])model.Covars!.Clone();
        covars[1, 2, 0, 0] = 1.0;
        covars[1, 2, 0, 1] = 1.0;
        covars[1, 2, 1, 0] = 1.0;
        covars[1, 2, 1, 1] = 1.0; // {1, 1; 1, 1}: eigenvalues 2 and 0
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.Covars = covars;
        model.RunCheck();
        Assert.Contains(warnings, w => w.Contains("null eigenvalue"));
    }

    [Theory]
    [InlineData(CovarianceType.Full, null)]
    [InlineData(CovarianceType.Tied, null)]
    [InlineData(CovarianceType.Diag, null)]
    [InlineData(CovarianceType.Spherical, null)]
    [InlineData(CovarianceType.Full, 1.7)]
    [InlineData(CovarianceType.Tied, -0.5)]
    [InlineData(CovarianceType.Diag, 2.0)]
    [InlineData(CovarianceType.Spherical, 0.25)]
    public void effective_covars_prior_is_the_default_or_the_user_value(CovarianceType type, double? userValue)
    {
        var model = new ExposedModel(NComponents, NMix, type, covarsPrior: userValue);
        Close.Equal(
            userValue ?? type switch
            {
                CovarianceType.Full => 0.0,
                CovarianceType.Tied => 0.0,
                CovarianceType.Diag => -1.5,
                _ => -(model.NMix + 2.0) / 2.0,
            },
            model.EffectiveCovarsPrior, "EffectiveCovarsPrior");
    }

    [Theory]
    [InlineData(CovarianceType.Full, null)]
    [InlineData(CovarianceType.Tied, null)]
    [InlineData(CovarianceType.Diag, null)]
    [InlineData(CovarianceType.Spherical, null)]
    [InlineData(CovarianceType.Full, 1.25)]
    [InlineData(CovarianceType.Tied, -3.0)]
    [InlineData(CovarianceType.Diag, 0.5)]
    [InlineData(CovarianceType.Spherical, -1.2)]
    public void effective_covars_weight_is_the_default_or_the_user_value(CovarianceType type, double? userValue)
    {
        var model = ValidModel(type); // sets Means and NFeatures, so the feature count is known
        if (userValue.HasValue)
            model.CovarsWeight = userValue;
        var nFeatures = model.NFeatures!.Value;
        Close.Equal(
            userValue ?? type switch
            {
                CovarianceType.Full => -(nFeatures + 2.0),
                CovarianceType.Tied => -(model.NMix + nFeatures + 1.0),
                _ => 0.0,
            },
            model.EffectiveCovarsWeight, "EffectiveCovarsWeight");
    }

    [Fact]
    public void the_constructor_stores_its_arguments()
    {
        var model = new ExposedModel(2, 3, CovarianceType.Tied, meansWeight: 2.5, meansPrior: 0.3,
            covarsPrior: -1.5, covarsWeight: 0.25);
        Assert.Equal(3, model.NMix);
        Assert.Equal(CovarianceType.Tied, model.CovarianceType);
        Close.Equal(2.5, model.MeansWeight, "MeansWeight");
        Close.Equal(0.3, model.MeansPrior, "MeansPrior");
        Close.Equal(-1.5, model.CovarsPrior!.Value, "CovarsPrior");
        Close.Equal(0.25, model.CovarsWeight!.Value, "CovarsWeight");
        Close.Equal(1e-3, model.MinCovar, "MinCovar");
        Close.Equal(1.0, model.WeightsPrior, "WeightsPrior");

        var defaults = new ExposedModel(2, 3, CovarianceType.Diag);
        Assert.Equal(3, defaults.NMix);
        Assert.Equal(CovarianceType.Diag, defaults.CovarianceType);
        Assert.Null(defaults.CovarsPrior);
        Assert.Null(defaults.CovarsWeight);
        Close.Equal(1.0, defaults.WeightsPrior, "WeightsPrior default");
        Close.Equal(0.0, defaults.MeansWeight, "MeansWeight default");
        Close.Equal(0.0, defaults.MeansPrior, "MeansPrior default");
        Close.Equal(1e-3, defaults.MinCovar, "MinCovar default");
    }
}
