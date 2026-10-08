using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm24;

/// <summary>
/// Acceptance tests for the null and shape-violation contract of <see cref="Stats.LogMultivariateNormalDensity"/> (ticket HMM-24,
/// criterion C1): a null argument throws ArgumentNullException (exact type), a shape violation throws ArgumentException (exact type).
/// </summary>
public class NullArgumentTests
{
    private static double[,,] ValidCovars(CovarianceType covarianceType) => covarianceType switch
    {
        CovarianceType.Spherical => new double[2, 1, 1] { { { 1.0 } }, { { 2.0 } } },
        CovarianceType.Diag => new double[2, 1, 2] { { { 1.0, 1.0 } }, { { 2.0, 3.0 } } },
        CovarianceType.Full => new double[2, 2, 2]
        {
            { { 1.0, 0.0 }, { 0.0, 1.0 } },
            { { 2.0, 0.5 }, { 0.5, 2.0 } }
        },
        _ => new double[1, 2, 2] { { { 1.0, 0.0 }, { 0.0, 1.0 } } }
    };

    private static readonly double[,] X = { { 0.0, 1.0 }, { 2.0, -1.0 }, { 0.5, 0.5 } };
    private static readonly double[,] Means = { { 0.0, 0.0 }, { 1.0, 1.0 } };

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void A_null_x_throws_ArgumentNullException(CovarianceType covarianceType)
    {
        var covars = ValidCovars(covarianceType);
        Assert.Throws<ArgumentNullException>(() => Stats.LogMultivariateNormalDensity(null!, Means, covars, covarianceType));
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void Null_means_throw_ArgumentNullException(CovarianceType covarianceType)
    {
        var covars = ValidCovars(covarianceType);
        Assert.Throws<ArgumentNullException>(() => Stats.LogMultivariateNormalDensity(X, null!, covars, covarianceType));
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void Null_covars_throw_ArgumentNullException(CovarianceType covarianceType)
    {
        Assert.Throws<ArgumentNullException>(() => Stats.LogMultivariateNormalDensity(X, Means, null!, covarianceType));
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void X_with_another_number_of_columns_than_the_means_throws_exactly_ArgumentException(CovarianceType covarianceType)
    {
        var x = new double[3, 3];
        var covars = ValidCovars(covarianceType);
        Assert.Throws<ArgumentException>(() => Stats.LogMultivariateNormalDensity(x, Means, covars, covarianceType));
    }

    [Theory]
    [InlineData(CovarianceType.Spherical, 1, 1, 1)]
    [InlineData(CovarianceType.Diag, 2, 1, 3)]
    [InlineData(CovarianceType.Full, 2, 3, 2)]
    [InlineData(CovarianceType.Tied, 2, 2, 2)]
    public void Covars_of_the_wrong_shape_throw_exactly_ArgumentException(CovarianceType covarianceType, int d0, int d1, int d2)
    {
        var covars = new double[d0, d1, d2];
        Assert.Throws<ArgumentException>(() => Stats.LogMultivariateNormalDensity(X, Means, covars, covarianceType));
    }
}
