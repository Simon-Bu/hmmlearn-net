using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm24;

/// <summary>
/// Acceptance tests for the null contract of <see cref="Stats.LogMultivariateNormalDensity"/> (ticket HMM-24, criterion C1):
/// a null argument throws ArgumentNullException (exact type). Shape violations are covered by HMM-3 ShapeChecksTests.
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
}
