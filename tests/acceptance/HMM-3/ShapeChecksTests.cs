using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm3;

/// <summary>
/// Acceptance tests for the input shape checks of <see cref="Stats.LogMultivariateNormalDensity"/> (ticket HMM-3,
/// criterion C3): a covariance array of the wrong shape for its type, or x with a different number of features than
/// the means, throws ArgumentException (hand-written cases; the number of components and features are taken from the
/// means, the feature count of x must match).
/// </summary>
public class ShapeChecksTests
{
    /// <summary>x with a different number of features than the means throws ArgumentException, whatever the
    /// covariance type (hand-made cases; the covars shape is the valid one for the type, so the only violation is
    /// the x/means mismatch).</summary>
    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void X_with_a_different_number_of_features_than_the_means_throws(CovarianceType covarianceType)
    {
        var x = new double[2, 3];
        var means = new double[2, 2];
        var covars = covarianceType switch
        {
            CovarianceType.Spherical => new double[2, 1, 1] { { { 1.0 } }, { { 1.0 } } },
            CovarianceType.Diag => new double[2, 1, 2] { { { 1.0, 1.0 } }, { { 1.0, 1.0 } } },
            CovarianceType.Full => new double[2, 2, 2]
            {
                { { 1.0, 0.0 }, { 0.0, 1.0 } },
                { { 1.0, 0.0 }, { 0.0, 1.0 } }
            },
            _ => new double[1, 2, 2] { { { 1.0, 0.0 }, { 0.0, 1.0 } } }
        };
        Assert.Throws<ArgumentException>(() => Stats.LogMultivariateNormalDensity(x, means, covars, covarianceType));
    }

    /// <summary>A covariance array whose shape does not match the type (and the nc/nf of the consistent x and means,
    /// here (2, 2)) throws ArgumentException, whatever the values (hand-made cases: spherical (nc,1,1), diag (nc,1,nf),
    /// full (nc,nf,nf), tied (1,nf,nf)).</summary>
    [Theory]
    [InlineData(CovarianceType.Spherical, 1, 1, 1)] // dim 0 is not nc = 2
    [InlineData(CovarianceType.Spherical, 2, 2, 1)] // dim 1 is not 1
    [InlineData(CovarianceType.Spherical, 2, 1, 2)] // dim 2 is not 1
    [InlineData(CovarianceType.Diag, 1, 1, 2)]      // dim 0 is not nc = 2
    [InlineData(CovarianceType.Diag, 2, 2, 2)]      // dim 1 is not 1
    [InlineData(CovarianceType.Diag, 2, 1, 3)]      // dim 2 is not nf = 2
    [InlineData(CovarianceType.Full, 1, 2, 2)]      // dim 0 is not nc = 2
    [InlineData(CovarianceType.Full, 2, 3, 2)]      // dim 1 is not nf = 2
    [InlineData(CovarianceType.Full, 2, 2, 3)]      // dim 2 is not nf = 2
    [InlineData(CovarianceType.Tied, 2, 2, 2)]      // dim 0 is not 1 (a valid full shape is not a tied shape)
    [InlineData(CovarianceType.Tied, 1, 3, 2)]      // dim 1 is not nf = 2
    [InlineData(CovarianceType.Tied, 1, 2, 3)]      // dim 2 is not nf = 2
    public void A_covariance_array_of_the_wrong_shape_for_its_type_throws(
        CovarianceType covarianceType, int d0, int d1, int d2)
    {
        var covars = new double[d0, d1, d2];
        var x = new double[2, 2];
        var means = new double[2, 2];
        Assert.Throws<ArgumentException>(() => Stats.LogMultivariateNormalDensity(x, means, covars, covarianceType));
    }

    /// <summary>The interface documents only ArgumentException and says nothing about null arguments (assumption,
    /// reported in .factory/out/tests.json): a null argument must still fail with an ArgumentException-family
    /// exception (ArgumentNullException is a subclass), never crash with a NullReferenceException. ThrowsAny accepts
    /// any subclass, so this test does not pin down the exact type.</summary>
    [Fact]
    public void A_null_argument_throws_an_ArgumentException_family_exception()
    {
        var x = new double[1, 2];
        var means = new double[1, 2];
        var covars = new double[1, 1, 2] { { { 1.0, 1.0 } } };
        Assert.ThrowsAny<ArgumentException>(() => Stats.LogMultivariateNormalDensity(null!, means, covars, CovarianceType.Diag));
        Assert.ThrowsAny<ArgumentException>(() => Stats.LogMultivariateNormalDensity(x, null!, covars, CovarianceType.Diag));
        Assert.ThrowsAny<ArgumentException>(() => Stats.LogMultivariateNormalDensity(x, means, null!, CovarianceType.Diag));
    }
}
