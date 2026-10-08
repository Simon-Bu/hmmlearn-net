using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm23;

/// <summary>
/// Acceptance tests for <see cref="InternalUtils.ValidateCovars"/> (ticket HMM-23, criterion C2): covariances without
/// features are vacuously symmetric and positive definite; shapes that do not match nComponents are still rejected.
/// </summary>
public class ZeroFeatureCovarsTests
{
    /// <summary>Full covariances (3, 0, 0) with three components are accepted.</summary>
    [Fact]
    public void Full_without_features_is_accepted()
    {
        var exception = Record.Exception(() => InternalUtils.ValidateCovars(new double[3, 0, 0], CovarianceType.Full, 3));
        Assert.Null(exception);
    }

    /// <summary>Tied covariances (1, 0, 0) are accepted.</summary>
    [Fact]
    public void Tied_without_features_is_accepted()
    {
        var exception = Record.Exception(() => InternalUtils.ValidateCovars(new double[1, 0, 0], CovarianceType.Tied, 3));
        Assert.Null(exception);
    }

    /// <summary>Control: a full tensor (2, 0, 0) with nComponents 3 still throws ArgumentException.</summary>
    [Fact]
    public void Full_with_wrong_component_count_throws_ArgumentException()
    {
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(new double[2, 0, 0], CovarianceType.Full, 3));
    }

    /// <summary>Control: a tied tensor (2, 0, 0) still throws ArgumentException.</summary>
    [Fact]
    public void Tied_with_two_matrices_throws_ArgumentException()
    {
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(new double[2, 0, 0], CovarianceType.Tied, 3));
    }
}
