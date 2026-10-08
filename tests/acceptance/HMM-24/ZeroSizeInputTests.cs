using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm24;

/// <summary>
/// Acceptance tests for the zero-size contract of <see cref="Stats.LogMultivariateNormalDensity"/> (ticket HMM-24, criterion C2):
/// no sample, no component or no feature gives a result of the expected shape, without any exception.
/// </summary>
public class ZeroSizeInputTests
{
    /// <summary>Identity-like valid covariance for the type, shaped for nc components and nf features.</summary>
    private static double[,,] Covars(CovarianceType covarianceType, int nc, int nf)
    {
        switch (covarianceType)
        {
            case CovarianceType.Spherical:
                var spherical = new double[nc, 1, 1];
                for (var c = 0; c < nc; c++) spherical[c, 0, 0] = 1.5;
                return spherical;
            case CovarianceType.Diag:
                var diag = new double[nc, 1, nf];
                for (var c = 0; c < nc; c++)
                    for (var f = 0; f < nf; f++) diag[c, 0, f] = 1.5;
                return diag;
            case CovarianceType.Full:
                var full = new double[nc, nf, nf];
                for (var c = 0; c < nc; c++)
                    for (var f = 0; f < nf; f++) full[c, f, f] = 1.5;
                return full;
            default:
                var tied = new double[1, nf, nf];
                for (var f = 0; f < nf; f++) tied[0, f, f] = 1.5;
                return tied;
        }
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void X_with_zero_rows_gives_an_empty_result_with_nComponents_columns(CovarianceType covarianceType)
    {
        var x = new double[0, 2];
        var means = new double[3, 2];
        var result = Stats.LogMultivariateNormalDensity(x, means, Covars(covarianceType, 3, 2), covarianceType);
        Assert.Equal(0, result.GetLength(0));
        Assert.Equal(3, result.GetLength(1));
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void Means_with_zero_rows_give_a_result_with_zero_columns(CovarianceType covarianceType)
    {
        var x = new double[4, 2];
        var means = new double[0, 2];
        var result = Stats.LogMultivariateNormalDensity(x, means, Covars(covarianceType, 0, 2), covarianceType);
        Assert.Equal(4, result.GetLength(0));
        Assert.Equal(0, result.GetLength(1));
    }

    [Theory]
    [InlineData(CovarianceType.Spherical)]
    [InlineData(CovarianceType.Diag)]
    [InlineData(CovarianceType.Full)]
    [InlineData(CovarianceType.Tied)]
    public void X_with_zero_features_gives_zeros(CovarianceType covarianceType)
    {
        var x = new double[4, 0];
        var means = new double[3, 0];
        var result = Stats.LogMultivariateNormalDensity(x, means, Covars(covarianceType, 3, 0), covarianceType);
        Assert.Equal(4, result.GetLength(0));
        Assert.Equal(3, result.GetLength(1));
        for (var i = 0; i < 4; i++)
            for (var c = 0; c < 3; c++)
                Assert.InRange(result[i, c], -1e-12, 1e-12);
    }
}
