using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm2;

/// <summary>
/// Acceptance tests for <see cref="InternalUtils.ValidateCovars"/> (ticket HMM-2, criterion C3): every case of
/// <c>validate_covars_cases</c> in oracle/golden/internal_utils.json, plus hand-checked edges (the compact
/// uniform shapes, non-positive variances, symmetry with the allclose tolerance, positive definiteness).
/// </summary>
public class ValidateCovarsTests
{
    /// <summary>Every golden case is reproduced: the valid ones pass, the invalid ones throw ArgumentException.</summary>
    [Fact]
    public void Every_validate_covars_case_of_the_golden_file_is_reproduced()
    {
        var cases = Golden.Load("internal_utils").GetProperty("validate_covars_cases").Items().ToList();
        Assert.NotEmpty(cases);
        foreach (var c in cases)
        {
            var name = c.GetProperty("name").AsString();
            var type = Enum.Parse<CovarianceType>(c.GetProperty("covariance_type").AsString(), ignoreCase: true);
            var covars = c.GetProperty("covars").AsTensor3();
            var nComponents = c.GetProperty("n_components").GetInt32();
            if (c.Has("raises") && c.GetProperty("raises").AsBool())
            {
                var thrown = Record.Exception(() => InternalUtils.ValidateCovars(covars, type, nComponents));
                Assert.True(thrown?.GetType() == typeof(ArgumentException), $"{name}: expected ArgumentException, got {thrown?.GetType().Name ?? "no exception"}");
            }
            else
            {
                InternalUtils.ValidateCovars(covars, type, nComponents);
            }
        }
    }

    /// <summary>Each of the four types accepts its valid compact uniform shape (hand-checked positive variances
    /// and positive-definite symmetric matrices).</summary>
    [Theory]
    [InlineData("spherical", 2)]
    [InlineData("diag", 2)]
    [InlineData("full", 2)]
    [InlineData("tied", 5)] // tied is one matrix, whatever the number of components
    public void Validate_covars_accepts_the_valid_compact_shape_of_each_type(string type, int nComponents)
    {
        var covarianceType = Enum.Parse<CovarianceType>(type, ignoreCase: true);
        var covars = covarianceType switch
        {
            CovarianceType.Spherical => Spherical(nComponents),
            CovarianceType.Diag => Diag(nComponents),
            CovarianceType.Full => Full(nComponents),
            _ => Tied(),
        };
        InternalUtils.ValidateCovars(covars, covarianceType, nComponents);
    }

    /// <summary>Spherical rejects a zero or a negative variance, in any component (hand-checked).</summary>
    [Fact]
    public void Validate_covars_spherical_rejects_zero_and_negative_variances()
    {
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(new double[1, 1, 1] { { { 0.0 } } }, CovarianceType.Spherical, 1));
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(new double[1, 1, 1] { { { -1.5 } } }, CovarianceType.Spherical, 1));
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(new double[2, 1, 1] { { { 1.0 } }, { { 0.0 } } }, CovarianceType.Spherical, 2));
    }

    /// <summary>Diag rejects a zero or a negative variance, wherever it is in the row (hand-checked).</summary>
    [Fact]
    public void Validate_covars_diag_rejects_zero_and_negative_variances()
    {
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(new double[1, 1, 2] { { { 0.0, 2.0 } } }, CovarianceType.Diag, 1));
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(new double[1, 1, 2] { { { 2.0, -2.0 } } }, CovarianceType.Diag, 1));
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(new double[2, 1, 2] { { { 1.0, 2.0 } }, { { 3.0, 0.0 } } }, CovarianceType.Diag, 2));
    }

    /// <summary>A shape that does not match the covariance type or nComponents throws ArgumentException, whatever the
    /// values: spherical (nc, 1, 1), diag (nc, 1, nf), full (nc, nf, nf), tied (1, nf, nf).</summary>
    [Theory]
    [InlineData("spherical", 3, 1, 2, 3)] // dim 2 is not 1
    [InlineData("spherical", 3, 2, 1, 3)] // dim 1 is not 1
    [InlineData("spherical", 2, 1, 1, 3)] // dim 0 is not nComponents
    [InlineData("diag", 3, 2, 2, 3)]      // dim 1 is not 1
    [InlineData("diag", 2, 1, 2, 3)]      // dim 0 is not nComponents
    [InlineData("full", 2, 2, 3, 2)]      // dim 1 is not dim 2
    [InlineData("full", 2, 2, 2, 3)]      // dim 0 is not nComponents
    [InlineData("tied", 2, 2, 2, 2)]      // dim 0 is not 1, although both matrices are valid
    [InlineData("tied", 1, 2, 3, 2)]      // dim 1 is not dim 2
    public void Validate_covars_rejects_a_shape_that_does_not_match_the_type_or_the_component_count(
        string type, int d0, int d1, int d2, int nComponents)
    {
        var covars = new double[d0, d1, d2];
        var covarianceType = Enum.Parse<CovarianceType>(type, ignoreCase: true);
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(covars, covarianceType, nComponents));
    }

    /// <summary>A zero matrix (all eigenvalues exactly 0) is not positive definite: tied and full reject it.</summary>
    [Theory]
    [InlineData("tied")]
    [InlineData("full")]
    public void Validate_covars_rejects_a_singular_matrix(string type)
    {
        var covarianceType = Enum.Parse<CovarianceType>(type, ignoreCase: true);
        var covars = new double[1, 2, 2] { { { 0.0, 0.0 }, { 0.0, 0.0 } } };
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(covars, covarianceType, 1));
    }

    /// <summary>
    /// The symmetry check uses the allclose tolerance |a[i,j] - a[j,i]| &lt;= 1e-8 + 1e-5 * |a[j,i]| (like numpy):
    /// a 1e-7 asymmetry is accepted (the matrix stays positive definite), a 1e-3 asymmetry is rejected.
    /// </summary>
    [Fact]
    public void Validate_covars_checks_symmetry_with_the_allclose_tolerance()
    {
        var withinTolerance = new double[1, 2, 2] { { { 1.0, 1.0 }, { 1.0000001, 2.0 } } };
        InternalUtils.ValidateCovars(withinTolerance, CovarianceType.Tied, 1);
        InternalUtils.ValidateCovars(withinTolerance, CovarianceType.Full, 1);

        var beyondTolerance = new double[1, 2, 2] { { { 1.0, 1.0 }, { 1.001, 2.0 } } };
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(beyondTolerance, CovarianceType.Tied, 1));
        Assert.Throws<ArgumentException>(() => InternalUtils.ValidateCovars(beyondTolerance, CovarianceType.Full, 1));
    }

    private static double[,,] Spherical(int nComponents)
    {
        var covars = new double[nComponents, 1, 1];
        for (var i = 0; i < nComponents; i++)
            covars[i, 0, 0] = 1.0 + i;
        return covars;
    }

    private static double[,,] Diag(int nComponents, int nFeatures = 3)
    {
        var covars = new double[nComponents, 1, nFeatures];
        for (var i = 0; i < nComponents; i++)
            for (var f = 0; f < nFeatures; f++)
                covars[i, 0, f] = 1.0 + f;
        return covars;
    }

    private static double[,,] Full(int nComponents)
    {
        var covars = new double[nComponents, 2, 2];
        for (var i = 0; i < nComponents; i++)
        {
            covars[i, 0, 0] = 2.0;
            covars[i, 0, 1] = 0.5;
            covars[i, 1, 0] = 0.5;
            covars[i, 1, 1] = 3.0;
        }
        return covars;
    }

    private static double[,,] Tied() => new double[1, 2, 2] { { { 2.0, 0.5 }, { 0.5, 3.0 } } };
}
