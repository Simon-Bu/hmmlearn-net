using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm1;

/// <summary>
/// Acceptance tests for <see cref="Utils.FillCovars"/> (ticket HMM-1, criterion C3): every case of
/// <c>fill_covars_cases</c> in oracle/golden/utils.json, plus hand-checked edges (other shapes, purity,
/// and shape mismatches rejected with ArgumentException).
/// </summary>
public class FillCovarsTests
{
    /// <summary>Every golden case (the four covariance types, compact shapes) is expanded to the expected (nc, nf, nf) result.</summary>
    [Fact]
    public void Every_fill_covars_case_of_the_golden_file_is_reproduced()
    {
        var cases = Golden.Load("utils").GetProperty("fill_covars_cases").Items().ToList();
        Assert.NotEmpty(cases);
        foreach (var c in cases)
        {
            var name = c.GetProperty("name").AsString();
            var type = c.GetProperty("covariance_type").AsString() switch
            {
                "spherical" => CovarianceType.Spherical,
                "diag" => CovarianceType.Diag,
                "full" => CovarianceType.Full,
                "tied" => CovarianceType.Tied,
                var unknown => throw new FormatException($"unknown covariance type {unknown} in the golden file"),
            };
            var covars = c.GetProperty("covars").AsTensor3();
            var nComponents = c.GetProperty("n_components").GetInt32();
            var nFeatures = c.GetProperty("n_features").GetInt32();
            var result = Utils.FillCovars(covars, type, nComponents, nFeatures);
            Close.Equal(c.GetProperty("expected").AsTensor3(), result, name);
        }
    }

    /// <summary>Spherical (nc, 1, 1): each component is the variance times the identity (hand-checked nc=2, nf=3).</summary>
    [Fact]
    public void Fill_covars_spherical_gives_the_variance_times_the_identity()
    {
        var covars = new double[2, 1, 1] { { { 4.0 } }, { { 9.0 } } };
        var result = Utils.FillCovars(covars, CovarianceType.Spherical, 2, 3);
        var expected = new double[2, 3, 3];
        expected[0, 0, 0] = 4.0;
        expected[0, 1, 1] = 4.0;
        expected[0, 2, 2] = 4.0;
        expected[1, 0, 0] = 9.0;
        expected[1, 1, 1] = 9.0;
        expected[1, 2, 2] = 9.0;
        Close.Equal(expected, result, "spherical (2, 3)");
    }

    /// <summary>Diag (nc, 1, nf): the variances go on the diagonal and zeros elsewhere (hand-checked nc=1, nf=3).</summary>
    [Fact]
    public void Fill_covars_diag_puts_the_variances_on_the_diagonal()
    {
        var covars = new double[1, 1, 3] { { { 7.0, 8.0, 9.0 } } };
        var result = Utils.FillCovars(covars, CovarianceType.Diag, 1, 3);
        var expected = new double[1, 3, 3];
        expected[0, 0, 0] = 7.0;
        expected[0, 1, 1] = 8.0;
        expected[0, 2, 2] = 9.0;
        Close.Equal(expected, result, "diag (1, 3)");
    }

    /// <summary>Tied (1, nf, nf): the single matrix is repeated for every component (hand-checked nc=2, nf=2).</summary>
    [Fact]
    public void Fill_covars_tied_repeats_the_single_matrix_for_every_component()
    {
        var covars = new double[1, 2, 2] { { { 5.0, 2.0 }, { 2.0, 3.0 } } };
        var result = Utils.FillCovars(covars, CovarianceType.Tied, 2, 2);
        var expected = new double[2, 2, 2] { { { 5.0, 2.0 }, { 2.0, 3.0 } }, { { 5.0, 2.0 }, { 2.0, 3.0 } } };
        Close.Equal(expected, result, "tied (2, 2)");
    }

    /// <summary>Full (nc, nf, nf): the matrices are kept (hand-checked the degenerate case nc=2, nf=1).</summary>
    [Fact]
    public void Fill_covars_full_keeps_the_matrices()
    {
        var covars = new double[2, 1, 1] { { { 4.0 } }, { { 9.0 } } };
        var result = Utils.FillCovars(covars, CovarianceType.Full, 2, 1);
        var expected = new double[2, 1, 1] { { { 4.0 } }, { { 9.0 } } };
        Close.Equal(expected, result, "full (2, 1)");
    }

    /// <summary>FillCovars is pure: the compact input is not modified (in particular the diag expansion must not write zeros into it).</summary>
    [Fact]
    public void Fill_covars_does_not_modify_the_input()
    {
        var diag = new double[2, 1, 2] { { { 1.0, 2.0 } }, { { 3.0, 4.0 } } };
        var full = new double[2, 2, 2] { { { 1.0, 0.2 }, { 0.2, 2.0 } }, { { 3.0, -0.5 }, { -0.5, 1.0 } } };
        Utils.FillCovars(diag, CovarianceType.Diag, 2, 2);
        Utils.FillCovars(full, CovarianceType.Full, 2, 2);
        Close.Equal(new double[2, 1, 2] { { { 1.0, 2.0 } }, { { 3.0, 4.0 } } }, diag, "diag input unchanged");
        Close.Equal(new double[2, 2, 2] { { { 1.0, 0.2 }, { 0.2, 2.0 } }, { { 3.0, -0.5 }, { -0.5, 1.0 } } }, full, "full input unchanged");
    }

    /// <summary>A shape that does not match the covariance type, nComponents or nFeatures throws ArgumentException.</summary>
    [Theory]
    [InlineData("spherical", 3, 1, 2, 3, 2)] // dim 2 != 1
    [InlineData("spherical", 2, 1, 1, 3, 2)] // dim 0 != nComponents
    [InlineData("spherical", 3, 2, 1, 3, 2)] // dim 1 != 1
    [InlineData("diag", 3, 1, 1, 3, 2)]      // dim 2 != nFeatures
    [InlineData("diag", 3, 2, 2, 3, 2)]      // dim 1 != 1
    [InlineData("diag", 2, 1, 2, 3, 2)]      // dim 0 != nComponents
    [InlineData("full", 3, 1, 2, 3, 2)]      // dim 1 != nFeatures
    [InlineData("full", 2, 2, 2, 3, 2)]      // dim 0 != nComponents
    [InlineData("full", 3, 2, 3, 3, 2)]      // dim 2 != nFeatures
    [InlineData("tied", 2, 2, 2, 3, 2)]      // dim 0 != 1
    [InlineData("tied", 1, 2, 3, 3, 2)]      // dim 2 != nFeatures
    public void Fill_covars_rejects_a_shape_that_does_not_match_the_type(string type, int d0, int d1, int d2, int nComponents, int nFeatures)
    {
        var covars = new double[d0, d1, d2];
        var covarianceType = type switch
        {
            "spherical" => CovarianceType.Spherical,
            "diag" => CovarianceType.Diag,
            "full" => CovarianceType.Full,
            _ => CovarianceType.Tied,
        };
        Assert.Throws<ArgumentException>(() => Utils.FillCovars(covars, covarianceType, nComponents, nFeatures));
    }
}
