using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm2;

/// <summary>
/// Acceptance tests for <see cref="InternalUtils.DistributeCovarMatrixToMatchCovarianceType"/> (ticket HMM-2,
/// criterion C4): every case of <c>distribute_covar_cases</c> in oracle/golden/internal_utils.json, plus
/// hand-checked cases (the mean of ALL entries for spherical, the diagonal for diag, the uniform result shapes,
/// the input left untouched).
/// </summary>
public class DistributeCovarMatrixTests
{
    /// <summary>Every golden case is reproduced for the four covariance types, one and three components.</summary>
    [Fact]
    public void Every_distribute_covar_case_of_the_golden_file_is_reproduced()
    {
        var cases = Golden.Load("internal_utils").GetProperty("distribute_covar_cases").Items().ToList();
        Assert.NotEmpty(cases);
        foreach (var c in cases)
        {
            var name = c.GetProperty("name").AsString();
            var type = Enum.Parse<CovarianceType>(c.GetProperty("covariance_type").AsString(), ignoreCase: true);
            var tiedCovar = c.GetProperty("tied_covar").AsMatrix();
            var nComponents = c.GetProperty("n_components").GetInt32();
            var result = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(tiedCovar, type, nComponents);
            Close.Equal(c.GetProperty("expected").AsTensor3(), result, name);
        }
    }

    /// <summary>
    /// Spherical takes the mean of ALL the entries of the template, off-diagonal ones included (hmmlearn quirk):
    /// (2 + 1 + 1 + 4) / 4 = 2 for one and for two states (hand-checked; a diagonal-only mean would give 3).
    /// </summary>
    [Fact]
    public void Distribute_spherical_uses_the_mean_of_all_entries_including_the_off_diagonals()
    {
        var template = new double[,] { { 2.0, 1.0 }, { 1.0, 4.0 } };
        var one = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(template, CovarianceType.Spherical, 1);
        Close.Equal(new double[1, 1, 1] { { { 2.0 } } }, one, "one state");
        var two = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(template, CovarianceType.Spherical, 2);
        Close.Equal(new double[2, 1, 1] { { { 2.0 } }, { { 2.0 } } }, two, "two states");
    }

    /// <summary>Diag takes the diagonal of the template, repeated for each state (hand-checked [2, 4], 2 states).</summary>
    [Fact]
    public void Distribute_diag_takes_the_diagonal_repeated_for_each_state()
    {
        var template = new double[,] { { 2.0, 1.0 }, { 1.0, 4.0 } };
        var result = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(template, CovarianceType.Diag, 2);
        Close.Equal(new double[2, 1, 2] { { { 2.0, 4.0 } }, { { 2.0, 4.0 } } }, result, "diag (2, 1, 2)");
    }

    /// <summary>Full repeats the template for each state (hand-checked, 3 states, shape (3, 2, 2)).</summary>
    [Fact]
    public void Distribute_full_repeats_the_template_for_each_state()
    {
        var template = new double[,] { { 2.0, 1.0 }, { 1.0, 4.0 } };
        var result = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(template, CovarianceType.Full, 3);
        Close.Equal(new double[3, 2, 2]
        {
            { { 2.0, 1.0 }, { 1.0, 4.0 } },
            { { 2.0, 1.0 }, { 1.0, 4.0 } },
            { { 2.0, 1.0 }, { 1.0, 4.0 } }
        }, result, "full (3, 2, 2)");
    }

    /// <summary>Tied keeps the single template, whatever nComponents (hand-checked with 3 states, shape (1, 2, 2)).</summary>
    [Fact]
    public void Distribute_tied_keeps_the_single_template_whatever_the_component_count()
    {
        var template = new double[,] { { 2.0, 1.0 }, { 1.0, 4.0 } };
        var result = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(template, CovarianceType.Tied, 3);
        Close.Equal(new double[1, 2, 2] { { { 2.0, 1.0 }, { 1.0, 4.0 } } }, result, "tied (1, 2, 2)");
    }

    /// <summary>The template is not modified by the distribution (hand-checked for the four types).</summary>
    [Fact]
    public void Distribute_does_not_modify_the_template()
    {
        var template = new double[,] { { 2.0, 1.0 }, { 1.0, 4.0 } };
        var original = (double[,])template.Clone();
        foreach (var type in new[] { CovarianceType.Spherical, CovarianceType.Diag, CovarianceType.Full, CovarianceType.Tied })
            InternalUtils.DistributeCovarMatrixToMatchCovarianceType(template, type, 2);
        Close.Equal(original, template, "template unchanged");
    }
}
