using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm3;

/// <summary>
/// Acceptance tests for the full and tied paths of <see cref="Stats.LogMultivariateNormalDensity"/> (ticket HMM-3,
/// criterion C2): every full and tied case of oracle/golden/stats.json is reproduced through the Cholesky factor
/// (the singular case retries once with 1e-7 on the diagonal, the non-positive-definite case raises), plus
/// hand-checked invariants and exact values on trivial inputs.
/// </summary>
public class FullTiedDensityTests
{
    /// <summary>Every full case of the golden file is reproduced, case by case; the cases with <c>raises: true</c>
    /// expect the ArgumentException with the documented message.</summary>
    [Fact]
    public void Reproduces_every_full_case_of_the_golden_file()
        => RunGoldenCases(CovarianceType.Full);

    /// <summary>Every tied case of the golden file is reproduced, case by case (the one matrix serves every
    /// component).</summary>
    [Fact]
    public void Reproduces_every_tied_case_of_the_golden_file()
        => RunGoldenCases(CovarianceType.Tied);

    /// <summary>The Cholesky path, exact value on a trivial input: the identity covariance at the mean, two features,
    /// gives -0.5 * 2 * log(2 pi) = -log(2 pi) (hand-checked, C2).</summary>
    [Fact]
    public void The_identity_covariance_at_the_mean_gives_minus_log_2_pi()
    {
        var actual = Stats.LogMultivariateNormalDensity(
            new double[,] { { 0.0, 0.0 } }, new double[,] { { 0.0, 0.0 } },
            new double[1, 2, 2] { { { 1.0, 0.0 }, { 0.0, 1.0 } } }, CovarianceType.Full)[0, 0];
        Close.Equal(-Math.Log(2.0 * Math.PI), actual, "identity covariance at the mean");
    }

    /// <summary>The retry once with 1e-7 on the diagonal (C2): the singular matrix [[1,1],[1,1]] (pivot 0) is not
    /// positive definite, yet the call must succeed and give the same density as the same matrix with 1e-7 added to
    /// the diagonal, which is positive definite and needs no retry (invariant, hand-made data).</summary>
    [Fact]
    public void A_singular_matrix_is_retried_once_with_1e_7_on_the_diagonal()
    {
        var x = new double[,] { { 0.0, 1.0 }, { 2.0, -1.0 } };
        var means = new double[,] { { 0.0, 1.0 }, { 1.0, 1.0 } };
        var singular = new double[2, 2, 2]
        {
            { { 1.0, 1.0 }, { 1.0, 1.0 } },
            { { 1.0, 0.0 }, { 0.0, 1.0 } }
        };
        var retried = new double[2, 2, 2]
        {
            { { 1.0 + 1.0E-7, 1.0 }, { 1.0, 1.0 + 1.0E-7 } },
            { { 1.0, 0.0 }, { 0.0, 1.0 } }
        };
        Close.Equal(Stats.LogMultivariateNormalDensity(x, means, retried, CovarianceType.Full),
            Stats.LogMultivariateNormalDensity(x, means, singular, CovarianceType.Full),
            "singular matrix with the 1e-7 retry");
    }

    /// <summary>When the matrix is still not positive definite after the retry, throw ArgumentException with the
    /// documented message (C2, hand-made): [[1,2],[2,1]] has the eigenvalue -1 and 1e-7 on the diagonal does not fix
    /// it.</summary>
    [Fact]
    public void A_matrix_still_not_positive_definite_after_the_retry_throws()
    {
        var thrown = Assert.Throws<ArgumentException>(() => Stats.LogMultivariateNormalDensity(
            new double[,] { { 0.0, 0.0 } }, new double[,] { { 0.0, 0.0 } },
            new double[1, 2, 2] { { { 1.0, 2.0 }, { 2.0, 1.0 } } }, CovarianceType.Full));
        Assert.Equal("'covars' must be symmetric, positive-definite", thrown.Message);
    }

    /// <summary>Tied uses its one matrix for every component: the tied density equals the full density with the same
    /// matrix repeated for every component (invariant, hand-made data, C2).</summary>
    [Fact]
    public void A_tied_matrix_serves_every_component_like_the_same_full_matrix_repeated()
    {
        var x = new double[,] { { 0.5, -1.25 }, { 3.0, 0.25 } };
        var means = new double[,] { { 0.0, 1.0 }, { 1.0, -1.0 }, { -1.0, 0.5 } };
        var tied = new double[1, 2, 2] { { { 2.0, 0.5 }, { 0.5, 3.0 } } };
        var repeated = new double[3, 2, 2]
        {
            { { 2.0, 0.5 }, { 0.5, 3.0 } },
            { { 2.0, 0.5 }, { 0.5, 3.0 } },
            { { 2.0, 0.5 }, { 0.5, 3.0 } }
        };
        Close.Equal(Stats.LogMultivariateNormalDensity(x, means, repeated, CovarianceType.Full),
            Stats.LogMultivariateNormalDensity(x, means, tied, CovarianceType.Tied), "tied vs repeated full");
    }

    /// <summary>The tied path shares the full retry: a singular tied matrix gives the same density as the same
    /// singular matrix repeated for every component (invariant, hand-made data, C2).</summary>
    [Fact]
    public void A_singular_tied_matrix_is_retried_like_full()
    {
        var x = new double[,] { { 0.0, 1.0 } };
        var means = new double[,] { { 0.0, 1.0 }, { 1.0, 1.0 } };
        var tied = new double[1, 2, 2] { { { 1.0, 1.0 }, { 1.0, 1.0 } } };
        var full = new double[2, 2, 2]
        {
            { { 1.0, 1.0 }, { 1.0, 1.0 } },
            { { 1.0, 1.0 }, { 1.0, 1.0 } }
        };
        Close.Equal(Stats.LogMultivariateNormalDensity(x, means, full, CovarianceType.Full),
            Stats.LogMultivariateNormalDensity(x, means, tied, CovarianceType.Tied), "singular tied vs full");
    }

    /// <summary>The method is pure: the retry with 1e-7 on the diagonal must not modify the caller's covars (hand-made
    /// singular case, constraint "static, pure method").</summary>
    [Fact]
    public void The_inputs_are_not_modified_in_place()
    {
        var x = new double[,] { { 0.0, 1.0 } };
        var means = new double[,] { { 0.0, 1.0 } };
        var covars = new double[1, 2, 2] { { { 1.0, 1.0 }, { 1.0, 1.0 } } };
        Stats.LogMultivariateNormalDensity(x, means, covars, CovarianceType.Full);
        Close.Equal(new double[1, 2, 2] { { { 1.0, 1.0 }, { 1.0, 1.0 } } }, covars, "covars after the call");
        Close.Equal(new double[,] { { 0.0, 1.0 } }, x, "x after the call");
        Close.Equal(new double[,] { { 0.0, 1.0 } }, means, "means after the call");
    }

    private static void RunGoldenCases(CovarianceType covarianceType)
    {
        var selected = Golden.Load("stats").GetProperty("cases").Items()
            .Where(c => Enum.Parse<CovarianceType>(c.GetProperty("covariance_type").AsString(), ignoreCase: true) == covarianceType)
            .ToList();
        Assert.NotEmpty(selected);
        foreach (var c in selected)
        {
            var name = c.GetProperty("name").AsString();
            var x = c.GetProperty("X").AsMatrix();
            var means = c.GetProperty("means").AsMatrix();
            var covars = c.GetProperty("covars").AsTensor3();
            if (c.Has("raises") && c.GetProperty("raises").AsBool())
            {
                var thrown = Assert.Throws<ArgumentException>(
                    () => Stats.LogMultivariateNormalDensity(x, means, covars, covarianceType));
                Assert.Equal("'covars' must be symmetric, positive-definite", thrown.Message);
            }
            else
                Close.Equal(c.GetProperty("expected").AsMatrix(),
                    Stats.LogMultivariateNormalDensity(x, means, covars, covarianceType), name);
        }
    }
}
