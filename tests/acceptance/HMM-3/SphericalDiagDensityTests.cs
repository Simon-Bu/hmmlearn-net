using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm3;

/// <summary>
/// Acceptance tests for the spherical and diag paths of
/// <see cref="Stats.LogMultivariateNormalDensity"/> (ticket HMM-3, criterion C1): every spherical and diag case of
/// oracle/golden/stats.json is reproduced (the floor on the variances and the -infinity overflow are golden cases),
/// plus hand-checked invariants and at most three exact numeric assertions on trivial inputs.
/// </summary>
public class SphericalDiagDensityTests
{
    /// <summary>numpy's finfo(float).tiny, the smallest normal double: variances are raised to at least this.</summary>
    private const double Tiny = 2.2250738585072014E-308;

    /// <summary>Every spherical case of the golden file is reproduced, case by case.</summary>
    [Fact]
    public void Reproduces_every_spherical_case_of_the_golden_file()
        => RunGoldenCases(CovarianceType.Spherical);

    /// <summary>Every diag case of the golden file is reproduced, case by case. In particular
    /// <c>diag-zero-variance-at-the-mean</c> checks the floor (a zero variance at the mean stays finite) and the
    /// overflow (a far-off sample gives "-Infinity").</summary>
    [Fact]
    public void Reproduces_every_diag_case_of_the_golden_file()
        => RunGoldenCases(CovarianceType.Diag);

    /// <summary>The floor on the variances, exact value on a trivial input: a zero variance at the mean is raised to
    /// tiny, so for one feature the density is -0.5 * (log(2 pi) + log(tiny)) (hand-checked, C1).</summary>
    [Fact]
    public void A_zero_variance_at_the_mean_is_raised_to_tiny()
    {
        var actual = Stats.LogMultivariateNormalDensity(
            new double[,] { { 0.0 } }, new double[,] { { 0.0 } },
            new double[1, 1, 1] { { { 0.0 } } }, CovarianceType.Diag)[0, 0];
        Close.Equal(-0.5 * (Math.Log(2.0 * Math.PI) + Math.Log(Tiny)), actual, "zero variance at the mean");
    }

    /// <summary>The floor is "at least tiny": a variance below tiny (1e-308, subnormal) gives the same density as tiny
    /// itself at the mean (invariant, hand-made data).</summary>
    [Fact]
    public void A_variance_below_tiny_gives_the_same_density_as_tiny_at_the_mean()
    {
        var below = Stats.LogMultivariateNormalDensity(
            new double[,] { { 0.0 } }, new double[,] { { 0.0 } },
            new double[1, 1, 1] { { { 1.0E-308 } } }, CovarianceType.Spherical)[0, 0];
        var atFloor = Stats.LogMultivariateNormalDensity(
            new double[,] { { 0.0 } }, new double[,] { { 0.0 } },
            new double[1, 1, 1] { { { Tiny } } }, CovarianceType.Spherical)[0, 0];
        Close.Equal(atFloor, below, "variance below the tiny floor");
    }

    /// <summary>The -infinity overflow case, exact value on a trivial input: (2^512 - 0)^2 = 2^1024 overflows the
    /// double range, so the squared term is +infinity and the density is -infinity, as in numpy (C1).</summary>
    [Fact]
    public void An_overflow_gives_negative_infinity()
    {
        var actual = Stats.LogMultivariateNormalDensity(
            new double[,] { { Math.Pow(2.0, 512) } }, new double[,] { { 0.0 } },
            new double[1, 1, 1] { { { 1.0 } } }, CovarianceType.Diag)[0, 0];
        Close.Equal(double.NegativeInfinity, actual, "overflow case");
    }

    /// <summary>The default covariance type is diag: the three-argument call gives the same result as the explicit diag
    /// call (invariant, hand-made data).</summary>
    [Fact]
    public void The_default_covariance_type_is_diag()
    {
        var x = new double[,] { { 0.5, -1.25 }, { 3.0, 0.25 } };
        var means = new double[,] { { 0.0, 1.0 }, { 1.0, -1.0 } };
        var covars = new double[2, 1, 2] { { { 0.5, 2.0 } }, { { 3.0, 1.0 } } };
        Close.Equal(Stats.LogMultivariateNormalDensity(x, means, covars, CovarianceType.Diag),
            Stats.LogMultivariateNormalDensity(x, means, covars), "default covariance type");
    }

    /// <summary>A spherical variance applies to every feature: the spherical density equals the diag density with the
    /// same variance on every feature (invariant, hand-made data).</summary>
    [Fact]
    public void A_spherical_variance_applies_to_every_feature()
    {
        var x = new double[,] { { 0.5, -1.25 }, { 3.0, 0.25 } };
        var means = new double[,] { { 0.0, 1.0 }, { 1.0, -1.0 } };
        var spherical = new double[2, 1, 1] { { { 0.5 } }, { { 2.0 } } };
        var diag = new double[2, 1, 2] { { { 0.5, 0.5 } }, { { 2.0, 2.0 } } };
        Close.Equal(Stats.LogMultivariateNormalDensity(x, means, diag, CovarianceType.Diag),
            Stats.LogMultivariateNormalDensity(x, means, spherical, CovarianceType.Spherical),
            "spherical vs diag with the same variance per feature");
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
                Assert.Throws<ArgumentException>(() => Stats.LogMultivariateNormalDensity(x, means, covars, covarianceType));
            else
                Close.Equal(c.GetProperty("expected").AsMatrix(),
                    Stats.LogMultivariateNormalDensity(x, means, covars, covarianceType), name);
        }
    }
}
