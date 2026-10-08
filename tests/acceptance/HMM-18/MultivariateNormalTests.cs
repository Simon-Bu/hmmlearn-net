using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm18;

/// <summary>
/// Acceptance tests of KlDivergence.KlMultivariateNormalDistribution (ticket HMM-18, criterion C3): it reproduces every
/// multivariate normal case of oracle/golden/kl.json (1, 2 and 3 dimensions), agrees with the univariate formula in one
/// dimension, is 0 for equal arguments (within 1e-9) and strictly positive for distinct ones, and throws ArgumentException
/// for inconsistent dimensions (means of different lengths, covariances of different sizes, a mean whose length does not
/// match its own covariance, a non-square covariance).
/// </summary>
public class MultivariateNormalTests
{
    public static IEnumerable<object[]> GoldenCases =>
        Golden.Load("kl").GetProperty("multivariate_normal_cases").Items()
            .Select(c => new object[] { c.GetProperty("name").AsString() });

    private static JsonElement Case(string name) =>
        Assert.Single(Golden.Load("kl").GetProperty("multivariate_normal_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    [Theory]
    [MemberData(nameof(GoldenCases))]
    public void kl_multivariate_normal_matches_the_oracle(string name)
    {
        var c = Case(name);
        Close.Equal(c.GetProperty("expected").AsDouble(),
            KlDivergence.KlMultivariateNormalDistribution(c.GetProperty("mean_q").AsVector(), c.GetProperty("covar_q").AsMatrix(),
                c.GetProperty("mean_p").AsVector(), c.GetProperty("covar_p").AsMatrix()),
            $"multivariate normal case {name}");
    }

    [Fact]
    public void kl_multivariate_normal_agrees_with_the_univariate_formula_in_one_dimension()
    {
        const double meanQ = 0.3, varianceQ = 0.8;
        const double meanP = -0.2, varianceP = 1.5;
        var univariate = KlDivergence.KlNormalDistribution(meanQ, varianceQ, meanP, varianceP);
        var multivariate = KlDivergence.KlMultivariateNormalDistribution(new[] { meanQ }, new[,] { { varianceQ } }, new[] { meanP }, new[,] { { varianceP } });
        Close.Equal(univariate, multivariate, "1-D multivariate normal vs univariate normal");
    }

    [Fact]
    public void kl_multivariate_normal_agrees_with_the_univariate_formula_on_the_oracle_one_dimension_case()
    {
        var c = Case("one-dimension");
        var meanQ = c.GetProperty("mean_q").AsVector();
        var covarQ = c.GetProperty("covar_q").AsMatrix();
        var meanP = c.GetProperty("mean_p").AsVector();
        var covarP = c.GetProperty("covar_p").AsMatrix();
        var univariate = KlDivergence.KlNormalDistribution(meanQ[0], covarQ[0, 0], meanP[0], covarP[0, 0]);
        var multivariate = KlDivergence.KlMultivariateNormalDistribution(meanQ, covarQ, meanP, covarP);
        Close.Equal(univariate, multivariate, "oracle one-dimension case vs univariate formula");
    }

    [Fact]
    public void kl_multivariate_normal_is_zero_for_equal_arguments()
    {
        var mean = new[] { 0.5, -0.5 };
        var covar = new double[,] { { 2.0, 0.5 }, { 0.5, 1.0 } };
        Close.Equal(0.0,
            KlDivergence.KlMultivariateNormalDistribution(mean, covar, (double[])mean.Clone(), (double[,])covar.Clone()),
            "KL(q, q)", rtol: 0.0, atol: 1e-9);
    }

    [Fact]
    public void kl_multivariate_normal_is_positive_for_distinct_arguments()
    {
        var identity = Identity(2);
        var kl = KlDivergence.KlMultivariateNormalDistribution(new[] { 1.0, 0.0 }, identity, new[] { 0.0, 0.0 }, (double[,])identity.Clone());
        Assert.True(kl > 0.0, $"KL of shifted 2-D normals = {kl:R} is not positive");
    }

    [Fact]
    public void kl_multivariate_normal_rejects_means_of_different_lengths()
    {
        Assert.Throws<ArgumentException>(() =>
            KlDivergence.KlMultivariateNormalDistribution(new[] { 0.0, 0.0 }, Identity(2), new[] { 0.0, 0.0, 0.0 }, Identity(3)));
    }

    [Fact]
    public void kl_multivariate_normal_rejects_covariances_of_different_sizes()
    {
        Assert.Throws<ArgumentException>(() =>
            KlDivergence.KlMultivariateNormalDistribution(new[] { 0.0, 0.0 }, Identity(2), new[] { 0.0, 0.0, 0.0 }, Identity(3)));
    }

    [Fact]
    public void kl_multivariate_normal_rejects_a_mean_inconsistent_with_its_covariance()
    {
        Assert.Throws<ArgumentException>(() =>
            KlDivergence.KlMultivariateNormalDistribution(new[] { 0.0, 0.0 }, Identity(3), new[] { 0.0 }, Identity(1)));
    }

    [Fact]
    public void kl_multivariate_normal_rejects_a_non_square_covariance()
    {
        var rectangular = new double[2, 3];
        for (var i = 0; i < 2; i++)
            for (var j = 0; j < 3; j++)
                rectangular[i, j] = 1.0;
        Assert.Throws<ArgumentException>(() =>
            KlDivergence.KlMultivariateNormalDistribution(new[] { 0.0, 0.0 }, rectangular, new[] { 0.0, 0.0 }, Identity(2)));
    }

    private static double[,] Identity(int n)
    {
        var result = new double[n, n];
        for (var i = 0; i < n; i++)
            result[i, i] = 1.0;
        return result;
    }
}
