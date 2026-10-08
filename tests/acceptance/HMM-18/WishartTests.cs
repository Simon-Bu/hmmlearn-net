using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm18;

/// <summary>
/// Acceptance tests of KlDivergence.KlWishartDistribution (ticket HMM-18, criterion C4): it reproduces every wishart case
/// of oracle/golden/kl.json (1, 2 and 3 dimensions, including the example of hmmlearn's test), is 0 for equal arguments
/// (within 1e-9) and strictly positive for distinct ones, and throws ArgumentException for scale matrices of different
/// sizes.
/// </summary>
public class WishartTests
{
    public static IEnumerable<object[]> GoldenCases =>
        Golden.Load("kl").GetProperty("wishart_cases").Items()
            .Select(c => new object[] { c.GetProperty("name").AsString() });

    private static JsonElement Case(string name) =>
        Assert.Single(Golden.Load("kl").GetProperty("wishart_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    [Theory]
    [MemberData(nameof(GoldenCases))]
    public void kl_wishart_matches_the_oracle(string name)
    {
        var c = Case(name);
        Close.Equal(c.GetProperty("expected").AsDouble(),
            KlDivergence.KlWishartDistribution(c.GetProperty("dof_q").AsDouble(), c.GetProperty("scale_q").AsMatrix(),
                c.GetProperty("dof_p").AsDouble(), c.GetProperty("scale_p").AsMatrix()),
            $"wishart case {name}");
    }

    [Fact]
    public void kl_wishart_is_zero_for_equal_arguments()
    {
        const double dof = 5.0;
        var scale = new double[,] { { 2.0, 0.5 }, { 0.5, 1.0 } };
        Close.Equal(0.0, KlDivergence.KlWishartDistribution(dof, scale, dof, (double[,])scale.Clone()), "KL(q, q)", rtol: 0.0, atol: 1e-9);
    }

    [Fact]
    public void kl_wishart_is_positive_for_distinct_arguments()
    {
        var scale = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var kl = KlDivergence.KlWishartDistribution(5.0, scale, 3.0, (double[,])scale.Clone());
        Assert.True(kl > 0.0, $"KL of two Wisharts with different dof = {kl:R} is not positive");
    }

    [Fact]
    public void kl_wishart_rejects_scale_matrices_of_different_sizes()
    {
        var scale2 = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var scale3 = new double[3, 3];
        for (var i = 0; i < 3; i++)
            scale3[i, i] = 1.0;
        Assert.Throws<ArgumentException>(() => KlDivergence.KlWishartDistribution(5.0, scale2, 3.0, scale3));
    }
}
