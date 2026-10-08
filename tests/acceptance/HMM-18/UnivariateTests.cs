using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm18;

/// <summary>
/// Acceptance tests of KlDivergence.KlNormalDistribution and KlDivergence.KlGammaDistribution (ticket HMM-18, criterion C2):
/// each reproduces every normal case and gamma case of oracle/golden/kl.json, and both are 0 for equal arguments (within
/// 1e-9) and strictly positive for distinct ones. The numbers come from the oracle only; the one exact numeric assertion
/// uses trivial inputs (ones and a shift of one) for which every intermediate is a power of two.
/// </summary>
public class UnivariateTests
{
    public static IEnumerable<object[]> NormalCases =>
        Golden.Load("kl").GetProperty("normal_cases").Items()
            .Select(c => new object[] { c.GetProperty("name").AsString() });

    public static IEnumerable<object[]> GammaCases =>
        Golden.Load("kl").GetProperty("gamma_cases").Items()
            .Select(c => new object[] { c.GetProperty("name").AsString() });

    private static JsonElement NormalCase(string name) =>
        Assert.Single(Golden.Load("kl").GetProperty("normal_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    private static JsonElement GammaCase(string name) =>
        Assert.Single(Golden.Load("kl").GetProperty("gamma_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    [Theory]
    [MemberData(nameof(NormalCases))]
    public void kl_normal_distribution_matches_the_oracle(string name)
    {
        var c = NormalCase(name);
        Close.Equal(c.GetProperty("expected").AsDouble(),
            KlDivergence.KlNormalDistribution(c.GetProperty("mean_q").AsDouble(), c.GetProperty("variance_q").AsDouble(),
                c.GetProperty("mean_p").AsDouble(), c.GetProperty("variance_p").AsDouble()),
            $"normal case {name}");
    }

    [Theory]
    [MemberData(nameof(GammaCases))]
    public void kl_gamma_distribution_matches_the_oracle(string name)
    {
        var c = GammaCase(name);
        Close.Equal(c.GetProperty("expected").AsDouble(),
            KlDivergence.KlGammaDistribution(c.GetProperty("b_q").AsDouble(), c.GetProperty("c_q").AsDouble(),
                c.GetProperty("b_p").AsDouble(), c.GetProperty("c_p").AsDouble()),
            $"gamma case {name}");
    }

    [Fact]
    public void kl_normal_is_zero_for_equal_arguments()
    {
        Close.Equal(0.0, KlDivergence.KlNormalDistribution(0.0, 1.0, 0.0, 1.0), "KL(N(0,1), N(0,1))", rtol: 0.0, atol: 1e-9);
    }

    [Fact]
    public void kl_normal_is_positive_for_distinct_arguments()
    {
        var kl = KlDivergence.KlNormalDistribution(0.0, 1.0, 1.0, 1.0);
        Assert.True(kl > 0.0, $"KL(N(0,1), N(1,1)) = {kl:R} is not positive");
    }

    [Fact]
    public void kl_normal_of_the_unit_shift_is_exactly_half()
    {
        // log(1/1)/2 + ((0 - 1)^2 + 1) / (2 * 1) - 1/2 = 0 + 1 - 1/2: every operand and intermediate is a power of two,
        // so any faithful implementation of the formula returns 0.5 exactly.
        Assert.Equal(0.5, KlDivergence.KlNormalDistribution(0.0, 1.0, 1.0, 1.0));
    }

    [Fact]
    public void kl_gamma_is_zero_for_equal_arguments()
    {
        Close.Equal(0.0, KlDivergence.KlGammaDistribution(1.0, 0.01, 1.0, 0.01), "KL(Gamma(1,0.01), Gamma(1,0.01))", rtol: 0.0, atol: 1e-9);
    }

    [Fact]
    public void kl_gamma_is_positive_for_distinct_arguments()
    {
        var otherShape = KlDivergence.KlGammaDistribution(1.0, 0.01, 2.0, 0.01);
        Assert.True(otherShape > 0.0, $"KL with other shape = {otherShape:R} is not positive");
        var otherRate = KlDivergence.KlGammaDistribution(1.0, 0.01, 1.0, 0.02);
        Assert.True(otherRate > 0.0, $"KL with other rate = {otherRate:R} is not positive");
    }
}
