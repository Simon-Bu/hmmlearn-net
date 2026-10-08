using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm18;

/// <summary>
/// Acceptance tests of KlDivergence.KlDirichlet (ticket HMM-18, criterion C1): it reproduces every dirichlet case of
/// oracle/golden/kl.json, is 0 for equal arguments (within 1e-9) and strictly positive for distinct ones, and rejects
/// parameter vectors of different lengths with ArgumentException. The numbers come from the oracle only; the hand-written
/// inputs test the invariants (zero for equal arguments, positivity, dimension check) with no expected value beyond them.
/// </summary>
public class DirichletTests
{
    public static IEnumerable<object[]> GoldenCases =>
        Golden.Load("kl").GetProperty("dirichlet_cases").Items()
            .Select(c => new object[] { c.GetProperty("name").AsString() });

    private static JsonElement Case(string name) =>
        Assert.Single(Golden.Load("kl").GetProperty("dirichlet_cases").Items(),
            c => c.GetProperty("name").AsString() == name);

    [Theory]
    [MemberData(nameof(GoldenCases))]
    public void kl_dirichlet_matches_the_oracle(string name)
    {
        var c = Case(name);
        Close.Equal(c.GetProperty("expected").AsDouble(),
            KlDivergence.KlDirichlet(c.GetProperty("q").AsVector(), c.GetProperty("p").AsVector()),
            $"dirichlet case {name}");
    }

    [Fact]
    public void kl_dirichlet_is_zero_for_equal_arguments()
    {
        var q = new[] { 1.5, 2.5, 0.75 };
        Close.Equal(0.0, KlDivergence.KlDirichlet(q, (double[])q.Clone()), "KL(q, q)", rtol: 0.0, atol: 1e-9);
    }

    [Fact]
    public void kl_dirichlet_is_positive_for_distinct_arguments()
    {
        var q = new[] { 1.0, 2.0, 3.0, 4.0 };
        var p = new[] { 4.0, 3.0, 2.0, 1.0 };
        var klQp = KlDivergence.KlDirichlet(q, p);
        var klPq = KlDivergence.KlDirichlet(p, q);
        Assert.True(klQp > 0.0, $"KL(q, p) = {klQp:R} is not positive");
        Assert.True(klPq > 0.0, $"KL(p, q) = {klPq:R} is not positive");
    }

    [Theory]
    [InlineData(2, 3)]
    [InlineData(3, 2)]
    public void kl_dirichlet_rejects_vectors_of_different_lengths(int qLength, int pLength)
    {
        Assert.Throws<ArgumentException>(() => KlDivergence.KlDirichlet(VectorOfOnes(qLength), VectorOfOnes(pLength)));
    }

    private static double[] VectorOfOnes(int length)
    {
        var vector = new double[length];
        for (var i = 0; i < length; i++)
            vector[i] = 1.0;
        return vector;
    }
}
