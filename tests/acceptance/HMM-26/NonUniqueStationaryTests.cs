using HmmLearn.Tests.Acceptance.Hmm9;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm26;

/// <summary>
/// Acceptance tests of ticket HMM-26: when the stationary distribution is not unique, GetStationaryDistribution returns the
/// stationary distribution of the closed class with the lowest state index (0 elsewhere), as in the non_unique_cases of
/// oracle/golden/stationary.json: finite, non-negative, sum 1, fixed point. The unique periodic matrix and the immutability
/// of TransMat are checked too.
/// </summary>
public class NonUniqueStationaryTests
{
    /// <summary>Every non-unique case of the oracle.</summary>
    public static IEnumerable<object[]> NonUniqueCases =>
        Golden.Load("stationary").GetProperty("non_unique_cases").Items()
            .Select(c => new object[] { c.GetProperty("name").AsString() });

    private static StubBaseHmm ModelOf(double[,] transMat) =>
        new(transMat.GetLength(0), 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "", Implementation.Log)
        { TransMat = transMat };

    [Theory]
    [MemberData(nameof(NonUniqueCases))]
    public void non_unique_case_reproduces_the_oracle(string name)
    {
        var c = Assert.Single(Golden.Load("stationary").GetProperty("non_unique_cases").Items(),
            e => e.GetProperty("name").AsString() == name);
        var transMat = c.GetProperty("transmat").AsMatrix();
        var expected = c.GetProperty("expected").AsVector();

        var stationary = ModelOf(transMat).GetStationaryDistribution();

        Close.Equal(expected, stationary, name + " stationary");
    }

    [Theory]
    [MemberData(nameof(NonUniqueCases))]
    public void non_unique_case_is_a_finite_non_negative_fixed_point_summing_to_one(string name)
    {
        var c = Assert.Single(Golden.Load("stationary").GetProperty("non_unique_cases").Items(),
            e => e.GetProperty("name").AsString() == name);
        var transMat = c.GetProperty("transmat").AsMatrix();
        var nc = transMat.GetLength(0);

        var stationary = ModelOf(transMat).GetStationaryDistribution();

        var sum = 0.0;
        for (var i = 0; i < nc; i++)
        {
            Assert.False(double.IsNaN(stationary[i]) || double.IsInfinity(stationary[i]), $"{name} entry {i} is not finite");
            Assert.True(stationary[i] >= 0.0, $"{name} entry {i} is negative: {stationary[i]}");
            sum += stationary[i];
        }
        Close.Equal(1.0, sum, name + " sum");
        for (var i = 0; i < nc; i++)
        {
            var product = 0.0;
            for (var j = 0; j < nc; j++)
                product += stationary[j] * transMat[j, i];
            Close.Equal(stationary[i], product, $"{name} invariant[{i}]");
        }
    }

    [Fact]
    public void unique_periodic_matrix_gives_the_uniform_distribution()
    {
        var stationary = ModelOf(new[,] { { 0.0, 1.0 }, { 1.0, 0.0 } }).GetStationaryDistribution();

        Close.Equal(new[] { 0.5, 0.5 }, stationary, "periodic stationary");
    }

    [Fact]
    public void trans_mat_is_left_unchanged()
    {
        var transMat = new[,] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.3, 0.3, 0.4 } };
        var copy = (double[,])transMat.Clone();
        var model = ModelOf(transMat);

        model.GetStationaryDistribution();

        Assert.Same(transMat, model.TransMat);
        Assert.Equal(copy, transMat);
    }
}
