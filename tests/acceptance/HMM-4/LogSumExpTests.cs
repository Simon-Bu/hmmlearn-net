using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm4;

/// <summary>
/// Acceptance tests for <see cref="Hmmc.LogSumExp"/> and <see cref="Hmmc.LogAddExp"/> (ticket HMM-4, criterion C4):
/// hand-written values only (the golden file holds no expectations for the two helpers), covering log 2 for two zeros,
/// stability around +1000 and -1000, -infinity propagation, and a few exact values on trivial inputs.
/// </summary>
public class LogSumExpTests
{
    /// <summary>Criterion C4: log 2 for two zeros.</summary>
    [Fact]
    public void LogSumExp_of_two_zeros_gives_log_2()
    {
        Close.Equal(Math.Log(2.0), Hmmc.LogSumExp(new double[] { 0.0, 0.0 }), "logsumexp([0, 0])");
    }

    /// <summary>Criterion C4: log 2 for two zeros, in the pairwise helper as well.</summary>
    [Fact]
    public void LogAddExp_of_two_zeros_gives_log_2()
    {
        Close.Equal(Math.Log(2.0), Hmmc.LogAddExp(0.0, 0.0), "logaddexp(0, 0)");
    }

    /// <summary>Criterion C4: the helper survives values around 1000 (no overflow) and around -1000 (no underflow to a
    /// wrong answer): the stable formula max + log(1 + exp(-|a - b|)) gives the expected values, computed from the same
    /// formula, not hardcoded.</summary>
    [Fact]
    public void LogSumExp_survives_values_around_1000_and_minus_1000()
    {
        var nearBig = 1000.0 + Math.Log(1.0 + Math.Exp(-1.0)); // (1000, 999)
        var big = Hmmc.LogSumExp(new double[] { 1000.0, 999.0 });
        Close.Equal(nearBig, big, "logsumexp([1000, 999])");
        Assert.True(double.IsFinite(big), "no overflow around 1000");

        Close.Equal(1000.0 + Math.Log(2.0), Hmmc.LogSumExp(new double[] { 1000.0, 1000.0 }), "logsumexp([1000, 1000])");

        var nearSmall = -999.0 + Math.Log(1.0 + Math.Exp(-1.0)); // (-1000, -999): the maximum is -999
        var small = Hmmc.LogSumExp(new double[] { -1000.0, -999.0 });
        Close.Equal(nearSmall, small, "logsumexp([-1000, -999])");
        Assert.True(double.IsFinite(small), "no underflow around -1000");

        Close.Equal(-1000.0 + Math.Log(2.0), Hmmc.LogSumExp(new double[] { -1000.0, -1000.0 }), "logsumexp([-1000, -1000])");
    }

    /// <summary>Criterion C4: the pairwise helper survives values around 1000 and -1000, symmetrically in both argument
    /// orders.</summary>
    [Fact]
    public void LogAddExp_survives_values_around_1000_and_minus_1000()
    {
        var nearBig = 1000.0 + Math.Log(1.0 + Math.Exp(-1.0));
        var nearSmall = -999.0 + Math.Log(1.0 + Math.Exp(-1.0)); // (-1000, -999): the maximum is -999
        Close.Equal(nearBig, Hmmc.LogAddExp(1000.0, 999.0), "logaddexp(1000, 999)");
        Close.Equal(nearBig, Hmmc.LogAddExp(999.0, 1000.0), "logaddexp(999, 1000)");
        Close.Equal(1000.0 + Math.Log(2.0), Hmmc.LogAddExp(1000.0, 1000.0), "logaddexp(1000, 1000)");
        Close.Equal(nearSmall, Hmmc.LogAddExp(-1000.0, -999.0), "logaddexp(-1000, -999)");
        Close.Equal(nearSmall, Hmmc.LogAddExp(-999.0, -1000.0), "logaddexp(-999, -1000)");
        Close.Equal(-1000.0 + Math.Log(2.0), Hmmc.LogAddExp(-1000.0, -1000.0), "logaddexp(-1000, -1000)");
    }

    /// <summary>Criterion C4: when one argument is -infinity the result is exactly the other argument (the C++ returns it
    /// as is, so exact equality is safe).</summary>
    [Theory]
    [InlineData(3.5)]
    [InlineData(0.0)]
    [InlineData(-1000.5)]
    public void LogAddExp_returns_the_other_argument_when_one_is_minus_infinity(double x)
    {
        Assert.True(Hmmc.LogAddExp(double.NegativeInfinity, x).Equals(x), "logaddexp(-inf, x) must be exactly x");
        Assert.True(Hmmc.LogAddExp(x, double.NegativeInfinity).Equals(x), "logaddexp(x, -inf) must be exactly x");
    }

    /// <summary>Criterion C4: -infinity when all inputs are -infinity (the maximum, when infinite, is returned as is; a
    /// missing guard would give NaN through exp(-inf - -inf)).</summary>
    [Fact]
    public void All_minus_infinity_inputs_give_minus_infinity()
    {
        Assert.Equal(double.NegativeInfinity, Hmmc.LogAddExp(double.NegativeInfinity, double.NegativeInfinity));
        Assert.Equal(double.NegativeInfinity, Hmmc.LogSumExp(new double[] { double.NegativeInfinity }));
        Assert.Equal(double.NegativeInfinity,
            Hmmc.LogSumExp(new double[] { double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity }));
    }

    /// <summary>The stub specifies "the maximum, when infinite, is returned as is": a +infinity maximum comes back as is.
    /// (Assumption noted in .factory/out/tests.json.)</summary>
    [Fact]
    public void LogSumExp_returns_the_maximum_when_it_is_positive_infinity()
    {
        Assert.Equal(double.PositiveInfinity, Hmmc.LogSumExp(new double[] { 0.5, double.PositiveInfinity }));
    }

    /// <summary>Trivial exact inputs: a single value is its own log-sum-exp (exactly), and on small values the helper really
    /// computes log(sum(exp(v))) (0.5 + 0.25 + 0.75 = 1.5).</summary>
    [Fact]
    public void Trivial_inputs_are_exact()
    {
        Assert.True(Hmmc.LogSumExp(new double[] { 1.5 }).Equals(1.5), "a single value is its own logsumexp");
        Close.Equal(Math.Log(1.5),
            Hmmc.LogSumExp(new double[] { Math.Log(0.5), Math.Log(0.25), Math.Log(0.75) }),
            "logsumexp(log([0.5, 0.25, 0.75])) = log(1.5)");
    }
}
