using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm25;

/// <summary>
/// Acceptance tests for ticket HMM-25: <see cref="Hmmc.ComputeLogXiSum"/> and <see cref="Hmmc.ComputeScalingXiSum"/> on a
/// sequence without samples (all lattices and frame probabilities are (0, nc)). The sum over t = 0..n_samples-2 is empty, so
/// the log version returns a matrix of -infinity (exact: it is the neutral element) and the scaling version a matrix of zeros.
/// </summary>
public class ZeroSampleXiSumTests
{
    private static double[,] Uniform(int nc)
    {
        var m = new double[nc, nc];
        for (var i = 0; i < nc; i++)
            for (var j = 0; j < nc; j++)
                m[i, j] = 1.0 / nc;
        return m;
    }

    /// <summary>With 0 samples the log xi sum is a new (nc, nc) matrix of exactly -infinity and does not throw.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Log_xi_sum_on_zero_samples_is_a_matrix_of_minus_infinity(int nc)
    {
        var result = Hmmc.ComputeLogXiSum(new double[0, nc], Uniform(nc), new double[0, nc], new double[0, nc]);

        Assert.Equal(nc, result.GetLength(0));
        Assert.Equal(nc, result.GetLength(1));
        foreach (var value in result)
            Assert.Equal(double.NegativeInfinity, value);
    }

    /// <summary>Each call returns a fresh matrix, never a shared instance.</summary>
    [Fact]
    public void Log_xi_sum_on_zero_samples_returns_a_new_matrix_each_call()
    {
        var first = Hmmc.ComputeLogXiSum(new double[0, 2], Uniform(2), new double[0, 2], new double[0, 2]);
        var second = Hmmc.ComputeLogXiSum(new double[0, 2], Uniform(2), new double[0, 2], new double[0, 2]);

        Assert.NotSame(first, second);
    }

    /// <summary>0 samples gives the same result as one sample: both are the neutral matrix of -infinity.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Log_xi_sum_on_zero_samples_equals_the_one_sample_result(int nc)
    {
        var one = new double[1, nc];
        var zero = Hmmc.ComputeLogXiSum(new double[0, nc], Uniform(nc), new double[0, nc], new double[0, nc]);
        var single = Hmmc.ComputeLogXiSum(one, Uniform(nc), one, one);

        Assert.Equal(single, zero);
    }

    /// <summary>With 0 samples the scaling xi sum is an (nc, nc) matrix of zeros (unchanged behavior).</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Scaling_xi_sum_on_zero_samples_is_a_matrix_of_zeros(int nc)
    {
        var result = Hmmc.ComputeScalingXiSum(new double[0, nc], Uniform(nc), new double[0, nc], new double[0, nc]);

        Assert.Equal(nc, result.GetLength(0));
        Assert.Equal(nc, result.GetLength(1));
        foreach (var value in result)
            Assert.Equal(0.0, value);
    }

    /// <summary>0 samples is not a shape error by itself, but real shape errors still throw ArgumentException.</summary>
    [Fact]
    public void Shape_errors_on_zero_samples_still_throw_ArgumentException()
    {
        // fwdlattice (0, 3) with frame probabilities (0, 2)
        Assert.Throws<ArgumentException>(() =>
            Hmmc.ComputeLogXiSum(new double[0, 3], Uniform(2), new double[0, 2], new double[0, 2]));
        Assert.Throws<ArgumentException>(() =>
            Hmmc.ComputeScalingXiSum(new double[0, 3], Uniform(2), new double[0, 2], new double[0, 2]));

        // fwdlattice (1, 2) with frame probabilities (0, 2)
        Assert.Throws<ArgumentException>(() =>
            Hmmc.ComputeLogXiSum(new double[1, 2], Uniform(2), new double[0, 2], new double[0, 2]));
        Assert.Throws<ArgumentException>(() =>
            Hmmc.ComputeScalingXiSum(new double[1, 2], Uniform(2), new double[0, 2], new double[0, 2]));

        // bwdlattice and transmat with the wrong shape
        Assert.Throws<ArgumentException>(() =>
            Hmmc.ComputeLogXiSum(new double[0, 2], Uniform(2), new double[1, 2], new double[0, 2]));
        Assert.Throws<ArgumentException>(() =>
            Hmmc.ComputeLogXiSum(new double[0, 2], Uniform(3), new double[0, 2], new double[0, 2]));
    }
}
