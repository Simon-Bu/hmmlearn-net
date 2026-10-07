using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm9;

/// <summary>
/// Acceptance tests for the validation of <see cref="BaseHmm"/> (ticket HMM-9, criterion C1): Check accepts valid start
/// probabilities and transition matrices and throws ArgumentException when StartProb is null, has the wrong length or does
/// not sum to 1 (with the tolerance of CheckSum1, so a sum off by 5e-6 is accepted and one off by 2e-5 is not), and when
/// TransMat is null, has the wrong shape or a row that does not sum to 1; entries that are negative but still sum to 1 are
/// accepted (hmmlearn's _check only looks at the shape and the sums); and Score on a model with invalid parameters fails the
/// same way. The tests reach the protected hook through the CallCheck member of <see cref="StubBaseHmm"/>, the shared stub
/// model of the ticket.
/// </summary>
public class CheckTests
{
    /// <summary>A model with the given number of components and the given (possibly invalid) parameters; everything else
    /// stays at harmless defaults.</summary>
    private static StubBaseHmm Model(int nComponents, double[]? startProb, double[,]? transMat) =>
        new(nComponents, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "", Implementation.Log)
        { StartProb = startProb, TransMat = transMat };

    [Fact]
    public void check_accepts_valid_startprob_and_transmat()
    {
        var model = Model(2, new[] { 0.5, 0.5 }, new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } });
        model.CallCheck(); // must not throw
    }

    [Fact]
    public void check_throws_when_startprob_is_null()
    {
        var model = Model(2, null, new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } });
        var ex = Assert.Throws<ArgumentException>(() => model.CallCheck());
        Assert.Equal("startprob_ must have length n_components", ex.Message);
    }

    [Fact]
    public void check_throws_when_startprob_has_the_wrong_length()
    {
        var tooShort = Model(3, new[] { 0.5, 0.5 }, new double[,] { { 0.5, 0.5, 0.0 }, { 0.0, 0.5, 0.5 }, { 0.0, 0.0, 1.0 } });
        Assert.Equal("startprob_ must have length n_components", Assert.Throws<ArgumentException>(tooShort.CallCheck).Message);
        var tooLong = Model(3, new[] { 0.25, 0.25, 0.25, 0.25 }, new double[,] { { 0.5, 0.5, 0.0 }, { 0.0, 0.5, 0.5 }, { 0.0, 0.0, 1.0 } });
        Assert.Equal("startprob_ must have length n_components", Assert.Throws<ArgumentException>(tooLong.CallCheck).Message);
    }

    [Fact]
    public void check_throws_when_startprob_does_not_sum_to_1()
    {
        Assert.Throws<ArgumentException>(() => Model(2, new[] { 0.5, 0.7 }, new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } }).CallCheck());
        Assert.Throws<ArgumentException>(() => Model(2, new[] { 0.2, 0.2 }, new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } }).CallCheck());
    }

    [Fact]
    public void check_accepts_sums_within_the_check_sum_1_tolerance()
    {
        // The tolerance of CheckSum1 is 1e-8 + 1e-5: a sum off by 5e-6 is close enough to 1.
        Model(2, new[] { 1 + 5e-6, 0.0 }, new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } }).CallCheck(); // must not throw
        Model(2, new[] { 0.5, 0.5 }, new double[,] { { 0.7 + 5e-6, 0.3 - 5e-6 }, { 0.3, 0.7 } }).CallCheck(); // must not throw
    }

    [Fact]
    public void check_throws_when_sums_are_outside_the_check_sum_1_tolerance()
    {
        // Off by 2e-5, twice the tolerance.
        Assert.Throws<ArgumentException>(() => Model(2, new[] { 1 + 2e-5, 0.0 }, new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } }).CallCheck());
        Assert.Throws<ArgumentException>(() =>
            Model(2, new[] { 0.5, 0.5 }, new double[,] { { 0.7 + 2e-5, 0.3 }, { 0.3, 0.7 } }).CallCheck());
    }

    [Fact]
    public void check_throws_when_transmat_is_null()
    {
        Assert.Throws<ArgumentException>(() => Model(2, new[] { 0.5, 0.5 }, null).CallCheck());
    }

    [Fact]
    public void check_throws_when_transmat_has_the_wrong_shape()
    {
        var startProb = new[] { 0.25, 0.25, 0.25, 0.25 };
        Assert.Throws<ArgumentException>(() => Model(4, startProb, new double[3, 4]).CallCheck());
        Assert.Throws<ArgumentException>(() => Model(4, startProb, new double[4, 3]).CallCheck());
        Assert.Throws<ArgumentException>(() => Model(4, startProb, new double[3, 3]).CallCheck());
    }

    [Fact]
    public void check_throws_when_a_transmat_row_does_not_sum_to_1()
    {
        // only the middle row is off
        Assert.Throws<ArgumentException>(() =>
            Model(3, new[] { 0.4, 0.3, 0.3 }, new double[,] { { 0.2, 0.3, 0.5 }, { 0.4, 0.6, 0.2 }, { 0.3, 0.4, 0.3 } }).CallCheck());
        // every row is off
        Assert.Throws<ArgumentException>(() =>
            Model(2, new[] { 0.5, 0.5 }, new double[,] { { 0.7, 0.4 }, { 0.3, 0.7 } }).CallCheck());
    }

    [Fact]
    public void check_accepts_negative_entries_that_sum_to_1()
    {
        // hmmlearn's _check only validates the shape and the sums, not the non-negativity of the entries.
        Model(2, new[] { 1.5, -0.5 }, new double[,] { { 1.5, -0.5 }, { 0.3, 0.7 } }).CallCheck(); // must not throw
    }

    [Fact]
    public void score_with_invalid_parameters_fails_the_same_way()
    {
        var x = new double[,] { { -0.1, -0.2 }, { -0.3, -0.4 } };
        Assert.Throws<ArgumentException>(() =>
            Model(2, new[] { 0.5, 0.3 }, new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } }).Score(x));
        Assert.Throws<ArgumentException>(() =>
            Model(2, new[] { 0.5, 0.5 }, new double[,] { { 0.7, 0.4 }, { 0.3, 0.7 } }).Score(x));
    }
}

/// <summary>
/// The ticket's stub model: a <see cref="BaseHmm"/> whose observations ARE the log frame probabilities (ComputeLogLikelihood
/// returns a copy of x, like the oracle's stub), whose states are the observations (GenerateSampleFromState returns the state
/// itself) and whose number of free scalar parameters per letter is { 's' = nc - 1, 't' = nc * (nc - 1) }; it passes its
/// constructor arguments to <see cref="BaseHmm"/>. The Call* members give the acceptance tests of the ticket access to the
/// protected hooks of BaseHmm.
/// </summary>
internal sealed class StubBaseHmm : BaseHmm
{
    public StubBaseHmm(int nComponents, double startProbPrior, double transMatPrior, DecoderAlgorithm algorithm, int? randomState,
        int nIter, double tol, bool verbose, string @params, string initParams, Implementation implementation)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
    {
    }

    /// <summary>Runs the protected validation hook.</summary>
    public void CallCheck() => Check();

    /// <summary>Runs the protected initialization hook.</summary>
    public void CallInit(double[,] x, int[] lengths) => Init(x, lengths);

    /// <summary>Runs the protected M-step hook.</summary>
    public void CallDoMStep(SufficientStatistics stats) => DoMStep(stats);

    /// <summary>Runs the protected lower-bound hook.</summary>
    public double CallComputeLowerBound(double currLogProb) => ComputeLowerBound(currLogProb);

    protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

    protected override double[] GenerateSampleFromState(int state, Random random) => new[] { (double)state };

    protected override IReadOnlyDictionary<char, int> GetNFitScalarsPerParam()
    {
        var nc = NComponents;
        return new Dictionary<char, int> { ['s'] = nc - 1, ['t'] = nc * (nc - 1) };
    }
}
