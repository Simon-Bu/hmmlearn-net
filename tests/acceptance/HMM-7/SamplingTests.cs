using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm7;

/// <summary>
/// Acceptance tests for <see cref="AbstractHmm.Sample"/> (ticket HMM-7, criterion C3): it returns nSamples observations and
/// states with the right shape, the states follow StartProb and TransMat (statistical, never against numbers), the given
/// currState is the first state, sampling continues from the previous last state, the same seed gives the same sample, and
/// out-of-range nSamples/currState throw ArgumentOutOfRangeException. The stub's emission of state s is the single observation
/// s, so the observations must echo the states. All random draws come from fixed seeds passed by the tests.
/// </summary>
public class SamplingTests
{
    private static int NComponents => 3;

    private static double[] StartProb => new double[] { 0.5, 0.3, 0.2 };

    private static double[,] TransMat => new double[,]
    {
        { 0.6, 0.2, 0.2 },
        { 0.2, 0.5, 0.3 },
        { 0.1, 0.3, 0.6 },
    };

    private static StubHmm MakeSamplingModel()
    {
        var model = new StubHmm(NComponents, Implementation.Log);
        model.StartProb = (double[])StartProb.Clone();
        model.TransMat = (double[,])TransMat.Clone();
        return model;
    }

    /// <summary>Sample returns nSamples observations and states: the shape is (nSamples, 1) for the stub (one observation per
    /// sample), the states are in range, and every observation is exactly its own state.</summary>
    [Fact]
    public void Sample_returns_n_samples_observations_and_states()
    {
        const int nSamples = 37;
        var (x, states) = MakeSamplingModel().Sample(nSamples, random: new Random(12345));
        Assert.Equal(nSamples, x.GetLength(0));
        Assert.Equal(1, x.GetLength(1));
        Assert.Equal(nSamples, states.Length);
        for (var i = 0; i < nSamples; i++)
        {
            Assert.InRange(states[i], 0, NComponents - 1);
            Close.Equal((double)states[i], x[i, 0], $"sample {i}: the observation must be its own state");
        }
    }

    /// <summary>The first state of each sample is drawn from StartProb: 20000 independent single samples (independent seeds) give
    /// frequencies within 5 standard deviations of the start probabilities, which the test computes from its own inputs.</summary>
    [Fact]
    public void Sample_draws_the_first_state_from_StartProb()
    {
        const int n = 20000;
        var startProb = StartProb;
        var counts = new int[NComponents];
        for (var i = 0; i < n; i++)
        {
            var (_, states) = MakeSamplingModel().Sample(1, random: new Random(987654 + i));
            counts[states[0]]++;
        }
        for (var s = 0; s < NComponents; s++)
        {
            var p = startProb[s];
            var frequency = counts[s] / (double)n;
            var bound = 5.0 * Math.Sqrt(p * (1.0 - p) / n);
            Assert.True(Math.Abs(frequency - p) <= bound,
                $"state {s}: first-state frequency {frequency} is more than 5 standard deviations from {p} (bound {bound}, n {n})");
        }
    }

    /// <summary>The states of one long chain follow TransMat: for every source state the conditional transition frequencies are
    /// within 5 standard deviations of the transition-matrix row, with the bounds computed from the test's own inputs and the
    /// observed number of departures from each state.</summary>
    [Fact]
    public void Sample_draws_transitions_from_TransMat()
    {
        const int nSamples = 60000;
        var transMat = TransMat;
        var (_, states) = MakeSamplingModel().Sample(nSamples, random: new Random(12345));
        var departures = new int[NComponents];
        var counts = new int[NComponents, NComponents];
        for (var t = 1; t < states.Length; t++)
        {
            counts[states[t - 1], states[t]]++;
            departures[states[t - 1]]++;
        }
        for (var i = 0; i < NComponents; i++)
        {
            Assert.True(departures[i] > 1000,
                $"state {i} was a source state only {departures[i]} times in a {nSamples}-sample chain; the statistical check cannot run");
            for (var j = 0; j < NComponents; j++)
            {
                var p = transMat[i, j];
                var frequency = counts[i, j] / (double)departures[i];
                var bound = 5.0 * Math.Sqrt(p * (1.0 - p) / departures[i]);
                Assert.True(Math.Abs(frequency - p) <= bound,
                    $"state {i} -> {j}: transition frequency {frequency} is more than 5 standard deviations from {p} (bound {bound}, departures {departures[i]})");
            }
        }
    }

    /// <summary>The given currState is the first state of the sample, and its observation is emitted from that state.</summary>
    [Fact]
    public void Sample_honours_the_given_currState()
    {
        var (x, states) = MakeSamplingModel().Sample(9, random: new Random(12345), currState: 2);
        Assert.Equal(2, states[0]);
        Assert.Equal(2.0, x[0, 0]);
    }

    /// <summary>Sampling continues from the previous last state, like the reference test_generate_samples: a fresh Sample(10) with
    /// seed 42 followed by a Sample(10, currState: last) with the same seed gives the same first state (the previous last) and
    /// continues the chain.</summary>
    [Fact]
    public void Sample_continues_from_the_previous_last_state()
    {
        var model = MakeSamplingModel();
        var (_, z0) = model.Sample(10, random: new Random(42));
        var (_, z1) = model.Sample(10, random: new Random(42), currState: z0[^1]);
        Assert.Equal(z0[^1], z1[0]);
        // The draws are consumed in order from the generator: a 20-sample draw from the same seed starts like the 10-sample one.
        // (The continuation z1 is NOT the tail of the 20-sample draw: with a given currState the first uniform number is used
        // for the second state, so it follows a different stream.)
        var (_, z20) = model.Sample(20, random: new Random(42));
        for (var i = 0; i < 10; i++)
            Assert.True(z0[i] == z20[i], $"state {i}: a 20-sample draw must start like the 10-sample draw of the same seed");
        Assert.Equal(10, z1.Length);
        foreach (var state in z1)
            Assert.InRange(state, 0, NComponents - 1);
    }

    /// <summary>The same random argument seed gives the same sample: two fresh Random(12345) give identical states and
    /// observations.</summary>
    [Fact]
    public void The_same_random_argument_seed_gives_the_same_sample()
    {
        var (x1, z1) = MakeSamplingModel().Sample(15, random: new Random(12345));
        var (x2, z2) = MakeSamplingModel().Sample(15, random: new Random(12345));
        Close.Equal(x1, x2, "observations of two same-seed samples");
        Assert.True(z1.SequenceEqual(z2), "states of two same-seed samples");
    }

    /// <summary>A model with a RandomState seed gives the same sample from call to call (a fresh generator seeded with the value
    /// per call), and the same sample as an explicit new Random with that seed.</summary>
    [Fact]
    public void The_RandomState_seed_gives_the_same_sample()
    {
        const int seed = 12345;
        var model = MakeSamplingModel();
        model.RandomState = seed;
        var (x1, z1) = model.Sample(15);
        var (x2, z2) = model.Sample(15);
        Close.Equal(x1, x2, "observations of two calls on the same RandomState model");
        Assert.True(z1.SequenceEqual(z2), "states of two calls on the same RandomState model");
        var (x3, z3) = model.Sample(15, random: new Random(seed));
        Close.Equal(x1, x3, "observations of the RandomState model and an explicit generator with the same seed");
        Assert.True(z1.SequenceEqual(z3), "states of the RandomState model and an explicit generator with the same seed");
    }

    /// <summary>nSamples below 1 throws ArgumentOutOfRangeException.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void Sample_throws_argument_out_of_range_for_bad_n_samples(int nSamples)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MakeSamplingModel().Sample(nSamples, random: new Random(12345)));
    }

    /// <summary>A currState outside 0..NComponents-1 throws ArgumentOutOfRangeException.</summary>
    [Theory]
    [InlineData(-1)]
    [InlineData(3)]
    public void Sample_throws_argument_out_of_range_for_a_curr_state_outside_the_states(int currState)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MakeSamplingModel().Sample(5, random: new Random(12345), currState: currState));
    }

    /// <summary>Sample throws InvalidOperationException while StartProb is null (the model is neither set nor fitted).</summary>
    [Fact]
    public void Sample_throws_invalid_operation_when_startprob_is_null()
    {
        var model = new StubHmm(NComponents, Implementation.Log);
        Assert.Throws<InvalidOperationException>(() => model.Sample(5, random: new Random(12345)));
    }

    /// <summary>The stub model of the ticket: Check does nothing and the emission of state s is the single observation s, so
    /// every observation must echo its own state.</summary>
    private sealed class StubHmm : AbstractHmm
    {
        public StubHmm(int nComponents, Implementation implementation)
            : base(nComponents, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "st", "st", implementation)
        {
        }

        protected override void Check()
        {
        }

        protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

        protected override double[] GenerateSampleFromState(int state, Random random) => new double[] { state };
    }
}
