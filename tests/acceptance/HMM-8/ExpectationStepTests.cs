using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm8;

/// <summary>
/// Acceptance tests for the E-step of <see cref="AbstractHmm.Fit(double[,], int[])"/> (ticket HMM-8, criterion C2): the statistics
/// given to DoMStep match the oracle for both implementations (NObs one per sequence, Start the sum of the first-sample posteriors,
/// Trans the xi sums of the sequences); two copies of the same sequence give twice the statistics of one; sequences of length one
/// add to Start but not to Trans; Start is filled only when 's' is in Params and Trans only when 't' is; the lattice given to
/// AccumulateSufficientStatistics holds probabilities with the Scaling implementation and log probabilities with Log; the recorded
/// posteriors match the oracle posteriors; and a model overriding the fit virtuals (FitStartProb, FitTransMat) sees them in the
/// forward pass and in the xi sums.
/// </summary>
public class ExpectationStepTests
{
    private static JsonElement InferenceCase(string name) =>
        Assert.Single(Golden.Load("base").GetProperty("inference_cases").Items(), c => c.GetProperty("name").AsString() == name);

    private static JsonElement HmmcCase(string name) =>
        Assert.Single(Golden.Load("hmmc").GetProperty("cases").Items(), c => c.GetProperty("name").AsString() == name);

    private sealed record Wikipedia
    {
        public required double[,] X { get; init; }

        public required double[] StartProb { get; init; }

        public required double[,] TransMat { get; init; }

        public required double[,] FrameProb { get; init; }

        public required double[,] LogFrameProb { get; init; }

        public required double[,] ScalingXiSum { get; init; }

        public required double Score { get; init; }

        public required double[,] Posteriors { get; init; }
    }

    private static Wikipedia WikipediaCase()
    {
        var b = InferenceCase("wikipedia");
        var h = HmmcCase("wikipedia");
        var expected = b.GetProperty("expected");
        return new Wikipedia
        {
            X = b.GetProperty("X").AsMatrix(),
            StartProb = b.GetProperty("startprob").AsVector(),
            TransMat = b.GetProperty("transmat").AsMatrix(),
            FrameProb = h.GetProperty("frameprob").AsMatrix(),
            LogFrameProb = h.GetProperty("log_frameprob").AsMatrix(),
            ScalingXiSum = h.GetProperty("scaling_xi_sum").AsMatrix(),
            Score = expected.GetProperty("score").AsDouble(),
            Posteriors = expected.GetProperty("posteriors").AsMatrix(),
        };
    }

    /// <summary>The one-sample case of oracle/golden/hmmc.json (base.json has no such case): a single row of observations with
    /// its own startprob and transmat. The model's observations ARE the log frame probabilities.</summary>
    private sealed record OneSample
    {
        public required double[] StartProb { get; init; }

        public required double[,] TransMat { get; init; }

        public required double[,] FrameProb { get; init; }

        public required double[,] LogFrameProb { get; init; }
    }

    private static OneSample OneSampleCase()
    {
        var h = HmmcCase("one-sample");
        return new OneSample
        {
            StartProb = h.GetProperty("startprob").AsVector(),
            TransMat = h.GetProperty("transmat").AsMatrix(),
            FrameProb = h.GetProperty("frameprob").AsMatrix(),
            LogFrameProb = h.GetProperty("log_frameprob").AsMatrix(),
        };
    }

    private static double[,] Stack(double[,] first, double[,] second)
    {
        var rows = first.GetLength(0) + second.GetLength(0);
        var result = new double[rows, first.GetLength(1)];
        for (var r = 0; r < first.GetLength(0); r++)
            for (var c = 0; c < first.GetLength(1); c++)
                result[r, c] = first[r, c];
        for (var r = 0; r < second.GetLength(0); r++)
            for (var c = 0; c < second.GetLength(1); c++)
                result[first.GetLength(0) + r, c] = second[r, c];
        return result;
    }

    private static double[,] Copy(double[,] matrix)
    {
        var result = new double[matrix.GetLength(0), matrix.GetLength(1)];
        for (var r = 0; r < matrix.GetLength(0); r++)
            for (var c = 0; c < matrix.GetLength(1); c++)
                result[r, c] = matrix[r, c];
        return result;
    }

    private static SufficientStatistics CopyStats(SufficientStatistics stats)
    {
        var copy = new SufficientStatistics(stats.Start.Length);
        copy.NObs = stats.NObs;
        copy.Start = (double[])stats.Start.Clone();
        copy.Trans = (double[,])stats.Trans.Clone();
        return copy;
    }

    /// <summary>A single-iteration model that records the per-sequence arguments of AccumulateSufficientStatistics and the
    /// statistics of the M-step; Init installs the given parameters.</summary>
    private class EstepRecorder : AbstractHmm
    {
        private readonly double[] _initialStartProb;
        private readonly double[,] _initialTransMat;
        private readonly string _params;

        public sealed class SequenceCapture
        {
            public required double[,] X { get; init; }

            public required double[,] Lattice { get; init; }

            public required double[,] Posteriors { get; init; }

            public required double[,] Fwd { get; init; }

            public required double[,] Bwd { get; init; }
        }

        public List<SequenceCapture> Sequences { get; } = new();

        public SufficientStatistics? LastStats { get; private set; }

        public EstepRecorder(Implementation implementation, string @params, double[] initialStartProb, double[,] initialTransMat)
            : base(2, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, @params, "st", implementation)
        {
            _initialStartProb = initialStartProb;
            _initialTransMat = initialTransMat;
            _params = @params;
        }

        protected override void Init(double[,] x, int[] lengths)
        {
            StartProb = (double[])_initialStartProb.Clone();
            TransMat = (double[,])_initialTransMat.Clone();
            NFeatures = x.GetLength(1);
        }

        protected override void Check()
        {
        }

        protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

        protected override double[] GenerateSampleFromState(int state, Random random) => new[] { (double)state };

        // The default ComputeLowerBound throws NotImplementedException: this model monitors the log probability itself.
        protected override double ComputeLowerBound(double currLogProb) => currLogProb;

        protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
            double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice)
        {
            Sequences.Add(new SequenceCapture
            {
                X = Copy(x),
                Lattice = Copy(lattice),
                Posteriors = Copy(posteriors),
                Fwd = Copy(fwdlattice),
                Bwd = Copy(bwdlattice),
            });
            base.AccumulateSufficientStatistics(stats, x, lattice, posteriors, fwdlattice, bwdlattice);
        }

        protected override void DoMStep(SufficientStatistics stats) => LastStats = CopyStats(stats);
    }

    /// <summary>Like EstepRecorder but overriding the fit virtuals: null means fall back to the base implementation.</summary>
    private sealed class FitVirtualOverrides : EstepRecorder
    {
        private readonly double[]? _overrideStartProb;
        private readonly double[,]? _overrideTransMat;

        public FitVirtualOverrides(Implementation implementation, double[] initialStartProb, double[,] initialTransMat,
            double[]? overrideStartProb, double[,]? overrideTransMat)
            : base(implementation, "st", initialStartProb, initialTransMat)
        {
            _overrideStartProb = overrideStartProb;
            _overrideTransMat = overrideTransMat;
        }

        protected override double[] FitStartProb => _overrideStartProb ?? base.FitStartProb;

        protected override double[,] FitTransMat => _overrideTransMat ?? base.FitTransMat;
    }

    /// <summary>The statistics given to DoMStep match the oracle of a single sequence: NObs 1, Start the first-sample posteriors,
    /// Trans the xi sums, and the monitor history holds the oracle score.</summary>
    [Theory]
    [InlineData(Implementation.Scaling)]
    [InlineData(Implementation.Log)]
    public void the_statistics_given_to_do_m_step_match_the_oracle(Implementation implementation)
    {
        var data = WikipediaCase();
        var model = new EstepRecorder(implementation, "st", data.StartProb, data.TransMat);
        model.Fit(data.X);
        Assert.NotNull(model.LastStats);
        var stats = model.LastStats!;
        Assert.Equal(1, stats.NObs);
        var expectedStart = new[] { data.Posteriors[0, 0], data.Posteriors[0, 1] };
        Close.Equal(expectedStart, stats.Start, "Start (sum of the first-sample posteriors)");
        Close.Equal(data.ScalingXiSum, stats.Trans, "Trans (xi sums)");
        Assert.Single(model.Monitor.History);
        Close.Equal(data.Score, model.Monitor.History[0], "lower bound (oracle score)");
        Assert.Single(model.Sequences);
        Close.Equal(data.Posteriors, model.Sequences[0].Posteriors, "recorded posteriors");
    }

    /// <summary>Two copies of the same sequence (lengths [n, n]) give twice the statistics of one copy, for both implementations.</summary>
    [Theory]
    [InlineData(Implementation.Scaling)]
    [InlineData(Implementation.Log)]
    public void two_copies_of_a_sequence_give_twice_the_statistics(Implementation implementation)
    {
        var data = WikipediaCase();
        var rows = data.X.GetLength(0);
        var model = new EstepRecorder(implementation, "st", data.StartProb, data.TransMat);
        model.Fit(Stack(data.X, data.X), new[] { rows, rows });
        Assert.NotNull(model.LastStats);
        var stats = model.LastStats!;
        Assert.Equal(2, stats.NObs);
        var expectedStart = new[] { 2 * data.Posteriors[0, 0], 2 * data.Posteriors[0, 1] };
        Close.Equal(expectedStart, stats.Start, "Start (sum of the first-sample posteriors of both copies)");
        var expectedTrans = Copy(data.ScalingXiSum);
        for (var r = 0; r < expectedTrans.GetLength(0); r++)
            for (var c = 0; c < expectedTrans.GetLength(1); c++)
                expectedTrans[r, c] = 2 * data.ScalingXiSum[r, c];
        Close.Equal(expectedTrans, stats.Trans, "Trans (xi sums of both copies)");
        Close.Equal(2 * data.Score, model.Monitor.History[0], "lower bound (oracle score of both copies)");
        Assert.Equal(2, model.Sequences.Count);
        foreach (var seq in model.Sequences)
            Close.Equal(data.Posteriors, seq.Posteriors, "recorded posteriors of a copy");
    }

    /// <summary>A sequence of length one adds to Start (normalized startprob * first observation) but never to Trans, for both
    /// implementations.</summary>
    [Theory]
    [InlineData(Implementation.Scaling)]
    [InlineData(Implementation.Log)]
    public void a_sequence_of_length_one_adds_to_start_but_not_to_trans(Implementation implementation)
    {
        var data = OneSampleCase();
        var x = (double[,])data.LogFrameProb.Clone(); // the observations ARE the log frame probabilities
        var model = new EstepRecorder(implementation, "st", data.StartProb, data.TransMat);
        model.Fit(x, new[] { 1 });
        Assert.NotNull(model.LastStats);
        var stats = model.LastStats!;
        Assert.Equal(1, stats.NObs);
        var unnormalized = new[] { data.StartProb[0] * data.FrameProb[0, 0], data.StartProb[1] * data.FrameProb[0, 1] };
        var total = 0.0;
        for (var s = 0; s < unnormalized.Length; s++)
            total += unnormalized[s];
        var expectedStart = new double[unnormalized.Length];
        for (var s = 0; s < unnormalized.Length; s++)
            expectedStart[s] = unnormalized[s] / total;
        Close.Equal(expectedStart, stats.Start, "Start (normalized startprob times the observation probability)");
        var allZero = new double[2, 2];
        Close.Equal(allZero, stats.Trans, "Trans (no transitions in a length-one sequence)");
    }

    /// <summary>Start is filled only when 's' is in Params: with Params = "s" Trans stays zero.</summary>
    [Fact]
    public void start_is_filled_only_when_s_is_in_params()
    {
        var data = WikipediaCase();
        var model = new EstepRecorder(Implementation.Scaling, "s", data.StartProb, data.TransMat);
        model.Fit(data.X);
        Assert.NotNull(model.LastStats);
        var stats = model.LastStats!;
        var expectedStart = new[] { data.Posteriors[0, 0], data.Posteriors[0, 1] };
        Close.Equal(expectedStart, stats.Start, "Start (filled because 's' is in params)");
        var allZero = new double[2, 2];
        Close.Equal(allZero, stats.Trans, "Trans (zero because 't' is not in params)");
    }

    /// <summary>Trans is filled only when 't' is in Params: with Params = "t" Start stays zero.</summary>
    [Fact]
    public void trans_is_filled_only_when_t_is_in_params()
    {
        var data = WikipediaCase();
        var model = new EstepRecorder(Implementation.Scaling, "t", data.StartProb, data.TransMat);
        model.Fit(data.X);
        Assert.NotNull(model.LastStats);
        var stats = model.LastStats!;
        Close.Equal(data.ScalingXiSum, stats.Trans, "Trans (filled because 't' is in params)");
        var allZero = new double[2];
        Close.Equal(allZero, stats.Start, "Start (zero because 's' is not in params)");
    }

    /// <summary>The lattice given to AccumulateSufficientStatistics holds probabilities with the Scaling implementation and log
    /// probabilities with the Log implementation.</summary>
    [Theory]
    [InlineData(Implementation.Scaling)]
    [InlineData(Implementation.Log)]
    public void the_lattice_given_to_accumulate_is_probabilities_with_scaling_and_log_probabilities_with_log(Implementation implementation)
    {
        var data = WikipediaCase();
        var model = new EstepRecorder(implementation, "st", data.StartProb, data.TransMat);
        model.Fit(data.X);
        Assert.Single(model.Sequences);
        if (implementation == Implementation.Scaling)
        {
            Close.Equal(data.FrameProb, model.Sequences[0].Lattice, "lattice with the Scaling implementation");
        }
        else
        {
            Close.Equal(data.LogFrameProb, model.Sequences[0].Lattice, "lattice with the Log implementation");
        }
    }

    /// <summary>A model overriding FitTransMat sees the override in the forward pass and in the xi sums given to DoMStep.</summary>
    [Fact]
    public void a_model_overriding_fit_trans_mat_sees_it_in_the_pass_and_the_xi_sums()
    {
        var data = WikipediaCase();
        var overrideTrans = new double[2, 2]
        {
            { 0.5, 0.5 },
            { 0.5, 0.5 },
        };
        var model = new FitVirtualOverrides(Implementation.Scaling, data.StartProb, data.TransMat, null, overrideTrans);
        model.Fit(data.X);
        Assert.Single(model.Sequences);
        var seq = model.Sequences[0];
        Close.Equal(Hmmc.ForwardScaling(data.StartProb, overrideTrans, seq.Lattice).FwdLattice, seq.Fwd, "forward lattice with the overridden transmat");
        Assert.NotNull(model.LastStats);
        var expectedTrans = Hmmc.ComputeScalingXiSum(seq.Fwd, overrideTrans, seq.Bwd, seq.Lattice);
        Close.Equal(expectedTrans, model.LastStats!.Trans, "Trans (xi sums with the overridden transmat)");
    }

    /// <summary>A model overriding FitStartProb sees the override in the forward pass given to AccumulateSufficientStatistics.</summary>
    [Fact]
    public void a_model_overriding_fit_start_prob_sees_it_in_the_forward_pass()
    {
        var data = WikipediaCase();
        var overrideStart = new[] { 0.9, 0.1 };
        var model = new FitVirtualOverrides(Implementation.Scaling, data.StartProb, data.TransMat, overrideStart, null);
        model.Fit(data.X);
        Assert.Single(model.Sequences);
        var seq = model.Sequences[0];
        Close.Equal(Hmmc.ForwardScaling(overrideStart, data.TransMat, seq.Lattice).FwdLattice, seq.Fwd, "forward lattice with the overridden startprob");
    }
}
