using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm8;

/// <summary>
/// Acceptance tests for the EM loop of <see cref="AbstractHmm.Fit(double[,], int[])"/> (ticket HMM-8, criterion C1): Fit runs Init
/// and Check and then at most NIter iterations, each one an E-step (InitializeSufficientStatistics, EstepBegin, one
/// AccumulateSufficientStatistics per sequence), ComputeLowerBound, DoMStep and Monitor.Report in that order; it stops after the
/// iteration in which the monitor first converges (the M-step of that iteration has run) and otherwise runs exactly NIter
/// iterations; NIter = 0 only initializes and checks; the monitor receives Tol, NIter and Verbose and is reset at the start of each
/// Fit, and its history holds the lower bounds; null lengths mean one sequence of all the rows; lengths that do not sum to the
/// rows and invalid observations throw ArgumentException before Init runs; and Fit returns the same instance without mutating the
/// caller's arrays.
/// </summary>
public class FitLoopTests
{
    private static JsonElement InferenceCase(string name) =>
        Assert.Single(Golden.Load("base").GetProperty("inference_cases").Items(), c => c.GetProperty("name").AsString() == name);

    private sealed record Wikipedia
    {
        public required double[,] X { get; init; }

        public required double[] StartProb { get; init; }

        public required double[,] TransMat { get; init; }

        public required double Score { get; init; }
    }

    private static Wikipedia WikipediaCase()
    {
        var c = InferenceCase("wikipedia");
        return new Wikipedia
        {
            X = c.GetProperty("X").AsMatrix(),
            StartProb = c.GetProperty("startprob").AsVector(),
            TransMat = c.GetProperty("transmat").AsMatrix(),
            Score = c.GetProperty("expected").GetProperty("score").AsDouble(),
        };
    }

    private static int Count(IEnumerable<string> calls, string name) => calls.Count(c => c == name);

    /// <summary>A model that records the fit hooks it is called through; Init installs the oracle parameters, the M-step does not
    /// change anything, and ComputeLowerBound returns the log probability of the E-step (or the next scripted value).</summary>
    private sealed class RecordingHmm : AbstractHmm
    {
        private readonly double[] _initialStartProb;
        private readonly double[,] _initialTransMat;
        private readonly Queue<double>? _scriptedLowerBounds;

        public List<string> Calls { get; } = new();

        public int[]? InitLengths { get; private set; }

        public SufficientStatistics? LastStats { get; private set; }

        public RecordingHmm(Implementation implementation, int nIter, double tol, double[] initialStartProb, double[,] initialTransMat,
            Queue<double>? scriptedLowerBounds = null)
            : base(2, DecoderAlgorithm.Viterbi, null, nIter, tol, false, "st", "st", implementation)
        {
            _initialStartProb = initialStartProb;
            _initialTransMat = initialTransMat;
            _scriptedLowerBounds = scriptedLowerBounds;
        }

        protected override void Init(double[,] x, int[] lengths)
        {
            Calls.Add("Init");
            InitLengths = lengths;
            StartProb = (double[])_initialStartProb.Clone();
            TransMat = (double[,])_initialTransMat.Clone();
            NFeatures = x.GetLength(1);
        }

        protected override void Check() => Calls.Add("Check");

        protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

        protected override double[] GenerateSampleFromState(int state, Random random) => new[] { (double)state };

        protected override SufficientStatistics InitializeSufficientStatistics()
        {
            Calls.Add("InitializeSufficientStatistics");
            return base.InitializeSufficientStatistics();
        }

        protected override void EstepBegin() => Calls.Add("EstepBegin");

        protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
            double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice)
        {
            Calls.Add("AccumulateSufficientStatistics");
            base.AccumulateSufficientStatistics(stats, x, lattice, posteriors, fwdlattice, bwdlattice);
        }

        protected override double ComputeLowerBound(double currLogProb)
        {
            Calls.Add("ComputeLowerBound");
            return _scriptedLowerBounds is { Count: > 0 } ? _scriptedLowerBounds.Dequeue() : currLogProb;
        }

        protected override void DoMStep(SufficientStatistics stats)
        {
            Calls.Add("DoMStep");
            LastStats = CopyStats(stats);
        }

        private static SufficientStatistics CopyStats(SufficientStatistics stats)
        {
            var copy = new SufficientStatistics(stats.Start.Length);
            copy.NObs = stats.NObs;
            copy.Start = (double[])stats.Start.Clone();
            copy.Trans = (double[,])stats.Trans.Clone();
            return copy;
        }
    }

    /// <summary>Fit runs Init, Check, then one iteration with the E-step hooks, ComputeLowerBound and DoMStep, in that order, and
    /// reports once to the monitor.</summary>
    [Fact]
    public void fit_runs_init_check_then_each_iteration_with_the_estep_hooks_compute_lower_bound_and_do_m_step()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 1, 1e-2, data.StartProb, data.TransMat);
        model.Fit(data.X);
        Assert.Equal(
            new[] { "Init", "Check", "InitializeSufficientStatistics", "EstepBegin", "AccumulateSufficientStatistics", "ComputeLowerBound", "DoMStep" },
            model.Calls);
        Assert.Single(model.Monitor.History);
        Assert.Equal(1, model.Monitor.Iter);
    }

    /// <summary>Each sequence described by lengths gets its own AccumulateSufficientStatistics call, in order, inside one E-step.</summary>
    [Fact]
    public void fit_accumulates_every_sequence_of_lengths_in_the_estep()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 1, 1e-2, data.StartProb, data.TransMat);
        model.Fit(data.X, new[] { 2, 3 });
        Assert.Equal(
            new[] { "Init", "Check", "InitializeSufficientStatistics", "EstepBegin", "AccumulateSufficientStatistics", "AccumulateSufficientStatistics", "ComputeLowerBound", "DoMStep" },
            model.Calls);
        Assert.NotNull(model.LastStats);
        Assert.Equal(2, model.LastStats!.NObs);
    }

    /// <summary>Fit stops after the iteration in which the monitor first converges: with a monitor that does not change the
    /// parameters the lower bound is constant, the monitor converges after two reports, so exactly two iterations run and the
    /// M-step of the second one has run.</summary>
    [Fact]
    public void fit_stops_after_the_iteration_where_the_monitor_first_converges()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 5, 1e-2, data.StartProb, data.TransMat);
        model.Fit(data.X);
        Assert.Equal(2, Count(model.Calls, "InitializeSufficientStatistics"));
        Assert.Equal(2, Count(model.Calls, "EstepBegin"));
        Assert.Equal(2, Count(model.Calls, "AccumulateSufficientStatistics"));
        Assert.Equal(2, Count(model.Calls, "ComputeLowerBound"));
        Assert.Equal(2, Count(model.Calls, "DoMStep"));
        Assert.Equal("DoMStep", model.Calls[^1]);
        Assert.Equal(2, model.Monitor.History.Count);
        Assert.Equal(2, model.Monitor.Iter);
    }

    /// <summary>When the monitor never converges (the lower bound keeps improving by more than Tol on every report), Fit runs
    /// exactly NIter iterations.</summary>
    [Fact]
    public void fit_runs_exactly_n_iter_iterations_when_the_monitor_never_converges()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 3, 1e-2, data.StartProb, data.TransMat,
            scriptedLowerBounds: new Queue<double>(new[] { -3.0, -2.0, -1.0 }));
        model.Fit(data.X);
        Assert.Equal(3, Count(model.Calls, "InitializeSufficientStatistics"));
        Assert.Equal(3, Count(model.Calls, "DoMStep"));
        Assert.Equal(new double[] { -3.0, -2.0, -1.0 }, model.Monitor.History);
        Assert.Equal(3, model.Monitor.Iter);
        Assert.Equal("DoMStep", model.Calls[^1]);
    }

    /// <summary>NIter = 0 only initializes and checks: no E-step, no M-step, no report.</summary>
    [Fact]
    public void fit_with_n_iter_zero_only_initializes_and_checks()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 0, 1e-2, data.StartProb, data.TransMat);
        model.Fit(data.X);
        Assert.Equal(new[] { "Init", "Check" }, model.Calls);
        Assert.Empty(model.Monitor.History);
        Assert.Equal(0, model.Monitor.Iter);
    }

    /// <summary>Fit passes the current Tol, NIter and Verbose to the monitor, even when they changed after construction.</summary>
    [Fact]
    public void fit_passes_tol_n_iter_and_verbose_to_the_monitor()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 4, 0.037, data.StartProb, data.TransMat);
        model.Tol = 0.123;
        model.NIter = 1;
        model.Verbose = true;
        model.Fit(data.X);
        Assert.Equal(0.123, model.Monitor.Tol);
        Assert.Equal(1, model.Monitor.NIter);
        Assert.True(model.Monitor.Verbose);
    }

    /// <summary>The monitor history is reset at the start of each Fit: the second fit reports into an empty history.</summary>
    [Fact]
    public void fit_resets_the_monitor_history_at_the_start_of_each_fit()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 3, 1e-2, data.StartProb, data.TransMat,
            scriptedLowerBounds: new Queue<double>(new[] { -3.0, -2.0, -1.0 }));
        model.Fit(data.X);
        Assert.Equal(3, model.Monitor.History.Count);
        model.NIter = 1;
        model.Fit(data.X);
        Assert.Single(model.Monitor.History);
        Assert.Equal(1, model.Monitor.Iter);
    }

    /// <summary>The monitor history holds the lower bounds of the data: with the oracle parameters and a no-op M-step the lower
    /// bound is the oracle score on every iteration.</summary>
    [Fact]
    public void the_monitor_history_holds_the_lower_bounds_of_the_data()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 5, 1e-2, data.StartProb, data.TransMat);
        model.Fit(data.X);
        Assert.Equal(2, model.Monitor.History.Count);
        Close.Equal(data.Score, model.Monitor.History[0], "lower bound after the first iteration");
        Close.Equal(data.Score, model.Monitor.History[1], "lower bound after the second iteration");
    }

    /// <summary>Fit returns the same instance.</summary>
    [Fact]
    public void fit_returns_the_same_instance()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 0, 1e-2, data.StartProb, data.TransMat);
        Assert.Same(model, model.Fit(data.X));
    }

    /// <summary>Null lengths mean one sequence of all the rows: Init receives [n_rows] and the E-step accumulates one sequence.</summary>
    [Fact]
    public void null_lengths_means_one_sequence_of_all_the_rows()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 1, 1e-2, data.StartProb, data.TransMat);
        model.Fit(data.X);
        Assert.Equal(new[] { 5 }, (int[]?)model.InitLengths);
        Assert.Equal(1, Count(model.Calls, "AccumulateSufficientStatistics"));
        Assert.NotNull(model.LastStats);
        Assert.Equal(1, model.LastStats!.NObs);
    }

    /// <summary>Lengths that do not sum to the number of rows throw ArgumentException before Init runs.</summary>
    [Fact]
    public void lengths_that_do_not_sum_to_the_rows_throw_before_init()
    {
        var data = WikipediaCase();
        var model = new RecordingHmm(Implementation.Scaling, 1, 1e-2, data.StartProb, data.TransMat);
        Assert.Throws<ArgumentException>(() => model.Fit(data.X, new[] { 2, 2 }));
        Assert.Empty(model.Calls);
    }

    /// <summary>Invalid observations (a NaN value) throw ArgumentException before Init runs.</summary>
    [Fact]
    public void invalid_observations_throw_before_init()
    {
        var data = WikipediaCase();
        var x = (double[,])data.X.Clone();
        x[2, 1] = double.NaN;
        var model = new RecordingHmm(Implementation.Scaling, 1, 1e-2, data.StartProb, data.TransMat);
        Assert.Throws<ArgumentException>(() => model.Fit(x));
        Assert.Empty(model.Calls);
    }

    /// <summary>Fit does not mutate the caller's x or lengths.</summary>
    [Fact]
    public void fit_does_not_mutate_the_callers_x_and_lengths()
    {
        var data = WikipediaCase();
        var x = data.X;
        var lengths = new[] { 2, 3 };
        var x0 = (double[,])x.Clone();
        var l0 = (int[])lengths.Clone();
        var model = new RecordingHmm(Implementation.Scaling, 1, 1e-2, data.StartProb, data.TransMat);
        model.Fit(x, lengths);
        for (var r = 0; r < x.GetLength(0); r++)
            for (var c = 0; c < x.GetLength(1); c++)
                Assert.Equal(x0[r, c], x[r, c]);
        Assert.Equal(l0, lengths);
    }
}
