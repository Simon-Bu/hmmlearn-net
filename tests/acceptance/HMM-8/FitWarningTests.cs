using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm8;

/// <summary>
/// Acceptance tests for warning forwarding of <see cref="AbstractHmm.Fit(double[,], int[])"/> (ticket HMM-8, criterion C4): Fit
/// forwards the monitor's warnings to the model's Warning event while fitting; it emits the documented zero-row warning when an
/// M-step leaves a row of the transmat with zero sum; and it unsubscribes from the monitor's Warning both when the fit finishes
/// normally and when it throws.
/// </summary>
public class FitWarningTests
{
    private static JsonElement InferenceCase(string name) =>
        Assert.Single(Golden.Load("base").GetProperty("inference_cases").Items(), c => c.GetProperty("name").AsString() == name);

    private sealed record Wikipedia
    {
        public required double[,] X { get; init; }

        public required double[] StartProb { get; init; }

        public required double[,] TransMat { get; init; }
    }

    private static Wikipedia WikipediaCase()
    {
        var c = InferenceCase("wikipedia");
        return new Wikipedia
        {
            X = c.GetProperty("X").AsMatrix(),
            StartProb = c.GetProperty("startprob").AsVector(),
            TransMat = c.GetProperty("transmat").AsMatrix(),
        };
    }

    /// <summary>A model with the Log implementation (safe with a zero transmat row): Init installs the oracle parameters; the
    /// M-step optionally zeroes the first row of the transmat once; ComputeLowerBound optionally throws.</summary>
    private sealed class WarningRecorder : AbstractHmm
    {
        private readonly double[] _initialStartProb;
        private readonly double[,] _initialTransMat;
        private int _mstepCount;

        public bool ThrowInComputeLowerBound { get; }

        public WarningRecorder(double[] initialStartProb, double[,] initialTransMat, int nIter, double tol,
            bool corruptTransMatOnFirstMStep = false, bool throwInComputeLowerBound = false)
            : base(2, DecoderAlgorithm.Viterbi, null, nIter, tol, false, "st", "st", Implementation.Log)
        {
            _initialStartProb = initialStartProb;
            _initialTransMat = initialTransMat;
            ThrowInComputeLowerBound = throwInComputeLowerBound;
            _corruptTransMatOnFirstMStep = corruptTransMatOnFirstMStep;
        }

        private readonly bool _corruptTransMatOnFirstMStep;

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

        protected override double ComputeLowerBound(double currLogProb)
        {
            if (ThrowInComputeLowerBound)
                throw new InvalidOperationException("boom");
            return currLogProb;
        }

        protected override void DoMStep(SufficientStatistics stats)
        {
            if (_corruptTransMatOnFirstMStep && _mstepCount++ == 0)
            {
                TransMat![0, 0] = 0.0;
                TransMat![0, 1] = 0.0;
            }
        }
    }

    /// <summary>Fit forwards the monitor's warnings to the model's Warning event: the M-step zeroes a row of the transmat, so the
    /// zero-row warning fires on the first iteration and the monitor's non-convergence warning on the second report (the lower
    /// bound drops far below the first one with the corrupted transmat).</summary>
    [Fact]
    public void fit_forwards_the_monitor_warnings_and_the_zero_row_warning_while_fitting()
    {
        var data = WikipediaCase();
        var model = new WarningRecorder(data.StartProb, data.TransMat, 2, 1e-2, corruptTransMatOnFirstMStep: true);
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.Fit(data.X);
        Assert.Equal(2, warnings.Count);
        Assert.Equal("Some rows of transmat_ have zero sum because no transition from the state was ever observed.", warnings[0]);
        Assert.StartsWith("Model is not converging.", warnings[1]);
    }

    /// <summary>The zero-row warning message is the documented fixed text.</summary>
    [Fact]
    public void the_zero_row_warning_message_is_the_documented_text()
    {
        var data = WikipediaCase();
        var model = new WarningRecorder(data.StartProb, data.TransMat, 2, 1e-2, corruptTransMatOnFirstMStep: true);
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.Fit(data.X);
        Assert.Contains("Some rows of transmat_ have zero sum because no transition from the state was ever observed.", warnings);
    }

    /// <summary>A converging fit with no zero transmat row emits no warning at all.</summary>
    [Fact]
    public void a_converging_fit_with_no_zero_transmat_row_emits_no_warning()
    {
        var data = WikipediaCase();
        var model = new WarningRecorder(data.StartProb, data.TransMat, 5, 1e-2);
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.Fit(data.X);
        Assert.Empty(warnings);
    }

    /// <summary>Two reports far below any previous lower bound make the monitor raise its non-convergence warning; if Fit still
    /// subscribes to the monitor after the fit, the warning is forwarded to the model's Warning event.</summary>
    private static void RaiseMonitorWarning(ConvergenceMonitor monitor)
    {
        monitor.Report(-1e300);
        monitor.Report(-2e300);
    }

    /// <summary>Fit unsubscribes from the monitor's Warning when the fit finishes: a later monitor warning is not forwarded.</summary>
    [Fact]
    public void fit_unsubscribes_from_the_monitor_when_the_fit_finishes()
    {
        var data = WikipediaCase();
        var model = new WarningRecorder(data.StartProb, data.TransMat, 1, 1e-2);
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        model.Fit(data.X);
        RaiseMonitorWarning(model.Monitor);
        Assert.Empty(warnings);
    }

    /// <summary>Fit unsubscribes from the monitor's Warning even when the fit throws: a later monitor warning is not forwarded.</summary>
    [Fact]
    public void fit_unsubscribes_from_the_monitor_even_when_the_fit_throws()
    {
        var data = WikipediaCase();
        var model = new WarningRecorder(data.StartProb, data.TransMat, 3, 1e-2, throwInComputeLowerBound: true);
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        Assert.Throws<InvalidOperationException>(() => model.Fit(data.X));
        RaiseMonitorWarning(model.Monitor);
        Assert.Empty(warnings);
    }
}
