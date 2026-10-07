using System.Globalization;
using System.Text;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm6;

/// <summary>
/// Acceptance tests for the reset, warning and subclassing of <see cref="ConvergenceMonitor"/> (ticket HMM-6, criterion C3):
/// Reset empties the history and sets Iter back to 0; the Warning event is raised exactly when a report is lower than the
/// previous one by more than sqrt(machine epsilon) (about 1.49e-8) — not for an improvement, not for a tiny decrease, not for
/// a decrease of exactly the threshold — and a subclass can override Converged.
/// </summary>
public class ResetAndWarningTests
{
    private const double Previous = -0.5;

    /// <summary>The ticket's threshold: sqrt(2.220446049250313e-16) = 1.4901161193847656e-8 (exactly 2^-26 in IEEE double).</summary>
    private static double Threshold => Math.Sqrt(2.220446049250313e-16);

    private static List<string> Lines(StringWriter writer)
    {
        var text = writer.ToString().Replace("\r\n", "\n");
        if (text.EndsWith('\n'))
            text = text[..^1];
        return text.Split('\n').ToList();
    }

    /// <summary>
    /// Reset empties the history and sets Iter back to 0 (Tol, NIter and Verbose untouched): the next report is again a first
    /// report — its verbose line is iteration 1 again and carries the "+nan" improvement — and without a previous value no
    /// Warning can fire.
    /// </summary>
    [Fact]
    public void Reset_empties_the_history_and_sets_Iter_back_to_zero()
    {
        var monitor = new ConvergenceMonitor(0.001, 10, verbose: false);
        monitor.Report(-0.01);
        monitor.Report(-0.02);
        Assert.True(monitor.Iter == 2, $"Iter after two reports is 2, got {monitor.Iter}");
        Close.Equal(new double[] { -0.01, -0.02 }, monitor.History.ToArray(), "history before Reset");
        monitor.Reset();
        Assert.True(monitor.Iter == 0, $"Iter after Reset is 0, got {monitor.Iter}");
        Assert.Empty(monitor.History);
        Assert.True(monitor.Tol == 0.001 && monitor.NIter == 10 && !monitor.Verbose, "Reset keeps Tol, NIter and Verbose");
        Assert.False(monitor.Converged, "a reset monitor (no reports since the reset) is not converged");

        var writer = new StringWriter();
        var verbose = new ConvergenceMonitor(0.001, 10, verbose: true, writer);
        verbose.Report(-0.01);
        verbose.Report(-0.02);
        verbose.Reset();
        verbose.Report(-0.03);
        var lines = Lines(writer);
        Assert.True(lines.Count == 3, $"2 lines before Reset + 1 after, got {lines.Count}");
        var nanTail = new string(' ', 16 - "+nan".Length) + "+nan";
        Assert.True(lines[2].StartsWith(new string(' ', 10 - 1) + "1 "), $"the report after Reset is iteration 1 again, got \"{lines[2]}\"");
        Assert.True(lines[2].EndsWith(nanTail), $"the report after Reset is a first report again (delta \"+nan\"), got \"{lines[2]}\"");
    }

    /// <summary>
    /// No Warning for an improvement (delta +0.25), for a decrease of half the threshold (2^-27, about 7.45e-9) or for a
    /// decrease of exactly the threshold (2^-26): the event fires only when the delta is strictly below -sqrt(eps), as in the
    /// Python reference (delta &lt; -precision).
    /// </summary>
    [Fact]
    public void No_warning_for_an_improvement_a_half_threshold_decrease_or_exactly_the_threshold()
    {
        var threshold = Threshold;
        var scenarios = new (double Delta, string What)[]
        {
            (0.25, "an improvement"),
            (-(threshold / 2), "a decrease of half the threshold"),
            (-threshold, "a decrease of exactly the threshold"),
        };
        foreach (var (delta, what) in scenarios)
        {
            var messages = new List<string>();
            var monitor = new ConvergenceMonitor(0.001, 10, verbose: false);
            monitor.Warning += messages.Add;
            monitor.Report(Previous);
            monitor.Report(Previous + delta);
            Assert.True(messages.Count == 0, $"{what} (delta {delta:R}): no Warning, got {messages.Count}");
        }
    }

    /// <summary>
    /// A report lower than the previous one by more than the threshold (delta 2^-25, about 2.98e-8) raises the Warning exactly
    /// once per such report; every message starts with "Model is not converging." and carries the current value, the previous
    /// one and the delta; and the event fires before the value is appended (inside the handler, Iter and History are still the
    /// pre-report ones).
    /// </summary>
    [Fact]
    public void The_warning_fires_exactly_once_per_dropping_report_and_carries_the_numbers()
    {
        var threshold = Threshold;
        var messages = new List<string>();
        var stateDuringWarning = new List<(int Iter, double[] History)>();
        var monitor = new ConvergenceMonitor(0.001, 10, verbose: false);
        monitor.Warning += message =>
        {
            messages.Add(message);
            stateDuringWarning.Add((monitor.Iter, monitor.History.ToArray()));
        };
        monitor.Report(Previous);
        var firstDrop = Previous - 2 * threshold;
        var secondDrop = Previous - 4 * threshold;
        monitor.Report(firstDrop);
        monitor.Report(secondDrop);
        Assert.True(messages.Count == 2, $"one Warning per dropping report, got {messages.Count}");
        AssertContainsNumber(messages[0], Previous, "the previous value");
        AssertContainsNumber(messages[0], firstDrop, "the current value");
        AssertContainsNumber(messages[0], firstDrop - Previous, "the delta");
        AssertContainsNumber(messages[1], firstDrop, "the previous value");
        AssertContainsNumber(messages[1], secondDrop, "the current value");
        AssertContainsNumber(messages[1], secondDrop - firstDrop, "the delta");
        foreach (var m in messages)
            Assert.True(m.StartsWith("Model is not converging."), $"the message must start with \"Model is not converging.\", got: {m}");
        Assert.True(stateDuringWarning[0].Iter == 1, $"the Warning fires before Iter is incremented, got Iter {stateDuringWarning[0].Iter}");
        Close.Equal(new double[] { Previous }, stateDuringWarning[0].History, "History at Warning time (the new value is not appended yet)");
    }

    /// <summary>
    /// The first report has no previous value: it never raises the Warning, but it still counts (Iter 1, one value in the
    /// history).
    /// </summary>
    [Fact]
    public void The_first_report_never_raises_the_warning()
    {
        var messages = new List<string>();
        var monitor = new ConvergenceMonitor(0.001, 10, verbose: false);
        monitor.Warning += messages.Add;
        monitor.Report(-100.0);
        Assert.True(messages.Count == 0, $"no previous value: no Warning on the first report, got {messages.Count}");
        Assert.True(monitor.Iter == 1, $"the first report still counts, Iter expected 1, got {monitor.Iter}");
        Assert.True(monitor.History.Count == 1, "the first report still appends");
    }

    private sealed class AlwaysConvergedMonitor : ConvergenceMonitor
    {
        public AlwaysConvergedMonitor() : base(0.001, 10, false)
        {
        }

        public override bool Converged => true;
    }

    private sealed class NeverConvergedMonitor : ConvergenceMonitor
    {
        public NeverConvergedMonitor() : base(0.001, 2, false)
        {
        }

        public override bool Converged => false;
    }

    /// <summary>
    /// A subclass can override Converged (the base criterion being virtual), in both directions: an override that always
    /// converges wins before any report (the base criterion would say false), and one that never converges wins even when the
    /// base criterion (Iter == NIter) is met; the rest of the monitor (Report, History, Iter) keeps working in the subclass.
    /// </summary>
    [Fact]
    public void A_subclass_can_override_Converged()
    {
        var always = new AlwaysConvergedMonitor();
        Assert.True(always.Converged, "the override wins before any report (the base criterion would say false)");
        always.Report(-0.5);
        Assert.True(always.History.Count == 1 && always.Iter == 1, "Report keeps working in the subclass");
        Close.Equal(new double[] { -0.5 }, always.History.ToArray(), "history in the subclass");

        var never = new NeverConvergedMonitor();
        never.Report(-0.5);
        never.Report(-0.25);
        Assert.True(never.Iter == 2 && never.History.Count == 2, "two reports in the subclass");
        Assert.False(never.Converged, "the override wins even though Iter == NIter == 2 would converge the base criterion");
    }

    /// <summary>True when the message carries the value in its default (shortest round-trip) invariant string form, or in the
    /// Python f-string style (lowercase exponent).</summary>
    private static void AssertContainsNumber(string message, double value, string what)
    {
        var canonical = value.ToString(CultureInfo.InvariantCulture);
        var pythonStyle = canonical.Replace("E-", "e-").Replace("E+", "e+");
        Assert.True(message.Contains(canonical) || message.Contains(pythonStyle),
            $"{what} {canonical} must appear in the message: {message}");
    }
}
