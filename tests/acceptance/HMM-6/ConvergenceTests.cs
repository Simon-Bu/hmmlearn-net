using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm6;

/// <summary>
/// Acceptance tests for the convergence state of <see cref="ConvergenceMonitor"/> (ticket HMM-6, criterion C1): after each
/// Report of every scenario of oracle/golden/monitor.json, Converged, Iter and History equal the expected steps — convergence
/// by iteration count, by an improvement below the tolerance, by a decrease — and the scenarios that must not converge do not.
/// Hand-written cases: a fresh monitor and the live Tol/NIter setters.
/// </summary>
public class ConvergenceTests
{
    private static List<JsonElement> Cases() => Golden.Load("monitor").GetProperty("cases").Items().ToList();

    /// <summary>
    /// Every scenario of oracle/golden/monitor.json: after each Report, Iter, History (oldest first) and Converged equal the
    /// expected step — "converged-by-iterations" stops at the iteration count, "converged-by-log-prob" at an improvement below
    /// the tolerance, "decreasing-log-prob-is-converged-and-warned" at a decrease (a decrease counts as converged), and
    /// "never-converged" and "verbose-format" are not converged before the step the oracle marks converged.
    /// </summary>
    [Fact]
    public void Every_scenario_of_the_golden_file_reaches_the_expected_state_after_each_report()
    {
        var cases = Cases();
        Assert.True(cases.Count >= 5, $"expected the 5 scenarios of oracle/golden/monitor.json, saw {cases.Count}");
        foreach (var c in cases)
        {
            var name = c.GetProperty("name").AsString();
            var monitor = new ConvergenceMonitor(c.GetProperty("tol").AsDouble(), c.GetProperty("n_iter").AsInt(), verbose: false);
            var step = 0;
            foreach (var s in c.GetProperty("steps").Items())
            {
                monitor.Report(s.GetProperty("log_prob").AsDouble());
                step++;
                var what = $"{name}: after report {step}";
                var expectedIter = s.GetProperty("iter").AsInt();
                Assert.True(monitor.Iter == expectedIter, $"{what}: Iter expected {expectedIter}, got {monitor.Iter}");
                Close.Equal(s.GetProperty("history").AsVector(), monitor.History.ToArray(), $"{what}: History");
                var expectedConverged = s.GetProperty("converged").AsBool();
                Assert.True(monitor.Converged == expectedConverged, $"{what}: Converged expected {expectedConverged}, got {monitor.Converged}");
            }
        }
    }

    /// <summary>
    /// Before any report, a monitor of the golden scenarios has no history, Iter 0 and is not converged (the scenarios all have
    /// n_iter &gt; 1, so neither the iteration count nor the two-value criterion can fire yet).
    /// </summary>
    [Fact]
    public void A_fresh_monitor_has_no_history_no_iter_and_is_not_converged()
    {
        foreach (var c in Cases())
        {
            var name = c.GetProperty("name").AsString();
            var monitor = new ConvergenceMonitor(c.GetProperty("tol").AsDouble(), c.GetProperty("n_iter").AsInt(), verbose: false);
            Assert.True(monitor.NIter > 1, $"{name}: n_iter {monitor.NIter} must be > 1 for this test's precondition");
            Assert.True(monitor.Iter == 0, $"{name}: a fresh monitor has Iter 0, got {monitor.Iter}");
            Assert.Empty(monitor.History);
            Assert.False(monitor.Converged, $"{name}: a fresh monitor is not converged");
        }
    }

    /// <summary>
    /// The Tol and NIter setters are live for Converged: with an improvement of 0.5, a tight tol keeps the monitor
    /// unconverged, raising Tol past the improvement converges it, lowering it undoes that, and setting NIter to the current
    /// Iter converges by iteration count.
    /// </summary>
    [Fact]
    public void The_Tol_and_NIter_setters_are_live_for_Converged()
    {
        var monitor = new ConvergenceMonitor(1e-9, 10, verbose: false);
        monitor.Report(-1.0);
        monitor.Report(-0.5);
        Assert.False(monitor.Converged, "improvement 0.5 with tol 1e-9 and 2 of 10 iterations: not converged");
        monitor.Tol = 1.0;
        Assert.True(monitor.Converged, "raising Tol to 1.0 makes the 0.5 improvement converge");
        monitor.Tol = 1e-9;
        Assert.False(monitor.Converged, "lowering Tol back below the improvement undoes the convergence");
        monitor.NIter = 2;
        Assert.True(monitor.Converged, "setting NIter to the current Iter (2) converges by iteration count");
    }
}
