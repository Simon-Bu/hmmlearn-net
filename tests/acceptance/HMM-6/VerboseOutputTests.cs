using System.Text;
using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm6;

/// <summary>
/// Acceptance tests for the verbose output of <see cref="ConvergenceMonitor"/> (ticket HMM-6, criterion C2): with verbose true
/// the monitor writes exactly the stderr_lines of every scenario of oracle/golden/monitor.json to the given TextWriter (the
/// first line's "+nan" and the 10/16/16 alignment included), and with verbose false it writes nothing. Hand-written case:
/// toggling the Verbose property at report time.
/// </summary>
public class VerboseOutputTests
{
    private static List<JsonElement> Cases() => Golden.Load("monitor").GetProperty("cases").Items().ToList();

    /// <summary>The lines written to the writer, whatever the newline convention ("\n" or "\r\n"); a single trailing newline
    /// is not a line of its own.</summary>
    private static List<string> Lines(StringWriter writer)
    {
        var text = writer.ToString().Replace("\r\n", "\n");
        if (text.EndsWith('\n'))
            text = text[..^1];
        return text.Split('\n').ToList();
    }

    /// <summary>
    /// Every scenario of oracle/golden/monitor.json, verbose true: the given TextWriter receives exactly the golden
    /// stderr_lines, one line per report, compared by string equality — including the first line, whose improvement is not a
    /// number and is written as "+nan" right-aligned on 16 characters (the line ends with spaces then "+nan").
    /// </summary>
    [Fact]
    public void Every_scenario_prints_exactly_the_golden_stderr_lines_when_verbose()
    {
        var cases = Cases();
        Assert.True(cases.Count >= 5, $"expected the 5 scenarios of oracle/golden/monitor.json, saw {cases.Count}");
        var nanTail = new string(' ', 16 - "+nan".Length) + "+nan";
        foreach (var c in cases)
        {
            var name = c.GetProperty("name").AsString();
            var writer = new StringWriter();
            var monitor = new ConvergenceMonitor(c.GetProperty("tol").AsDouble(), c.GetProperty("n_iter").AsInt(), verbose: true, writer);
            foreach (var s in c.GetProperty("steps").Items())
                monitor.Report(s.GetProperty("log_prob").AsDouble());
            var expected = c.GetProperty("stderr_lines").Items().Select(e => e.AsString()).ToList();
            var actual = Lines(writer);
            Assert.True(actual.Count == expected.Count, $"{name}: expected {expected.Count} lines, got {actual.Count}: \"{string.Join(" | ", actual)}\"");
            for (var i = 0; i < expected.Count; i++)
                Assert.True(actual[i] == expected[i], $"{name}: line {i + 1} expected \"{expected[i]}\", got \"{actual[i]}\"");
            Assert.True(actual[0].EndsWith(nanTail), $"{name}: the first line must end with the right-aligned \"+nan\", got \"{actual[0]}\"");
        }
    }

    /// <summary>Every scenario, verbose false: nothing at all is written to the given TextWriter, however many reports.</summary>
    [Fact]
    public void Nothing_is_written_when_verbose_is_false()
    {
        foreach (var c in Cases())
        {
            var name = c.GetProperty("name").AsString();
            var writer = new StringWriter();
            var monitor = new ConvergenceMonitor(c.GetProperty("tol").AsDouble(), c.GetProperty("n_iter").AsInt(), verbose: false, writer);
            foreach (var s in c.GetProperty("steps").Items())
                monitor.Report(s.GetProperty("log_prob").AsDouble());
            Assert.True(writer.ToString() == string.Empty, $"{name}: verbose false writes nothing, got \"{writer.ToString()}\"");
        }
    }

    /// <summary>
    /// The Verbose property is read at report time: with verbose off the first report writes nothing; switching it on makes the
    /// following reports write one line each (iterations 2 and 3 of the running count); switching it off again writes nothing
    /// further.
    /// </summary>
    [Fact]
    public void Switching_Verbose_on_and_off_controls_which_reports_are_written()
    {
        var writer = new StringWriter();
        var monitor = new ConvergenceMonitor(0.001, 10, verbose: false, writer);
        monitor.Report(-0.5);
        Assert.True(writer.ToString() == string.Empty, "verbose false: the first report writes nothing");
        monitor.Verbose = true;
        monitor.Report(-0.25);
        monitor.Report(-0.125);
        var lines = Lines(writer);
        Assert.True(lines.Count == 2, $"expected 2 lines after switching verbose on, got {lines.Count}");
        Assert.True(lines[0].StartsWith(new string(' ', 10 - 1) + "2 "), $"the first written line is iteration 2, got \"{lines[0]}\"");
        Assert.True(lines[1].StartsWith(new string(' ', 10 - 1) + "3 "), $"the second written line is iteration 3, got \"{lines[1]}\"");
        monitor.Verbose = false;
        monitor.Report(0.0);
        Assert.True(Lines(writer).Count == 2, "switching verbose off again writes nothing further");
    }
}
