namespace HmmLearn;

/// <summary>
/// Monitors and reports the convergence of EM (port of ConvergenceMonitor of hmmlearn/base.py). Subclass it and override
/// <see cref="Converged"/> to use another criterion.
/// </summary>
public class ConvergenceMonitor
{
    /// <summary>
    /// Creates a monitor. <paramref name="tol"/>: EM has converged when the improvement of the log probability between two
    /// consecutive iterations is below it; <paramref name="nIter"/>: maximum number of iterations; <paramref name="verbose"/>:
    /// whether each report is also written to <paramref name="output"/> (default: <see cref="Console.Error"/>).
    /// </summary>
    public ConvergenceMonitor(double tol, int nIter, bool verbose, TextWriter? output = null) => throw new NotImplementedException();

    /// <summary>Convergence threshold.</summary>
    public double Tol { get; set; }

    /// <summary>Maximum number of iterations.</summary>
    public int NIter { get; set; }

    /// <summary>Whether <see cref="Report"/> writes a line per iteration.</summary>
    public bool Verbose { get; set; }

    /// <summary>The log probabilities reported since the last reset, oldest first.</summary>
    public IReadOnlyList<double> History => throw new NotImplementedException();

    /// <summary>Number of reports since the last reset.</summary>
    public int Iter => throw new NotImplementedException();

    /// <summary>Raised with a message when a report is lower than the previous one by more than sqrt(machine epsilon) (about 1.49e-8).</summary>
    public event Action<string>? Warning;

    /// <summary>Forgets the history and sets <see cref="Iter"/> back to 0.</summary>
    public void Reset() => throw new NotImplementedException();

    /// <summary>
    /// Records the log probability of the data of the current iteration. If <see cref="Verbose"/>, first writes one line to the
    /// output: iteration number (<see cref="Iter"/> + 1) right-aligned on 10 characters, a space, the log probability with 8
    /// decimals right-aligned on 16, a space, the improvement over the previous report with an explicit sign (+ or -), 8 decimals,
    /// right-aligned on 16; for the first report the improvement is not a number and is written "+nan". Numbers use the invariant
    /// culture. Then, if the new value is lower than the previous one by more than sqrt(eps), raises <see cref="Warning"/>.
    /// Finally appends the value to the history and increments <see cref="Iter"/>.
    /// </summary>
    public void Report(double logProb) => throw new NotImplementedException();

    /// <summary>
    /// Whether EM has converged: <see cref="Iter"/> equals <see cref="NIter"/>, or there are at least two reports and the last
    /// minus the one before is lower than <see cref="Tol"/>.
    /// </summary>
    public virtual bool Converged => throw new NotImplementedException();
}
