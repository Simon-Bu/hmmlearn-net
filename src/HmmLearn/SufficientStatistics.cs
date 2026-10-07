namespace HmmLearn;

/// <summary>
/// The sufficient statistics that the E-step accumulates over the sequences and the M-step consumes (hmmlearn's <c>stats</c>
/// dictionary). Models derive from this class to add the statistics of their emissions.
/// </summary>
public class SufficientStatistics
{
    /// <summary>Allocates <see cref="Start"/> (n_components zeros) and <see cref="Trans"/> (n_components x n_components zeros).</summary>
    public SufficientStatistics(int nComponents)
    {
        NObs = 0;
        Start = new double[nComponents];
        Trans = new double[nComponents, nComponents];
    }

    /// <summary>Number of sequences accumulated so far (hmmlearn's 'nobs').</summary>
    public int NObs { get; set; }

    /// <summary>Sum over the sequences of the posterior probability of each state at the first sample ('start').</summary>
    public double[] Start { get; set; } = null!;

    /// <summary>Sum over the sequences of the expected number of transitions between each pair of states ('trans').</summary>
    public double[,] Trans { get; set; } = null!;
}
