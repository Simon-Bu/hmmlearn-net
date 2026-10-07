namespace HmmLearn;

/// <summary>
/// Port of ext/_hmmc.cpp, second part: the sums of transition posteriors (xi) used by the M-step, and the Viterbi algorithm.
/// The forward and backward passes and the log helpers are in Hmmc.ForwardBackward.cs.
/// </summary>
public static partial class Hmmc
{
    /// <summary>
    /// Sum over the sequence of the posterior probability of each transition, from scaled lattices:
    /// xi_sum[i,j] = sum over t = 0..n_samples-2 of fwdlattice[t,i] * transmat[i,j] * frameprob[t+1,j] * bwdlattice[t+1,j].
    /// Result shape (nc, nc), zeros for a single sample. fwdlattice and bwdlattice must be (n_samples, nc), transmat (nc, nc),
    /// frameprob (n_samples, nc); otherwise <see cref="ArgumentException"/> "shape mismatch".
    /// </summary>
    public static double[,] ComputeScalingXiSum(double[,] fwdlattice, double[,] transmat, double[,] bwdlattice, double[,] frameprob)
        => throw new NotImplementedException();

    /// <summary>
    /// Same sum in log space, from log lattices: logProb = LogSumExp of the last row of fwdlattice; then
    /// log_xi_sum[i,j] = LogAddExp over t = 0..n_samples-2 of
    /// (fwdlattice[t,i] + log(transmat[i,j]) + logFrameprob[t+1,j] + bwdlattice[t+1,j] - logProb), starting from -infinity
    /// (so a single sample gives a matrix of -infinity). Result shape (nc, nc). Shape errors as in <see cref="ComputeScalingXiSum"/>.
    /// </summary>
    public static double[,] ComputeLogXiSum(double[,] fwdlattice, double[,] transmat, double[,] bwdlattice, double[,] logFrameprob)
        => throw new NotImplementedException();

    /// <summary>
    /// Most likely state sequence. lattice[0,i] = log(startprob[i]) + logFrameprob[0,i]; lattice[t,i] = max over j of
    /// (lattice[t-1,j] + log(transmat[j,i])) + logFrameprob[t,i]. The last state is the FIRST index of the maximum of the last row;
    /// going backwards, the state at t is the index i that maximizes lattice[t,i] + log(transmat[i, state at t+1]), and among equal
    /// maxima the LARGEST index wins (the C++ compares (value, index) pairs; a value of -infinity ties with itself, so an all
    /// -infinity candidate list also gives the largest index). Returns the log probability of that path (the maximum of the last
    /// row) and the states. Shapes: startprob length nc, transmat nc x nc, at least one sample; otherwise
    /// <see cref="ArgumentException"/> "shape mismatch".
    /// </summary>
    public static (double LogProb, int[] StateSequence) Viterbi(double[] startprob, double[,] transmat, double[,] logFrameprob)
        => throw new NotImplementedException();
}
