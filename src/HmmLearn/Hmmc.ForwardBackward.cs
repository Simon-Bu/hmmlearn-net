namespace HmmLearn;

/// <summary>
/// Port of ext/_hmmc.cpp, the numerical core of hmmlearn. This file holds the forward and backward passes, in two flavours:
/// "scaling" (probabilities, renormalized at every step) and "log" (log probabilities). The xi sums and the Viterbi algorithm are
/// in Hmmc.ViterbiXi.cs. Lattices have shape (n_samples, n_components); frame probabilities are the emission probabilities of
/// each sample under each state.
/// </summary>
public static partial class Hmmc
{
    /// <summary>
    /// log(exp(a) + exp(b)) without overflow: if a is -infinity the result is b, if b is -infinity the result is a, otherwise
    /// max(a, b) + log(1 + exp(-|b - a|)) (hmmlearn uses log1p; double.LogP1 or Math.Log(1 + x) are both fine here).
    /// </summary>
    internal static double LogAddExp(double a, double b) => throw new NotImplementedException();

    /// <summary>
    /// log(sum(exp(v))) without overflow: factor out the maximum; if the maximum is infinite (-infinity for an all -infinity input)
    /// it is returned as is.
    /// </summary>
    internal static double LogSumExp(ReadOnlySpan<double> v) => throw new NotImplementedException();

    /// <summary>
    /// Forward pass with scaling. fwd[0,i] = startprob[i] * frameprob[0,i]; fwd[t,j] = (sum over i of fwd[t-1,i] * transmat[i,j]) *
    /// frameprob[t,j]; after each step the row is multiplied by scale = 1 / (sum of the row) and scale is stored in the scaling
    /// vector. Returns the log probability of the sequence (minus the sum of the logs of the scales), the scaled lattice and the
    /// scaling vector. If the sum of a row is below 1e-300 it throws <see cref="ArithmeticException"/> with the message
    /// "forward pass failed with underflow; consider using implementation='log' instead". Shapes that do not agree (startprob of
    /// length nc, transmat nc x nc, at least one sample) throw <see cref="ArgumentException"/> with the message "shape mismatch".
    /// </summary>
    public static (double LogProb, double[,] FwdLattice, double[] Scaling) ForwardScaling(
        double[] startprob, double[,] transmat, double[,] frameprob) => throw new NotImplementedException();

    /// <summary>
    /// Forward pass in log space: fwd[0,i] = log(startprob[i]) + logFrameprob[0,i]; fwd[t,j] = LogSumExp over i of
    /// (fwd[t-1,i] + log(transmat[i,j])) + logFrameprob[t,j]. log(0) is -infinity. Returns the log probability of the sequence
    /// (LogSumExp of the last row) and the lattice. Shape errors as in <see cref="ForwardScaling"/>.
    /// </summary>
    public static (double LogProb, double[,] FwdLattice) ForwardLog(
        double[] startprob, double[,] transmat, double[,] logFrameprob) => throw new NotImplementedException();

    /// <summary>
    /// Backward pass with scaling, using the scaling vector of <see cref="ForwardScaling"/>: bwd[ns-1,i] = scaling[ns-1];
    /// bwd[t,i] = (sum over j of transmat[i,j] * frameprob[t+1,j] * bwd[t+1,j]) * scaling[t]. startprob only takes part in the
    /// shape check; scaling must have one element per sample. Shape errors throw <see cref="ArgumentException"/> "shape mismatch".
    /// </summary>
    public static double[,] BackwardScaling(
        double[] startprob, double[,] transmat, double[,] frameprob, double[] scaling) => throw new NotImplementedException();

    /// <summary>
    /// Backward pass in log space: bwd[ns-1,i] = 0; bwd[t,i] = LogSumExp over j of
    /// (log(transmat[i,j]) + logFrameprob[t+1,j] + bwd[t+1,j]). Shape errors as in <see cref="ForwardScaling"/>.
    /// </summary>
    public static double[,] BackwardLog(
        double[] startprob, double[,] transmat, double[,] logFrameprob) => throw new NotImplementedException();
}
