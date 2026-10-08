using System.Runtime.InteropServices;

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
    {
        var ns = frameprob.GetLength(0);
        var nc = frameprob.GetLength(1);
        CheckXiShapes(fwdlattice, transmat, bwdlattice, ns, nc);
        var xiSum = new double[nc, nc];
        for (var t = 0; t < ns - 1; t++)
        {
            for (var i = 0; i < nc; i++)
            {
                for (var j = 0; j < nc; j++)
                    xiSum[i, j] += fwdlattice[t, i] * transmat[i, j] * frameprob[t + 1, j] * bwdlattice[t + 1, j];
            }
        }
        return xiSum;
    }

    /// <summary>
    /// Same sum in log space, from log lattices: logProb = LogSumExp of the last row of fwdlattice; then
    /// log_xi_sum[i,j] = LogAddExp over t = 0..n_samples-2 of
    /// (fwdlattice[t,i] + log(transmat[i,j]) + logFrameprob[t+1,j] + bwdlattice[t+1,j] - logProb), starting from -infinity
    /// (so a single sample gives a matrix of -infinity). With 0 samples (all inputs (0, nc)) the sum is empty too: the result is a
    /// new (nc, nc) matrix of -infinity and the last row of fwdlattice is not read. Result shape (nc, nc). Shape errors as in <see cref="ComputeScalingXiSum"/>.
    /// </summary>
    public static double[,] ComputeLogXiSum(double[,] fwdlattice, double[,] transmat, double[,] bwdlattice, double[,] logFrameprob)
    {
        var ns = logFrameprob.GetLength(0);
        var nc = logFrameprob.GetLength(1);
        CheckXiShapes(fwdlattice, transmat, bwdlattice, ns, nc);
        var logXiSum = new double[nc, nc];
        for (var i = 0; i < nc; i++)
            for (var j = 0; j < nc; j++)
                logXiSum[i, j] = double.NegativeInfinity;
        if (ns == 0) return logXiSum;
        var logTransmat = LogMatrix(transmat);
        var logProb = LogSumExp(MemoryMarshal.CreateReadOnlySpan(ref fwdlattice[ns - 1, 0], nc));
        for (var t = 0; t < ns - 1; t++)
        {
            for (var i = 0; i < nc; i++)
            {
                for (var j = 0; j < nc; j++)
                    logXiSum[i, j] = LogAddExp(
                        logXiSum[i, j],
                        fwdlattice[t, i] + logTransmat[i, j] + logFrameprob[t + 1, j] + bwdlattice[t + 1, j] - logProb);
            }
        }
        return logXiSum;
    }

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
    {
        var ns = logFrameprob.GetLength(0);
        var nc = logFrameprob.GetLength(1);
        CheckShapes(startprob, transmat, ns, nc);
        var logStartprob = new double[nc];
        for (var i = 0; i < nc; i++) logStartprob[i] = Math.Log(startprob[i]);
        var logTransmat = LogMatrix(transmat);
        var lattice = new double[ns, nc];
        for (var i = 0; i < nc; i++) lattice[0, i] = logStartprob[i] + logFrameprob[0, i];
        for (var t = 1; t < ns; t++)
        {
            for (var i = 0; i < nc; i++)
            {
                var max = double.NegativeInfinity;
                for (var j = 0; j < nc; j++) max = Math.Max(max, lattice[t - 1, j] + logTransmat[j, i]);
                lattice[t, i] = max + logFrameprob[t, i];
            }
        }
        var states = new int[ns];
        var last = 0;
        var logProb = lattice[ns - 1, 0];
        for (var i = 1; i < nc; i++)
            if (lattice[ns - 1, i] > logProb)
            {
                logProb = lattice[ns - 1, i];
                last = i;
            }
        states[ns - 1] = last;
        for (var t = ns - 2; t >= 0; t--)
        {
            var best = double.NegativeInfinity;
            var chosen = 0;
            for (var i = 0; i < nc; i++)
            {
                var candidate = lattice[t, i] + logTransmat[i, last];
                if (candidate >= best)
                {
                    best = candidate;
                    chosen = i;
                }
            }
            states[t] = last = chosen;
        }
        return (logProb, states);
    }

    /// <summary>
    /// Throws ArgumentException "shape mismatch" when fwdlattice and bwdlattice are not (ns, nc) or transmat is not nc x nc.
    /// </summary>
    private static void CheckXiShapes(double[,] fwdlattice, double[,] transmat, double[,] bwdlattice, int ns, int nc)
    {
        if (fwdlattice.GetLength(0) != ns || fwdlattice.GetLength(1) != nc
            || transmat.GetLength(0) != nc || transmat.GetLength(1) != nc
            || bwdlattice.GetLength(0) != ns || bwdlattice.GetLength(1) != nc)
            throw new ArgumentException("shape mismatch");
    }

    /// <summary>Elementwise natural log of a matrix (log of 0 is -infinity, as in the C++ helper).</summary>
    private static double[,] LogMatrix(double[,] m)
    {
        var rows = m.GetLength(0);
        var cols = m.GetLength(1);
        var result = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                result[i, j] = Math.Log(m[i, j]);
        return result;
    }
}
