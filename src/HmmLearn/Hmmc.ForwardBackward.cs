using System.Runtime.InteropServices;

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
    internal static double LogAddExp(double a, double b)
    {
        if (a == double.NegativeInfinity) return b;
        if (b == double.NegativeInfinity) return a;
        return Math.Max(a, b) + Math.Log(1.0 + Math.Exp(-Math.Abs(b - a)));
    }

    /// <summary>
    /// log(sum(exp(v))) without overflow: factor out the maximum; if the maximum is infinite (-infinity for an all -infinity input)
    /// it is returned as is; an empty input is the log of the empty sum, -infinity.
    /// </summary>
    internal static double LogSumExp(ReadOnlySpan<double> v)
    {
        if (v.IsEmpty) return double.NegativeInfinity;
        var max = v[0];
        for (var i = 1; i < v.Length; i++) if (v[i] > max) max = v[i];
        if (double.IsInfinity(max)) return max;
        var acc = 0.0;
        for (var i = 0; i < v.Length; i++) acc += Math.Exp(v[i] - max);
        return Math.Log(acc) + max;
    }

    /// <summary>
    /// Forward pass with scaling. fwd[0,i] = startprob[i] * frameprob[0,i]; fwd[t,j] = (sum over i of fwd[t-1,i] * transmat[i,j]) *
    /// frameprob[t,j]; after each step the row is multiplied by scale = 1 / (sum of the row) and scale is stored in the scaling
    /// vector. Returns the log probability of the sequence (minus the sum of the logs of the scales), the scaled lattice and the
    /// scaling vector. If the sum of a row is below 1e-300 it throws <see cref="ArithmeticException"/> with the message
    /// "forward pass failed with underflow; consider using implementation='log' instead". Shapes that do not agree (startprob of
    /// length nc, transmat nc x nc, at least one sample) throw <see cref="ArgumentException"/> with the message "shape mismatch".
    /// </summary>
    public static (double LogProb, double[,] FwdLattice, double[] Scaling) ForwardScaling(
        double[] startprob, double[,] transmat, double[,] frameprob)
    {
        var ns = frameprob.GetLength(0);
        var nc = frameprob.GetLength(1);
        CheckShapes(startprob, transmat, ns, nc);
        var fwd = new double[ns, nc];
        var scaling = new double[ns];
        var logProb = 0.0;
        for (var i = 0; i < nc; i++) fwd[0, i] = startprob[i] * frameprob[0, i];
        var scale = ScaleRow(fwd, 0, nc, ref logProb);
        scaling[0] = scale;
        for (var i = 0; i < nc; i++) fwd[0, i] *= scale;
        for (var t = 1; t < ns; t++)
        {
            for (var j = 0; j < nc; j++)
            {
                var acc = 0.0;
                for (var i = 0; i < nc; i++) acc += fwd[t - 1, i] * transmat[i, j];
                fwd[t, j] = acc * frameprob[t, j];
            }
            scale = ScaleRow(fwd, t, nc, ref logProb);
            scaling[t] = scale;
            for (var j = 0; j < nc; j++) fwd[t, j] *= scale;
        }
        return (logProb, fwd, scaling);
    }

    /// <summary>
    /// Forward pass in log space: fwd[0,i] = log(startprob[i]) + logFrameprob[0,i]; fwd[t,j] = LogSumExp over i of
    /// (fwd[t-1,i] + log(transmat[i,j])) + logFrameprob[t,j]. log(0) is -infinity. Returns the log probability of the sequence
    /// (LogSumExp of the last row) and the lattice. Shape errors as in <see cref="ForwardScaling"/>.
    /// </summary>
    public static (double LogProb, double[,] FwdLattice) ForwardLog(
        double[] startprob, double[,] transmat, double[,] logFrameprob)
    {
        var ns = logFrameprob.GetLength(0);
        var nc = logFrameprob.GetLength(1);
        CheckShapes(startprob, transmat, ns, nc);
        var fwd = new double[ns, nc];
        var buf = new double[nc];
        for (var i = 0; i < nc; i++) fwd[0, i] = Math.Log(startprob[i]) + logFrameprob[0, i];
        for (var t = 1; t < ns; t++)
        {
            for (var j = 0; j < nc; j++)
            {
                for (var i = 0; i < nc; i++) buf[i] = fwd[t - 1, i] + Math.Log(transmat[i, j]);
                fwd[t, j] = LogSumExp(buf.AsSpan()) + logFrameprob[t, j];
            }
        }
        var logProb = LogSumExp(MemoryMarshal.CreateReadOnlySpan(ref fwd[ns - 1, 0], nc));
        return (logProb, fwd);
    }

    /// <summary>
    /// Backward pass with scaling, using the scaling vector of <see cref="ForwardScaling"/>: bwd[ns-1,i] = scaling[ns-1];
    /// bwd[t,i] = (sum over j of transmat[i,j] * frameprob[t+1,j] * bwd[t+1,j]) * scaling[t]. startprob only takes part in the
    /// shape check; scaling must have one element per sample. Shape errors throw <see cref="ArgumentException"/> "shape mismatch".
    /// </summary>
    public static double[,] BackwardScaling(
        double[] startprob, double[,] transmat, double[,] frameprob, double[] scaling)
    {
        var ns = frameprob.GetLength(0);
        var nc = frameprob.GetLength(1);
        CheckShapes(startprob, transmat, ns, nc);
        if (scaling.Length != ns) throw new ArgumentException("shape mismatch");
        var bwd = new double[ns, nc];
        for (var i = 0; i < nc; i++) bwd[ns - 1, i] = scaling[ns - 1];
        for (var t = ns - 2; t >= 0; t--)
        {
            for (var i = 0; i < nc; i++)
            {
                var acc = 0.0;
                for (var j = 0; j < nc; j++) acc += transmat[i, j] * frameprob[t + 1, j] * bwd[t + 1, j];
                bwd[t, i] = acc * scaling[t];
            }
        }
        return bwd;
    }

    /// <summary>
    /// Backward pass in log space: bwd[ns-1,i] = 0; bwd[t,i] = LogSumExp over j of
    /// (log(transmat[i,j]) + logFrameprob[t+1,j] + bwd[t+1,j]). Shape errors as in <see cref="ForwardScaling"/>.
    /// </summary>
    public static double[,] BackwardLog(
        double[] startprob, double[,] transmat, double[,] logFrameprob)
    {
        var ns = logFrameprob.GetLength(0);
        var nc = logFrameprob.GetLength(1);
        CheckShapes(startprob, transmat, ns, nc);
        var bwd = new double[ns, nc];
        var buf = new double[nc];
        for (var t = ns - 2; t >= 0; t--)
        {
            for (var i = 0; i < nc; i++)
            {
                for (var j = 0; j < nc; j++) buf[j] = Math.Log(transmat[i, j]) + logFrameprob[t + 1, j] + bwd[t + 1, j];
                bwd[t, i] = LogSumExp(buf.AsSpan());
            }
        }
        return bwd;
    }

    /// <summary>
    /// Throws ArgumentException "shape mismatch" when the shapes do not agree: startprob of length nc, transmat nc x nc, at
    /// least one sample.
    /// </summary>
    private static void CheckShapes(double[] startprob, double[,] transmat, int ns, int nc)
    {
        if (ns == 0 || startprob.Length != nc || transmat.GetLength(0) != nc || transmat.GetLength(1) != nc)
            throw new ArgumentException("shape mismatch");
    }

    /// <summary>
    /// Sums row t of the lattice, throws the documented ArithmeticException when the sum is below 1e-300, adds -log(scale) to
    /// logProb and returns scale = 1 / (row sum).
    /// </summary>
    private static double ScaleRow(double[,] fwd, int t, int nc, ref double logProb)
    {
        const double minSum = 1e-300;
        var sum = 0.0;
        for (var i = 0; i < nc; i++) sum += fwd[t, i];
        if (sum < minSum)
            throw new ArithmeticException("forward pass failed with underflow; consider using implementation='log' instead");
        var scale = 1.0 / sum;
        logProb -= Math.Log(scale);
        return scale;
    }
}
