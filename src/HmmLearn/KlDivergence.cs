using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace HmmLearn;

/// <summary>
/// Kullback-Leibler divergences between the distributions used by the variational models (port of hmmlearn/_kl_divergence.py,
/// based on http://www.fil.ion.ucl.ac.uk/~wpenny/publications/densities.ps). Each method returns KL(q || p). Special functions
/// (log-gamma, digamma), matrix inverses and traces may come from MathNet.Numerics; log-determinants from <see cref="InternalUtils.LogDet"/>.
/// None of the methods checks that the result is not negative (hmmlearn asserts it for the normal and gamma cases; a tiny negative
/// value caused by rounding is returned as it is).
/// </summary>
public static class KlDivergence
{
    /// <summary>
    /// KL between two Dirichlet distributions with parameter vectors q and p of the same length: with qsum and psum the sums,
    /// logGamma(qsum) - logGamma(psum) - sum(logGamma(q[j]) - logGamma(p[j])) + sum((q[j] - p[j]) * (digamma(q[j]) - digamma(qsum))).
    /// Throws <see cref="ArgumentException"/> if the lengths differ.
    /// </summary>
    public static double KlDirichlet(double[] q, double[] p)
    {
        if (q.Length != p.Length)
            throw new ArgumentException($"q and p must have the same length, got {q.Length} and {p.Length}", nameof(q));
        var qsum = 0.0;
        var psum = 0.0;
        for (var j = 0; j < q.Length; j++)
        {
            qsum += q[j];
            psum += p[j];
        }
        var gammaSum = 0.0;
        var linearSum = 0.0;
        var digammaQsum = SpecialFunctions.DiGamma(qsum);
        for (var j = 0; j < q.Length; j++)
        {
            gammaSum += SpecialFunctions.GammaLn(q[j]) - SpecialFunctions.GammaLn(p[j]);
            linearSum += (q[j] - p[j]) * (SpecialFunctions.DiGamma(q[j]) - digammaQsum);
        }
        return SpecialFunctions.GammaLn(qsum) - SpecialFunctions.GammaLn(psum) - gammaSum + linearSum;
    }

    /// <summary>KL between two univariate normal distributions: log(varianceP / varianceQ) / 2 + ((meanQ - meanP)^2 + varianceQ) / (2 * varianceP) - 1/2.</summary>
    public static double KlNormalDistribution(double meanQ, double varianceQ, double meanP, double varianceP)
        => Math.Log(varianceP / varianceQ) / 2
           + ((meanQ - meanP) * (meanQ - meanP) + varianceQ) / (2 * varianceP)
           - 0.5;

    /// <summary>
    /// KL between two multivariate normal distributions of dimension D: 1/2 * (logdet(covarP) - logdet(covarQ) + trace(inverse(covarP) *
    /// covarQ) + (meanQ - meanP)^T * inverse(covarP) * (meanQ - meanP) - D). Throws <see cref="ArgumentException"/> for inconsistent dimensions.
    /// </summary>
    public static double KlMultivariateNormalDistribution(double[] meanQ, double[,] covarQ, double[] meanP, double[,] covarP)
    {
        var n = meanQ.Length;
        if (meanP.Length != n)
            throw new ArgumentException($"meanQ and meanP must have the same length, got {n} and {meanP.Length}", nameof(meanP));
        if (covarQ.GetLength(0) != covarQ.GetLength(1) || covarQ.GetLength(0) != n)
            throw new ArgumentException($"covarQ must be a square {n} x {n} matrix, got {covarQ.GetLength(0)} x {covarQ.GetLength(1)}", nameof(covarQ));
        if (covarP.GetLength(0) != covarP.GetLength(1) || covarP.GetLength(0) != n)
            throw new ArgumentException($"covarP must be a square {n} x {n} matrix, got {covarP.GetLength(0)} x {covarP.GetLength(1)}", nameof(covarP));
        var precisionP = Matrix<double>.Build.DenseOfArray(covarP).Inverse();
        var trace = (precisionP * Matrix<double>.Build.DenseOfArray(covarQ)).Trace();
        var quadratic = 0.0;
        for (var i = 0; i < n; i++)
        {
            var diffI = meanQ[i] - meanP[i];
            for (var j = 0; j < n; j++)
                quadratic += diffI * precisionP[i, j] * (meanQ[j] - meanP[j]);
        }
        return 0.5 * (InternalUtils.LogDet(covarP) - InternalUtils.LogDet(covarQ) + trace + quadratic - n);
    }

    /// <summary>
    /// KL between two gamma distributions given by (b, c) pairs: (bQ - bP) * digamma(bQ) - logGamma(bQ) + logGamma(bP) +
    /// bP * (log(cQ) - log(cP)) + bQ * (cP - cQ) / cQ.
    /// </summary>
    public static double KlGammaDistribution(double bQ, double cQ, double bP, double cP)
        => (bQ - bP) * SpecialFunctions.DiGamma(bQ)
           - SpecialFunctions.GammaLn(bQ) + SpecialFunctions.GammaLn(bP)
           + bP * (Math.Log(cQ) - Math.Log(cP))
           + bQ * (cP - cQ) / cQ;

    /// <summary>
    /// KL between two Wishart distributions of dimension D (degrees of freedom dof, scale matrices D x D):
    /// (dofQ - dofP) / 2 * E(dofQ, scaleQ) - D * dofQ / 2 + dofQ / 2 * trace(scaleP * inverse(scaleQ)) + logZ(dofP, scaleP) - logZ(dofQ, scaleQ),
    /// where E(dof, scale) = -logdet(scale / 2) + sum over i = 0..D-1 of digamma((dof - i) / 2) and
    /// logZ(dof, scale) = D * (D - 1) / 4 * log(pi) - dof / 2 * logdet(scale / 2) + sum over i = 0..D-1 of logGamma((dof - i) / 2).
    /// Reference: Ji, Krishnapuram and Carin, IEEE TPAMI 28(4), 2006. Throws <see cref="ArgumentException"/> for scale matrices that are
    /// not square of the same size.
    /// </summary>
    public static double KlWishartDistribution(double dofQ, double[,] scaleQ, double dofP, double[,] scaleP)
    {
        var d = scaleQ.GetLength(0);
        if (scaleQ.GetLength(1) != d || scaleP.GetLength(0) != d || scaleP.GetLength(1) != d)
            throw new ArgumentException(
                $"scaleQ and scaleP must be square matrices of the same size, got {scaleQ.GetLength(0)} x {scaleQ.GetLength(1)} and {scaleP.GetLength(0)} x {scaleP.GetLength(1)}",
                nameof(scaleQ));
        var trace = (Matrix<double>.Build.DenseOfArray(scaleP) * Matrix<double>.Build.DenseOfArray(scaleQ).Inverse()).Trace();
        return (dofQ - dofP) / 2 * WishartE(dofQ, scaleQ)
               - d * dofQ / 2
               + dofQ / 2 * trace
               + WishartLogZ(dofP, scaleP) - WishartLogZ(dofQ, scaleQ);
    }

    /// <summary>E(dof, scale) = -logdet(scale / 2) + the sum over i = 0..D-1 of digamma((dof - i) / 2).</summary>
    private static double WishartE(double dof, double[,] scale)
    {
        var sum = 0.0;
        for (var i = 0; i < scale.GetLength(0); i++)
            sum += SpecialFunctions.DiGamma((dof - i) / 2);
        return -InternalUtils.LogDet(Half(scale)) + sum;
    }

    /// <summary>logZ(dof, scale) = D * (D - 1) / 4 * log(pi) - dof / 2 * logdet(scale / 2) + the sum over i = 0..D-1 of logGamma((dof - i) / 2).</summary>
    private static double WishartLogZ(double dof, double[,] scale)
    {
        var d = scale.GetLength(0);
        var sum = 0.0;
        for (var i = 0; i < d; i++)
            sum += SpecialFunctions.GammaLn((dof - i) / 2);
        return d * (d - 1) / 4.0 * Math.Log(Math.PI) - dof / 2 * InternalUtils.LogDet(Half(scale)) + sum;
    }

    /// <summary>Every entry of a divided by 2 (a new array).</summary>
    private static double[,] Half(double[,] a)
    {
        var result = new double[a.GetLength(0), a.GetLength(1)];
        for (var i = 0; i < a.GetLength(0); i++)
            for (var j = 0; j < a.GetLength(1); j++)
                result[i, j] = a[i, j] / 2;
        return result;
    }
}
