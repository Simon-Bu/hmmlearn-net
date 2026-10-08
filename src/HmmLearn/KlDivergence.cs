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
    public static double KlDirichlet(double[] q, double[] p) => throw new NotImplementedException();

    /// <summary>KL between two univariate normal distributions: log(varianceP / varianceQ) / 2 + ((meanQ - meanP)^2 + varianceQ) / (2 * varianceP) - 1/2.</summary>
    public static double KlNormalDistribution(double meanQ, double varianceQ, double meanP, double varianceP) => throw new NotImplementedException();

    /// <summary>
    /// KL between two multivariate normal distributions of dimension D: 1/2 * (logdet(covarP) - logdet(covarQ) + trace(inverse(covarP) *
    /// covarQ) + (meanQ - meanP)^T * inverse(covarP) * (meanQ - meanP) - D). Throws <see cref="ArgumentException"/> for inconsistent dimensions.
    /// </summary>
    public static double KlMultivariateNormalDistribution(double[] meanQ, double[,] covarQ, double[] meanP, double[,] covarP)
        => throw new NotImplementedException();

    /// <summary>
    /// KL between two gamma distributions given by (b, c) pairs: (bQ - bP) * digamma(bQ) - logGamma(bQ) + logGamma(bP) +
    /// bP * (log(cQ) - log(cP)) + bQ * (cP - cQ) / cQ.
    /// </summary>
    public static double KlGammaDistribution(double bQ, double cQ, double bP, double cP) => throw new NotImplementedException();

    /// <summary>
    /// KL between two Wishart distributions of dimension D (degrees of freedom dof, scale matrices D x D):
    /// (dofQ - dofP) / 2 * E(dofQ, scaleQ) - D * dofQ / 2 + dofQ / 2 * trace(scaleP * inverse(scaleQ)) + logZ(dofP, scaleP) - logZ(dofQ, scaleQ),
    /// where E(dof, scale) = -logdet(scale / 2) + sum over i = 0..D-1 of digamma((dof - i) / 2) and
    /// logZ(dof, scale) = D * (D - 1) / 4 * log(pi) - dof / 2 * logdet(scale / 2) + sum over i = 0..D-1 of logGamma((dof - i) / 2).
    /// Reference: Ji, Krishnapuram and Carin, IEEE TPAMI 28(4), 2006.
    /// </summary>
    public static double KlWishartDistribution(double dofQ, double[,] scaleQ, double dofP, double[,] scaleP) => throw new NotImplementedException();
}
