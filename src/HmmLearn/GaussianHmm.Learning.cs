namespace HmmLearn;

/// <summary>The learning half of GaussianHmm: initialization and M-step (port of GaussianHMM._init and ._do_mstep of hmmlearn/hmm.py).</summary>
public partial class GaussianHmm
{
    /// <summary>
    /// Initializes before fitting. Base initialization (start probabilities, transition matrix, number of features); then, if
    /// <c>NeedsInit('m', "means_", Means != null)</c>, Means = the centers of <see cref="KMeans.Fit"/> with NComponents clusters on x;
    /// then, if <c>NeedsInit('c', "covars_", Covars != null)</c>, the data covariance of x (the sample covariance of the columns, dividing
    /// by n - 1; zeros if x has a single row) plus MinCovar on the diagonal is distributed to the covariance type with
    /// <see cref="InternalUtils.DistributeCovarMatrixToMatchCovarianceType"/> and stored in Covars.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths) => throw new NotImplementedException();

    /// <summary>
    /// M-step. First the base update of the start probabilities and the transition matrix. Then, with denom[c] = stats.Post[c]:
    /// <list type="bullet">
    /// <item>If Params contains 'm': Means[c, f] = (MeansWeight * MeansPrior + stats.Obs[c, f]) / (MeansWeight + denom[c]).</item>
    /// <item>If Params contains 'c', with meandiff = Means - MeansPrior (the means just updated if 'm' is in Params, else the current ones):
    /// for Spherical and Diag: c_n[c, f] = MeansWeight * meandiff^2 + stats.Obs2[c, f] - 2 * Means[c, f] * stats.Obs[c, f] + Means[c, f]^2 * denom[c];
    /// c_d[c] = max(CovarsWeight - 1, 0) + denom[c]; covar[c, f] = (CovarsPrior + c_n[c, f]) / max(c_d[c], 1e-5); for Spherical the covariance
    /// of a state is then the mean over the features of covar[c, f] (a single number per state).
    /// For Tied and Full: c_n[c] = MeansWeight * outer(meandiff[c], meandiff[c]) + stats.ObsObsT[c] - outer(stats.Obs[c], Means[c]) -
    /// transpose of that + outer(Means[c], Means[c]) * denom[c]; cvweight = max(CovarsWeight - n_features, 0); Tied: covar =
    /// (CovarsPrior + sum over c of c_n[c]) / (cvweight + sum of denom), one (nf, nf) matrix; Full: covar[c] = (CovarsPrior + c_n[c]) /
    /// (cvweight + denom[c]). CovarsPrior is a scalar added to EVERY entry of the matrix, off-diagonal ones included (hmmlearn does this).</item>
    /// </list>
    /// The result is stored in Covars in the compact uniform shapes (Spherical (nc, 1, 1), Diag (nc, 1, nf), Full (nc, nf, nf), Tied (1, nf, nf)).
    /// </summary>
    protected override void DoMStep(SufficientStatistics stats) => throw new NotImplementedException();
}
