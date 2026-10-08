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
    protected override void Init(double[,] x, int[] lengths)
    {
        base.Init(x, lengths);
        if (NeedsInit('m', "means_", Means != null))
            Means = KMeans.Fit(x, NComponents).Centers;
        if (NeedsInit('c', "covars_", Covars != null))
            Covars = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(DataCovariance(x), CovarianceType, NComponents);
    }

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
    protected override void DoMStep(SufficientStatistics stats)
    {
        base.DoMStep(stats);
        var gs = (GaussianStatistics)stats;
        if (Params.Contains('m'))
            UpdateMeans(gs);
        if (Params.Contains('c'))
            Covars = CovarianceType is CovarianceType.Spherical or CovarianceType.Diag
                ? UpdateSphericalOrDiagCovars(gs)
                : UpdateTiedOrFullCovars(gs);
    }

    /// <summary>The means update of the M-step (hmmlearn's <c>means_weight * means_prior + obs</c> over <c>means_weight + post</c>), in a new matrix.</summary>
    private void UpdateMeans(GaussianStatistics gs)
    {
        var nFeatures = NFeatures!.Value;
        var means = new double[NComponents, nFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var f = 0; f < nFeatures; f++)
                means[c, f] = (MeansWeight * MeansPrior + gs.Obs[c, f]) / (MeansWeight + gs.Post[c]);
        Means = means;
    }

    /// <summary>
    /// The Spherical and Diag covariance update: per feature c_n = meansWeight * meandiff^2 + Obs2 - 2 * means * Obs + means^2 * post,
    /// over max(max(covarsWeight - 1, 0) + post, 1e-5) plus covarsPrior; Spherical keeps the mean over the features.
    /// </summary>
    private double[,,] UpdateSphericalOrDiagCovars(GaussianStatistics gs)
    {
        var nFeatures = NFeatures!.Value;
        var diag = new double[NComponents, nFeatures];
        for (var c = 0; c < NComponents; c++)
        {
            var denom = Math.Max(CovarsWeight - 1.0, 0.0) + gs.Post[c];
            for (var f = 0; f < nFeatures; f++)
            {
                var meandiff = Means![c, f] - MeansPrior;
                var c_n = MeansWeight * meandiff * meandiff
                    + gs.Obs2[c, f]
                    - 2.0 * Means[c, f] * gs.Obs[c, f]
                    + Means[c, f] * Means[c, f] * gs.Post[c];
                diag[c, f] = (CovarsPrior + c_n) / Math.Max(denom, 1e-5);
            }
        }

        if (CovarianceType == CovarianceType.Spherical)
        {
            var result = new double[NComponents, 1, 1];
            for (var c = 0; c < NComponents; c++)
            {
                var mean = 0.0;
                for (var f = 0; f < nFeatures; f++)
                    mean += diag[c, f];
                result[c, 0, 0] = mean / nFeatures;
            }
            return result;
        }

        var compact = new double[NComponents, 1, nFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var f = 0; f < nFeatures; f++)
                compact[c, 0, f] = diag[c, f];
        return compact;
    }

    /// <summary>
    /// The Tied and Full covariance update: per state c_n = meansWeight * outer(meandiff, meandiff) + ObsObsT - outer(Obs, means) -
    /// its transpose + outer(means, means) * post, over max(covarsWeight - n_features, 0) + post (the sum of the posts for Tied) plus the
    /// scalar covarsPrior added to every entry.
    /// </summary>
    private double[,,] UpdateTiedOrFullCovars(GaussianStatistics gs)
    {
        var nFeatures = NFeatures!.Value;
        var per = new double[NComponents, nFeatures, nFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var i = 0; i < nFeatures; i++)
                for (var j = 0; j < nFeatures; j++)
                {
                    var meandiffI = Means![c, i] - MeansPrior;
                    var meandiffJ = Means[c, j] - MeansPrior;
                    per[c, i, j] = MeansWeight * meandiffI * meandiffJ
                        + gs.ObsObsT![c, i, j]
                        - gs.Obs[c, i] * Means[c, j]
                        - gs.Obs[c, j] * Means[c, i]
                        + Means[c, i] * Means[c, j] * gs.Post[c];
                }
        var cvweight = Math.Max(CovarsWeight - nFeatures, 0.0);

        if (CovarianceType == CovarianceType.Tied)
        {
            var totalPost = 0.0;
            for (var c = 0; c < NComponents; c++)
                totalPost += gs.Post[c];
            var result = new double[1, nFeatures, nFeatures];
            for (var i = 0; i < nFeatures; i++)
                for (var j = 0; j < nFeatures; j++)
                {
                    var sum = 0.0;
                    for (var c = 0; c < NComponents; c++)
                        sum += per[c, i, j];
                    result[0, i, j] = (CovarsPrior + sum) / (cvweight + totalPost);
                }
            return result;
        }

        var full = new double[NComponents, nFeatures, nFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var i = 0; i < nFeatures; i++)
                for (var j = 0; j < nFeatures; j++)
                    full[c, i, j] = (CovarsPrior + per[c, i, j]) / (cvweight + gs.Post[c]);
        return full;
    }

    /// <summary>
    /// The data covariance used to initialize the covariances (hmmlearn's <c>np.cov(X.T) + min_covar * eye</c>): the sample covariance
    /// of the columns of x, dividing by n - 1 (zeros when x has a single row), plus MinCovar on the diagonal.
    /// </summary>
    private double[,] DataCovariance(double[,] x)
    {
        var n = x.GetLength(0);
        var nf = x.GetLength(1);
        var mean = new double[nf];
        for (var i = 0; i < n; i++)
            for (var f = 0; f < nf; f++)
                mean[f] += x[i, f] / n;
        var cov = new double[nf, nf];
        for (var f = 0; f < nf; f++)
            for (var g = 0; g < nf; g++)
            {
                if (n > 1)
                {
                    var sum = 0.0;
                    for (var i = 0; i < n; i++)
                        sum += (x[i, f] - mean[f]) * (x[i, g] - mean[g]);
                    cov[f, g] = sum / (n - 1);
                }
                if (f == g)
                    cov[f, g] += MinCovar;
            }
        return cov;
    }
}
