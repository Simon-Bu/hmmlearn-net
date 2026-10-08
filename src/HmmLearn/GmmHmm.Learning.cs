namespace HmmLearn;

/// <summary>The learning half of GmmHmm: initialization and M-step (port of GMMHMM._init and ._do_mstep of hmmlearn/hmm.py).</summary>
public partial class GmmHmm
{
    /// <summary>
    /// Initializes before fitting. Base initialization (start probabilities, transition matrix, number of features); then
    /// (all deterministic except the fallback below): the data covariance cv = sample covariance of the columns of x (dividing by n - 1;
    /// zeros for a single row) plus MinCovar on the diagonal; x is clustered into NComponents clusters with <see cref="KMeans.Fit"/>,
    /// each cluster with at least NMix samples is clustered again into NMix clusters whose centers are the means of that state; a smaller
    /// cluster gets NMix draws from a Gaussian centered on the mean of the NComponents centers with covariance cv (a <see cref="Random"/>
    /// seeded like the base, Cholesky factor of cv). Then, each only if needed (<c>NeedsInit</c>): Weights = 1 / NMix everywhere
    /// ('w', "weights_"); Means = the centers found above ('m', "means_"); Covars = cv distributed to the covariance type ('c', "covars_"):
    /// tied and full give the matrix itself, diag its diagonal, spherical the mean of ALL entries of cv, the same value for every state
    /// and mixture component, in the compact rank-4 shapes.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths) => throw new NotImplementedException();

    /// <summary>
    /// M-step; follow hmm.py lines 734-809 exactly. First the base update of the start probabilities and transition matrix. With
    /// nm the number of mixture components, nf the number of features and s = stats:
    /// <list type="bullet">
    /// <item>'w': w_n = s.PostMixSum + (WeightsPrior - 1); w_d[c] = s.PostSum[c] + sum over m of (WeightsPrior - 1); Weights = w_n / w_d.</item>
    /// <item>'m': m_d = s.PostMixSum + MeansWeight; entries where the mixture weight is exactly 0 and the whole row of s.MN is 0 get m_d = 1
    /// (so an unused component gets mean 0, not NaN); Means = s.MN / m_d.</item>
    /// <item>'c' (with the means just updated; mu = MeansPrior, lambda = MeansWeight, centered_means = Means - mu, ndot = outer product):
    /// full: c_n = prior + lambda * ndot(centered_means) + s.CN; c_d = s.PostMixSum + 1 + nu + nf + 1; Covars = c_n / c_d;
    /// diag: c_n = lambda * centered_means^2 + 2 * beta + s.CN; c_d = s.PostMixSum + 1 + 2 * (alpha + 1);
    /// spherical: c_n = lambda * squared norm of centered_means + 2 * beta + s.CN; c_d = nf * (s.PostMixSum + 1) + 2 * (alpha + 1);
    /// tied: c_n = sum over m of lambda * ndot(centered_means) + prior + s.CN; c_d = s.PostSum + nm + nu + nf + 1 (per state),
    /// where for full and tied (alpha, beta) are unused and "prior" is <see cref="EffectiveCovarsPrior"/> added to EVERY entry of the
    /// matrix, nu = <see cref="EffectiveCovarsWeight"/>; for diag and spherical alpha = <see cref="EffectiveCovarsPrior"/> and
    /// beta = <see cref="EffectiveCovarsWeight"/>. Check the exact roles of the names in hmm.py before writing.</item>
    /// </list>
    /// The results replace Weights, Means and Covars (new arrays, compact rank-4 shapes).
    /// </summary>
    protected override void DoMStep(SufficientStatistics stats) => throw new NotImplementedException();
}
