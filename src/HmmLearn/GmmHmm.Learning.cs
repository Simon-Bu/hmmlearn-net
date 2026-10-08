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
    protected override void Init(double[,] x, int[] lengths)
    {
        base.Init(x, lengths);
        var nc = NComponents;
        var nf = NFeatures!.Value;
        var nm = NMix;

        if (NeedsInit('w', "weights_", Weights != null))
            Weights = UniformWeights(nc, nm);

        var meansNeeded = NeedsInit('m', "means_", Means != null);
        var covarsNeeded = NeedsInit('c', "covars_", Covars != null);
        if (meansNeeded || covarsNeeded)
        {
            var (means, cv) = InitializeMeans(x, nc, nf, nm);
            if (meansNeeded)
                Means = means;
            if (covarsNeeded)
                Covars = DistributeCovariance(cv ?? DataCovariance(x), nc, nm);
        }
    }

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
    protected override void DoMStep(SufficientStatistics stats)
    {
        base.DoMStep(stats);
        var gs = (GmmStatistics)stats;
        var nc = NComponents;
        var nf = NFeatures!.Value;
        var nm = NMix;

        if (Params.Contains('w'))
        {
            var weights = new double[nc, nm];
            for (var c = 0; c < nc; c++)
            {
                var wD = gs.PostSum[c] + nm * (WeightsPrior - 1.0);
                for (var m = 0; m < nm; m++)
                    weights[c, m] = (gs.PostMixSum[c, m] + (WeightsPrior - 1.0)) / wD;
            }
            Weights = weights;
        }

        if (Params.Contains('m'))
        {
            var means = new double[nc, nm, nf];
            for (var c = 0; c < nc; c++)
                for (var m = 0; m < nm; m++)
                {
                    var mD = gs.PostMixSum[c, m] + MeansWeight;
                    if (Weights![c, m] == 0.0)
                    {
                        var allZero = true;
                        for (var f = 0; f < nf; f++)
                            if (gs.MN[c, m, f] != 0.0)
                            {
                                allZero = false;
                                break;
                            }
                        if (allZero)
                            mD = 1.0;
                    }
                    for (var f = 0; f < nf; f++)
                        means[c, m, f] = gs.MN[c, m, f] / mD;
                }
            Means = means;
        }

        if (Params.Contains('c'))
            Covars = UpdateCovars(gs, nc, nm, nf);
    }

    /// <summary>1 / NMix in a new (nc, nm) matrix.</summary>
    private static double[,] UniformWeights(int nc, int nm)
    {
        var weights = new double[nc, nm];
        for (var c = 0; c < nc; c++)
            for (var m = 0; m < nm; m++)
                weights[c, m] = 1.0 / nm;
        return weights;
    }

    /// <summary>
    /// The two-level k-means of the means (hmm.py's _init): x clustered into NComponents clusters; each cluster with at least NMix
    /// samples is clustered again into NMix clusters whose centers are that state's means; a smaller cluster gets NMix draws from the
    /// Gaussian centered on the mean of the NComponents centers with covariance cv (the data covariance, computed lazily). Returns the
    /// means (nc, nm, nf) and cv (null when no draw was needed).
    /// </summary>
    private (double[,,] Means, double[,]? Covariance) InitializeMeans(double[,] x, int nc, int nf, int nm)
    {
        var (mainCenters, labels) = KMeans.Fit(x, nc);
        var mainCentroid = new double[nf];
        for (var f = 0; f < nf; f++)
            for (var c = 0; c < nc; c++)
                mainCentroid[f] += mainCenters[c, f] / nc;

        var means = new double[nc, nm, nf];
        var random = RandomState.HasValue ? new Random(RandomState.Value) : Random.Shared;
        double[,]? cv = null;
        for (var label = 0; label < nc; label++)
        {
            var rows = new List<int>();
            for (var i = 0; i < x.GetLength(0); i++)
                if (labels[i] == label)
                    rows.Add(i);
            if (rows.Count >= nm)
            {
                var cluster = new double[rows.Count, nf];
                for (var i = 0; i < rows.Count; i++)
                    for (var f = 0; f < nf; f++)
                        cluster[i, f] = x[rows[i], f];
                var centers = KMeans.Fit(cluster, nm).Centers;
                for (var m = 0; m < nm; m++)
                    for (var f = 0; f < nf; f++)
                        means[label, m, f] = centers[m, f];
            }
            else
            {
                cv ??= DataCovariance(x);
                for (var m = 0; m < nm; m++)
                {
                    var draw = GaussianDraw(mainCentroid, cv, random);
                    for (var f = 0; f < nf; f++)
                        means[label, m, f] = draw[f];
                }
            }
        }
        return (means, cv);
    }

    /// <summary>
    /// One draw from the Gaussian of mean <paramref name="mean"/> and covariance <paramref name="covar"/>: the mean plus L z, with z
    /// standard normal numbers (Box-Muller) and L the lower Cholesky factor of covar.
    /// </summary>
    private static double[] GaussianDraw(double[] mean, double[,] covar, Random random)
    {
        var nf = mean.Length;
        var factor = new double[nf, nf];
        for (var i = 0; i < nf; i++)
            for (var j = 0; j <= i; j++)
            {
                var sum = covar[i, j];
                for (var k = 0; k < j; k++)
                    sum -= factor[i, k] * factor[j, k];
                factor[i, j] = j == i ? Math.Sqrt(sum) : sum / factor[j, j];
            }

        var z = new double[nf];
        var hasSpare = false;
        var spare = 0.0;
        for (var i = 0; i < nf; i++)
        {
            if (hasSpare)
            {
                z[i] = spare;
                hasSpare = false;
                continue;
            }
            var u1 = 1.0 - random.NextDouble(); // (0, 1], so log(u1) is finite
            var u2 = random.NextDouble();
            var radius = Math.Sqrt(-2.0 * Math.Log(u1));
            z[i] = radius * Math.Cos(2.0 * Math.PI * u2);
            spare = radius * Math.Sin(2.0 * Math.PI * u2);
            hasSpare = true;
        }

        var sample = new double[nf];
        for (var i = 0; i < nf; i++)
        {
            var value = mean[i];
            for (var k = 0; k <= i; k++)
                value += factor[i, k] * z[k];
            sample[i] = value;
        }
        return sample;
    }

    /// <summary>
    /// cv in the compact rank-4 shape of the covariance type: tied and full give the matrix itself (tied: one per state, no mixture
    /// axis), diag its diagonal, spherical the mean of ALL entries of cv, the same value for every state and mixture component.
    /// </summary>
    private double[,,,] DistributeCovariance(double[,] cv, int nc, int nm)
    {
        var nf = cv.GetLength(0);
        var (d1, d2, d3) = ExpectedCovarsShape(nm, nf);
        var result = new double[nc, d1, d2, d3];
        switch (CovarianceType)
        {
            case CovarianceType.Spherical:
                var mean = 0.0;
                for (var f = 0; f < nf; f++)
                    for (var g = 0; g < nf; g++)
                        mean += cv[f, g];
                mean /= nf * nf;
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                        result[c, m, 0, 0] = mean;
                break;
            case CovarianceType.Diag:
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                        for (var f = 0; f < nf; f++)
                            result[c, m, 0, f] = cv[f, f];
                break;
            case CovarianceType.Full:
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                        for (var f = 0; f < nf; f++)
                            for (var g = 0; g < nf; g++)
                                result[c, m, f, g] = cv[f, g];
                break;
            default: // Tied: one matrix per state (no mixture axis).
                for (var c = 0; c < nc; c++)
                    for (var f = 0; f < nf; f++)
                        for (var g = 0; g < nf; g++)
                            result[c, 0, f, g] = cv[f, g];
                break;
        }
        return result;
    }

    /// <summary>
    /// The data covariance of x: the sample covariance of the columns (dividing by n - 1; zeros for a single row) plus MinCovar on the
    /// diagonal.
    /// </summary>
    private double[,] DataCovariance(double[,] x)
    {
        var n = x.GetLength(0);
        var nf = x.GetLength(1);
        var mean = new double[nf];
        for (var i = 0; i < n; i++)
            for (var f = 0; f < nf; f++)
                mean[f] += x[i, f];
        var cov = new double[nf, nf];
        for (var f = 0; f < nf; f++)
            for (var g = 0; g < nf; g++)
            {
                if (n > 1)
                {
                    var sum = 0.0;
                    for (var i = 0; i < n; i++)
                        sum += (x[i, f] - mean[f] / n) * (x[i, g] - mean[g] / n);
                    cov[f, g] = sum / (n - 1);
                }
                if (f == g)
                    cov[f, g] += MinCovar;
            }
        return cov;
    }

    /// <summary>
    /// The covariance update of the M-step (hmm.py lines 759-809), with the means just updated if 'm' is in Params: centered_means =
    /// Means - MeansPrior, mu the prior mean and lambda the prior precision.
    /// <list type="bullet">
    /// <item>Full: (prior + lambda * centered_i * centered_j + s.CN[c, m, i, j]) / (s.PostMixSum[c, m] + 1 + nu + nf + 1), the prior
    /// added to every entry.</item>
    /// <item>Diag: (lambda * centered_f^2 + 2 * beta + s.CN[c, m, 0, f]) / (s.PostMixSum[c, m] + 1 + 2 * (alpha + 1)).</item>
    /// <item>Spherical: (lambda * ||centered||^2 + 2 * beta + s.CN[c, m, 0, 0]) / (nf * (s.PostMixSum[c, m] + 1) + 2 * (alpha + 1)).</item>
    /// <item>Tied: (prior + sum over m of lambda * centered_{c,m,i} * centered_{c,m,j} + s.CN[c, 0, i, j]) / (s.PostSum[c] + nm + nu + nf + 1),
    /// one matrix per state at index 0 of the mixture axis.</item>
    /// </list>
    /// (alpha, beta) are EffectiveCovarsPrior/Weight for diag and spherical; (prior, nu) for full and tied, the prior a scalar added to
    /// every entry of the matrix.
    /// </summary>
    private double[,,,] UpdateCovars(GmmStatistics gs, int nc, int nm, int nf)
    {
        var means = Means!;
        var mu = MeansPrior;
        var lambda = MeansWeight;
        var (d1, d2, d3) = ExpectedCovarsShape(nm, nf);
        var result = new double[nc, d1, d2, d3];

        switch (CovarianceType)
        {
            case CovarianceType.Full:
            {
                var prior = EffectiveCovarsPrior;
                var nu = EffectiveCovarsWeight;
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                        for (var i = 0; i < nf; i++)
                        {
                            var ci = means[c, m, i] - mu;
                            for (var j = 0; j < nf; j++)
                            {
                                var cj = means[c, m, j] - mu;
                                result[c, m, i, j] =
                                    (prior + lambda * ci * cj + gs.CN[c, m, i, j]) /
                                    (gs.PostMixSum[c, m] + 1.0 + nu + nf + 1.0);
                            }
                        }
                break;
            }
            case CovarianceType.Diag:
            {
                var alpha = EffectiveCovarsPrior;
                var beta = EffectiveCovarsWeight;
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                        for (var f = 0; f < nf; f++)
                        {
                            var ci = means[c, m, f] - mu;
                            result[c, m, 0, f] =
                                (lambda * ci * ci + 2.0 * beta + gs.CN[c, m, 0, f]) /
                                (gs.PostMixSum[c, m] + 1.0 + 2.0 * (alpha + 1.0));
                        }
                break;
            }
            case CovarianceType.Spherical:
            {
                var alpha = EffectiveCovarsPrior;
                var beta = EffectiveCovarsWeight;
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                    {
                        var norm2 = 0.0;
                        for (var f = 0; f < nf; f++)
                        {
                            var ci = means[c, m, f] - mu;
                            norm2 += ci * ci;
                        }
                        result[c, m, 0, 0] =
                            (lambda * norm2 + 2.0 * beta + gs.CN[c, m, 0, 0]) /
                            (nf * (gs.PostMixSum[c, m] + 1.0) + 2.0 * (alpha + 1.0));
                    }
                break;
            }
            default: // Tied: one matrix per state, the sum over the mixture components.
            {
                var prior = EffectiveCovarsPrior;
                var nu = EffectiveCovarsWeight;
                for (var c = 0; c < nc; c++)
                {
                    var cD = gs.PostSum[c] + nm + nu + nf + 1.0;
                    for (var i = 0; i < nf; i++)
                        for (var j = 0; j < nf; j++)
                        {
                            var sum = prior + gs.CN[c, 0, i, j];
                            for (var m = 0; m < nm; m++)
                                sum += lambda * (means[c, m, i] - mu) * (means[c, m, j] - mu);
                            result[c, 0, i, j] = sum / cD;
                        }
                }
                break;
            }
        }
        return result;
    }
}
