using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace HmmLearn;

/// <summary>
/// Hidden Markov model with Gaussian emissions learned by Variational Bayes (port of VariationalGaussianHMM of hmmlearn/vhmm.py and of
/// BaseGaussianHMM of hmmlearn/_emissions.py). Each state has a Normal prior and posterior on its mean (precision scale beta) and a
/// Wishart prior and posterior on its precision (degrees of freedom dof, inverse scale matrix scale); the E-step uses the subnormalized
/// emission likelihood built from the posteriors, the M-step (VariationalGaussianHmm.Learning.cs) updates them. <see cref="Means"/> is the
/// posterior mean and <see cref="Covars"/> the point estimate scale / dof, both used for scoring, decoding and sampling.
/// Shapes: means (n_components, n_features); beta (n_components); dof (n_components), or one number for Tied; scale, covars: the compact
/// uniform shapes of oracle/README.md: spherical (nc, 1, 1), diag (nc, 1, nf), full (nc, nf, nf), tied (1, nf, nf).
/// </summary>
public partial class VariationalGaussianHmm : VariationalBaseHmm
{
    private double[,,]? _covars;

    /// <summary>
    /// Creates the model; the defaults are those of hmmlearn (full covariances, parameter letters "stmc"). The prior arguments are
    /// copied by <see cref="Init"/> when it is time to initialize and null means the default documented there.
    /// </summary>
    public VariationalGaussianHmm(int nComponents = 1, CovarianceType covarianceType = CovarianceType.Full, double? startProbPrior = null,
        double? transMatPrior = null, double[,]? meansPrior = null, double[]? betaPrior = null, double[]? dofPrior = null,
        double[,,]? scalePrior = null, DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi, int? randomState = null, int nIter = 100,
        double tol = 1e-6, bool verbose = false, string @params = "stmc", string initParams = "stmc",
        Implementation implementation = Implementation.Log)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
    {
        CovarianceType = covarianceType;
        MeansPrior = meansPrior;
        BetaPrior = betaPrior;
        DofPrior = dofPrior;
        ScalePrior = scalePrior;
    }

    /// <summary>The structure of the covariance parameters.</summary>
    public CovarianceType CovarianceType { get; set; }

    /// <summary>Prior of the means used by <see cref="Init"/>, shape (n_components, n_features); null means the mean of the data for every state.</summary>
    public double[,]? MeansPrior { get; set; }

    /// <summary>Prior of beta used by <see cref="Init"/>, length n_components; null means 1 for every state.</summary>
    public double[]? BetaPrior { get; set; }

    /// <summary>Prior of the degrees of freedom used by <see cref="Init"/>: n_components numbers (one number for Tied); null means n_features for every state (n_features for Tied).</summary>
    public double[]? DofPrior { get; set; }

    /// <summary>
    /// Prior of the Wishart scale used by <see cref="Init"/>, in the compact uniform shape of the covariance type; null means 1e-3 times the
    /// identity matrix for Full and Tied and 1e-3 for Diag and Spherical.
    /// </summary>
    public double[,,]? ScalePrior { get; set; }

    /// <summary>Mean of the Normal prior of the means, shape (n_components, n_features) (hmmlearn's means_prior_); null until initialized or set.</summary>
    public double[,]? MeansPriorValues { get; set; }

    /// <summary>Mean of the Normal posterior of the means, shape (n_components, n_features) (hmmlearn's means_posterior_); null until initialized or set.</summary>
    public double[,]? MeansPosterior { get; set; }

    /// <summary>Prior scale of the variance over the means, length n_components (hmmlearn's beta_prior_); null until initialized or set.</summary>
    public double[]? BetaPriorValues { get; set; }

    /// <summary>Posterior scale of the variance over the means, length n_components (hmmlearn's beta_posterior_); null until initialized or set.</summary>
    public double[]? BetaPosterior { get; set; }

    /// <summary>Prior degrees of freedom of the Wishart distributions: n_components numbers, one number for Tied (hmmlearn's dof_prior_); null until initialized or set.</summary>
    public double[]? DofPriorValues { get; set; }

    /// <summary>Posterior degrees of freedom of the Wishart distributions: n_components numbers, one number for Tied (hmmlearn's dof_posterior_); null until initialized or set.</summary>
    public double[]? DofPosterior { get; set; }

    /// <summary>Prior inverse scale of the Wishart distributions, compact uniform shape (hmmlearn's scale_prior_); null until initialized or set.</summary>
    public double[,,]? ScalePriorValues { get; set; }

    /// <summary>Posterior inverse scale of the Wishart distributions, compact uniform shape (hmmlearn's scale_posterior_); null until initialized or set.</summary>
    public double[,,]? ScalePosterior { get; set; }

    /// <summary>
    /// Point estimate of the covariances (hmmlearn's _covars_), compact uniform shape; null until initialized or set. The setter validates
    /// the value with <see cref="InternalUtils.ValidateCovars"/> (ArgumentException if invalid) and stores a copy; the M-step also assigns
    /// through it (hmmlearn stores the M-step result unchecked: a documented deviation for degenerate fits).
    /// </summary>
    public double[,,]? Covars
    {
        get => _covars == null ? null : Clone3D(_covars);
        set
        {
            if (value != null)
            {
                InternalUtils.ValidateCovars(value, CovarianceType, NComponents);
                _covars = Clone3D(value);
            }
            else
                _covars = null;
        }
    }

    /// <summary>
    /// The covariances as full matrices (n_components, n_features, n_features) for every covariance type
    /// (<see cref="Utils.FillCovars"/>). Throws <see cref="InvalidOperationException"/> if Covars or the number of features is unknown.
    /// </summary>
    public double[,,] FullCovars
    {
        get
        {
            if (_covars == null || NFeatures == null)
                throw new InvalidOperationException();
            return Utils.FillCovars(_covars, CovarianceType, NComponents, NFeatures.Value);
        }
    }

    /// <summary>The means used for scoring and sampling: <see cref="MeansPosterior"/> (hmmlearn's means_ is the posterior mean).</summary>
    public double[,]? Means => MeansPosterior;

    /// <summary>
    /// Initializes before fitting (hmmlearn's VariationalGaussianHMM._init). First the base initialization (number of features, start and
    /// transition posteriors). Then k-means on x with NComponents clusters (<see cref="KMeans.Fit"/>), whose centers and cluster sizes
    /// seed the posteriors, and the column means of x. Then, if one of <c>NeedsInit('m', ...)</c> for "means_prior_", "means_posterior_",
    /// "beta_prior_", "beta_posterior_" is true (evaluated in this order, stopping at the first true as the Python <c>or</c> does):
    /// MeansPriorValues = MeansPrior or, when null, the mean of x for every state; MeansPosterior = the k-means centers (a copy);
    /// BetaPriorValues = BetaPrior or ones; BetaPosterior = the cluster sizes. Then, if one of <c>NeedsInit('c', ...)</c> for "dof_prior_",
    /// "dof_posterior_", "scale_prior_", "scale_posterior_" is true: for Full, Diag and Spherical DofPriorValues = DofPrior or NFeatures for every
    /// state and DofPosterior = the cluster sizes; for Tied DofPriorValues = DofPrior or { NFeatures } and DofPosterior = { number of samples };
    /// cv = the sample covariance of the columns of x (dividing by n - 1) plus 1e-3 on the diagonal; Covars = cv distributed to the covariance
    /// type with <see cref="InternalUtils.DistributeCovarMatrixToMatchCovarianceType"/>; ScalePriorValues = ScalePrior or the default
    /// above; ScalePosterior = Covars times DofPosterior of the state (Full and Diag: matrix or variances of the state times its dof;
    /// Spherical: the single variance of the state times its dof; Tied: the one matrix times the one dof).
    /// The random numbers of the base class come from a <see cref="Random"/> seeded with <see cref="AbstractHmm.RandomState"/> when it has a value.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths)
    {
        base.Init(x, lengths);
        var n = x.GetLength(0);
        var nc = NComponents;
        var nf = NFeatures!.Value;
        var (centers, labels) = KMeans.Fit(x, nc);
        var sizes = new double[nc];
        for (var i = 0; i < labels.Length; i++)
            sizes[labels[i]] += 1.0;

        if (NeedsInit('m', "means_prior_", MeansPriorValues != null)
            || NeedsInit('m', "means_posterior_", MeansPosterior != null)
            || NeedsInit('m', "beta_prior_", BetaPriorValues != null)
            || NeedsInit('m', "beta_posterior_", BetaPosterior != null))
        {
            MeansPriorValues = MeansPrior != null ? Clone2D(MeansPrior) : ColumnMeans(x, nc, nf);
            MeansPosterior = Clone2D(centers);
            if (BetaPrior != null)
            {
                BetaPriorValues = new double[BetaPrior.Length];
                Array.Copy(BetaPrior, BetaPriorValues, BetaPrior.Length);
            }
            else
            {
                BetaPriorValues = new double[nc];
                Array.Fill(BetaPriorValues, 1.0);
            }
            BetaPosterior = (double[])sizes.Clone();
        }

        if (NeedsInit('c', "dof_prior_", DofPriorValues != null)
            || NeedsInit('c', "dof_posterior_", DofPosterior != null)
            || NeedsInit('c', "scale_prior_", ScalePriorValues != null)
            || NeedsInit('c', "scale_posterior_", ScalePosterior != null))
        {
            if (DofPrior != null)
            {
                DofPriorValues = new double[DofPrior.Length];
                Array.Copy(DofPrior, DofPriorValues, DofPrior.Length);
            }
            else
                DofPriorValues = CovarianceType == CovarianceType.Tied ? new double[] { nf } : FillDof(nc, nf);
            DofPosterior = CovarianceType == CovarianceType.Tied ? new double[] { n } : (double[])sizes.Clone();

            var cv = SampleCovariance(x, nf);
            for (var f = 0; f < nf; f++)
                cv[f, f] += 1e-3;
            Covars = InternalUtils.DistributeCovarMatrixToMatchCovarianceType(cv, CovarianceType, nc);

            ScalePriorValues = ScalePrior != null ? Clone3D(ScalePrior) : DefaultScalePrior(nc, nf);
            ScalePosterior = ScaleByDof(_covars!, DofPosterior!);
        }
    }

    /// <summary>
    /// Base checks, then: CovarianceType must be a defined value ("{covariance type} is invalid"); MeansPriorValues and MeansPosterior must be
    /// non-null (NComponents x NFeatures); BetaPriorValues and BetaPosterior non-null of length NComponents; DofPriorValues and DofPosterior
    /// non-null of length NComponents (length 1 for Tied); ScalePriorValues and ScalePosterior non-null in the compact uniform shape of the type
    /// ((nc, 1, 1), (nc, 1, nf), (nc, nf, nf) or (1, nf, nf)); each violation throws <see cref="ArgumentException"/>.
    /// </summary>
    protected override void Check()
    {
        base.Check();
        if (CovarianceType is not (CovarianceType.Spherical or CovarianceType.Diag or CovarianceType.Full or CovarianceType.Tied))
            throw new ArgumentException($"{CovarianceType} is invalid");
        var nc = NComponents;
        var nf = NFeatures!.Value;

        if (MeansPriorValues == null || MeansPriorValues.GetLength(0) != nc || MeansPriorValues.GetLength(1) != nf)
            throw new ArgumentException("means_prior_ must have shape (n_components, n_features)");
        if (MeansPosterior == null || MeansPosterior.GetLength(0) != nc || MeansPosterior.GetLength(1) != nf)
            throw new ArgumentException("means_posterior_ must have shape (n_components, n_features)");
        if (BetaPriorValues == null || BetaPriorValues.Length != nc)
            throw new ArgumentException("beta_prior_ must have length n_components");
        if (BetaPosterior == null || BetaPosterior.Length != nc)
            throw new ArgumentException("beta_posterior_ must have length n_components");
        var dofLength = CovarianceType == CovarianceType.Tied ? 1 : nc;
        if (DofPriorValues == null || DofPriorValues.Length != dofLength)
            throw new ArgumentException("dof_prior_ must have length n_components");
        if (DofPosterior == null || DofPosterior.Length != dofLength)
            throw new ArgumentException("dof_posterior_ must have length n_components");
        var (d0, d1, d2) = CompactShape(nc, nf);
        if (ScalePriorValues == null || ScalePriorValues.GetLength(0) != d0 || ScalePriorValues.GetLength(1) != d1
            || ScalePriorValues.GetLength(2) != d2)
            throw new ArgumentException($"scale_prior_ must have shape ({d0}, {d1}, {d2})");
        if (ScalePosterior == null || ScalePosterior.GetLength(0) != d0 || ScalePosterior.GetLength(1) != d1
            || ScalePosterior.GetLength(2) != d2)
            throw new ArgumentException($"scale_posterior_ must have shape ({d0}, {d1}, {d2})");
    }

    /// <summary>
    /// Free scalar parameters per letter: 's' = nc - 1, 't' = nc * (nc - 1), 'm' = nc * nf + nc, and 'c' = nc + nc * nf * (nf + 1) / 2 (Full),
    /// 1 + nf * (nf + 1) / 2 (Tied), nc + nc * nf (Diag), nc + nc (Spherical). Throws <see cref="ArgumentException"/> ("{type} is invalid") for an
    /// undefined covariance type; requires the number of features to be known.
    /// </summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam()
    {
        if (NFeatures == null)
            return null;
        var nc = NComponents;
        var nf = NFeatures.Value;
        var c = CovarianceType switch
        {
            CovarianceType.Full => nc + nc * nf * (nf + 1) / 2,
            CovarianceType.Tied => 1 + nf * (nf + 1) / 2,
            CovarianceType.Diag => nc + nc * nf,
            CovarianceType.Spherical => nc + nc,
            _ => throw new ArgumentException($"{CovarianceType} is invalid"),
        };
        return new Dictionary<char, int>
        {
            ['s'] = nc - 1,
            ['t'] = nc * (nc - 1),
            ['m'] = nc * nf + nc,
            ['c'] = c,
        };
    }

    /// <summary>
    /// The subnormalized log likelihood of each sample under each state, shape (n_samples, n_components): with nf features, W[c] the inverse of
    /// the posterior scale of the state (as a full matrix, <see cref="Utils.FillCovars"/>; Tied: the same one for every state), dof[c] its
    /// posterior degrees of freedom (Tied: the one number): term1[c] = (sum over i = 0..nf-1 of digamma((dof[c] - i) / 2) + nf * ln 2 +
    /// logdet(W[c])) / 2, term3[c] = nf / BetaPosterior[c], delta = x[i] - MeansPosterior[c], dots = dof[c] * delta^T W[c] delta, and
    /// result[i, c] = term1[c] - (dots + term3[c]) / 2 (the constant nf * ln(2 pi) / 2 is left out, as hmmlearn does).
    /// </summary>
    protected override double[,] ComputeSubnormLogLikelihood(double[,] x)
    {
        var n = x.GetLength(0);
        var nc = NComponents;
        var nf = NFeatures!.Value;
        var scaleFull = Utils.FillCovars(ScalePosterior!, CovarianceType, nc, nf);
        var result = new double[n, nc];
        for (var c = 0; c < nc; c++)
        {
            var w = Inverse(TakeMatrix(scaleFull, c, nf));
            var dof = DofPosterior![c % DofPosterior!.Length];
            var digammaSum = 0.0;
            for (var i = 0; i < nf; i++)
                digammaSum += SpecialFunctions.DiGamma(0.5 * (dof - i));
            var term1 = (digammaSum + nf * Math.Log(2.0) + InternalUtils.LogDet(w)) / 2.0;
            var term3 = (double)nf / BetaPosterior![c];
            for (var i = 0; i < n; i++)
            {
                var dots = 0.0;
                for (var a = 0; a < nf; a++)
                    for (var b = 0; b < nf; b++)
                        dots += (x[i, a] - MeansPosterior![c, a]) * w[a, b] * (x[i, b] - MeansPosterior![c, b]);
                result[i, c] = term1 - (dots * dof + term3) / 2.0;
            }
        }
        return result;
    }

    /// <summary>The lower bound of the base class (log probability minus the Dirichlet KL terms) minus, for every state c: the KL divergence of the posterior from the prior of the mean, <see cref="KlDivergence.KlMultivariateNormalDistribution"/>(MeansPosterior[c], inverse(BetaPosterior[c] * P[c]), MeansPriorValues[c], inverse(BetaPriorValues[c] * P[c])) with P[c] = dof[c] * inverse(scale posterior matrix of the state), and the KL divergence of the Wishart posterior from its prior, <see cref="KlDivergence.KlWishartDistribution"/>(DofPosterior[c], scale posterior matrix, DofPriorValues[c], scale prior matrix) of each state (Tied: only for the first state, once, since there is one Wishart; the scale matrices are the full matrices from <see cref="Utils.FillCovars"/>).</summary>
    protected override double ComputeLowerBound(double currLogProb)
    {
        var bound = base.ComputeLowerBound(currLogProb);
        var nc = NComponents;
        var nf = NFeatures!.Value;
        var scalePosteriorFull = Utils.FillCovars(ScalePosterior!, CovarianceType, nc, nf);
        var scalePriorFull = Utils.FillCovars(ScalePriorValues!, CovarianceType, nc, nf);
        var tied = CovarianceType == CovarianceType.Tied;
        for (var c = 0; c < nc; c++)
        {
            var scale = TakeMatrix(scalePosteriorFull, c, nf);
            var dof = DofPosterior![c % DofPosterior!.Length];
            var precision = Scale(Inverse(scale), dof);
            var covarQ = Inverse(Scale(precision, BetaPosterior![c]));
            var covarP = Inverse(Scale(precision, BetaPriorValues![c]));
            bound -= KlDivergence.KlMultivariateNormalDistribution(Row(MeansPosterior!, c, nf), covarQ,
                Row(MeansPriorValues!, c, nf), covarP);
            if (!tied || c == 0)
                bound -= KlDivergence.KlWishartDistribution(dof, scale, DofPriorValues![c % DofPriorValues!.Length],
                    TakeMatrix(scalePriorFull, c, nf));
        }
        return bound;
    }

    /// <summary>The log density of each sample under each state: <see cref="Stats.LogMultivariateNormalDensity"/> with <see cref="Means"/> and <see cref="Covars"/>.</summary>
    protected override double[,] ComputeLogLikelihood(double[,] x) => Stats.LogMultivariateNormalDensity(x, Means!, Covars!, CovarianceType);

    /// <summary>A <see cref="GaussianStatistics"/> for NComponents, the number of features and CovarianceType.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => new GaussianStatistics(NComponents, NFeatures!.Value, CovarianceType);

    /// <summary>Base accumulation, then <see cref="GaussianEmission.Accumulate"/> with <see cref="NeedsStatisticsForMeans"/> and <see cref="NeedsStatisticsForCovars"/>.</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice)
    {
        base.AccumulateSufficientStatistics(stats, x, lattice, posteriors, fwdlattice, bwdlattice);
        GaussianEmission.Accumulate((GaussianStatistics)stats, x, posteriors, CovarianceType, NeedsStatisticsForMeans, NeedsStatisticsForCovars);
    }

    /// <summary>Whether the statistics of the means are accumulated: Params contains 'm' or 'c' (the covariance update needs the posterior masses).</summary>
    protected virtual bool NeedsStatisticsForMeans => Params.Contains('m') || Params.Contains('c');

    /// <summary>Whether the statistics of the covariances are accumulated: Params contains 'c'.</summary>
    protected virtual bool NeedsStatisticsForCovars => Params.Contains('c');

    /// <summary>One observation from the Gaussian of the state (<see cref="GaussianEmission.GenerateSample"/> with <see cref="Means"/> and <see cref="FullCovars"/>).</summary>
    protected override double[] GenerateSampleFromState(int state, Random random) => GaussianEmission.GenerateSample(Means!, FullCovars, state, random);

    /// <summary>The compact uniform shape of the covariance type: spherical (nc, 1, 1), diag (nc, 1, nf), full (nc, nf, nf), tied (1, nf, nf).</summary>
    private (int, int, int) CompactShape(int nc, int nf) => CovarianceType switch
    {
        CovarianceType.Full => (nc, nf, nf),
        CovarianceType.Tied => (1, nf, nf),
        CovarianceType.Diag => (nc, 1, nf),
        CovarianceType.Spherical => (nc, 1, 1),
        _ => throw new ArgumentException($"{CovarianceType} is invalid"),
    };

    /// <summary>The default scale prior of the type: 1e-3 times the identity matrix for Full and Tied, 1e-3 for every entry for Diag and Spherical.</summary>
    private double[,,] DefaultScalePrior(int nc, int nf)
    {
        var (d0, d1, d2) = CompactShape(nc, nf);
        var result = new double[d0, d1, d2];
        var identity = CovarianceType is CovarianceType.Full or CovarianceType.Tied;
        for (var i = 0; i < d0; i++)
            for (var a = 0; a < d1; a++)
                for (var b = 0; b < d2; b++)
                    result[i, a, b] = identity && a != b ? 0.0 : 1e-3;
        return result;
    }

    /// <summary>NFeatures for every state.</summary>
    private static double[] FillDof(int nc, int nf)
    {
        var result = new double[nc];
        Array.Fill(result, nf);
        return result;
    }

    /// <summary>The mean of the columns of x repeated for every state.</summary>
    private static double[,] ColumnMeans(double[,] x, int nc, int nf)
    {
        var n = x.GetLength(0);
        var mean = new double[nf];
        for (var i = 0; i < n; i++)
            for (var f = 0; f < nf; f++)
                mean[f] += x[i, f] / n;
        var result = new double[nc, nf];
        for (var c = 0; c < nc; c++)
            for (var f = 0; f < nf; f++)
                result[c, f] = mean[f];
        return result;
    }

    /// <summary>Sample covariance of the columns of x, dividing by n - 1 (numpy's cov).</summary>
    private static double[,] SampleCovariance(double[,] x, int nf)
    {
        var n = x.GetLength(0);
        var mean = new double[nf];
        for (var i = 0; i < n; i++)
            for (var f = 0; f < nf; f++)
                mean[f] += x[i, f] / n;
        var result = new double[nf, nf];
        for (var i = 0; i < n; i++)
            for (var a = 0; a < nf; a++)
                for (var b = 0; b < nf; b++)
                    result[a, b] += (x[i, a] - mean[a]) * (x[i, b] - mean[b]);
        for (var a = 0; a < nf; a++)
            for (var b = 0; b < nf; b++)
                result[a, b] /= n - 1;
        return result;
    }

    /// <summary>Every entry of <paramref name="covars"/> times the dof of its state (Tied: the one dof for the one matrix).</summary>
    private static double[,,] ScaleByDof(double[,,] covars, double[] dof)
    {
        var nc = covars.GetLength(0);
        var nf = covars.GetLength(1);
        var ng = covars.GetLength(2);
        var result = new double[nc, nf, ng];
        for (var c = 0; c < nc; c++)
            for (var a = 0; a < nf; a++)
                for (var b = 0; b < ng; b++)
                    result[c, a, b] = covars[c, a, b] * dof[c % dof.Length];
        return result;
    }

    /// <summary>Row c of a matrix with nf columns, in a new array.</summary>
    private static double[] Row(double[,] a, int c, int nf)
    {
        var result = new double[nf];
        for (var f = 0; f < nf; f++)
            result[f] = a[c, f];
        return result;
    }

    /// <summary>The (nf, nf) matrix of state c of a full (nc, nf, nf) tensor, in a new array.</summary>
    private static double[,] TakeMatrix(double[,,] full, int c, int nf)
    {
        var result = new double[nf, nf];
        for (var i = 0; i < nf; i++)
            for (var j = 0; j < nf; j++)
                result[i, j] = full[c, i, j];
        return result;
    }

    /// <summary>Every entry of a matrix times a factor, in a new array.</summary>
    private static double[,] Scale(double[,] a, double factor)
    {
        var n = a.GetLength(0);
        var result = new double[n, n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                result[i, j] = a[i, j] * factor;
        return result;
    }

    /// <summary>The inverse of a square matrix.</summary>
    private static double[,] Inverse(double[,] a) => Matrix<double>.Build.DenseOfArray(a).Inverse().ToArray();

    /// <summary>A copy of a 2D array.</summary>
    private static double[,] Clone2D(double[,] a) => (double[,])a.Clone();

    /// <summary>A copy of a 3D array.</summary>
    private static double[,,] Clone3D(double[,,] a) => (double[,,])a.Clone();
}
