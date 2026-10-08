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
        => throw new NotImplementedException();

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
        get => throw new NotImplementedException();
        set => throw new NotImplementedException();
    }

    /// <summary>
    /// The covariances as full matrices (n_components, n_features, n_features) for every covariance type
    /// (<see cref="Utils.FillCovars"/>). Throws <see cref="InvalidOperationException"/> if Covars or the number of features is unknown.
    /// </summary>
    public double[,,] FullCovars => throw new NotImplementedException();

    /// <summary>The means used for scoring and sampling: <see cref="MeansPosterior"/> (hmmlearn's means_ is the posterior mean).</summary>
    public double[,]? Means => throw new NotImplementedException();

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
    protected override void Init(double[,] x, int[] lengths) => throw new NotImplementedException();

    /// <summary>
    /// Base checks, then: CovarianceType must be a defined value ("{covariance type} is invalid"); MeansPriorValues and MeansPosterior must be
    /// non-null (NComponents x NFeatures); BetaPriorValues and BetaPosterior non-null of length NComponents; DofPriorValues and DofPosterior
    /// non-null of length NComponents (length 1 for Tied); ScalePriorValues and ScalePosterior non-null in the compact uniform shape of the type
    /// ((nc, 1, 1), (nc, 1, nf), (nc, nf, nf) or (1, nf, nf)); each violation throws <see cref="ArgumentException"/>.
    /// </summary>
    protected override void Check() => throw new NotImplementedException();

    /// <summary>
    /// Free scalar parameters per letter: 's' = nc - 1, 't' = nc * (nc - 1), 'm' = nc * nf + nc, and 'c' = nc + nc * nf * (nf + 1) / 2 (Full),
    /// 1 + nf * (nf + 1) / 2 (Tied), nc + nc * nf (Diag), nc + nc (Spherical). Throws <see cref="ArgumentException"/> ("{type} is invalid") for an
    /// undefined covariance type; requires the number of features to be known.
    /// </summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam() => throw new NotImplementedException();

    /// <summary>
    /// The subnormalized log likelihood of each sample under each state, shape (n_samples, n_components): with nf features, W[c] the inverse of
    /// the posterior scale of the state (as a full matrix, <see cref="Utils.FillCovars"/>; Tied: the same one for every state), dof[c] its
    /// posterior degrees of freedom (Tied: the one number): term1[c] = (sum over i = 0..nf-1 of digamma((dof[c] - i) / 2) + nf * ln 2 +
    /// logdet(W[c])) / 2, term3[c] = nf / BetaPosterior[c], delta = x[i] - MeansPosterior[c], dots = dof[c] * delta^T W[c] delta, and
    /// result[i, c] = term1[c] - (dots + term3[c]) / 2 (the constant nf * ln(2 pi) / 2 is left out, as hmmlearn does).
    /// </summary>
    protected override double[,] ComputeSubnormLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>The lower bound of the base class (log probability minus the Dirichlet KL terms) minus, for every state c: the KL divergence of the posterior from the prior of the mean, <see cref="KlDivergence.KlMultivariateNormalDistribution"/>(MeansPosterior[c], inverse(BetaPosterior[c] * P[c]), MeansPriorValues[c], inverse(BetaPriorValues[c] * P[c])) with P[c] = dof[c] * inverse(scale posterior matrix of the state), and the KL divergence of the Wishart posterior from its prior, <see cref="KlDivergence.KlWishartDistribution"/>(DofPosterior[c], scale posterior matrix, DofPriorValues[c], scale prior matrix) of each state (Tied: only for the first state, once, since there is one Wishart; the scale matrices are the full matrices from <see cref="Utils.FillCovars"/>).</summary>
    protected override double ComputeLowerBound(double currLogProb) => throw new NotImplementedException();

    /// <summary>The log density of each sample under each state: <see cref="Stats.LogMultivariateNormalDensity"/> with <see cref="Means"/> and <see cref="Covars"/>.</summary>
    protected override double[,] ComputeLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>A <see cref="GaussianStatistics"/> for NComponents, the number of features and CovarianceType.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => throw new NotImplementedException();

    /// <summary>Base accumulation, then <see cref="GaussianEmission.Accumulate"/> with <see cref="NeedsStatisticsForMeans"/> and <see cref="NeedsStatisticsForCovars"/>.</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice) => throw new NotImplementedException();

    /// <summary>Whether the statistics of the means are accumulated: Params contains 'm' or 'c' (the covariance update needs the posterior masses).</summary>
    protected virtual bool NeedsStatisticsForMeans => throw new NotImplementedException();

    /// <summary>Whether the statistics of the covariances are accumulated: Params contains 'c'.</summary>
    protected virtual bool NeedsStatisticsForCovars => throw new NotImplementedException();

    /// <summary>One observation from the Gaussian of the state (<see cref="GaussianEmission.GenerateSample"/> with <see cref="Means"/> and <see cref="FullCovars"/>).</summary>
    protected override double[] GenerateSampleFromState(int state, Random random) => throw new NotImplementedException();
}
