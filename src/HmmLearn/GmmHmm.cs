namespace HmmLearn;

/// <summary>
/// The sufficient statistics of a mixture model (hmmlearn's 'post_mix_sum', 'post_sum', 'm_n' and 'c_n'): the base ones plus the
/// posterior mass of each mixture component and the weighted sums the M-step needs. <see cref="MN"/> already holds the prior part
/// (MeansWeight * MeansPrior) when the statistics are created.
/// </summary>
public class GmmStatistics : SufficientStatistics
{
    /// <summary>
    /// Allocates the base statistics, <see cref="PostMixSum"/> (nc x nm), <see cref="PostSum"/> (nc), <see cref="MN"/> (nc x nm x nf,
    /// every entry meansWeight * meansPrior) and <see cref="CN"/> (zeros of the compact rank-4 shape of the covariance type, given by
    /// <paramref name="covarsShape"/> = its four dimensions), all other entries zero.
    /// </summary>
    public GmmStatistics(int nComponents, int nMix, int nFeatures, double meansWeight, double meansPrior, int[] covarsShape)
        : base(nComponents) => throw new NotImplementedException();

    /// <summary>PostMixSum[c, m]: sum over the samples of the posterior of state c times the posterior of mixture m within c.</summary>
    public double[,] PostMixSum { get; set; } = null!;

    /// <summary>PostSum[c]: sum over the samples of the posterior of state c.</summary>
    public double[] PostSum { get; set; } = null!;

    /// <summary>MN[c, m, f]: meansWeight * meansPrior + sum over the samples of the joint posterior times x[f].</summary>
    public double[,,] MN { get; set; } = null!;

    /// <summary>
    /// CN: sum over the samples of the joint posterior times the centered second moment, in the compact rank-4 shapes: spherical
    /// (nc, nm, 1, 1) with the squared norm of the centered sample, diag (nc, nm, 1, nf) with its squared coordinates, full (nc, nm, nf, nf)
    /// with its outer product, tied (nc, 1, nf, nf) with the outer product summed over the mixture components. Centered means x minus
    /// the mean of the mixture component.
    /// </summary>
    public double[,,,] CN { get; set; } = null!;
}

/// <summary>
/// Hidden Markov model whose emissions are Gaussian mixtures (port of GMMHMM of hmmlearn/hmm.py and BaseGMMHMM of
/// hmmlearn/_emissions.py). This file holds the parameters, the checks, the likelihood, the sampling and the statistics; the
/// initialization and the M-step are in GmmHmm.Learning.cs (same partial class). Covariances are stored in the compact rank-4 uniform
/// shapes of oracle/README.md: spherical (nc, nm, 1, 1), diag (nc, nm, 1, nf), full (nc, nm, nf, nf), tied (nc, 1, nf, nf)
/// (tied: one matrix per state, shared by its mixture components).
/// </summary>
public partial class GmmHmm : BaseHmm
{
    /// <summary>Creates the model; the defaults are those of hmmlearn (diag covariances, parameter letters "stmcw").</summary>
    public GmmHmm(int nComponents = 1, int nMix = 1, double minCovar = 1e-3, double startProbPrior = 1.0, double transMatPrior = 1.0,
        double weightsPrior = 1.0, double meansPrior = 0.0, double meansWeight = 0.0, double? covarsPrior = null,
        double? covarsWeight = null, DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi,
        CovarianceType covarianceType = CovarianceType.Diag, int? randomState = null, int nIter = 10, double tol = 1e-2,
        bool verbose = false, string @params = "stmcw", string initParams = "stmcw", Implementation implementation = Implementation.Log)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
        => throw new NotImplementedException();

    /// <summary>Number of mixture components of each state.</summary>
    public int NMix { get; set; }

    /// <summary>The structure of the covariance parameters.</summary>
    public CovarianceType CovarianceType { get; set; }

    /// <summary>Floor added to the diagonal of the data covariance when the covariances are initialized.</summary>
    public double MinCovar { get; set; }

    /// <summary>Parameter of the Dirichlet prior of the mixture weights of each state.</summary>
    public double WeightsPrior { get; set; }

    /// <summary>Mean of the Normal prior of the means (scalar).</summary>
    public double MeansPrior { get; set; }

    /// <summary>Weight of the prior of the means (scalar).</summary>
    public double MeansWeight { get; set; }

    /// <summary>Scalar prior of the covariances; null means the default of the covariance type (see <see cref="EffectiveCovarsPrior"/>).</summary>
    public double? CovarsPrior { get; set; }

    /// <summary>Scalar weight of the prior of the covariances; null means the default of the covariance type (see <see cref="EffectiveCovarsWeight"/>).</summary>
    public double? CovarsWeight { get; set; }

    /// <summary>Mixture weights of each state, shape (n_components, n_mix); null until set or fitted.</summary>
    public double[,]? Weights { get; set; }

    /// <summary>Mean of each mixture component, shape (n_components, n_mix, n_features); null until set or fitted.</summary>
    public double[,,]? Means { get; set; }

    /// <summary>Compact covariances in the rank-4 uniform shapes above; null until set or fitted. Validated by <see cref="Check"/>, not by the setter.</summary>
    public double[,,,]? Covars { get; set; }

    /// <summary>
    /// The covariance prior actually used: <see cref="CovarsPrior"/> if it has a value, else by type: full 0, tied 0, diag -1.5,
    /// spherical -(NMix + 2) / 2.
    /// </summary>
    public double EffectiveCovarsPrior => throw new NotImplementedException();

    /// <summary>
    /// The covariance prior weight actually used: <see cref="CovarsWeight"/> if it has a value, else by type: full -(n_features + 2),
    /// tied -(NMix + n_features + 1), diag 0, spherical 0. Requires the number of features to be known for full and tied.
    /// </summary>
    public double EffectiveCovarsWeight => throw new NotImplementedException();

    /// <summary>
    /// Base checks, then: Weights (NComponents x NMix) with rows summing to 1 (CheckSum1); Means (NComponents x NMix x nf); Covars of
    /// the rank-4 shape of the covariance type, else <see cref="ArgumentException"/> with a message like "'diag' mixture covars must
    /// have shape (nc, nm, 1, nf)"; spherical and diag variances must be non-negative (a zero one raises the warning "Degenerate
    /// mixture covariance"); every tied and full matrix must be symmetric (numpy.allclose) with a smallest eigenvalue not negative (zero
    /// raises a warning "Covariance of state #i has a null eigenvalue"); NFeatures is taken from Means when unknown.
    /// </summary>
    protected override void Check() => throw new NotImplementedException();

    /// <summary>
    /// Log density of each sample under each state: for state c, the log-sum-exp over its mixture components of (log weight + log
    /// Gaussian density), the density of the components being <see cref="Stats.LogMultivariateNormalDensity"/> with that state's
    /// means and covariances (for tied, the state's single matrix).
    /// </summary>
    protected override double[,] ComputeLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>A <see cref="GmmStatistics"/> for the current parameters (MN starts at MeansWeight * MeansPrior).</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => throw new NotImplementedException();

    /// <summary>
    /// Base accumulation, then for each state the posterior of each mixture component (the log weighted densities log-normalized over the
    /// mixture components, exponentiated), the joint posterior (state posterior times mixture posterior), and: PostMixSum, PostSum always;
    /// if Params contains 'm', MN += joint posterior times x; if Params contains 'c', CN += the centered second moments (see <see cref="GmmStatistics.CN"/>).
    /// </summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice) => throw new NotImplementedException();

    /// <summary>
    /// One observation: a mixture component drawn from the weights of the state (one uniform number, cumulative weights), then a draw
    /// from that component's Gaussian (<see cref="GaussianEmission.GenerateSample"/>; for tied, the state's matrix).
    /// </summary>
    protected override double[] GenerateSampleFromState(int state, Random random) => throw new NotImplementedException();

    /// <summary>'s' = nc - 1, 't' = nc * (nc - 1), 'm' = nc * nm * nf, 'c' = nc * nm (spherical), nc * nm * nf (diag), nc * nm * nf * (nf + 1) / 2 (full), nc * nf * (nf + 1) / 2 (tied), 'w' = nm - 1.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam() => throw new NotImplementedException();
}
