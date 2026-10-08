namespace HmmLearn;

/// <summary>
/// Hidden Markov model with categorical (discrete) emissions learned by Variational Bayes (port of VariationalCategoricalHMM of
/// hmmlearn/vhmm.py): each row of the emission probabilities has a Dirichlet prior and posterior. Observations have shape
/// (n_samples, 1): one non-negative integer symbol per sample, stored as a double. <see cref="EmissionProb"/> holds the normalized
/// posterior, used for scoring, decoding and sampling; the E-step uses the subnormalized (digamma) emission likelihoods.
/// </summary>
public class VariationalCategoricalHmm : VariationalBaseHmm
{
    /// <summary>
    /// Creates the model. <paramref name="emissionProbPrior"/>: value of every parameter of the Dirichlet prior of each emission row
    /// (null means 1 / number of features); <paramref name="nFeatures"/>: number of symbols, inferred from the data (largest symbol + 1)
    /// if null. The defaults are those of hmmlearn.
    /// </summary>
    public VariationalCategoricalHmm(int nComponents = 1, double? startProbPrior = null, double? transMatPrior = null,
        double? emissionProbPrior = null, int? nFeatures = null, DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi,
        int? randomState = null, int nIter = 100, double tol = 1e-6, bool verbose = false, string @params = "ste",
        string initParams = "ste", Implementation implementation = Implementation.Log)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
        => throw new NotImplementedException();

    /// <summary>Value of every parameter of the Dirichlet prior of each emission row used by <see cref="Init"/>; null means 1 / NFeatures.</summary>
    public double? EmissionProbPrior { get; set; }

    /// <summary>Parameters of the Dirichlet prior of each emission row, shape (n_components, n_features) (hmmlearn's emissionprob_prior_); null until initialized or set.</summary>
    public double[,]? EmissionProbPriorValues { get; set; }

    /// <summary>Parameters of the Dirichlet posterior of each emission row, shape (n_components, n_features) (hmmlearn's emissionprob_posterior_); null until initialized or set.</summary>
    public double[,]? EmissionProbPosterior { get; set; }

    /// <summary>The emission probabilities: each row of <see cref="EmissionProbPosterior"/> divided by its sum, set by the M-step when Params contains 'e'; null until then (hmmlearn's emissionprob_).</summary>
    public double[,]? EmissionProb { get; set; }

    /// <summary>digamma(posterior) - digamma(row sum of the posterior), shape (n_components, n_features), refreshed at the start of every E-step; null before the first one.</summary>
    protected double[,]? EmissionProbLogSubnorm { get; set; }

    /// <summary>
    /// Initializes before fitting: the base initialization, then, if <c>NeedsInit('e', "emissionprob_posterior_", EmissionProbPosterior != null)</c>:
    /// <see cref="EmissionProbPriorValues"/> = (NComponents x NFeatures) copies of EmissionProbPrior (1 / NFeatures when null) and
    /// <see cref="EmissionProbPosterior"/> = one draw per row of the Dirichlet distribution with every parameter equal to that same value,
    /// multiplied by (sum of lengths) / NComponents. The random numbers come from a <see cref="Random"/> seeded with
    /// <see cref="AbstractHmm.RandomState"/> when it has a value (a new generator at each call), else from <see cref="Random.Shared"/>.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths) => throw new NotImplementedException();

    /// <summary>
    /// Base checks, then <see cref="EmissionProbPriorValues"/> and <see cref="EmissionProbPosterior"/> must be non-null with the same
    /// shape, else <see cref="ArgumentException"/> ("emissionprob_prior_ and emissionprob_posterior_ must have shape (n_components,
    /// n_features)"); if NFeatures is null it becomes the number of columns of the posterior; the posterior must be (NComponents x
    /// NFeatures), else ArgumentException ("emissionprob_ must have shape (n_components, n_features)").
    /// </summary>
    protected override void Check() => throw new NotImplementedException();

    /// <summary>Start of every E-step: the base subnormalized start and transition parameters, then <see cref="EmissionProbLogSubnorm"/> from the emission posterior.</summary>
    protected override void EstepBegin() => throw new NotImplementedException();

    /// <summary>The subnormalized log likelihood of each sample under each state: result[i, c] = EmissionProbLogSubnorm[c, symbol of sample i], shape (n_samples, n_components).</summary>
    protected override double[,] ComputeSubnormLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>Base M-step, then if Params contains 'e': <see cref="EmissionProbPosterior"/> = prior values + Obs and <see cref="EmissionProb"/> = each row divided by its sum (new arrays).</summary>
    protected override void DoMStep(SufficientStatistics stats) => throw new NotImplementedException();

    /// <summary>The base lower bound (log probability minus the KL of the start and transition posteriors) minus the sum over the states of the KL divergence of the emission row posterior from its prior.</summary>
    protected override double ComputeLowerBound(double currLogProb) => throw new NotImplementedException();

    /// <summary>Probability of each sample under each state from <see cref="EmissionProb"/> (<see cref="CategoricalEmission.Likelihood"/>).</summary>
    protected override double[,] ComputeLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>Natural logarithm of <see cref="ComputeLikelihood"/> (log of 0 is -infinity).</summary>
    protected override double[,] ComputeLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>A <see cref="CategoricalStatistics"/>.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => throw new NotImplementedException();

    /// <summary>Base accumulation, then if Params contains 'e': <see cref="CategoricalEmission.AccumulateObs"/> into Obs.</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice) => throw new NotImplementedException();

    /// <summary>One symbol drawn from the emission probabilities of the state (<see cref="CategoricalEmission.GenerateSample"/>).</summary>
    protected override double[] GenerateSampleFromState(int state, Random random) => throw new NotImplementedException();

    /// <summary><see cref="CategoricalEmission.CheckAndSetNFeatures"/> on <see cref="AbstractHmm.NFeatures"/>.</summary>
    protected override void CheckAndSetNFeatures(double[,] x) => throw new NotImplementedException();

    /// <summary><see cref="CategoricalEmission.NFitScalars"/>; requires NFeatures to be known.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam() => throw new NotImplementedException();
}
