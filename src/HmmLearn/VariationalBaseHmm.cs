namespace HmmLearn;

/// <summary>
/// Base class of the hidden Markov models learned by Variational Bayes (port of VariationalBaseHMM of hmmlearn/base.py): the start
/// probabilities and each row of the transition matrix have a Dirichlet prior and a Dirichlet posterior. The E-step runs the
/// forward-backward passes with subnormalized parameters (exp of digamma differences) and subnormalized emission likelihoods; the
/// M-step adds the statistics to the priors; the monitored quantity is the variational lower bound. <see cref="AbstractHmm.StartProb"/>
/// and <see cref="AbstractHmm.TransMat"/> hold the normalized posteriors, for scoring and decoding. Models add their emission
/// posteriors on top.
/// </summary>
public abstract class VariationalBaseHmm : AbstractHmm
{
    /// <summary>
    /// Creates the model. <paramref name="startProbPrior"/> and <paramref name="transMatPrior"/> are the values of every parameter of
    /// the symmetric Dirichlet priors of the start probabilities and of each transition row; null means 1 / NComponents.
    /// </summary>
    protected VariationalBaseHmm(int nComponents, double? startProbPrior, double? transMatPrior, DecoderAlgorithm algorithm, int? randomState,
        int nIter, double tol, bool verbose, string @params, string initParams, Implementation implementation)
        : base(nComponents, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
        => throw new NotImplementedException();

    /// <summary>Value of every parameter of the Dirichlet prior of the start probabilities used by <see cref="Init"/>; null means 1 / NComponents.</summary>
    public double? StartProbPrior { get; set; }

    /// <summary>Value of every parameter of the Dirichlet prior of each transition row used by <see cref="Init"/>; null means 1 / NComponents.</summary>
    public double? TransMatPrior { get; set; }

    /// <summary>Parameters of the Dirichlet prior of the start probabilities, length NComponents (hmmlearn's startprob_prior_); null until initialized or set.</summary>
    public double[]? StartProbPriorValues { get; set; }

    /// <summary>Parameters of the Dirichlet posterior of the start probabilities, length NComponents (hmmlearn's startprob_posterior_); null until initialized or set.</summary>
    public double[]? StartProbPosterior { get; set; }

    /// <summary>Parameters of the Dirichlet prior of each transition row, shape (NComponents, NComponents) (hmmlearn's transmat_prior_); null until initialized or set.</summary>
    public double[,]? TransMatPriorValues { get; set; }

    /// <summary>Parameters of the Dirichlet posterior of each transition row, shape (NComponents, NComponents) (hmmlearn's transmat_posterior_); null until initialized or set.</summary>
    public double[,]? TransMatPosterior { get; set; }

    /// <summary>Subnormalized start probabilities of the current E-step: exp(digamma(posterior) - digamma(sum of the posterior)); null before the first E-step.</summary>
    protected double[]? StartProbSubnorm { get; set; }

    /// <summary>Subnormalized transition matrix of the current E-step: row by row exp(digamma(posterior) - digamma(sum of the row)); null before the first E-step.</summary>
    protected double[,]? TransMatSubnorm { get; set; }

    /// <summary>
    /// Log of the subnormalized likelihood of each sample under each state, shape (n_samples, n_components): the emission term of the
    /// E-step, computed from the emission posteriors (expected log likelihood).
    /// </summary>
    protected abstract double[,] ComputeSubnormLogLikelihood(double[,] x);

    /// <summary>
    /// Initializes before fitting (hmmlearn's VariationalBaseHMM._init): <c>CheckAndSetNFeatures(x)</c>; then, if
    /// <c>NeedsInit('s', "startprob_posterior_", StartProbPosterior != null)</c> or (only evaluated when the first is false)
    /// <c>NeedsInit('s', "startprob_prior_", StartProbPriorValues != null)</c>: <see cref="StartProbPriorValues"/> = NComponents copies of
    /// StartProbPrior (1 / NComponents when null) and <see cref="StartProbPosterior"/> = a draw of the Dirichlet distribution with every
    /// parameter 1 / NComponents, multiplied by the number of sequences (lengths.Length); likewise for 't' with
    /// <see cref="TransMatPriorValues"/> (NComponents x NComponents copies of TransMatPrior, 1 / NComponents when null) and
    /// <see cref="TransMatPosterior"/> = one such draw per row, multiplied by (sum of lengths) / NComponents. The random numbers come
    /// from a <see cref="Random"/> seeded with <see cref="AbstractHmm.RandomState"/> when it has a value (a new generator at each
    /// call), else from <see cref="Random.Shared"/>. Then the same degenerate-fit warning as BaseHmm when
    /// <see cref="AbstractHmm.GetNFitScalarsPerParam"/> is not null and x has fewer values than the free scalars of the letters of Params.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths) => throw new NotImplementedException();

    /// <summary>
    /// Validates the priors and posteriors; throws <see cref="ArgumentException"/> if <see cref="StartProbPriorValues"/> or
    /// <see cref="StartProbPosterior"/> is null or does not have length NComponents (messages "startprob_prior_ must have length
    /// n_components", "startprob_posterior_ must have length n_components"), or if <see cref="TransMatPriorValues"/> or
    /// <see cref="TransMatPosterior"/> is null or not (NComponents x NComponents) ("transmat_prior_ must have shape (n_components,
    /// n_components)", "transmat_posterior_ must have shape (n_components, n_components)"). The values are not otherwise checked.
    /// </summary>
    protected override void Check() => throw new NotImplementedException();

    /// <summary>Start of every E-step: refreshes <see cref="StartProbSubnorm"/> and <see cref="TransMatSubnorm"/> from the posteriors with the digamma function.</summary>
    protected override void EstepBegin() => throw new NotImplementedException();

    /// <summary><see cref="StartProbSubnorm"/>: the E-step passes use the subnormalized start probabilities.</summary>
    protected override double[] FitStartProb => throw new NotImplementedException();

    /// <summary><see cref="TransMatSubnorm"/>: the E-step passes and the xi sums use the subnormalized transition matrix.</summary>
    protected override double[,] FitTransMat => throw new NotImplementedException();

    /// <summary><see cref="ComputeSubnormLogLikelihood"/>: the E-step log passes use the subnormalized emission likelihoods.</summary>
    protected override double[,] ComputeFitLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>The exponential of <see cref="ComputeSubnormLogLikelihood"/>: the E-step scaling passes use the subnormalized emission likelihoods.</summary>
    protected override double[,] ComputeFitLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>
    /// M-step of the start probabilities and transition matrix. If Params contains 's': <see cref="StartProbPosterior"/> = prior values +
    /// stats.Start and <see cref="AbstractHmm.StartProb"/> = the posterior divided by its sum. If Params contains 't':
    /// <see cref="TransMatPosterior"/> = prior values + stats.Trans and <see cref="AbstractHmm.TransMat"/> = each row of the posterior
    /// divided by its sum. New arrays replace the old ones.
    /// </summary>
    protected override void DoMStep(SufficientStatistics stats) => throw new NotImplementedException();

    /// <summary>
    /// The variational lower bound contribution of the start probabilities, the transitions and the data: the log probability of the data
    /// computed with the subnormalized parameters, minus <see cref="KlDivergence.KlDirichlet"/> of the start posterior from its prior, minus the
    /// sum over the states of the same divergence for each transition row.
    /// </summary>
    protected override double ComputeLowerBound(double currLogProb) => throw new NotImplementedException();
}
