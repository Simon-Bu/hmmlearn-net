namespace HmmLearn;

/// <summary>
/// Hidden Markov model with multinomial emissions (port of MultinomialHMM of hmmlearn/hmm.py and BaseMultinomialHMM of
/// hmmlearn/_emissions.py). An observation is a vector of n_features non-negative integer counts (stored as doubles); its total is the
/// number of trials of that sample. Unlike <see cref="CategoricalHmm"/> (one symbol per sample), a sample counts how many times each
/// symbol came out of n trials. The statistics reuse <see cref="CategoricalStatistics"/> (Obs[state, symbol]).
/// </summary>
public class MultinomialHmm : BaseHmm
{
    /// <summary>
    /// Creates the model. <paramref name="nTrials"/>: if it has a value, the total count of every sample must equal it and sampling uses it;
    /// if null, the totals may differ from sample to sample (sampling then needs NTrials to be set). The defaults are those of hmmlearn.
    /// </summary>
    public MultinomialHmm(int nComponents = 1, int? nTrials = null, double startProbPrior = 1.0, double transMatPrior = 1.0,
        DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi, int? randomState = null, int nIter = 10, double tol = 1e-2,
        bool verbose = false, string @params = "ste", string initParams = "ste", Implementation implementation = Implementation.Log)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
        => throw new NotImplementedException();

    /// <summary>Number of trials of every sample; null when the totals are free.</summary>
    public int? NTrials { get; set; }

    /// <summary>Probability of each symbol in each state, shape (n_components, n_features); null until set or fitted.</summary>
    public double[,]? EmissionProb { get; set; }

    /// <summary>
    /// Base initialization; then, if InitParams contains 'e' (always re-initialized, without the "will be overwritten" warning),
    /// EmissionProb is uniform random numbers in [0, 1) (n_components x n_features, same seeding rule as the base) with each row normalized.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths) => throw new NotImplementedException();

    /// <summary>
    /// Base checks; EmissionProb must be non-null and (NComponents x NFeatures), else <see cref="ArgumentException"/>
    /// ("emissionprob_ must have shape (n_components, n_features)"); if NFeatures is null it becomes the number of columns; rows sum to 1.
    /// </summary>
    protected override void Check() => throw new NotImplementedException();

    /// <summary>Base M-step, then if Params contains 'e': EmissionProb = Obs with each row divided by its sum (no prior; a row of zeros stays zeros).</summary>
    protected override void DoMStep(SufficientStatistics stats) => throw new NotImplementedException();

    /// <summary>
    /// Probability of each sample under each state: the multinomial probability of the counts of the sample with n = their total and
    /// p = the row of EmissionProb, that is n! / (prod of count!) * prod of p^count, with 0^0 = 1.
    /// </summary>
    protected override double[,] ComputeLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>Natural logarithm of the same probability, computed directly (log-gamma for the factorials, count * log(p) with 0 * log(0) = 0 and count * log(0) = -infinity for a positive count).</summary>
    protected override double[,] ComputeLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>A <see cref="CategoricalStatistics"/>.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => throw new NotImplementedException();

    /// <summary>Base accumulation, then if Params contains 'e': Obs += transpose(posteriors) * x (n_components x n_features).</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice) => throw new NotImplementedException();

    /// <summary>
    /// Counts of NTrials independent draws from the emission probabilities of the state. Throws <see cref="InvalidOperationException"/> when NTrials is null.
    /// </summary>
    protected override double[] GenerateSampleFromState(int state, Random random) => throw new NotImplementedException();

    /// <summary>
    /// Base check of the number of features; every value must be a non-negative integer ("Symbol counts should be nonnegative integers");
    /// when NTrials has a value, every sample must total NTrials ("Total count for each sample should add up to the number of trials").
    /// All failures are <see cref="ArgumentException"/>.
    /// </summary>
    protected override void CheckAndSetNFeatures(double[,] x) => throw new NotImplementedException();

    /// <summary>'s' = nc - 1, 't' = nc * (nc - 1), 'e' = nc * (nf - 1); requires NFeatures to be known.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam() => throw new NotImplementedException();
}
