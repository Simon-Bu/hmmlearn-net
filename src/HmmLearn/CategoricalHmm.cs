namespace HmmLearn;

/// <summary>The sufficient statistics of a categorical model: the base ones plus Obs.</summary>
public class CategoricalStatistics : SufficientStatistics
{
    /// <summary>Allocates the base statistics and <see cref="Obs"/> (n_components x n_features zeros).</summary>
    public CategoricalStatistics(int nComponents, int nFeatures) : base(nComponents) => throw new NotImplementedException();

    /// <summary>Obs[state, symbol]: expected number of times the symbol was emitted by the state.</summary>
    public double[,] Obs { get; set; } = null!;
}

/// <summary>
/// Hidden Markov model with categorical (discrete) emissions (port of CategoricalHMM of hmmlearn/hmm.py). Observations have shape
/// (n_samples, 1): one non-negative integer symbol per sample, stored as a double.
/// </summary>
public class CategoricalHmm : BaseHmm
{
    /// <summary>
    /// Creates the model. <paramref name="emissionProbPrior"/>: parameter of the symmetric Dirichlet prior of each row of
    /// <see cref="EmissionProb"/>; <paramref name="nFeatures"/>: number of symbols, inferred from the data (largest symbol + 1) if null.
    /// The defaults are those of hmmlearn.
    /// </summary>
    public CategoricalHmm(int nComponents = 1, double startProbPrior = 1.0, double transMatPrior = 1.0, double emissionProbPrior = 1.0,
        int? nFeatures = null, DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi, int? randomState = null, int nIter = 10,
        double tol = 1e-2, bool verbose = false, string @params = "ste", string initParams = "ste",
        Implementation implementation = Implementation.Log)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
        => throw new NotImplementedException();

    /// <summary>Parameter of the Dirichlet prior of each row of <see cref="EmissionProb"/>.</summary>
    public double EmissionProbPrior { get; set; }

    /// <summary>Probability of each symbol in each state, shape (n_components, n_features); null until set or fitted.</summary>
    public double[,]? EmissionProb { get; set; }

    /// <summary>
    /// Initializes before fitting: the base initialization (start probabilities, transition matrix, number of features), then, if
    /// <c>NeedsInit('e', "emissionprob_", EmissionProb != null)</c>, the emission probabilities are uniform random numbers in [0, 1)
    /// (<see cref="Random.NextDouble"/>, n_components x n_features, from the same seeding rule as the base) with each row normalized to sum 1.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths) => throw new NotImplementedException();

    /// <summary>
    /// Base checks, then: if NFeatures is null it becomes the number of columns of EmissionProb; EmissionProb must be non-null and
    /// (NComponents x NFeatures), else <see cref="ArgumentException"/> ("emissionprob_ must have shape (n_components, n_features)");
    /// its rows must sum to 1 (CheckSum1).
    /// </summary>
    protected override void Check() => throw new NotImplementedException();

    /// <summary>Base M-step, then if Params contains 'e': EmissionProb = max(EmissionProbPrior - 1 + Obs, 0), each row normalized to sum 1.</summary>
    protected override void DoMStep(SufficientStatistics stats) => throw new NotImplementedException();

    /// <summary>Probability of each sample under each state (<see cref="CategoricalEmission.Likelihood"/>).</summary>
    protected override double[,] ComputeLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>Natural logarithm of <see cref="ComputeLikelihood"/> (log of 0 is -infinity).</summary>
    protected override double[,] ComputeLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>A <see cref="CategoricalStatistics"/>.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => throw new NotImplementedException();

    /// <summary>Base accumulation, then if Params contains 'e': <see cref="CategoricalEmission.AccumulateObs"/> into Obs.</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice) => throw new NotImplementedException();

    /// <summary>One symbol drawn from the emission probabilities of the state.</summary>
    protected override double[] GenerateSampleFromState(int state, Random random) => throw new NotImplementedException();

    /// <summary><see cref="CategoricalEmission.CheckAndSetNFeatures"/> on <see cref="AbstractHmm.NFeatures"/>.</summary>
    protected override void CheckAndSetNFeatures(double[,] x) => throw new NotImplementedException();

    /// <summary><see cref="CategoricalEmission.NFitScalars"/>; requires NFeatures to be known.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam() => throw new NotImplementedException();
}
