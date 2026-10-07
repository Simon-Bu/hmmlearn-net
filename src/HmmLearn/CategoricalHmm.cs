namespace HmmLearn;

/// <summary>The sufficient statistics of a categorical model: the base ones plus Obs.</summary>
public class CategoricalStatistics : SufficientStatistics
{
    /// <summary>Allocates the base statistics and <see cref="Obs"/> (n_components x n_features zeros).</summary>
    public CategoricalStatistics(int nComponents, int nFeatures) : base(nComponents)
    {
        Obs = new double[nComponents, nFeatures];
    }

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
    {
        EmissionProbPrior = emissionProbPrior;
        NFeatures = nFeatures;
    }

    /// <summary>Parameter of the Dirichlet prior of each row of <see cref="EmissionProb"/>.</summary>
    public double EmissionProbPrior { get; set; }

    /// <summary>Probability of each symbol in each state, shape (n_components, n_features); null until set or fitted.</summary>
    public double[,]? EmissionProb { get; set; }

    /// <summary>
    /// Initializes before fitting: the base initialization (start probabilities, transition matrix, number of features), then, if
    /// <c>NeedsInit('e', "emissionprob_", EmissionProb != null)</c>, the emission probabilities are uniform random numbers in [0, 1)
    /// (<see cref="Random.NextDouble"/>, n_components x n_features, from the same seeding rule as the base) with each row normalized to sum 1.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths)
    {
        base.Init(x, lengths);
        if (!NeedsInit('e', "emissionprob_", EmissionProb != null))
            return;

        var random = RandomState.HasValue ? new Random(RandomState.Value) : Random.Shared;
        var nFeatures = NFeatures!.Value;
        var emissionProb = new double[NComponents, nFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var j = 0; j < nFeatures; j++)
                emissionProb[c, j] = random.NextDouble();
        Utils.Normalize(emissionProb, 1);
        EmissionProb = emissionProb;
    }

    /// <summary>
    /// Base checks, then: if NFeatures is null it becomes the number of columns of EmissionProb; EmissionProb must be non-null and
    /// (NComponents x NFeatures), else <see cref="ArgumentException"/> ("emissionprob_ must have shape (n_components, n_features)");
    /// its rows must sum to 1 (CheckSum1).
    /// </summary>
    protected override void Check()
    {
        base.Check();
        if (EmissionProb == null)
            throw new ArgumentException("emissionprob_ must have shape (n_components, n_features)");
        if (NFeatures is null)
            NFeatures = EmissionProb.GetLength(1);
        if (EmissionProb.GetLength(0) != NComponents || EmissionProb.GetLength(1) != NFeatures)
            throw new ArgumentException("emissionprob_ must have shape (n_components, n_features)");
        CheckSum1("emissionprob_", EmissionProb);
    }

    /// <summary>Base M-step, then if Params contains 'e': EmissionProb = max(EmissionProbPrior - 1 + Obs, 0), each row normalized to sum 1.</summary>
    protected override void DoMStep(SufficientStatistics stats)
    {
        base.DoMStep(stats);
        if (!Params.Contains('e'))
            return;

        var obs = ((CategoricalStatistics)stats).Obs;
        var rows = obs.GetLength(0);
        var cols = obs.GetLength(1);
        var emissionProb = new double[rows, cols];
        for (var c = 0; c < rows; c++)
            for (var j = 0; j < cols; j++)
                emissionProb[c, j] = Math.Max(EmissionProbPrior - 1.0 + obs[c, j], 0.0);
        Utils.Normalize(emissionProb, 1);
        EmissionProb = emissionProb;
    }

    /// <summary>Probability of each sample under each state (<see cref="CategoricalEmission.Likelihood"/>).</summary>
    protected override double[,] ComputeLikelihood(double[,] x) => CategoricalEmission.Likelihood(EmissionProb!, x);

    /// <summary>Natural logarithm of <see cref="ComputeLikelihood"/> (log of 0 is -infinity).</summary>
    protected override double[,] ComputeLogLikelihood(double[,] x)
    {
        var likelihood = CategoricalEmission.Likelihood(EmissionProb!, x);
        var rows = likelihood.GetLength(0);
        var cols = likelihood.GetLength(1);
        var logLikelihood = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                logLikelihood[i, j] = Math.Log(likelihood[i, j]);
        return logLikelihood;
    }

    /// <summary>A <see cref="CategoricalStatistics"/>.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => new CategoricalStatistics(NComponents, NFeatures!.Value);

    /// <summary>Base accumulation, then if Params contains 'e': <see cref="CategoricalEmission.AccumulateObs"/> into Obs.</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice)
    {
        base.AccumulateSufficientStatistics(stats, x, lattice, posteriors, fwdlattice, bwdlattice);
        if (Params.Contains('e'))
            CategoricalEmission.AccumulateObs(((CategoricalStatistics)stats).Obs, x, posteriors);
    }

    /// <summary>One symbol drawn from the emission probabilities of the state.</summary>
    protected override double[] GenerateSampleFromState(int state, Random random) => CategoricalEmission.GenerateSample(EmissionProb!, state, random);

    /// <summary><see cref="CategoricalEmission.CheckAndSetNFeatures"/> on <see cref="AbstractHmm.NFeatures"/>.</summary>
    protected override void CheckAndSetNFeatures(double[,] x)
    {
        var nFeatures = NFeatures;
        CategoricalEmission.CheckAndSetNFeatures(x, ref nFeatures);
        NFeatures = nFeatures;
    }

    /// <summary><see cref="CategoricalEmission.NFitScalars"/>; requires NFeatures to be known.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam()
        => NFeatures is null ? null : CategoricalEmission.NFitScalars(NComponents, NFeatures.Value);
}
