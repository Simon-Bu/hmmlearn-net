using MathNet.Numerics;

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
    {
        NTrials = nTrials;
    }

    /// <summary>Number of trials of every sample; null when the totals are free.</summary>
    public int? NTrials { get; set; }

    /// <summary>Probability of each symbol in each state, shape (n_components, n_features); null until set or fitted.</summary>
    public double[,]? EmissionProb { get; set; }

    /// <summary>
    /// Base initialization; then, if InitParams contains 'e' (always re-initialized, without the "will be overwritten" warning),
    /// EmissionProb is uniform random numbers in [0, 1) (n_components x n_features, same seeding rule as the base) with each row normalized.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths)
    {
        base.Init(x, lengths);
        if (!InitParams.Contains('e'))
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
    /// Base checks; EmissionProb must be non-null and (NComponents x NFeatures), else <see cref="ArgumentException"/>
    /// ("emissionprob_ must have shape (n_components, n_features)"); if NFeatures is null it becomes the number of columns; rows sum to 1.
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

    /// <summary>Base M-step, then if Params contains 'e': EmissionProb = Obs with each row divided by its sum (no prior; a row of zeros stays zeros).</summary>
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
                emissionProb[c, j] = obs[c, j];
        Utils.Normalize(emissionProb, 1);
        EmissionProb = emissionProb;
    }

    /// <summary>
    /// Probability of each sample under each state: the multinomial probability of the counts of the sample with n = their total and
    /// p = the row of EmissionProb, that is n! / (prod of count!) * prod of p^count, with 0^0 = 1.
    /// </summary>
    protected override double[,] ComputeLikelihood(double[,] x)
    {
        var logLikelihood = ComputeLogLikelihood(x);
        var rows = logLikelihood.GetLength(0);
        var cols = logLikelihood.GetLength(1);
        var likelihood = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                likelihood[i, j] = Math.Exp(logLikelihood[i, j]);
        return likelihood;
    }

    /// <summary>Natural logarithm of the same probability, computed directly (log-gamma for the factorials, count * log(p) with 0 * log(0) = 0 and count * log(0) = -infinity for a positive count).</summary>
    protected override double[,] ComputeLogLikelihood(double[,] x)
    {
        var nSamples = x.GetLength(0);
        var nFeatures = x.GetLength(1);
        var emissionProb = EmissionProb!;
        var nComponents = emissionProb.GetLength(0);
        var logLikelihood = new double[nSamples, nComponents];
        for (var i = 0; i < nSamples; i++)
        {
            var n = 0.0;
            for (var j = 0; j < nFeatures; j++)
                n += x[i, j];
            if (n == 0.0)
            {
                // 0! / (prod 0!) * prod p^0 = 1 exactly (and scipy's gammaln(1) is exactly 0, so the log is exactly 0).
                for (var c = 0; c < nComponents; c++)
                    logLikelihood[i, c] = 0.0;
                continue;
            }
            var logCoefficient = SpecialFunctions.GammaLn(n + 1.0);
            for (var j = 0; j < nFeatures; j++)
                logCoefficient -= SpecialFunctions.GammaLn(x[i, j] + 1.0);
            for (var c = 0; c < nComponents; c++)
            {
                var logProb = logCoefficient;
                for (var j = 0; j < nFeatures; j++)
                {
                    var count = x[i, j];
                    if (count <= 0.0)
                        continue;
                    var p = emissionProb[c, j];
                    logProb += p > 0.0 ? count * Math.Log(p) : double.NegativeInfinity;
                }
                logLikelihood[i, c] = logProb;
            }
        }
        return logLikelihood;
    }

    /// <summary>A <see cref="CategoricalStatistics"/>.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => new CategoricalStatistics(NComponents, NFeatures!.Value);

    /// <summary>Base accumulation, then if Params contains 'e': Obs += transpose(posteriors) * x (n_components x n_features).</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice)
    {
        base.AccumulateSufficientStatistics(stats, x, lattice, posteriors, fwdlattice, bwdlattice);
        if (!Params.Contains('e'))
            return;

        var obs = ((CategoricalStatistics)stats).Obs;
        var nSamples = x.GetLength(0);
        var nFeatures = x.GetLength(1);
        var nComponents = posteriors.GetLength(1);
        for (var i = 0; i < nSamples; i++)
            for (var c = 0; c < nComponents; c++)
            {
                var posterior = posteriors[i, c];
                for (var j = 0; j < nFeatures; j++)
                    obs[c, j] += posterior * x[i, j];
            }
    }

    /// <summary>
    /// Counts of NTrials independent draws from the emission probabilities of the state. Throws <see cref="InvalidOperationException"/> when NTrials is null.
    /// </summary>
    protected override double[] GenerateSampleFromState(int state, Random random)
    {
        var nTrials = NTrials ?? throw new InvalidOperationException("NTrials must be set to sample a MultinomialHmm");
        var emissionProb = EmissionProb!;
        var nFeatures = emissionProb.GetLength(1);
        var counts = new double[nFeatures];
        for (var t = 0; t < nTrials; t++)
        {
            var u = random.NextDouble();
            var cumulative = 0.0;
            var symbol = nFeatures - 1;
            for (var j = 0; j < nFeatures; j++)
            {
                cumulative += emissionProb[state, j];
                if (cumulative > u)
                {
                    symbol = j;
                    break;
                }
            }
            counts[symbol]++;
        }
        return counts;
    }

    /// <summary>
    /// Base check of the number of features; every value must be a non-negative integer ("Symbol counts should be nonnegative integers");
    /// when NTrials has a value, every sample must total NTrials ("Total count for each sample should add up to the number of trials").
    /// All failures are <see cref="ArgumentException"/>.
    /// </summary>
    protected override void CheckAndSetNFeatures(double[,] x)
    {
        base.CheckAndSetNFeatures(x);
        var rows = x.GetLength(0);
        var cols = x.GetLength(1);
        for (var i = 0; i < rows; i++)
        {
            var total = 0.0;
            for (var j = 0; j < cols; j++)
            {
                var value = x[i, j];
                if (value < 0.0 || value != Math.Truncate(value))
                    throw new ArgumentException("Symbol counts should be nonnegative integers");
                total += value;
            }
            if (NTrials.HasValue && total != NTrials.Value)
                throw new ArgumentException("Total count for each sample should add up to the number of trials");
        }
    }

    /// <summary>'s' = nc - 1, 't' = nc * (nc - 1), 'e' = nc * (nf - 1); requires NFeatures to be known.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam()
        => NFeatures is null ? null : CategoricalEmission.NFitScalars(NComponents, NFeatures.Value);
}
