namespace HmmLearn;

/// <summary>The sufficient statistics of a Poisson model: the base ones plus Post and Obs.</summary>
public class PoissonStatistics : SufficientStatistics
{
    /// <summary>Allocates the base statistics, <see cref="Post"/> (n_components) and <see cref="Obs"/> (n_components x n_features), all zeros.</summary>
    public PoissonStatistics(int nComponents, int nFeatures) : base(nComponents) => throw new NotImplementedException();

    /// <summary>Post[c]: sum over the samples of the posterior probability of state c.</summary>
    public double[] Post { get; set; } = null!;

    /// <summary>Obs[c, f]: sum over the samples of posterior(c) * x[f].</summary>
    public double[,] Obs { get; set; } = null!;
}

/// <summary>
/// Hidden Markov model with Poisson emissions (port of PoissonHMM of hmmlearn/hmm.py and BasePoissonHMM of hmmlearn/_emissions.py).
/// An observation is a vector of n_features non-negative integer counts (stored as doubles); the features are independent given the state.
/// </summary>
public class PoissonHmm : BaseHmm
{
    /// <summary>
    /// Creates the model. <paramref name="lambdasPrior"/> and <paramref name="lambdasWeight"/> are the alpha and beta of the gamma
    /// prior of the rates. The defaults are those of hmmlearn (parameter letters "stl").
    /// </summary>
    public PoissonHmm(int nComponents = 1, double startProbPrior = 1.0, double transMatPrior = 1.0, double lambdasPrior = 0.0,
        double lambdasWeight = 0.0, DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi, int? randomState = null, int nIter = 10,
        double tol = 1e-2, bool verbose = false, string @params = "stl", string initParams = "stl",
        Implementation implementation = Implementation.Log)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
        => throw new NotImplementedException();

    /// <summary>Alpha of the gamma prior of the rates.</summary>
    public double LambdasPrior { get; set; }

    /// <summary>Beta of the gamma prior of the rates.</summary>
    public double LambdasWeight { get; set; }

    /// <summary>Rate of each feature in each state, shape (n_components, n_features); null until set or fitted.</summary>
    public double[,]? Lambdas { get; set; }

    /// <summary>
    /// Initializes before fitting: the base initialization, then, if <c>NeedsInit('l', "lambdas_", Lambdas != null)</c>, the method of
    /// moments on all the values of x: mean = average of all entries, variance = their population variance (divide by the count); each
    /// of the n_components x n_features rates is a draw from a gamma distribution of shape mean^2 / variance and scale variance / mean
    /// (rate mean / variance), using a <see cref="Random"/> seeded like the base. Throws <see cref="ArgumentException"/> when the mean or
    /// the variance of x is not positive.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths) => throw new NotImplementedException();

    /// <summary>
    /// Base checks; Lambdas must be non-null and (NComponents x n_features) with n_features = NFeatures when known (else the number of
    /// columns, which is then recorded), else <see cref="ArgumentException"/> ("lambdas_ must have shape (n_components, n_features)").
    /// </summary>
    protected override void Check() => throw new NotImplementedException();

    /// <summary>
    /// Base M-step, then if Params contains 'l': with n = the sum of ALL entries of stats.Post (the total mass of every state, as
    /// hmmlearn does), y_bar[c, f] = stats.Obs[c, f] / stats.Post[c]: Lambdas[c, f] = (LambdasPrior + n * y_bar[c, f]) / (LambdasWeight + n).
    /// </summary>
    protected override void DoMStep(SufficientStatistics stats) => throw new NotImplementedException();

    /// <summary>Probability of each sample under each state: the product over the features of the Poisson probability of the count.</summary>
    protected override double[,] ComputeLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>
    /// Log probability of each sample under each state: the sum over the features of count * log(rate) - rate - log(count!)
    /// (log-gamma), with 0 * log(0) = 0 so a zero count under a zero rate gives 0, and -infinity for a positive count under a zero rate.
    /// </summary>
    protected override double[,] ComputeLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>A <see cref="PoissonStatistics"/>.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => throw new NotImplementedException();

    /// <summary>Base accumulation, then if Params contains 'l': Post += column sums of the posteriors, Obs += transpose(posteriors) * x.</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice) => throw new NotImplementedException();

    /// <summary>One count per feature, each drawn from a Poisson distribution of the rate of the state.</summary>
    protected override double[] GenerateSampleFromState(int state, Random random) => throw new NotImplementedException();

    /// <summary>
    /// Base check of the number of features; every value must be a non-negative integer (<see cref="ArgumentException"/> otherwise:
    /// a stricter rule than hmmlearn's, which gives probability 0 to such values).
    /// </summary>
    protected override void CheckAndSetNFeatures(double[,] x) => throw new NotImplementedException();

    /// <summary>'s' = nc - 1, 't' = nc * (nc - 1), 'l' = nc * nf; requires NFeatures to be known.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam() => throw new NotImplementedException();
}
