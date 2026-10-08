using MathNet.Numerics;
using MathNet.Numerics.Distributions;

namespace HmmLearn;

/// <summary>The sufficient statistics of a Poisson model: the base ones plus Post and Obs.</summary>
public class PoissonStatistics : SufficientStatistics
{
    /// <summary>Allocates the base statistics, <see cref="Post"/> (n_components) and <see cref="Obs"/> (n_components x n_features), all zeros.</summary>
    public PoissonStatistics(int nComponents, int nFeatures) : base(nComponents)
    {
        Post = new double[nComponents];
        Obs = new double[nComponents, nFeatures];
    }

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
    {
        LambdasPrior = lambdasPrior;
        LambdasWeight = lambdasWeight;
    }

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
    protected override void Init(double[,] x, int[] lengths)
    {
        base.Init(x, lengths);
        if (!NeedsInit('l', "lambdas_", Lambdas != null))
            return;

        var mean = 0.0;
        var rows = x.GetLength(0);
        var cols = x.GetLength(1);
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                mean += x[i, j];
        mean /= (double)(rows * cols);

        var variance = 0.0;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                variance += (x[i, j] - mean) * (x[i, j] - mean);
        variance /= (double)(rows * cols);
        if (mean <= 0.0 || variance <= 0.0)
            throw new ArgumentException($"the mean ({mean}) and the variance ({variance}) of x must be positive to initialize the rates");

        var random = RandomState.HasValue ? new Random(RandomState.Value) : Random.Shared;
        var gamma = Gamma.WithShapeScale(mean * mean / variance, variance / mean, random);
        var nFeatures = NFeatures!.Value;
        var lambdas = new double[NComponents, nFeatures];
        for (var c = 0; c < NComponents; c++)
            for (var f = 0; f < nFeatures; f++)
                lambdas[c, f] = gamma.Sample();
        Lambdas = lambdas;
    }

    /// <summary>
    /// Base checks; Lambdas must be non-null and (NComponents x n_features) with n_features = NFeatures when known (else the number of
    /// columns, which is then recorded), else <see cref="ArgumentException"/> ("lambdas_ must have shape (n_components, n_features)");
    /// every rate must be non-negative (a zero rate is legal, a negative one is not).
    /// </summary>
    protected override void Check()
    {
        base.Check();
        if (Lambdas == null)
            throw new ArgumentException("lambdas_ must have shape (n_components, n_features)");
        var nFeatures = NFeatures ?? Lambdas.GetLength(1);
        if (Lambdas.GetLength(0) != NComponents || Lambdas.GetLength(1) != nFeatures)
            throw new ArgumentException("lambdas_ must have shape (n_components, n_features)");
        NFeatures = nFeatures;
        var rows = Lambdas.GetLength(0);
        var cols = Lambdas.GetLength(1);
        for (var c = 0; c < rows; c++)
            for (var f = 0; f < cols; f++)
                if (Lambdas[c, f] < 0.0)
                    throw new ArgumentException($"lambdas_[{c}, {f}] is {Lambdas[c, f]}; rates must be non-negative");
    }

    /// <summary>
    /// Base M-step, then if Params contains 'l': with n = the sum of ALL entries of stats.Post (the total mass of every state, as
    /// hmmlearn does), y_bar[c, f] = stats.Obs[c, f] / stats.Post[c]: Lambdas[c, f] = (LambdasPrior + n * y_bar[c, f]) / (LambdasWeight + n).
    /// </summary>
    protected override void DoMStep(SufficientStatistics stats)
    {
        base.DoMStep(stats);
        if (!Params.Contains('l'))
            return;

        var post = ((PoissonStatistics)stats).Post;
        var obs = ((PoissonStatistics)stats).Obs;
        var n = 0.0;
        for (var c = 0; c < post.Length; c++)
            n += post[c];
        var rows = obs.GetLength(0);
        var cols = obs.GetLength(1);
        var lambdas = new double[rows, cols];
        for (var c = 0; c < rows; c++)
            for (var f = 0; f < cols; f++)
                lambdas[c, f] = (LambdasPrior + n * obs[c, f] / post[c]) / (LambdasWeight + n);
        Lambdas = lambdas;
    }

    /// <summary>Probability of each sample under each state: the product over the features of the Poisson probability of the count.</summary>
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

    /// <summary>
    /// Log probability of each sample under each state: the sum over the features of count * log(rate) - rate - log(count!)
    /// (log-gamma), with 0 * log(0) = 0 so a zero count under a zero rate gives 0, and -infinity for a positive count under a zero rate.
    /// </summary>
    protected override double[,] ComputeLogLikelihood(double[,] x)
    {
        var rows = x.GetLength(0);
        var nFeatures = x.GetLength(1);
        var lambdas = Lambdas!;
        var nComponents = lambdas.GetLength(0);
        var logLikelihood = new double[rows, nComponents];
        for (var c = 0; c < nComponents; c++)
        {
            for (var i = 0; i < rows; i++)
            {
                var logProb = 0.0;
                for (var f = 0; f < nFeatures; f++)
                {
                    var count = x[i, f];
                    var rate = lambdas[c, f];
                    if (count <= 0.0)
                        logProb -= rate;
                    else
                        logProb += count * Math.Log(rate) - rate - SpecialFunctions.GammaLn(count + 1.0);
                }
                logLikelihood[i, c] = logProb;
            }
        }
        return logLikelihood;
    }

    /// <summary>A <see cref="PoissonStatistics"/>.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics() => new PoissonStatistics(NComponents, NFeatures!.Value);

    /// <summary>Base accumulation, then if Params contains 'l': Post += column sums of the posteriors, Obs += transpose(posteriors) * x.</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice)
    {
        base.AccumulateSufficientStatistics(stats, x, lattice, posteriors, fwdlattice, bwdlattice);
        if (!Params.Contains('l'))
            return;

        var post = ((PoissonStatistics)stats).Post;
        var obs = ((PoissonStatistics)stats).Obs;
        var rows = x.GetLength(0);
        var nFeatures = x.GetLength(1);
        var nComponents = posteriors.GetLength(1);
        for (var i = 0; i < rows; i++)
        {
            for (var c = 0; c < nComponents; c++)
            {
                var posterior = posteriors[i, c];
                post[c] += posterior;
                for (var f = 0; f < nFeatures; f++)
                    obs[c, f] += posterior * x[i, f];
            }
        }
    }

    /// <summary>One count per feature, each drawn from a Poisson distribution of the rate of the state (a zero rate draws exactly 0).</summary>
    protected override double[] GenerateSampleFromState(int state, Random random)
    {
        var lambdas = Lambdas!;
        var nFeatures = lambdas.GetLength(1);
        var counts = new double[nFeatures];
        for (var f = 0; f < nFeatures; f++)
            counts[f] = lambdas[state, f] == 0.0 ? 0.0 : Poisson.Sample(random, lambdas[state, f]);
        return counts;
    }

    /// <summary>
    /// Base check of the number of features; every value must be a non-negative integer (<see cref="ArgumentException"/> otherwise:
    /// a stricter rule than hmmlearn's, which gives probability 0 to such values).
    /// </summary>
    protected override void CheckAndSetNFeatures(double[,] x)
    {
        base.CheckAndSetNFeatures(x);
        var rows = x.GetLength(0);
        var cols = x.GetLength(1);
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
            {
                var value = x[i, j];
                if (value < 0.0 || value != Math.Truncate(value))
                    throw new ArgumentException("Poisson observations must be non-negative integer counts");
            }
    }

    /// <summary>'s' = nc - 1, 't' = nc * (nc - 1), 'l' = nc * nf; requires NFeatures to be known.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam()
    {
        if (NFeatures is null)
            return null;
        var nc = NComponents;
        var nf = NFeatures.Value;
        return new Dictionary<char, int> { ['s'] = nc - 1, ['t'] = nc * (nc - 1), ['l'] = nc * nf };
    }
}
