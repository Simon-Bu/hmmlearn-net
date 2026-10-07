namespace HmmLearn;

/// <summary>
/// Base class of the hidden Markov models learned by Expectation-Maximization or Variational Bayes (port of the inference half
/// of _AbstractHMM of hmmlearn/base.py). This file holds the parameters, the hooks that models override and the inference
/// methods (score, decode, predict, sample); the fitting half is in AbstractHmm.Fit.cs.
/// Observations are always a matrix <c>double[,]</c> of shape (n_samples, n_features); several sequences are concatenated in one
/// matrix and described by <c>lengths</c> (their lengths, summing to n_samples; null means one sequence).
/// </summary>
public abstract partial class AbstractHmm
{
    /// <summary>Sets the parameters and creates <see cref="Monitor"/> from <paramref name="tol"/>, <paramref name="nIter"/> and <paramref name="verbose"/>.</summary>
    protected AbstractHmm(int nComponents, DecoderAlgorithm algorithm, int? randomState, int nIter, double tol, bool verbose,
        string @params, string initParams, Implementation implementation) => throw new NotImplementedException();

    /// <summary>Number of states.</summary>
    public int NComponents { get; set; }

    /// <summary>Default decoder of <see cref="Decode"/> and <see cref="Predict"/>.</summary>
    public DecoderAlgorithm Algorithm { get; set; }

    /// <summary>Seed of the random numbers used by <see cref="Sample"/> and by initializations; null means non-deterministic.</summary>
    public int? RandomState { get; set; }

    /// <summary>Maximum number of EM iterations.</summary>
    public int NIter { get; set; }

    /// <summary>Convergence threshold: EM stops when the gain in log-likelihood is below it.</summary>
    public double Tol { get; set; }

    /// <summary>Whether each EM iteration is reported by <see cref="Monitor"/>.</summary>
    public bool Verbose { get; set; }

    /// <summary>The parameters updated during fitting, as letters ('s' start probabilities, 't' transitions, others per model).</summary>
    public string Params { get; set; } = string.Empty;

    /// <summary>The parameters initialized before fitting, as letters.</summary>
    public string InitParams { get; set; } = string.Empty;

    /// <summary>Forward-backward flavour.</summary>
    public Implementation Implementation { get; set; }

    /// <summary>Number of features of the observations; null until known.</summary>
    public int? NFeatures { get; set; }

    /// <summary>Initial state occupation distribution, length <see cref="NComponents"/>; null until set or fitted.</summary>
    public double[]? StartProb { get; set; }

    /// <summary>Transition probabilities between states, shape (n_components, n_components); null until set or fitted.</summary>
    public double[,]? TransMat { get; set; }

    /// <summary>Monitor of the EM convergence, created by the constructor.</summary>
    public ConvergenceMonitor Monitor { get; set; } = null!;

    /// <summary>Raised with a message for each warning (for instance a degenerate fit).</summary>
    public event Action<string>? Warning;

    /// <summary>Raises <see cref="Warning"/> if anybody listens.</summary>
    protected void RaiseWarning(string message) => throw new NotImplementedException();

    /// <summary>Validates the parameters; throws <see cref="ArgumentException"/> if one is invalid. Called before scoring, decoding, sampling and fitting.</summary>
    protected abstract void Check();

    /// <summary>Log probability of each sample under each state, shape (n_samples, n_components): log p(x | state).</summary>
    protected abstract double[,] ComputeLogLikelihood(double[,] x);

    /// <summary>Probability of each sample under each state; by default the exponential of <see cref="ComputeLogLikelihood"/>.</summary>
    protected virtual double[,] ComputeLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>One observation (n_features numbers) drawn from the emission distribution of <paramref name="state"/>.</summary>
    protected abstract double[] GenerateSampleFromState(int state, Random random);

    /// <summary>
    /// Throws <see cref="ArgumentException"/> if the matrix has no row, no column, or a value that is NaN or infinite
    /// (scikit-learn's check_array). Does not copy.
    /// </summary>
    protected static void CheckObservations(double[,] x) => throw new NotImplementedException();

    /// <summary>Throws <see cref="InvalidOperationException"/> if <see cref="StartProb"/> is null (the model is neither set nor fitted).</summary>
    protected void CheckIsFitted() => throw new NotImplementedException();

    /// <summary>
    /// Posterior probability of each state for each sample from scaled lattices: forward * backward element by element, then each
    /// row normalized to sum 1 (<see cref="Utils.Normalize(double[,], int?)"/> on axis 1). Both lattices have the same shape
    /// (n_samples, n_components); the result is a new matrix.
    /// </summary>
    internal static double[,] ComputePosteriorsScaling(double[,] fwdlattice, double[,] bwdlattice) => throw new NotImplementedException();

    /// <summary>
    /// Posterior probability of each state for each sample from log lattices: forward + backward, each row log-normalized
    /// (<see cref="Utils.LogNormalize(double[,], int?)"/> on axis 1, so the rows are normalized again even if pruning left them unnormalized),
    /// then exponentiated. The result is a new matrix.
    /// </summary>
    internal static double[,] ComputePosteriorsLog(double[,] fwdlattice, double[,] bwdlattice) => throw new NotImplementedException();

    /// <summary>
    /// Log probability of the data under the model: the sum over sequences of the log probability given by the forward pass
    /// (<see cref="Hmmc"/>), with the flavour of <see cref="Implementation"/>.
    /// </summary>
    public double Score(double[,] x, int[]? lengths = null) => throw new NotImplementedException();

    /// <summary>
    /// The log probability of <see cref="Score"/> and the posterior probability of each state for each sample, shape
    /// (n_samples, n_components); each sequence is processed on its own and the posteriors are concatenated in order.
    /// Posteriors with scaling: forward * backward, each row normalized to sum 1. In log space: forward + backward, each row
    /// log-normalized (<see cref="Utils"/>), then exponentiated.
    /// </summary>
    public (double LogProb, double[,] Posteriors) ScoreSamples(double[,] x, int[]? lengths = null) => throw new NotImplementedException();

    /// <summary>
    /// Finds the most likely state sequence of each sequence of x; returns the sum of their log probabilities and the states
    /// concatenated. <paramref name="algorithm"/> null: <see cref="Algorithm"/>. Viterbi: <see cref="Hmmc.Viterbi"/> on the log
    /// likelihoods of the sequence. Map: the posteriors of <see cref="ScoreSamples"/> of the sequence on its own; the log
    /// probability is the sum of the largest posterior of each sample (as hmmlearn does) and the state is the first index of the maximum.
    /// </summary>
    public (double LogProb, int[] States) Decode(double[,] x, int[]? lengths = null, DecoderAlgorithm? algorithm = null)
        => throw new NotImplementedException();

    /// <summary>The state sequence of <see cref="Decode"/> with the default algorithm.</summary>
    public int[] Predict(double[,] x, int[]? lengths = null) => throw new NotImplementedException();

    /// <summary>The posteriors of <see cref="ScoreSamples"/>.</summary>
    public double[,] PredictProba(double[,] x, int[]? lengths = null) => throw new NotImplementedException();

    /// <summary>
    /// Generates <paramref name="nSamples"/> observations and the states behind them. The first state is
    /// <paramref name="currState"/> if given, else drawn from <see cref="StartProb"/>; each next state is drawn from the row of
    /// <see cref="TransMat"/> of the previous one; each observation comes from <see cref="GenerateSampleFromState"/>. A state is
    /// drawn with one uniform number u from the cumulative distribution: the first index whose cumulative probability is greater
    /// than u (the last index if rounding leaves none). <paramref name="random"/> null: a generator seeded with
    /// <see cref="RandomState"/> if it has a value, else a shared one. The order of the draws is: first state (unless given), its
    /// observation, then for each next sample the state and the observation.
    /// Throws <see cref="ArgumentOutOfRangeException"/> for nSamples below 1 or a currState outside 0..NComponents-1.
    /// </summary>
    public (double[,] X, int[] States) Sample(int nSamples = 1, Random? random = null, int? currState = null)
        => throw new NotImplementedException();
}
