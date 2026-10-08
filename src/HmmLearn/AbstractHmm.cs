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
    /// <summary>Shared generator for <see cref="Sample"/> when neither a random argument nor a <see cref="RandomState"/> seed is given.</summary>
    private static readonly Random SharedRandom = new();

    /// <summary>Sets the parameters and creates <see cref="Monitor"/> from <paramref name="tol"/>, <paramref name="nIter"/> and <paramref name="verbose"/>.</summary>
    protected AbstractHmm(int nComponents, DecoderAlgorithm algorithm, int? randomState, int nIter, double tol, bool verbose,
        string @params, string initParams, Implementation implementation)
    {
        NComponents = nComponents;
        Algorithm = algorithm;
        RandomState = randomState;
        NIter = nIter;
        Tol = tol;
        Verbose = verbose;
        Params = @params;
        InitParams = initParams;
        Implementation = implementation;
        Monitor = new ConvergenceMonitor(tol, nIter, verbose);
    }

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
    protected void RaiseWarning(string message) => Warning?.Invoke(message);

    /// <summary>Validates the parameters; throws <see cref="ArgumentException"/> if one is invalid. Called before scoring, decoding, sampling and fitting.</summary>
    protected abstract void Check();

    /// <summary>Log probability of each sample under each state, shape (n_samples, n_components): log p(x | state).</summary>
    protected abstract double[,] ComputeLogLikelihood(double[,] x);

    /// <summary>Probability of each sample under each state; by default the exponential of <see cref="ComputeLogLikelihood"/>.</summary>
    protected virtual double[,] ComputeLikelihood(double[,] x)
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

    /// <summary>One observation (n_features numbers) drawn from the emission distribution of <paramref name="state"/>.</summary>
    protected abstract double[] GenerateSampleFromState(int state, Random random);

    /// <summary>
    /// Throws <see cref="ArgumentException"/> if the matrix has no row, no column, or a value that is NaN or infinite
    /// (scikit-learn's check_array). Does not copy.
    /// </summary>
    protected static void CheckObservations(double[,] x)
    {
        var rows = x.GetLength(0);
        var cols = x.GetLength(1);
        if (rows == 0 || cols == 0)
            throw new ArgumentException($"x has shape ({rows}, {cols}); observations must have at least one row and one column", nameof(x));
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                if (double.IsNaN(x[i, j]) || double.IsInfinity(x[i, j]))
                    throw new ArgumentException($"x[{i}, {j}] is {x[i, j]}; observations must be finite", nameof(x));
    }

    /// <summary>Number of columns that inference accepts; <see cref="NFeatures"/> by default (null: not known, no check).</summary>
    protected virtual int? ObservationColumns => NFeatures;

    /// <summary>
    /// Validates the matrix for inference: <see cref="CheckObservations"/>, then its number of columns must equal
    /// <see cref="ObservationColumns"/> (not checked while that is null), else <see cref="ArgumentException"/> naming both numbers.
    /// </summary>
    private void CheckInferenceInput(double[,] x)
    {
        CheckObservations(x);
        var expected = ObservationColumns;
        var actual = x.GetLength(1);
        if (expected.HasValue && actual != expected.Value)
            throw new ArgumentException($"x has {actual} column(s) but the model expects {expected.Value}", nameof(x));
    }

    /// <summary>Throws <see cref="InvalidOperationException"/> if <see cref="StartProb"/> is null (the model is neither set nor fitted).</summary>
    protected void CheckIsFitted()
    {
        if (StartProb == null)
            throw new InvalidOperationException("the model is neither set nor fitted: StartProb is null");
    }

    /// <summary>
    /// Posterior probability of each state for each sample from scaled lattices: forward * backward element by element, then each
    /// row normalized to sum 1 (<see cref="Utils.Normalize(double[,], int?)"/> on axis 1). Both lattices have the same shape
    /// (n_samples, n_components); the result is a new matrix.
    /// </summary>
    internal static double[,] ComputePosteriorsScaling(double[,] fwdlattice, double[,] bwdlattice)
    {
        var rows = fwdlattice.GetLength(0);
        var cols = fwdlattice.GetLength(1);
        var posteriors = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                posteriors[i, j] = fwdlattice[i, j] * bwdlattice[i, j];
        Utils.Normalize(posteriors, 1);
        return posteriors;
    }

    /// <summary>
    /// Posterior probability of each state for each sample from log lattices: forward + backward, each row log-normalized
    /// (<see cref="Utils.LogNormalize(double[,], int?)"/> on axis 1, so the rows are normalized again even if pruning left them unnormalized),
    /// then exponentiated. The result is a new matrix.
    /// </summary>
    internal static double[,] ComputePosteriorsLog(double[,] fwdlattice, double[,] bwdlattice)
    {
        var rows = fwdlattice.GetLength(0);
        var cols = fwdlattice.GetLength(1);
        var posteriors = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                posteriors[i, j] = fwdlattice[i, j] + bwdlattice[i, j];
        Utils.LogNormalize(posteriors, 1);
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                posteriors[i, j] = Math.Exp(posteriors[i, j]);
        return posteriors;
    }

    /// <summary>
    /// Log probability of the data under the model: the sum over sequences of the log probability given by the forward pass
    /// (<see cref="Hmmc"/>), with the flavour of <see cref="Implementation"/>.
    /// Throws <see cref="ArgumentException"/> (parameter x) if x has a number of columns other than the model expects.
    /// </summary>
    public double Score(double[,] x, int[]? lengths = null)
    {
        CheckIsFitted();
        Check();
        CheckInferenceInput(x);
        var logProb = 0.0;
        foreach (var subX in InternalUtils.SplitXLengths(x, lengths))
            logProb += ForwardLogProb(subX);
        return logProb;
    }

    /// <summary>
    /// The log probability of <see cref="Score"/> and the posterior probability of each state for each sample, shape
    /// (n_samples, n_components); each sequence is processed on its own and the posteriors are concatenated in order.
    /// Posteriors with scaling: forward * backward, each row normalized to sum 1. In log space: forward + backward, each row
    /// log-normalized (<see cref="Utils"/>), then exponentiated.
    /// Throws <see cref="ArgumentException"/> (parameter x) if x has a number of columns other than the model expects.
    /// </summary>
    public (double LogProb, double[,] Posteriors) ScoreSamples(double[,] x, int[]? lengths = null)
    {
        CheckIsFitted();
        Check();
        CheckInferenceInput(x);
        var total = x.GetLength(0);
        var posteriors = new double[total, NComponents];
        var logProb = 0.0;
        var start = 0;
        foreach (var subX in InternalUtils.SplitXLengths(x, lengths))
        {
            (double subLogProb, double[,] subPosteriors) = ScoreSequence(subX);
            logProb += subLogProb;
            var rows = subX.GetLength(0);
            for (var r = 0; r < rows; r++)
                for (var c = 0; c < NComponents; c++)
                    posteriors[start + r, c] = subPosteriors[r, c];
            start += rows;
        }
        return (logProb, posteriors);
    }

    /// <summary>
    /// Finds the most likely state sequence of each sequence of x; returns the sum of their log probabilities and the states
    /// concatenated. <paramref name="algorithm"/> null: <see cref="Algorithm"/>. Viterbi: <see cref="Hmmc.Viterbi"/> on the log
    /// likelihoods of the sequence. Map: the posteriors of <see cref="ScoreSamples"/> of the sequence on its own; the log
    /// probability is the sum of the largest posterior of each sample (as hmmlearn does) and the state is the first index of the maximum.
    /// Throws <see cref="ArgumentException"/> (parameter x) if x has a number of columns other than the model expects.
    /// </summary>
    public (double LogProb, int[] States) Decode(double[,] x, int[]? lengths = null, DecoderAlgorithm? algorithm = null)
    {
        CheckIsFitted();
        Check();
        CheckInferenceInput(x);
        algorithm ??= Algorithm;
        var total = x.GetLength(0);
        var states = new int[total];
        var logProb = 0.0;
        var start = 0;
        foreach (var subX in InternalUtils.SplitXLengths(x, lengths))
        {
            (double subLogProb, int[] subStates) = algorithm == DecoderAlgorithm.Viterbi
                ? DecodeViterbi(subX)
                : DecodeMap(subX);
            logProb += subLogProb;
            Array.Copy(subStates, 0, states, start, subStates.Length);
            start += subStates.Length;
        }
        return (logProb, states);
    }

    /// <summary>The state sequence of <see cref="Decode"/> with the default algorithm.</summary>
    public int[] Predict(double[,] x, int[]? lengths = null) => Decode(x, lengths).States;

    /// <summary>The posteriors of <see cref="ScoreSamples"/>.</summary>
    public double[,] PredictProba(double[,] x, int[]? lengths = null) => ScoreSamples(x, lengths).Posteriors;

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
    {
        CheckIsFitted();
        Check();
        if (nSamples < 1)
            throw new ArgumentOutOfRangeException(nameof(nSamples), nSamples, "nSamples must be at least 1");
        if (currState != null && (currState < 0 || currState >= NComponents))
            throw new ArgumentOutOfRangeException(nameof(currState), currState, "currState must be between 0 and NComponents - 1");
        random ??= RandomState.HasValue ? new Random(RandomState.Value) : SharedRandom;

        var states = new int[nSamples];
        var firstState = currState ?? DrawState(CumulativeVector(StartProb!), random);
        states[0] = firstState;
        var firstObservation = GenerateSampleFromState(firstState, random);
        var nFeatures = firstObservation.Length;
        var x = new double[nSamples, nFeatures];
        for (var j = 0; j < nFeatures; j++)
            x[0, j] = firstObservation[j];

        var transMatCdf = CumulativeRows(TransMat!);
        for (var t = 1; t < nSamples; t++)
        {
            var state = DrawState(transMatCdf, states[t - 1], random);
            states[t] = state;
            var observation = GenerateSampleFromState(state, random);
            for (var j = 0; j < nFeatures; j++)
                x[t, j] = observation[j];
        }
        return (x, states);
    }

    /// <summary>Forward pass of one sequence with the flavour of <see cref="Implementation"/>; its log probability.</summary>
    private double ForwardLogProb(double[,] subX) => Implementation == Implementation.Log
        ? Hmmc.ForwardLog(StartProb!, TransMat!, ComputeLogLikelihood(subX)).LogProb
        : Hmmc.ForwardScaling(StartProb!, TransMat!, ComputeLikelihood(subX)).LogProb;

    /// <summary>Log probability and posteriors of one sequence, with the flavour of <see cref="Implementation"/>.</summary>
    private (double LogProb, double[,] Posteriors) ScoreSequence(double[,] subX) => Implementation == Implementation.Log
        ? ScoreSequenceLog(subX)
        : ScoreSequenceScaling(subX);

    /// <summary>Forward and backward in log space, then the log posteriors of the sequence.</summary>
    private (double LogProb, double[,] Posteriors) ScoreSequenceLog(double[,] subX)
    {
        var logFrameprob = ComputeLogLikelihood(subX);
        var (logProb, fwdlattice) = Hmmc.ForwardLog(StartProb!, TransMat!, logFrameprob);
        var bwdlattice = Hmmc.BackwardLog(StartProb!, TransMat!, logFrameprob);
        return (logProb, ComputePosteriorsLog(fwdlattice, bwdlattice));
    }

    /// <summary>Forward and backward with scaling, then the scaled posteriors of the sequence.</summary>
    private (double LogProb, double[,] Posteriors) ScoreSequenceScaling(double[,] subX)
    {
        var frameprob = ComputeLikelihood(subX);
        var (logProb, fwdlattice, scaling) = Hmmc.ForwardScaling(StartProb!, TransMat!, frameprob);
        var bwdlattice = Hmmc.BackwardScaling(StartProb!, TransMat!, frameprob, scaling);
        return (logProb, ComputePosteriorsScaling(fwdlattice, bwdlattice));
    }

    /// <summary>Viterbi decode of one sequence, always in log space (as hmmlearn does).</summary>
    private (double LogProb, int[] States) DecodeViterbi(double[,] subX) =>
        Hmmc.Viterbi(StartProb!, TransMat!, ComputeLogLikelihood(subX));

    /// <summary>MAP decode of one sequence: per-sample maximum posterior; the log probability is the sum of those maxima.</summary>
    private (double LogProb, int[] States) DecodeMap(double[,] subX)
    {
        var (_, posteriors) = ScoreSamples(subX);
        var rows = posteriors.GetLength(0);
        var cols = posteriors.GetLength(1);
        var logProb = 0.0;
        var states = new int[rows];
        for (var i = 0; i < rows; i++)
        {
            var best = posteriors[i, 0];
            var state = 0;
            for (var j = 1; j < cols; j++)
            {
                if (posteriors[i, j] > best)
                {
                    best = posteriors[i, j];
                    state = j;
                }
            }
            logProb += best;
            states[i] = state;
        }
        return (logProb, states);
    }

    /// <summary>Cumulative sum of a vector, in a new array.</summary>
    private static double[] CumulativeVector(double[] values)
    {
        var cdf = new double[values.Length];
        var acc = 0.0;
        for (var i = 0; i < values.Length; i++)
        {
            acc += values[i];
            cdf[i] = acc;
        }
        return cdf;
    }

    /// <summary>Cumulative sum of each row of a matrix, in a new matrix.</summary>
    private static double[,] CumulativeRows(double[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var cdf = new double[rows, cols];
        for (var i = 0; i < rows; i++)
        {
            var acc = 0.0;
            for (var j = 0; j < cols; j++)
            {
                acc += matrix[i, j];
                cdf[i, j] = acc;
            }
        }
        return cdf;
    }

    /// <summary>One state from the cumulative distribution of a vector: the first index whose cumulative probability is greater
    /// than one uniform draw, the last index if rounding leaves none.</summary>
    private static int DrawState(double[] cdf, Random random)
    {
        var u = random.NextDouble();
        for (var i = 0; i < cdf.Length; i++)
            if (cdf[i] > u)
                return i;
        return cdf.Length - 1;
    }

    /// <summary>One state from the cumulative distribution of row <paramref name="row"/> of a cumulative matrix: the first index
    /// whose cumulative probability is greater than one uniform draw, the last index if rounding leaves none.</summary>
    private static int DrawState(double[,] cdf, int row, Random random)
    {
        var u = random.NextDouble();
        var cols = cdf.GetLength(1);
        for (var i = 0; i < cols; i++)
            if (cdf[row, i] > u)
                return i;
        return cols - 1;
    }
}
