namespace HmmLearn;

/// <summary>
/// Base class of the hidden Markov models learned by Expectation-Maximization (port of BaseHMM of hmmlearn/base.py): it checks and
/// randomly initializes the start probabilities and the transition matrix, updates them in the M-step with Dirichlet priors,
/// and offers the stationary distribution and the information criteria. Models add their emission parameters on top.
/// </summary>
public abstract class BaseHmm : AbstractHmm
{
    /// <summary>
    /// Creates the model. <paramref name="startProbPrior"/> and <paramref name="transMatPrior"/> are the parameters of the
    /// symmetric Dirichlet priors of the start probabilities and of each row of the transition matrix (1 means no prior).
    /// </summary>
    protected BaseHmm(int nComponents, double startProbPrior, double transMatPrior, DecoderAlgorithm algorithm, int? randomState,
        int nIter, double tol, bool verbose, string @params, string initParams, Implementation implementation)
        : base(nComponents, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
        => throw new NotImplementedException();

    /// <summary>Parameter of the Dirichlet prior of <see cref="AbstractHmm.StartProb"/>.</summary>
    public double StartProbPrior { get; set; }

    /// <summary>Parameter of the Dirichlet prior of each row of <see cref="AbstractHmm.TransMat"/>.</summary>
    public double TransMatPrior { get; set; }

    /// <summary>
    /// Initializes before fitting (hmmlearn's BaseHMM._init): <c>CheckAndSetNFeatures(x)</c>; the start probabilities, if
    /// <c>NeedsInit('s', "startprob_", StartProb != null)</c>, are drawn from a Dirichlet distribution with every parameter equal to
    /// 1 / NComponents; the transition matrix, if <c>NeedsInit('t', "transmat_", TransMat != null)</c>, gets one such draw per row;
    /// the random numbers come from a <see cref="Random"/> seeded with <see cref="AbstractHmm.RandomState"/> when it has a value
    /// (a new generator at each call), else from <see cref="Random.Shared"/>. Then, if <see cref="AbstractHmm.GetNFitScalarsPerParam"/>
    /// is not null and the number of values in x (rows times columns) is lower than the sum of the entries of the letters of
    /// <see cref="AbstractHmm.Params"/> (letters absent from the dictionary count 0), raises a warning: "Fitting a model with {n} free
    /// scalar parameters with only {size} data points will result in a degenerate solution."
    /// </summary>
    protected override void Init(double[,] x, int[] lengths) => throw new NotImplementedException();

    /// <summary>
    /// Validates the start probabilities and the transition matrix; throws <see cref="ArgumentException"/> if
    /// <see cref="AbstractHmm.StartProb"/> is null or its length is not NComponents ("startprob_ must have length n_components"),
    /// if it does not sum to 1, if <see cref="AbstractHmm.TransMat"/> is null or not (NComponents x NComponents), or if a row does not sum
    /// to 1 (the sum tolerance is that of CheckSum1).
    /// </summary>
    protected override void Check() => throw new NotImplementedException();

    /// <summary>
    /// M-step of the start probabilities and transition matrix. If Params contains 's': new = max(StartProbPrior - 1 + stats.Start, 0),
    /// but an entry whose current StartProb is exactly 0 stays 0, then the vector is normalized to sum 1. If Params contains 't': the
    /// same with TransMatPrior and stats.Trans, row by row, each row normalized (a row that sums to 0 stays zeros). The updated
    /// arrays replace StartProb and TransMat (new arrays).
    /// </summary>
    protected override void DoMStep(SufficientStatistics stats) => throw new NotImplementedException();

    /// <summary>The log probability itself: EM maximizes it.</summary>
    protected override double ComputeLowerBound(double currLogProb) => throw new NotImplementedException();

    /// <summary>
    /// The stationary distribution of the transition matrix: the left eigenvector for the eigenvalue of largest real part (1),
    /// normalized to sum 1 (for example through an eigen decomposition of the transposed matrix with MathNet.Numerics, or by solving
    /// the linear system). Throws <see cref="InvalidOperationException"/> if <see cref="AbstractHmm.TransMat"/> is null.
    /// </summary>
    public double[] GetStationaryDistribution() => throw new NotImplementedException();

    /// <summary>Akaike information criterion on x: -2 * Score(x, lengths) + 2 * (sum of the values of GetNFitScalarsPerParam()). Lower is better. Throws <see cref="InvalidOperationException"/> if GetNFitScalarsPerParam() is null.</summary>
    public double Aic(double[,] x, int[]? lengths = null) => throw new NotImplementedException();

    /// <summary>Bayesian information criterion on x: -2 * Score(x, lengths) + (sum of the values of GetNFitScalarsPerParam()) * ln(number of rows of x). Lower is better. Throws <see cref="InvalidOperationException"/> if GetNFitScalarsPerParam() is null.</summary>
    public double Bic(double[,] x, int[]? lengths = null) => throw new NotImplementedException();
}
