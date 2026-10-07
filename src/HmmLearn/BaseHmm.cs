using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;

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
    {
        StartProbPrior = startProbPrior;
        TransMatPrior = transMatPrior;
    }

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
    protected override void Init(double[,] x, int[] lengths)
    {
        CheckAndSetNFeatures(x);
        var random = RandomState.HasValue ? new Random(RandomState.Value) : Random.Shared;
        var alpha = new double[NComponents];
        Array.Fill(alpha, 1.0 / NComponents);
        if (NeedsInit('s', "startprob_", StartProb != null))
            StartProb = DrawDirichlet(random, alpha);
        if (NeedsInit('t', "transmat_", TransMat != null))
            TransMat = DrawDirichletRows(random, alpha, NComponents);

        var nFitScalarsPerParam = GetNFitScalarsPerParam();
        if (nFitScalarsPerParam != null)
        {
            var nFitScalars = 0;
            foreach (var letter in Params)
                nFitScalars += nFitScalarsPerParam.GetValueOrDefault(letter);
            var size = x.GetLength(0) * x.GetLength(1);
            if (size < nFitScalars)
                RaiseWarning($"Fitting a model with {nFitScalars} free scalar parameters with only {size} data points will result in a degenerate solution.");
        }
    }

    /// <summary>
    /// Validates the start probabilities and the transition matrix; throws <see cref="ArgumentException"/> if
    /// <see cref="AbstractHmm.StartProb"/> is null or its length is not NComponents ("startprob_ must have length n_components"),
    /// if it does not sum to 1, if <see cref="AbstractHmm.TransMat"/> is null or not (NComponents x NComponents), or if a row does not sum
    /// to 1 (the sum tolerance is that of CheckSum1).
    /// </summary>
    protected override void Check()
    {
        if (StartProb == null || StartProb.Length != NComponents)
            throw new ArgumentException("startprob_ must have length n_components");
        CheckSum1("startprob_", StartProb);

        if (TransMat == null || TransMat.GetLength(0) != NComponents || TransMat.GetLength(1) != NComponents)
            throw new ArgumentException("transmat_ must have shape (n_components, n_components)");
        CheckSum1("transmat_", TransMat);
    }

    /// <summary>
    /// M-step of the start probabilities and transition matrix. If Params contains 's': new = max(StartProbPrior - 1 + stats.Start, 0),
    /// but an entry whose current StartProb is exactly 0 stays 0, then the vector is normalized to sum 1. If Params contains 't': the
    /// same with TransMatPrior and stats.Trans, row by row, each row normalized (a row that sums to 0 stays zeros). The updated
    /// arrays replace StartProb and TransMat (new arrays).
    /// </summary>
    protected override void DoMStep(SufficientStatistics stats)
    {
        if (Params.Contains('s'))
        {
            var current = StartProb!;
            var length = current.Length;
            var startProb = new double[length];
            for (var i = 0; i < length; i++)
                startProb[i] = current[i] == 0.0 ? 0.0 : Math.Max(StartProbPrior - 1.0 + stats.Start[i], 0.0);
            Utils.Normalize(startProb);
            StartProb = startProb;
        }
        if (Params.Contains('t'))
        {
            var current = TransMat!;
            var rows = current.GetLength(0);
            var cols = current.GetLength(1);
            var transMat = new double[rows, cols];
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++)
                    transMat[i, j] = current[i, j] == 0.0 ? 0.0 : Math.Max(TransMatPrior - 1.0 + stats.Trans[i, j], 0.0);
            Utils.Normalize(transMat, 1);
            TransMat = transMat;
        }
    }

    /// <summary>The log probability itself: EM maximizes it.</summary>
    protected override double ComputeLowerBound(double currLogProb) => currLogProb;

    /// <summary>
    /// The stationary distribution of the transition matrix: the left eigenvector for the eigenvalue of largest real part (1),
    /// normalized to sum 1 (for example through an eigen decomposition of the transposed matrix with MathNet.Numerics, or by solving
    /// the linear system). Throws <see cref="InvalidOperationException"/> if <see cref="AbstractHmm.TransMat"/> is null.
    /// </summary>
    public double[] GetStationaryDistribution()
    {
        if (TransMat == null)
            throw new InvalidOperationException("the model is neither set nor fitted: TransMat is null");

        var nc = NComponents;
        var system = new DenseMatrix(nc, nc);
        for (var i = 0; i < nc; i++)
            for (var j = 0; j < nc; j++)
                system[i, j] = TransMat[j, i] - (i == j ? 1.0 : 0.0);

        // The last equation is replaced by the normalization: the entries sum to 1 (for a stochastic matrix the eigenvalue of
        // largest real part is 1, so this is the left eigenvector of the transposed matrix, as in hmmlearn).
        for (var j = 0; j < nc; j++)
            system[nc - 1, j] = 1.0;
        var right = new DenseMatrix(nc, 1);
        right[nc - 1, 0] = 1.0;
        var result = system.LU().Solve(right);
        var stationary = new double[nc];
        for (var i = 0; i < nc; i++)
            stationary[i] = result[i, 0];
        return stationary;
    }

    /// <summary>Akaike information criterion on x: -2 * Score(x, lengths) + 2 * (sum of the values of GetNFitScalarsPerParam()). Lower is better. Throws <see cref="InvalidOperationException"/> if GetNFitScalarsPerParam() is null.</summary>
    public double Aic(double[,] x, int[]? lengths = null)
    {
        var nParams = NFitScalars();
        return -2.0 * Score(x, lengths) + 2.0 * nParams;
    }

    /// <summary>Bayesian information criterion on x: -2 * Score(x, lengths) + (sum of the values of GetNFitScalarsPerParam()) * ln(number of rows of x). Lower is better. Throws <see cref="InvalidOperationException"/> if GetNFitScalarsPerParam() is null.</summary>
    public double Bic(double[,] x, int[]? lengths = null)
    {
        var nParams = NFitScalars();
        return -2.0 * Score(x, lengths) + nParams * Math.Log(x.GetLength(0));
    }

    /// <summary>The sum of ALL the values of <see cref="AbstractHmm.GetNFitScalarsPerParam"/> (not only the letters of <see cref="AbstractHmm.Params"/>).</summary>
    private int NFitScalars()
    {
        var perParam = GetNFitScalarsPerParam();
        if (perParam == null)
            throw new InvalidOperationException("the number of free scalar parameters per parameter letter is unknown");
        var total = 0;
        foreach (var value in perParam.Values)
            total += value;
        return total;
    }

    /// <summary>One draw from the symmetric Dirichlet distribution of <paramref name="alpha"/>, in a new array.</summary>
    private static double[] DrawDirichlet(Random random, double[] alpha) => Dirichlet.Sample(random, alpha);

    /// <summary>One Dirichlet draw per row, in a new matrix of shape (rows, rows).</summary>
    private static double[,] DrawDirichletRows(Random random, double[] alpha, int rows)
    {
        var matrix = new double[rows, rows];
        for (var i = 0; i < rows; i++)
        {
            var row = DrawDirichlet(random, alpha);
            for (var j = 0; j < rows; j++)
                matrix[i, j] = row[j];
        }
        return matrix;
    }
}
