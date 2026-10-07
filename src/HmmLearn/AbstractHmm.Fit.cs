using System.Globalization;

namespace HmmLearn;

/// <summary>
/// The fitting half of AbstractHmm (port of _AbstractHMM.fit and its helpers in hmmlearn/base.py): the EM loop, the E-step, the
/// base sufficient statistics, and the helpers models use when they initialize. Every member added here is virtual or non-abstract
/// (the classes that derive from AbstractHmm in earlier acceptance tests must keep compiling).
/// </summary>
public abstract partial class AbstractHmm
{
    /// <summary>The zero-row warning, raised by <see cref="Fit(double[,], int[])"/> after an iteration that did not converge
    /// leaves a row of <see cref="TransMat"/> with zero sum.</summary>
    private const string ZeroRowWarning = "Some rows of transmat_ have zero sum because no transition from the state was ever observed.";

    /// <summary>The tolerance of <see cref="CheckSum1(string, double[])"/> and <see cref="CheckSum1(string, double[,])"/>:
    /// |sum - 1| &lt;= 1e-8 + 1e-5 (numpy.allclose against 1).</summary>
    private const double Sum1Tolerance = 1e-8 + 1e-5;

    /// <summary>
    /// Estimates the parameters from the observations (several sequences described by <paramref name="lengths"/>; null: one
    /// sequence). Steps: <see cref="CheckObservations"/>; lengths validated (null: one sequence of all rows); <see cref="Init(double[,], int[])"/>;
    /// <see cref="Check"/>; <see cref="Monitor"/> receives <see cref="Tol"/>, <see cref="NIter"/> and <see cref="Verbose"/> and is
    /// reset; then at most <see cref="NIter"/> times: E-step, <see cref="ComputeLowerBound(double)"/> of the log probability,
    /// <see cref="DoMStep(SufficientStatistics)"/>, <c>Monitor.Report(lowerBound)</c>, stop if <c>Monitor.Converged</c>, otherwise
    /// raise <see cref="ZeroRowWarning"/> when a row of <see cref="TransMat"/> sums to 0. The warnings of the monitor are forwarded
    /// to <see cref="RaiseWarning"/> while fitting; the handler is unsubscribed in a finally block, so the monitor is left as it
    /// was found. The M-step runs even on the iteration that converges. Returns this instance. NIter = 0 only initializes and
    /// checks.
    /// </summary>
    public AbstractHmm Fit(double[,] x, int[]? lengths = null)
    {
        CheckObservations(x);
        if (lengths == null)
            lengths = new[] { x.GetLength(0) };
        else
            ValidateLengths(x, lengths);

        Init(x, lengths);
        Check();
        Monitor.Tol = Tol;
        Monitor.NIter = NIter;
        Monitor.Verbose = Verbose;
        Monitor.Reset();

        void OnMonitorWarning(string message) => RaiseWarning(message);
        Monitor.Warning += OnMonitorWarning;
        try
        {
            for (var iteration = 0; iteration < NIter; iteration++)
            {
                var (stats, currLogProb) = DoEstep(x, lengths);
                var lowerBound = ComputeLowerBound(currLogProb);
                DoMStep(stats);
                Monitor.Report(lowerBound);
                if (Monitor.Converged)
                    break;
                if (HasZeroRow(TransMat))
                    RaiseWarning(ZeroRowWarning);
            }
        }
        finally
        {
            Monitor.Warning -= OnMonitorWarning;
        }
        return this;
    }

    /// <summary>
    /// One E-step (hmmlearn's _do_estep): <see cref="InitializeSufficientStatistics"/>, <see cref="EstepBegin"/>, then for each
    /// sequence from <see cref="InternalUtils.SplitXLengths(double[,], int[])"/> the forward-backward pass of
    /// <see cref="Implementation"/> and <see cref="AccumulateSufficientStatistics"/>. Returns the statistics and the sum of the
    /// log probabilities of the sequences.
    /// </summary>
    private (SufficientStatistics Stats, double CurrLogProb) DoEstep(double[,] x, int[] lengths)
    {
        var stats = InitializeSufficientStatistics();
        EstepBegin();
        var currLogProb = 0.0;
        foreach (var sequence in InternalUtils.SplitXLengths(x, lengths))
        {
            var (logProb, lattice, posteriors, fwdlattice, bwdlattice) = Implementation == Implementation.Log
                ? FitSequenceLog(sequence)
                : FitSequenceScaling(sequence);
            AccumulateSufficientStatistics(stats, sequence, lattice, posteriors, fwdlattice, bwdlattice);
            currLogProb += logProb;
        }
        return (stats, currLogProb);
    }

    /// <summary>
    /// The log pass of one sequence (hmmlearn's _fit_log): <see cref="ComputeFitLogLikelihood(double[,])"/>, then
    /// <see cref="Hmmc.ForwardLog"/> and <see cref="Hmmc.BackwardLog"/> with <see cref="FitStartProb"/> and <see cref="FitTransMat"/>,
    /// then <see cref="ComputePosteriorsLog(double[,], double[,])"/>. Returns the log probability, the log lattice, the posteriors
    /// and the forward and backward lattices.
    /// </summary>
    private (double LogProb, double[,] Lattice, double[,] Posteriors, double[,] FwdLattice, double[,] BwdLattice) FitSequenceLog(double[,] sequence)
    {
        var logFrame = ComputeFitLogLikelihood(sequence);
        var (logProb, fwdlattice) = Hmmc.ForwardLog(FitStartProb, FitTransMat, logFrame);
        var bwdlattice = Hmmc.BackwardLog(FitStartProb, FitTransMat, logFrame);
        return (logProb, logFrame, ComputePosteriorsLog(fwdlattice, bwdlattice), fwdlattice, bwdlattice);
    }

    /// <summary>
    /// The scaling pass of one sequence (hmmlearn's _fit_scaling): <see cref="ComputeFitLikelihood(double[,])"/>, then
    /// <see cref="Hmmc.ForwardScaling"/> and <see cref="Hmmc.BackwardScaling"/> with <see cref="FitStartProb"/> and
    /// <see cref="FitTransMat"/>, then <see cref="ComputePosteriorsScaling(double[,], double[,])"/>. Returns the log probability,
    /// the lattice of probabilities, the posteriors and the forward and backward lattices.
    /// </summary>
    private (double LogProb, double[,] Lattice, double[,] Posteriors, double[,] FwdLattice, double[,] BwdLattice) FitSequenceScaling(double[,] sequence)
    {
        var frame = ComputeFitLikelihood(sequence);
        var (logProb, fwdlattice, scaling) = Hmmc.ForwardScaling(FitStartProb, FitTransMat, frame);
        var bwdlattice = Hmmc.BackwardScaling(FitStartProb, FitTransMat, frame, scaling);
        return (logProb, frame, ComputePosteriorsScaling(fwdlattice, bwdlattice), fwdlattice, bwdlattice);
    }

    /// <summary>
    /// Throws <see cref="ArgumentException"/> when <paramref name="lengths"/> has a negative element or does not sum to the number
    /// of rows of <paramref name="x"/> (the same messages as <see cref="InternalUtils.SplitXLengths(double[,], int[])"/>).
    /// </summary>
    private static void ValidateLengths(double[,] x, int[] lengths)
    {
        var total = 0;
        for (var i = 0; i < lengths.Length; i++)
        {
            if (lengths[i] < 0)
                throw new ArgumentException($"lengths[{i}] is {lengths[i]}, lengths must be >= 0", nameof(lengths));
            total += lengths[i];
        }
        if (total != x.GetLength(0))
            throw new ArgumentException($"lengths {string.Join(", ", lengths)} sum to {total}, but x has {x.GetLength(0)} rows", nameof(lengths));
    }

    /// <summary>True when <paramref name="matrix"/> has a row whose sum is exactly 0 (null: false).</summary>
    private static bool HasZeroRow(double[,]? matrix)
    {
        if (matrix == null)
            return false;
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        for (var i = 0; i < rows; i++)
        {
            var sum = 0.0;
            for (var j = 0; j < cols; j++)
                sum += matrix[i, j];
            if (sum == 0.0)
                return true;
        }
        return false;
    }

    /// <summary>Element-wise exponential of <paramref name="matrix"/> (the exponential of -infinity is 0), in a new matrix.</summary>
    private static double[,] ExpMatrix(double[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var result = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                result[i, j] = Math.Exp(matrix[i, j]);
        return result;
    }

    /// <summary>Initializes the parameters before fitting (hmmlearn's _init); models override it. The default throws NotImplementedException("Must be overridden in subclass").</summary>
    protected virtual void Init(double[,] x, int[] lengths) => throw new NotImplementedException("Must be overridden in subclass");

    /// <summary>Called at the start of each E-step, before any sequence (the variational models refresh their subnormalized parameters there). Does nothing by default.</summary>
    protected virtual void EstepBegin()
    {
    }

    /// <summary>Creates the statistics of one E-step: <c>new SufficientStatistics(NComponents)</c> by default; models return a derived object.</summary>
    protected virtual SufficientStatistics InitializeSufficientStatistics() => new(NComponents);

    /// <summary>
    /// Adds one sequence to the statistics. <paramref name="lattice"/> holds the probabilities (Implementation.Scaling) or the log
    /// probabilities (Implementation.Log) of each sample under each state; <paramref name="posteriors"/> the state posteriors;
    /// <paramref name="fwdlattice"/> and <paramref name="bwdlattice"/> the forward and backward lattices. The base version does:
    /// NObs += 1; if Params contains 's', Start += posteriors of the first sample; if Params contains 't' and the sequence has more
    /// than one sample, Trans += the xi sum (<see cref="Hmmc.ComputeScalingXiSum"/> for Scaling, the exponential of
    /// <see cref="Hmmc.ComputeLogXiSum"/> for Log) computed with <see cref="FitTransMat"/>. Models call it first, then add their own
    /// statistics.
    /// </summary>
    protected virtual void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice)
    {
        stats.NObs++;
        if (Params.Contains('s'))
        {
            var cols = posteriors.GetLength(1);
            for (var c = 0; c < cols; c++)
                stats.Start[c] += posteriors[0, c];
        }
        if (Params.Contains('t') && lattice.GetLength(0) > 1)
        {
            var xiSum = Implementation == Implementation.Scaling
                ? Hmmc.ComputeScalingXiSum(fwdlattice, FitTransMat, bwdlattice, lattice)
                : ExpMatrix(Hmmc.ComputeLogXiSum(fwdlattice, FitTransMat, bwdlattice, lattice));
            var rows = xiSum.GetLength(0);
            var cols = xiSum.GetLength(1);
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++)
                    stats.Trans[i, j] += xiSum[i, j];
        }
    }

    /// <summary>Updates the parameters from the statistics (M-step). Does nothing by default.</summary>
    protected virtual void DoMStep(SufficientStatistics stats)
    {
    }

    /// <summary>The quantity monitored for convergence, from the log probability of the data of the E-step. The default throws NotImplementedException("Must be overridden in subclass").</summary>
    protected virtual double ComputeLowerBound(double currLogProb) => throw new NotImplementedException("Must be overridden in subclass");

    /// <summary>Start probabilities used by the E-step passes (the variational models use subnormalized ones). <see cref="StartProb"/> by default.</summary>
    protected virtual double[] FitStartProb => StartProb!;

    /// <summary>Transition matrix used by the E-step passes and the xi sums. <see cref="TransMat"/> by default.</summary>
    protected virtual double[,] FitTransMat => TransMat!;

    /// <summary>Log likelihood used by the E-step log passes. <see cref="ComputeLogLikelihood"/> by default.</summary>
    protected virtual double[,] ComputeFitLogLikelihood(double[,] x) => ComputeLogLikelihood(x);

    /// <summary>Likelihood used by the E-step scaling passes. <see cref="ComputeLikelihood"/> by default.</summary>
    protected virtual double[,] ComputeFitLikelihood(double[,] x) => ComputeLikelihood(x);

    /// <summary>
    /// Number of free scalar parameters per parameter letter ('s', 't' and the model's own letters), used to warn about degenerate fits and by
    /// the information criteria. Null (the default) means unknown.
    /// </summary>
    protected virtual IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam() => null;

    /// <summary>
    /// Whether the parameter of letter <paramref name="code"/> must be initialized: true if <see cref="InitParams"/> contains the
    /// letter (and, when <paramref name="isSet"/> is already true, a warning is raised: "Even though the '{name}' attribute is set,
    /// it will be overwritten during initialization because 'init_params' contains '{code}'"), else true only if it is not set yet.
    /// </summary>
    protected bool NeedsInit(char code, string name, bool isSet)
    {
        if (InitParams.Contains(code))
        {
            if (isSet)
                RaiseWarning($"Even though the '{name}' attribute is set, it will be overwritten during initialization because 'init_params' contains '{code}'");
            return true;
        }
        return !isSet;
    }

    /// <summary>
    /// Records the number of features of x (its number of columns) in <see cref="NFeatures"/>; if it is already known and differs,
    /// throws <see cref="ArgumentException"/>("Unexpected number of dimensions, got {got} but expected {expected}"). Models override it.
    /// </summary>
    protected virtual void CheckAndSetNFeatures(double[,] x)
    {
        var nFeatures = x.GetLength(1);
        if (NFeatures.HasValue && NFeatures.Value != nFeatures)
            throw new ArgumentException($"Unexpected number of dimensions, got {nFeatures} but expected {NFeatures.Value}");
        NFeatures = nFeatures;
    }

    /// <summary>Throws <see cref="ArgumentException"/> unless the values sum to 1 (|sum - 1| &lt;= 1e-8 + 1e-5); the message is "{name} must sum to 1 (got {sum:F4})", formatted in the invariant culture.</summary>
    protected static void CheckSum1(string name, double[] values)
    {
        var sum = 0.0;
        for (var i = 0; i < values.Length; i++)
            sum += values[i];
        if (Math.Abs(sum - 1.0) > Sum1Tolerance)
            throw new ArgumentException($"{name} must sum to 1 (got {sum.ToString("F4", CultureInfo.InvariantCulture)})");
    }

    /// <summary>Throws <see cref="ArgumentException"/> unless every row sums to 1 (same tolerance); the message is "{name} rows must sum to 1 (got row sums of [...])", the row sums in the invariant culture.</summary>
    protected static void CheckSum1(string name, double[,] values)
    {
        var rows = values.GetLength(0);
        var cols = values.GetLength(1);
        var sums = new double[rows];
        for (var i = 0; i < rows; i++)
        {
            var sum = 0.0;
            for (var j = 0; j < cols; j++)
                sum += values[i, j];
            sums[i] = sum;
        }
        for (var i = 0; i < rows; i++)
        {
            if (Math.Abs(sums[i] - 1.0) > Sum1Tolerance)
            {
                var rowSums = string.Join(", ", sums.Select(s => s.ToString(CultureInfo.InvariantCulture)));
                throw new ArgumentException($"{name} rows must sum to 1 (got row sums of [{rowSums}])");
            }
        }
    }
}
