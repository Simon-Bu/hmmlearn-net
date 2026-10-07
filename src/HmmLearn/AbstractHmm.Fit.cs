namespace HmmLearn;

/// <summary>
/// The fitting half of AbstractHmm (port of _AbstractHMM.fit and its helpers in hmmlearn/base.py): the EM loop, the E-step, the
/// base sufficient statistics, and the helpers models use when they initialize. Every member added here is virtual or non-abstract
/// (the classes that derive from AbstractHmm in earlier acceptance tests must keep compiling).
/// </summary>
public abstract partial class AbstractHmm
{
    /// <summary>
    /// Estimates the parameters from the observations (several sequences described by <paramref name="lengths"/>; null: one
    /// sequence). Steps: <see cref="CheckObservations"/>; lengths default; <see cref="Init"/>; <see cref="Check"/>;
    /// <see cref="Monitor"/> receives <see cref="Tol"/>, <see cref="NIter"/> and <see cref="Verbose"/> and is reset; then at most
    /// <see cref="NIter"/> times: E-step, <see cref="ComputeLowerBound"/> of the log probability, <see cref="DoMStep"/>,
    /// <c>Monitor.Report(lowerBound)</c>, stop if <c>Monitor.Converged</c>, otherwise raise a warning (message "Some rows of
    /// transmat_ have zero sum because no transition from the state was ever observed.") when a row of <see cref="TransMat"/> sums to 0.
    /// The warnings of the monitor are forwarded to <see cref="RaiseWarning"/> while fitting. The M-step runs even on the iteration
    /// that converges. Returns this instance. NIter = 0 only initializes and checks.
    /// </summary>
    public AbstractHmm Fit(double[,] x, int[]? lengths = null) => throw new NotImplementedException();

    /// <summary>Initializes the parameters before fitting (hmmlearn's _init); models override it. The default throws NotImplementedException("Must be overridden in subclass").</summary>
    protected virtual void Init(double[,] x, int[] lengths) => throw new NotImplementedException("Must be overridden in subclass");

    /// <summary>Called at the start of each E-step, before any sequence (the variational models refresh their subnormalized parameters there). Does nothing by default.</summary>
    protected virtual void EstepBegin() => throw new NotImplementedException();

    /// <summary>Creates the statistics of one E-step: <c>new SufficientStatistics(NComponents)</c> by default; models return a derived object.</summary>
    protected virtual SufficientStatistics InitializeSufficientStatistics() => throw new NotImplementedException();

    /// <summary>
    /// Adds one sequence to the statistics. <paramref name="lattice"/> holds the probabilities (Implementation.Scaling) or the log
    /// probabilities (Implementation.Log) of each sample under each state; <paramref name="posteriors"/> the state posteriors;
    /// <paramref name="fwdlattice"/> and <paramref name="bwdlattice"/> the forward and backward lattices. The base version does:
    /// NObs += 1; if Params contains 's', Start += posteriors of the first sample; if Params contains 't' and the sequence has more
    /// than one sample, Trans += the xi sum (<see cref="Hmmc.ComputeScalingXiSum"/>, or exp of <see cref="Hmmc.ComputeLogXiSum"/>)
    /// computed with <see cref="FitTransMat"/>. Models call it first, then add their own statistics.
    /// </summary>
    protected virtual void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice) => throw new NotImplementedException();

    /// <summary>Updates the parameters from the statistics (M-step). Does nothing by default.</summary>
    protected virtual void DoMStep(SufficientStatistics stats) => throw new NotImplementedException();

    /// <summary>The quantity monitored for convergence, from the log probability of the data of the E-step. The default throws NotImplementedException("Must be overridden in subclass").</summary>
    protected virtual double ComputeLowerBound(double currLogProb) => throw new NotImplementedException("Must be overridden in subclass");

    /// <summary>Start probabilities used by the E-step passes (the variational models use subnormalized ones). <see cref="StartProb"/> by default.</summary>
    protected virtual double[] FitStartProb => throw new NotImplementedException();

    /// <summary>Transition matrix used by the E-step passes and the xi sums. <see cref="TransMat"/> by default.</summary>
    protected virtual double[,] FitTransMat => throw new NotImplementedException();

    /// <summary>Log likelihood used by the E-step log passes. <see cref="ComputeLogLikelihood"/> by default.</summary>
    protected virtual double[,] ComputeFitLogLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>Likelihood used by the E-step scaling passes. <see cref="ComputeLikelihood"/> by default.</summary>
    protected virtual double[,] ComputeFitLikelihood(double[,] x) => throw new NotImplementedException();

    /// <summary>
    /// Number of free scalar parameters per parameter letter ('s', 't' and the model's own letters), used to warn about degenerate fits and by
    /// the information criteria. Null (the default) means unknown.
    /// </summary>
    protected virtual IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam() => throw new NotImplementedException();

    /// <summary>
    /// Whether the parameter of letter <paramref name="code"/> must be initialized: true if <see cref="InitParams"/> contains the
    /// letter (and, when <paramref name="isSet"/> is already true, a warning is raised: "Even though the '{name}' attribute is set,
    /// it will be overwritten during initialization because 'init_params' contains '{code}'"), else true only if it is not set yet.
    /// </summary>
    protected bool NeedsInit(char code, string name, bool isSet) => throw new NotImplementedException();

    /// <summary>
    /// Records the number of features of x (its number of columns) in <see cref="NFeatures"/>; if it is already known and differs,
    /// throws <see cref="ArgumentException"/>("Unexpected number of dimensions, got {got} but expected {expected}"). Models override it.
    /// </summary>
    protected virtual void CheckAndSetNFeatures(double[,] x) => throw new NotImplementedException();

    /// <summary>Throws <see cref="ArgumentException"/> unless the values sum to 1 (|sum - 1| &lt;= 1e-8 + 1e-5); the message is "{name} must sum to 1 (got {sum:F4})".</summary>
    protected static void CheckSum1(string name, double[] values) => throw new NotImplementedException();

    /// <summary>Throws <see cref="ArgumentException"/> unless every row sums to 1 (same tolerance); the message is "{name} rows must sum to 1 (got row sums of [...])".</summary>
    protected static void CheckSum1(string name, double[,] values) => throw new NotImplementedException();
}
