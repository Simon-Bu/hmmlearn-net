using MathNet.Numerics;
using MathNet.Numerics.Distributions;

namespace HmmLearn;

/// <summary>
/// Base class of the hidden Markov models learned by Variational Bayes (port of VariationalBaseHMM of hmmlearn/base.py): the start
/// probabilities and each row of the transition matrix have a Dirichlet prior and a Dirichlet posterior. The E-step runs the
/// forward-backward passes with subnormalized parameters (exp of digamma differences) and subnormalized emission likelihoods; the
/// M-step adds the statistics to the priors; the monitored quantity is the variational lower bound. <see cref="AbstractHmm.StartProb"/>
/// and <see cref="AbstractHmm.TransMat"/> hold the normalized posteriors, for scoring and decoding. Models add their emission
/// posteriors on top.
/// </summary>
public abstract class VariationalBaseHmm : AbstractHmm
{
    /// <summary>
    /// Creates the model. <paramref name="startProbPrior"/> and <paramref name="transMatPrior"/> are the values of every parameter of
    /// the symmetric Dirichlet priors of the start probabilities and of each transition row; null means 1 / NComponents.
    /// </summary>
    protected VariationalBaseHmm(int nComponents, double? startProbPrior, double? transMatPrior, DecoderAlgorithm algorithm,
        int? randomState, int nIter, double tol, bool verbose, string @params, string initParams, Implementation implementation)
        : base(nComponents, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
    {
        StartProbPrior = startProbPrior;
        TransMatPrior = transMatPrior;
    }

    /// <summary>Value of every parameter of the Dirichlet prior of the start probabilities used by <see cref="Init"/>; null means 1 / NComponents.</summary>
    public double? StartProbPrior { get; set; }

    /// <summary>Value of every parameter of the Dirichlet prior of each transition row used by <see cref="Init"/>; null means 1 / NComponents.</summary>
    public double? TransMatPrior { get; set; }

    /// <summary>Parameters of the Dirichlet prior of the start probabilities, length NComponents (hmmlearn's startprob_prior_); null until initialized or set.</summary>
    public double[]? StartProbPriorValues { get; set; }

    /// <summary>Parameters of the Dirichlet posterior of the start probabilities, length NComponents (hmmlearn's startprob_posterior_); null until initialized or set.</summary>
    public double[]? StartProbPosterior { get; set; }

    /// <summary>Parameters of the Dirichlet prior of each transition row, shape (NComponents, NComponents) (hmmlearn's transmat_prior_); null until initialized or set.</summary>
    public double[,]? TransMatPriorValues { get; set; }

    /// <summary>Parameters of the Dirichlet posterior of each transition row, shape (NComponents, NComponents) (hmmlearn's transmat_posterior_); null until initialized or set.</summary>
    public double[,]? TransMatPosterior { get; set; }

    /// <summary>Subnormalized start probabilities of the current E-step: exp(digamma(posterior) - digamma(sum of the posterior)); null before the first E-step.</summary>
    protected double[]? StartProbSubnorm { get; set; }

    /// <summary>Subnormalized transition matrix of the current E-step: row by row exp(digamma(posterior) - digamma(sum of the row)); null before the first E-step.</summary>
    protected double[,]? TransMatSubnorm { get; set; }

    /// <summary>
    /// Log of the subnormalized likelihood of each sample under each state, shape (n_samples, n_components): the emission term of the
    /// E-step, computed from the emission posteriors (expected log likelihood).
    /// </summary>
    protected abstract double[,] ComputeSubnormLogLikelihood(double[,] x);

    /// <summary>
    /// Initializes before fitting (hmmlearn's VariationalBaseHMM._init): <c>CheckAndSetNFeatures(x)</c>; then, if
    /// <c>NeedsInit('s', "startprob_posterior_", StartProbPosterior != null)</c> or (only evaluated when the first is false)
    /// <c>NeedsInit('s', "startprob_prior_", StartProbPriorValues != null)</c>: <see cref="StartProbPriorValues"/> = NComponents copies of
    /// StartProbPrior (1 / NComponents when null) and <see cref="StartProbPosterior"/> = a draw of the Dirichlet distribution with every
    /// parameter 1 / NComponents, multiplied by the number of sequences (lengths.Length); likewise for 't' with
    /// <see cref="TransMatPriorValues"/> (NComponents x NComponents copies of TransMatPrior, 1 / NComponents when null) and
    /// <see cref="TransMatPosterior"/> = one such draw per row, multiplied by (sum of lengths) / NComponents. The random numbers come
    /// from a <see cref="Random"/> seeded with <see cref="AbstractHmm.RandomState"/> when it has a value (a new generator at each
    /// call), else from <see cref="Random.Shared"/>. Then the same degenerate-fit warning as BaseHmm when
    /// <see cref="AbstractHmm.GetNFitScalarsPerParam"/> is not null and x has fewer values than the free scalars of the letters of Params.
    /// </summary>
    protected override void Init(double[,] x, int[] lengths)
    {
        CheckAndSetNFeatures(x);
        var nc = NComponents;
        var uniformPrior = 1.0 / nc;
        var random = RandomState.HasValue ? new Random(RandomState.Value) : Random.Shared;
        if (NeedsInit('s', "startprob_posterior_", StartProbPosterior != null)
            || NeedsInit('s', "startprob_prior_", StartProbPriorValues != null))
        {
            var startProbInit = StartProbPrior ?? uniformPrior;
            var startProbPriorValues = new double[nc];
            Array.Fill(startProbPriorValues, startProbInit);
            StartProbPriorValues = startProbPriorValues;

            var alpha = new double[nc];
            Array.Fill(alpha, uniformPrior);
            var scale = lengths.Length;
            var draw = Dirichlet.Sample(random, alpha);
            var startProbPosterior = new double[nc];
            for (var i = 0; i < nc; i++)
                startProbPosterior[i] = draw[i] * scale;
            StartProbPosterior = startProbPosterior;
        }
        if (NeedsInit('t', "transmat_posterior_", TransMatPosterior != null)
            || NeedsInit('t', "transmat_prior_", TransMatPriorValues != null))
        {
            var transMatInit = TransMatPrior ?? uniformPrior;
            var transMatPriorValues = new double[nc, nc];
            for (var i = 0; i < nc; i++)
                for (var j = 0; j < nc; j++)
                    transMatPriorValues[i, j] = transMatInit;
            TransMatPriorValues = transMatPriorValues;

            var alpha = new double[nc];
            Array.Fill(alpha, uniformPrior);
            var totalLength = 0;
            for (var i = 0; i < lengths.Length; i++)
                totalLength += lengths[i];
            var scale = (double)totalLength / nc;
            var transMatPosterior = new double[nc, nc];
            for (var i = 0; i < nc; i++)
            {
                var draw = Dirichlet.Sample(random, alpha);
                for (var j = 0; j < nc; j++)
                    transMatPosterior[i, j] = draw[j] * scale;
            }
            TransMatPosterior = transMatPosterior;
        }

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
    /// Validates the priors and posteriors; throws <see cref="ArgumentException"/> if <see cref="StartProbPriorValues"/> or
    /// <see cref="StartProbPosterior"/> is null or does not have length NComponents (messages "startprob_prior_ must have length
    /// n_components", "startprob_posterior_ must have length n_components"), or if <see cref="TransMatPriorValues"/> or
    /// <see cref="TransMatPosterior"/> is null or not (NComponents x NComponents) ("transmat_prior_ must have shape (n_components,
    /// n_components)", "transmat_posterior_ must have shape (n_components, n_components)"). The values are not otherwise checked.
    /// </summary>
    protected override void Check()
    {
        var nc = NComponents;
        if (StartProbPriorValues == null || StartProbPriorValues.Length != nc)
            throw new ArgumentException("startprob_prior_ must have length n_components");
        if (StartProbPosterior == null || StartProbPosterior.Length != nc)
            throw new ArgumentException("startprob_posterior_ must have length n_components");
        if (TransMatPriorValues == null || TransMatPriorValues.GetLength(0) != nc || TransMatPriorValues.GetLength(1) != nc)
            throw new ArgumentException("transmat_prior_ must have shape (n_components, n_components)");
        if (TransMatPosterior == null || TransMatPosterior.GetLength(0) != nc || TransMatPosterior.GetLength(1) != nc)
            throw new ArgumentException("transmat_posterior_ must have shape (n_components, n_components)");
    }

    /// <summary>Start of every E-step: refreshes <see cref="StartProbSubnorm"/> and <see cref="TransMatSubnorm"/> from the posteriors with the digamma function.</summary>
    protected override void EstepBegin()
    {
        var posterior = StartProbPosterior!;
        var sum = 0.0;
        for (var i = 0; i < posterior.Length; i++)
            sum += posterior[i];
        var digammaSum = SpecialFunctions.DiGamma(sum);
        StartProbSubnorm = new double[posterior.Length];
        for (var i = 0; i < posterior.Length; i++)
            StartProbSubnorm[i] = Math.Exp(SpecialFunctions.DiGamma(posterior[i]) - digammaSum);

        var transPosterior = TransMatPosterior!;
        var rows = transPosterior.GetLength(0);
        var cols = transPosterior.GetLength(1);
        TransMatSubnorm = new double[rows, cols];
        for (var i = 0; i < rows; i++)
        {
            var rowSum = 0.0;
            for (var j = 0; j < cols; j++)
                rowSum += transPosterior[i, j];
            var digammaRowSum = SpecialFunctions.DiGamma(rowSum);
            for (var j = 0; j < cols; j++)
                TransMatSubnorm[i, j] = Math.Exp(SpecialFunctions.DiGamma(transPosterior[i, j]) - digammaRowSum);
        }
    }

    /// <summary><see cref="StartProbSubnorm"/>: the E-step passes use the subnormalized start probabilities.</summary>
    protected override double[] FitStartProb => StartProbSubnorm!;

    /// <summary><see cref="TransMatSubnorm"/>: the E-step passes and the xi sums use the subnormalized transition matrix.</summary>
    protected override double[,] FitTransMat => TransMatSubnorm!;

    /// <summary><see cref="ComputeSubnormLogLikelihood"/>: the E-step log passes use the subnormalized emission likelihoods.</summary>
    protected override double[,] ComputeFitLogLikelihood(double[,] x) => ComputeSubnormLogLikelihood(x);

    /// <summary>The exponential of <see cref="ComputeSubnormLogLikelihood"/>: the E-step scaling passes use the subnormalized emission likelihoods.</summary>
    protected override double[,] ComputeFitLikelihood(double[,] x)
    {
        var logLikelihood = ComputeSubnormLogLikelihood(x);
        var rows = logLikelihood.GetLength(0);
        var cols = logLikelihood.GetLength(1);
        var likelihood = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                likelihood[i, j] = Math.Exp(logLikelihood[i, j]);
        return likelihood;
    }

    /// <summary>
    /// M-step of the start probabilities and transition matrix. If Params contains 's': <see cref="StartProbPosterior"/> = prior values +
    /// stats.Start and <see cref="AbstractHmm.StartProb"/> = the posterior divided by its sum. If Params contains 't':
    /// <see cref="TransMatPosterior"/> = prior values + stats.Trans and <see cref="AbstractHmm.TransMat"/> = each row of the posterior
    /// divided by its sum. New arrays replace the old ones.
    /// </summary>
    protected override void DoMStep(SufficientStatistics stats)
    {
        if (Params.Contains('s'))
        {
            var prior = StartProbPriorValues!;
            var posterior = new double[prior.Length];
            for (var i = 0; i < prior.Length; i++)
                posterior[i] = prior[i] + stats.Start[i];
            StartProbPosterior = posterior;

            var sum = 0.0;
            for (var i = 0; i < posterior.Length; i++)
                sum += posterior[i];
            var startProb = new double[prior.Length];
            for (var i = 0; i < prior.Length; i++)
                startProb[i] = posterior[i] / sum;
            StartProb = startProb;
        }
        if (Params.Contains('t'))
        {
            var prior = TransMatPriorValues!;
            var rows = prior.GetLength(0);
            var cols = prior.GetLength(1);
            var posterior = new double[rows, cols];
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++)
                    posterior[i, j] = prior[i, j] + stats.Trans[i, j];
            TransMatPosterior = posterior;

            var transMat = new double[rows, cols];
            for (var i = 0; i < rows; i++)
            {
                var rowSum = 0.0;
                for (var j = 0; j < cols; j++)
                    rowSum += posterior[i, j];
                for (var j = 0; j < cols; j++)
                    transMat[i, j] = posterior[i, j] / rowSum;
            }
            TransMat = transMat;
        }
    }

    /// <summary>
    /// The variational lower bound contribution of the start probabilities, the transitions and the data: the log probability of the data
    /// computed with the subnormalized parameters, minus <see cref="KlDivergence.KlDirichlet"/> of the start posterior from its prior, minus the
    /// sum over the states of the same divergence for each transition row.
    /// </summary>
    protected override double ComputeLowerBound(double currLogProb)
    {
        var bound = currLogProb - KlDivergence.KlDirichlet(StartProbPosterior!, StartProbPriorValues!);
        var transPosterior = TransMatPosterior!;
        var transPrior = TransMatPriorValues!;
        var rows = transPosterior.GetLength(0);
        var cols = transPosterior.GetLength(1);
        var posteriorRow = new double[cols];
        var priorRow = new double[cols];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                posteriorRow[j] = transPosterior[i, j];
                priorRow[j] = transPrior[i, j];
            }
            bound -= KlDivergence.KlDirichlet(posteriorRow, priorRow);
        }
        return bound;
    }
}
