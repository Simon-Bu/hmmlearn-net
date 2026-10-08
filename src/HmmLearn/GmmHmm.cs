using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;

namespace HmmLearn;

/// <summary>
/// The sufficient statistics of a mixture model (hmmlearn's 'post_mix_sum', 'post_sum', 'm_n' and 'c_n'): the base ones plus the
/// posterior mass of each mixture component and the weighted sums the M-step needs. <see cref="MN"/> already holds the prior part
/// (MeansWeight * MeansPrior) when the statistics are created.
/// </summary>
public class GmmStatistics : SufficientStatistics
{
    /// <summary>
    /// Allocates the base statistics, <see cref="PostMixSum"/> (nc x nm), <see cref="PostSum"/> (nc), <see cref="MN"/> (nc x nm x nf,
    /// every entry meansWeight * meansPrior) and <see cref="CN"/> (zeros of the compact rank-4 shape of the covariance type, given by
    /// <paramref name="covarsShape"/> = its four dimensions), all other entries zero.
    /// </summary>
    public GmmStatistics(int nComponents, int nMix, int nFeatures, double meansWeight, double meansPrior, int[] covarsShape)
        : base(nComponents)
    {
        PostMixSum = new double[nComponents, nMix];
        PostSum = new double[nComponents];
        var prior = meansWeight * meansPrior;
        MN = new double[nComponents, nMix, nFeatures];
        for (var c = 0; c < nComponents; c++)
            for (var m = 0; m < nMix; m++)
                for (var f = 0; f < nFeatures; f++)
                    MN[c, m, f] = prior;
        CN = new double[covarsShape[0], covarsShape[1], covarsShape[2], covarsShape[3]];
    }

    /// <summary>PostMixSum[c, m]: sum over the samples of the posterior of state c times the posterior of mixture m within c.</summary>
    public double[,] PostMixSum { get; set; } = null!;

    /// <summary>PostSum[c]: sum over the samples of the posterior of state c.</summary>
    public double[] PostSum { get; set; } = null!;

    /// <summary>MN[c, m, f]: meansWeight * meansPrior + sum over the samples of the joint posterior times x[f].</summary>
    public double[,,] MN { get; set; } = null!;

    /// <summary>
    /// CN: sum over the samples of the joint posterior times the centered second moment, in the compact rank-4 shapes: spherical
    /// (nc, nm, 1, 1) with the squared norm of the centered sample, diag (nc, nm, 1, nf) with its squared coordinates, full (nc, nm, nf, nf)
    /// with its outer product, tied (nc, 1, nf, nf) with the outer product summed over the mixture components. Centered means x minus
    /// the mean of the mixture component.
    /// </summary>
    public double[,,,] CN { get; set; } = null!;
}

/// <summary>
/// Hidden Markov model whose emissions are Gaussian mixtures (port of GMMHMM of hmmlearn/hmm.py and BaseGMMHMM of
/// hmmlearn/_emissions.py). This file holds the parameters, the checks, the likelihood, the sampling and the statistics; the
/// initialization and the M-step are in GmmHmm.Learning.cs (same partial class). Covariances are stored in the compact rank-4 uniform
/// shapes of oracle/README.md: spherical (nc, nm, 1, 1), diag (nc, nm, 1, nf), full (nc, nm, nf, nf), tied (nc, 1, nf, nf)
/// (tied: one matrix per state, shared by its mixture components).
/// </summary>
public partial class GmmHmm : BaseHmm
{
    /// <summary>Creates the model; the defaults are those of hmmlearn (diag covariances, parameter letters "stmcw").</summary>
    public GmmHmm(int nComponents = 1, int nMix = 1, double minCovar = 1e-3, double startProbPrior = 1.0, double transMatPrior = 1.0,
        double weightsPrior = 1.0, double meansPrior = 0.0, double meansWeight = 0.0, double? covarsPrior = null,
        double? covarsWeight = null, DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi,
        CovarianceType covarianceType = CovarianceType.Diag, int? randomState = null, int nIter = 10, double tol = 1e-2,
        bool verbose = false, string @params = "stmcw", string initParams = "stmcw", Implementation implementation = Implementation.Log)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
    {
        NMix = nMix;
        MinCovar = minCovar;
        WeightsPrior = weightsPrior;
        MeansPrior = meansPrior;
        MeansWeight = meansWeight;
        CovarsPrior = covarsPrior;
        CovarsWeight = covarsWeight;
        CovarianceType = covarianceType;
    }

    /// <summary>Number of mixture components of each state.</summary>
    public int NMix { get; set; }

    /// <summary>The structure of the covariance parameters (spherical, diag, full, tied).</summary>
    public CovarianceType CovarianceType { get; set; }

    /// <summary>
    /// Floor on the diagonal of the covariance matrices (added on the diagonal when a Cholesky factorization fails in the sampling).
    /// </summary>
    public double MinCovar { get; set; }

    /// <summary>Parameter of the symmetric Dirichlet prior of each row of <see cref="Weights"/>.</summary>
    public double WeightsPrior { get; set; }

    /// <summary>Mean of the Gaussian prior on the means.</summary>
    public double MeansPrior { get; set; }

    /// <summary>Precision (inverse variance) of the Gaussian prior on the means.</summary>
    public double MeansWeight { get; set; }

    /// <summary>
    /// Prior of the Wishart distribution of the covariance (Full and Tied: the scale matrix as a number added to every entry;
    /// Diag and Spherical: the alpha parameter). Null until set or fitted; the effective value is <see cref="EffectiveCovarsPrior"/>.
    /// </summary>
    public double? CovarsPrior { get; set; }

    /// <summary>
    /// The nu parameter of the Wishart prior of the covariance. Null until set or fitted; the effective value is
    /// <see cref="EffectiveCovarsWeight"/>.
    /// </summary>
    public double? CovarsWeight { get; set; }

    /// <summary>
    /// Mixture weights of each state, shape (n_components, n_mix); the rows sum to 1. Null until set or fitted.
    /// </summary>
    public double[,]? Weights { get; set; }

    /// <summary>
    /// Means of the mixture components of each state, shape (n_components, n_mix, n_features). Null until set or fitted.
    /// </summary>
    public double[,,]? Means { get; set; }

    /// <summary>
    /// Covariances in the compact rank-4 shapes: spherical (nc, nm, 1, 1), diag (nc, nm, 1, nf), full (nc, nm, nf, nf),
    /// tied (nc, 1, nf, nf). Null until set or fitted.
    /// </summary>
    public double[,,,]? Covars { get; set; }

    /// <summary>
    /// <see cref="CovarsPrior"/> when it is set, else the hmmlearn default of the covariance type: 0 (Full, Tied), -1.5 (Diag),
    /// -(n_mix + 2) / 2 (Spherical).
    /// </summary>
    public double EffectiveCovarsPrior => CovarsPrior ?? CovarianceType switch
    {
        CovarianceType.Diag => -1.5,
        CovarianceType.Spherical => -(NMix + 2.0) / 2.0,
        _ => 0.0,
    };

    /// <summary>
    /// <see cref="CovarsWeight"/> when it is set, else the hmmlearn default of the covariance type: 0 (Diag, Spherical),
    /// -(n_features + 2) (Full), -(n_mix + n_features + 1) (Tied); Full and Tied require <see cref="NFeatures"/> known.
    /// </summary>
    public double EffectiveCovarsWeight
    {
        get
        {
            if (CovarsWeight.HasValue)
                return CovarsWeight.Value;
            return CovarianceType switch
            {
                CovarianceType.Full => -(RequireNFeatures() + 2.0),
                CovarianceType.Tied => -(NMix + RequireNFeatures() + 1.0),
                _ => 0.0,
            };
        }
    }

    /// <summary>
    /// Checks the start probabilities and transition matrix (base) and the emission parameters:
    /// <list type="bullet">
    /// <item><see cref="NFeatures"/> is taken from <see cref="Means"/> if unknown; a mismatch with the shape of <see cref="Means"/> throws.</item>
    /// <item><see cref="Weights"/> (shape (n_components, n_mix)), <see cref="Means"/> (shape (n_components, n_mix, n_features)) and
    /// <see cref="Covars"/> must be present and shaped; the rows of the weights must sum to 1.</item>
    /// <item><see cref="Covars"/> must have the compact rank-4 shape of the covariance type: spherical (nc, nm, 1, 1), diag
    /// (nc, nm, 1, nf), full (nc, nm, nf, nf), tied (nc, 1, nf, nf).</item>
    /// <item>Spherical and diag variances must be non-negative (a negative one throws); a zero one raises the "Degenerate mixture
    /// covariance" warning.</item>
    /// <item>Each tied or full matrix must be symmetric (numpy allclose) with a smallest eigenvalue not negative: a negative one
    /// throws, a null one raises a warning.</item>
    /// </list>
    /// A missing required parameter throws <see cref="ArgumentException"/>.
    /// </summary>
    protected override void Check()
    {
        base.Check();
        var means = Means ?? throw new ArgumentException("means_ must be set");
        var covars = Covars ?? throw new ArgumentException("covars_ must be set");
        var weights = Weights ?? throw new ArgumentException("weights_ must be set");

        NFeatures ??= means.GetLength(2);
        var nc = NComponents;
        var nf = NFeatures!.Value;
        var nm = NMix;

        if (means.GetLength(0) != nc || means.GetLength(1) != nm || means.GetLength(2) != nf)
            throw new ArgumentException(
                $"means_ must have shape ({nc}, {nm}, {nf}), actual shape: ({means.GetLength(0)}, {means.GetLength(1)}, {means.GetLength(2)})");

        if (weights.GetLength(0) != nc || weights.GetLength(1) != nm)
            throw new ArgumentException(
                $"weights_ must have shape ({nc}, {nm}), actual shape: ({weights.GetLength(0)}, {weights.GetLength(1)})");
        CheckSum1("weights_", weights);

        var (d1, d2, d3) = ExpectedCovarsShape(nm, nf);
        if (covars.GetLength(0) != nc || covars.GetLength(1) != d1 || covars.GetLength(2) != d2 || covars.GetLength(3) != d3)
            throw new ArgumentException(
                $"'{CovarianceType}' mixture covars must have shape ({nc}, {d1}, {d2}, {d3}), actual shape: " +
                $"({covars.GetLength(0)}, {covars.GetLength(1)}, {covars.GetLength(2)}, {covars.GetLength(3)})");

        switch (CovarianceType)
        {
            case CovarianceType.Spherical:
            case CovarianceType.Diag:
                CheckNonNegativeVariances(covars);
                break;
            case CovarianceType.Tied:
                for (var c = 0; c < nc; c++)
                    CheckCovarianceMatrix(covars, c, 0, $"state #{c}", nf);
                break;
            case CovarianceType.Full:
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                        CheckCovarianceMatrix(covars, c, m, $"state #{c}, mixture #{m}", nf);
                break;
        }
    }

    /// <summary>
    /// The log likelihood of each sample under each state: for a state, the log-sum-exp over its mixture components of
    /// log(weight) + log Gaussian density (hmmlearn's _compute_log_likelihood).
    /// </summary>
    protected override double[,] ComputeLogLikelihood(double[,] x)
    {
        var n = x.GetLength(0);
        var logprobs = new double[n, NComponents];
        for (var c = 0; c < NComponents; c++)
        {
            var logDenses = ComputeLogWeightedGaussianDensities(x, c);
            var nm = logDenses.GetLength(1);
            var row = new double[nm];
            for (var i = 0; i < n; i++)
            {
                for (var m = 0; m < nm; m++)
                    row[m] = logDenses[i, m];
                logprobs[i, c] = Hmmc.LogSumExp(row);
            }
        }
        return logprobs;
    }

    /// <summary>
    /// Allocates a <see cref="GmmStatistics"/>: the base statistics, zero <see cref="GmmStatistics.PostMixSum"/> and
    /// <see cref="GmmStatistics.PostSum"/>, <see cref="GmmStatistics.MN"/> at MeansWeight * MeansPrior in every entry and zero
    /// <see cref="GmmStatistics.CN"/> of the compact shape of the covariance type.
    /// </summary>
    protected override SufficientStatistics InitializeSufficientStatistics()
    {
        var nf = NFeatures!.Value;
        var (d1, d2, d3) = ExpectedCovarsShape(NMix, nf);
        return new GmmStatistics(NComponents, NMix, nf, MeansWeight, MeansPrior, new[] { NComponents, d1, d2, d3 });
    }

    /// <summary>
    /// Adds one sequence to the statistics (hmmlearn's _accumulate_sufficient_statistics): the base statistics; then, with the
    /// posterior of each mixture component within its state (the log weighted densities log-normalized over the components),
    /// PostSum += the column sums of the posteriors, PostMixSum += the sums of the joint posteriors, and, when 'm' is in Params,
    /// MN += the joint posterior weighted sums of x, and, when 'c' is in Params, CN += the joint posterior weighted sums of the
    /// centered second moment (per the covariance type).
    /// </summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice)
    {
        base.AccumulateSufficientStatistics(stats, x, lattice, posteriors, fwdlattice, bwdlattice);
        var gmm = (GmmStatistics)stats;
        var n = x.GetLength(0);
        var nc = NComponents;
        var nm = NMix;
        var nf = x.GetLength(1);

        // post_mix[i, c, m]: the posterior of mixture m within state c for sample i.
        var postMix = new double[n, nc, nm];
        for (var c = 0; c < nc; c++)
        {
            var logDenses = ComputeLogWeightedGaussianDensities(x, c);
            Utils.LogNormalize(logDenses, 1);
            for (var i = 0; i < n; i++)
                for (var m = 0; m < nm; m++)
                    postMix[i, c, m] = Math.Exp(logDenses[i, m]);
        }

        for (var i = 0; i < n; i++)
            for (var c = 0; c < nc; c++)
            {
                gmm.PostSum[c] += posteriors[i, c];
                for (var m = 0; m < nm; m++)
                    gmm.PostMixSum[c, m] += posteriors[i, c] * postMix[i, c, m];
            }

        if (Params.Contains('m'))
            for (var i = 0; i < n; i++)
                for (var c = 0; c < nc; c++)
                    for (var m = 0; m < nm; m++)
                    {
                        var weight = posteriors[i, c] * postMix[i, c, m];
                        if (weight == 0.0)
                            continue;
                        for (var f = 0; f < nf; f++)
                            gmm.MN[c, m, f] += weight * x[i, f];
                    }

        if (Params.Contains('c'))
            switch (CovarianceType)
            {
                case CovarianceType.Spherical:
                    for (var i = 0; i < n; i++)
                        for (var c = 0; c < nc; c++)
                            for (var m = 0; m < nm; m++)
                            {
                                var weight = posteriors[i, c] * postMix[i, c, m];
                                if (weight == 0.0)
                                    continue;
                                var norm2 = 0.0;
                                for (var f = 0; f < nf; f++)
                                {
                                    var d = x[i, f] - Means![c, m, f];
                                    norm2 += d * d;
                                }
                                gmm.CN[c, m, 0, 0] += weight * norm2;
                            }
                    break;
                case CovarianceType.Diag:
                    for (var i = 0; i < n; i++)
                        for (var c = 0; c < nc; c++)
                            for (var m = 0; m < nm; m++)
                            {
                                var weight = posteriors[i, c] * postMix[i, c, m];
                                if (weight == 0.0)
                                    continue;
                                for (var f = 0; f < nf; f++)
                                {
                                    var d = x[i, f] - Means![c, m, f];
                                    gmm.CN[c, m, 0, f] += weight * d * d;
                                }
                            }
                    break;
                case CovarianceType.Full:
                    for (var i = 0; i < n; i++)
                        for (var c = 0; c < nc; c++)
                            for (var m = 0; m < nm; m++)
                            {
                                var weight = posteriors[i, c] * postMix[i, c, m];
                                if (weight == 0.0)
                                    continue;
                                var d = new double[nf];
                                for (var f = 0; f < nf; f++)
                                    d[f] = x[i, f] - Means![c, m, f];
                                for (var j = 0; j < nf; j++)
                                    for (var k = 0; k < nf; k++)
                                        gmm.CN[c, m, j, k] += weight * d[j] * d[k];
                            }
                    break;
                case CovarianceType.Tied:
                    for (var i = 0; i < n; i++)
                        for (var c = 0; c < nc; c++)
                            for (var m = 0; m < nm; m++)
                            {
                                var weight = posteriors[i, c] * postMix[i, c, m];
                                if (weight == 0.0)
                                    continue;
                                for (var j = 0; j < nf; j++)
                                    for (var k = 0; k < nf; k++)
                                    {
                                        var dj = x[i, j] - Means![c, m, j];
                                        var dk = x[i, k] - Means![c, m, k];
                                        gmm.CN[c, 0, j, k] += weight * dj * dk;
                                    }
                            }
                    break;
            }
    }

    /// <summary>
    /// One observation: a mixture component drawn from the weights of the state (one uniform number, cumulative weights), then a draw
    /// from that component's Gaussian (<see cref="GaussianEmission.GenerateSample"/>; for tied, the state's matrix).
    /// </summary>
    protected override double[] GenerateSampleFromState(int state, Random random)
    {
        var nm = NMix;
        var nf = NFeatures!.Value;

        var u = random.NextDouble();
        var cumulative = 0.0;
        var iGauss = nm - 1;
        for (var m = 0; m < nm; m++)
        {
            cumulative += Weights![state, m];
            if (cumulative > u)
            {
                iGauss = m;
                break;
            }
        }

        var means = new double[1, nf];
        for (var f = 0; f < nf; f++)
            means[0, f] = Means![state, iGauss, f];
        var fullCovars = new double[1, nf, nf];
        switch (CovarianceType)
        {
            case CovarianceType.Spherical:
                for (var f = 0; f < nf; f++)
                    fullCovars[0, f, f] = Covars![state, iGauss, 0, 0];
                break;
            case CovarianceType.Diag:
                for (var f = 0; f < nf; f++)
                    fullCovars[0, f, f] = Covars![state, iGauss, 0, f];
                break;
            case CovarianceType.Full:
                for (var j = 0; j < nf; j++)
                    for (var k = 0; k < nf; k++)
                        fullCovars[0, j, k] = Covars![state, iGauss, j, k];
                break;
            default: // tied: the matrix of the state, shared by its mixture components
                for (var j = 0; j < nf; j++)
                    for (var k = 0; k < nf; k++)
                        fullCovars[0, j, k] = Covars![state, 0, j, k];
                break;
        }
        return GaussianEmission.GenerateSample(means, fullCovars, 0, random);
    }

    /// <summary>'s' = nc - 1, 't' = nc * (nc - 1), 'm' = nc * nm * nf, 'c' = nc * nm (spherical), nc * nm * nf (diag), nc * nm * nf * (nf + 1) / 2 (full), nc * nf * (nf + 1) / 2 (tied), 'w' = nm - 1.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam()
    {
        var nc = NComponents;
        var nm = NMix;
        var nf = NFeatures!.Value;
        return new Dictionary<char, int>
        {
            ['s'] = nc - 1,
            ['t'] = nc * (nc - 1),
            ['m'] = nc * nm * nf,
            ['c'] = CovarianceType switch
            {
                CovarianceType.Spherical => nc * nm,
                CovarianceType.Diag => nc * nm * nf,
                CovarianceType.Full => nc * nm * nf * (nf + 1) / 2,
                _ => nc * nf * (nf + 1) / 2,
            },
            ['w'] = nm - 1,
        };
    }

    /// <summary>The log weighted Gaussian densities of one state, (n_samples, n_mix): the log densities of its mixture components
    /// (<see cref="Stats.LogMultivariateNormalDensity"/>) plus the log of the mixture weights.</summary>
    private double[,] ComputeLogWeightedGaussianDensities(double[,] x, int state)
    {
        var n = x.GetLength(0);
        var nm = NMix;
        var nf = NFeatures!.Value;

        var means = new double[nm, nf];
        for (var m = 0; m < nm; m++)
            for (var f = 0; f < nf; f++)
                means[m, f] = Means![state, m, f];
        var densities = Stats.LogMultivariateNormalDensity(x, means, StateCovars(state), CovarianceType);
        var result = new double[n, nm];
        for (var i = 0; i < n; i++)
            for (var m = 0; m < nm; m++)
                result[i, m] = densities[i, m] + Math.Log(Weights![state, m]);
        return result;
    }

    /// <summary>The compact rank-3 covars of one state, as <see cref="Stats.LogMultivariateNormalDensity"/> expects them: spherical
    /// (nm, 1, 1), diag (nm, 1, nf), full (nm, nf, nf), tied (1, nf, nf).</summary>
    private double[,,] StateCovars(int state)
    {
        var nm = NMix;
        var nf = NFeatures!.Value;
        var covars = CovarianceType switch
        {
            CovarianceType.Spherical => new double[nm, 1, 1],
            CovarianceType.Diag => new double[nm, 1, nf],
            CovarianceType.Full => new double[nm, nf, nf],
            _ => new double[1, nf, nf],
        };
        switch (CovarianceType)
        {
            case CovarianceType.Spherical:
                for (var m = 0; m < nm; m++)
                    covars[m, 0, 0] = Covars![state, m, 0, 0];
                break;
            case CovarianceType.Diag:
                for (var m = 0; m < nm; m++)
                    for (var f = 0; f < nf; f++)
                        covars[m, 0, f] = Covars![state, m, 0, f];
                break;
            case CovarianceType.Full:
                for (var m = 0; m < nm; m++)
                    for (var j = 0; j < nf; j++)
                        for (var k = 0; k < nf; k++)
                            covars[m, j, k] = Covars![state, m, j, k];
                break;
            default: // tied
                for (var j = 0; j < nf; j++)
                    for (var k = 0; k < nf; k++)
                        covars[0, j, k] = Covars![state, 0, j, k];
                break;
        }
        return covars;
    }

    /// <summary>Raises <see cref="ArgumentException"/> when <see cref="NFeatures"/> is unknown.</summary>
    private int RequireNFeatures() =>
        NFeatures ?? throw new ArgumentException("the number of features is unknown; set NFeatures or Means first");

    /// <summary>The (d1, d2, d3) of the compact rank-4 covars shape for <paramref name="nm"/> components and <paramref name="nf"/>
    /// features: spherical (nm, 1, 1), diag (nm, 1, nf), full (nm, nf, nf), tied (1, nf, nf).</summary>
    private (int D1, int D2, int D3) ExpectedCovarsShape(int nm, int nf) => CovarianceType switch
    {
        CovarianceType.Spherical => (nm, 1, 1),
        CovarianceType.Diag => (nm, 1, nf),
        CovarianceType.Full => (nm, nf, nf),
        _ => (1, nf, nf),
    };

    /// <summary>Checks that every spherical or diag variance is non-negative; a negative one throws, a zero one raises the
    /// "Degenerate mixture covariance" warning.</summary>
    private void CheckNonNegativeVariances(double[,,,] covars)
    {
        var d0 = covars.GetLength(0);
        var d1 = covars.GetLength(1);
        var d2 = covars.GetLength(2);
        var d3 = covars.GetLength(3);
        var anyNegative = false;
        var anyZero = false;
        for (var c = 0; c < d0; c++)
            for (var m = 0; m < d1; m++)
                for (var a = 0; a < d2; a++)
                    for (var b = 0; b < d3; b++)
                    {
                        var v = covars[c, m, a, b];
                        if (v < 0.0)
                            anyNegative = true;
                        else if (v == 0.0)
                            anyZero = true;
                    }
        if (anyNegative)
            throw new ArgumentException($"'{CovarianceType}' mixture covars must be non-negative");
        if (anyZero)
            RaiseWarning("Degenerate mixture covariance");
    }

    /// <summary>Checks that the matrix at <c>covars[c, m]</c> is symmetric (numpy allclose) with a smallest eigenvalue not negative:
    /// a negative eigenvalue throws, a null one raises a warning; the messages use <paramref name="label"/> (for instance
    /// "state #0" or "state #1, mixture #2").</summary>
    private void CheckCovarianceMatrix(double[,,,] covars, int c, int m, string label, int nf)
    {
        var matrix = new double[nf, nf];
        for (var i = 0; i < nf; i++)
            for (var j = 0; j < nf; j++)
                matrix[i, j] = covars[c, m, i, j];
        for (var i = 0; i < nf; i++)
            for (var j = i + 1; j < nf; j++)
            {
                var symmetryTolerance = 1e-8 + 1e-5 * Math.Abs(matrix[j, i]);
                if (Math.Abs(matrix[i, j] - matrix[j, i]) > symmetryTolerance)
                    throw new ArgumentException($"Covariance of {label} is not symmetric");
            }
        for (var i = 0; i < nf; i++)
            for (var j = i + 1; j < nf; j++)
            {
                var v = 0.5 * (matrix[i, j] + matrix[j, i]);
                matrix[i, j] = v;
                matrix[j, i] = v;
            }
        var eigenvalues = Matrix<double>.Build.DenseOfArray(matrix).Evd(Symmetricity.Symmetric).EigenValues;
        var min = double.PositiveInfinity;
        var maxAbs = 0.0;
        for (var i = 0; i < eigenvalues.Count; i++)
        {
            var value = eigenvalues[i].Real;
            if (value < min)
                min = value;
            if (Math.Abs(value) > maxAbs)
                maxAbs = Math.Abs(value);
        }
        var tolerance = 1e-8 + 1e-5 * maxAbs;
        if (min < -tolerance)
            throw new ArgumentException($"Covariance of {label} is not positive definite");
        if (Math.Abs(min) <= tolerance)
            RaiseWarning($"Covariance of {label} has a null eigenvalue.");
    }
}
