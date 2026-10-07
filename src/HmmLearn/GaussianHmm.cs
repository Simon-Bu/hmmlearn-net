namespace HmmLearn;

/// <summary>
/// Hidden Markov model with Gaussian emissions (port of GaussianHMM of hmmlearn/hmm.py and BaseGaussianHMM of hmmlearn/_emissions.py).
/// This file holds the parameters, the checks, the likelihood, the sampling and the statistics; the initialization and the M-step are in
/// GaussianHmm.Learning.cs (same partial class). Covariances are stored in the compact uniform shapes of oracle/README.md:
/// spherical (nc, 1, 1), diag (nc, 1, nf), full (nc, nf, nf), tied (1, nf, nf).
/// </summary>
public partial class GaussianHmm : BaseHmm
{
    private double[,,]? _covars;

    /// <summary>Creates the model; the defaults are those of hmmlearn (diag covariances, parameter letters "stmc").</summary>
    public GaussianHmm(int nComponents = 1, CovarianceType covarianceType = CovarianceType.Diag, double minCovar = 1e-3,
        double startProbPrior = 1.0, double transMatPrior = 1.0, double meansPrior = 0, double meansWeight = 0,
        double covarsPrior = 1e-2, double covarsWeight = 1, DecoderAlgorithm algorithm = DecoderAlgorithm.Viterbi,
        int? randomState = null, int nIter = 10, double tol = 1e-2, bool verbose = false, string @params = "stmc",
        string initParams = "stmc", Implementation implementation = Implementation.Log)
        : base(nComponents, startProbPrior, transMatPrior, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
    {
        CovarianceType = covarianceType;
        MinCovar = minCovar;
        MeansPrior = meansPrior;
        MeansWeight = meansWeight;
        CovarsPrior = covarsPrior;
        CovarsWeight = covarsWeight;
    }

    /// <summary>The structure of the covariance parameters.</summary>
    public CovarianceType CovarianceType { get; set; }

    /// <summary>Floor added to the diagonal of the data covariance when the covariances are initialized.</summary>
    public double MinCovar { get; set; }

    /// <summary>Mean of the Normal prior of the means (scalar, applied to every feature of every state).</summary>
    public double MeansPrior { get; set; }

    /// <summary>Precision (weight) of the prior of the means.</summary>
    public double MeansWeight { get; set; }

    /// <summary>Scale parameter of the prior of the covariances.</summary>
    public double CovarsPrior { get; set; }

    /// <summary>Weight (degrees of freedom) of the prior of the covariances.</summary>
    public double CovarsWeight { get; set; }

    /// <summary>Mean of each state, shape (n_components, n_features); null until set or fitted.</summary>
    public double[,]? Means { get; set; }

    /// <summary>
    /// Compact covariances in the uniform shapes above; null until set or fitted. The setter validates the value with
    /// <see cref="InternalUtils.ValidateCovars"/> (ArgumentException if invalid) and, for Full and Tied, with
    /// <see cref="GaussianEmission.IsPositiveDefinite"/> (so positive semi-definite matrices are rejected, as in hmmlearn), and stores a copy.
    /// </summary>
    public double[,,]? Covars
    {
        get => _covars is null ? null : (double[,,])_covars.Clone();
        set
        {
            if (value != null)
            {
                InternalUtils.ValidateCovars(value, CovarianceType, NComponents);
                if (CovarianceType is CovarianceType.Full or CovarianceType.Tied)
                {
                    var n = value.GetLength(1);
                    for (var k = 0; k < value.GetLength(0); k++)
                    {
                        var matrix = new double[n, n];
                        for (var i = 0; i < n; i++)
                            for (var j = 0; j < n; j++)
                                matrix[i, j] = value[k, i, j];
                        if (!GaussianEmission.IsPositiveDefinite(matrix))
                            throw new ArgumentException($"covars of component {k} must be positive definite", nameof(value));
                    }
                }
            }
            _covars = value is null ? null : (double[,,])value.Clone();
        }
    }

    /// <summary>
    /// The covariances as full matrices (n_components, n_features, n_features) for every covariance type
    /// (<see cref="Utils.FillCovars"/>). Throws <see cref="InvalidOperationException"/> if Covars or the number of features is unknown.
    /// </summary>
    public double[,,] FullCovars
    {
        get
        {
            if (_covars == null || NFeatures == null)
                throw new InvalidOperationException("Covars or the number of features is unknown");
            return Utils.FillCovars(_covars, CovarianceType, NComponents, NFeatures.Value);
        }
    }

    /// <summary>
    /// Base checks, then: Means must be non-null and (NComponents x n_features) with n_features at least 1, else
    /// <see cref="ArgumentException"/>; NFeatures becomes the number of columns of Means; Covars must be non-null and valid
    /// (<see cref="InternalUtils.ValidateCovars"/>) with a number of features equal to that of Means (the last dimension of diag,
    /// the last two of full and tied); CovarianceType must be a defined value.
    /// </summary>
    protected override void Check()
    {
        base.Check();
        if (Means == null || Means.GetLength(0) != NComponents || Means.GetLength(1) < 1)
        {
            var shape = Means == null ? "null" : $"{Means.GetLength(0)} x {Means.GetLength(1)}";
            throw new ArgumentException($"means must have shape ({NComponents}, n_features) with at least one feature, got {shape}");
        }
        var nFeatures = Means.GetLength(1);
        NFeatures = nFeatures;
        if (CovarianceType < CovarianceType.Spherical || CovarianceType > CovarianceType.Tied)
            throw new ArgumentException($"covariance_type must be one of Spherical, Diag, Full, Tied, got {CovarianceType}");
        if (_covars == null)
            throw new ArgumentException("covars must be set");
        InternalUtils.ValidateCovars(_covars, CovarianceType, NComponents);
        switch (CovarianceType)
        {
            case CovarianceType.Diag:
                if (_covars.GetLength(2) != nFeatures)
                    throw new ArgumentException($"covars must have {nFeatures} features, got {_covars.GetLength(2)}");
                break;
            case CovarianceType.Full:
            case CovarianceType.Tied:
                if (_covars.GetLength(1) != nFeatures || _covars.GetLength(2) != nFeatures)
                    throw new ArgumentException($"covars must have {nFeatures} features, got {_covars.GetLength(1)} x {_covars.GetLength(2)}");
                break;
        }
    }

    /// <summary>The log density of each sample under each state: <see cref="Stats.LogMultivariateNormalDensity"/>.</summary>
    protected override double[,] ComputeLogLikelihood(double[,] x)
        => Stats.LogMultivariateNormalDensity(x, Means!, _covars!, CovarianceType);

    /// <summary>A <see cref="GaussianStatistics"/> for NComponents, the number of features and CovarianceType.</summary>
    protected override SufficientStatistics InitializeSufficientStatistics()
        => new GaussianStatistics(NComponents, NFeatures!.Value, CovarianceType);

    /// <summary>Base accumulation, then <see cref="GaussianEmission.Accumulate"/> with <see cref="NeedsStatisticsForMeans"/> and <see cref="NeedsStatisticsForCovars"/>.</summary>
    protected override void AccumulateSufficientStatistics(SufficientStatistics stats, double[,] x, double[,] lattice,
        double[,] posteriors, double[,] fwdlattice, double[,] bwdlattice)
    {
        base.AccumulateSufficientStatistics(stats, x, lattice, posteriors, fwdlattice, bwdlattice);
        GaussianEmission.Accumulate((GaussianStatistics)stats, x, posteriors, CovarianceType, NeedsStatisticsForMeans, NeedsStatisticsForCovars);
    }

    /// <summary>Whether the statistics of the means are accumulated: Params contains 'm' (as in hmmlearn, 'c' alone does not).</summary>
    protected virtual bool NeedsStatisticsForMeans => Params.Contains('m');

    /// <summary>Whether the statistics of the covariances are accumulated: Params contains 'c'.</summary>
    protected virtual bool NeedsStatisticsForCovars => Params.Contains('c');

    /// <summary>One observation from the Gaussian of the state (<see cref="GaussianEmission.GenerateSample"/> with <see cref="FullCovars"/>).</summary>
    protected override double[] GenerateSampleFromState(int state, Random random)
        => GaussianEmission.GenerateSample(Means!, FullCovars, state, random);

    /// <summary><see cref="GaussianEmission.NFitScalars"/>; requires the number of features to be known.</summary>
    protected override IReadOnlyDictionary<char, int>? GetNFitScalarsPerParam()
        => NFeatures.HasValue ? GaussianEmission.NFitScalars(NComponents, NFeatures.Value, CovarianceType) : null;
}
