namespace HmmLearn;

/// <summary>The structure of the covariance parameters of a Gaussian emission (hmmlearn's <c>covariance_type</c>).</summary>
public enum CovarianceType
{
    /// <summary>One variance per state (or mixture component), shared by all features.</summary>
    Spherical,

    /// <summary>One variance per feature, for each state (or mixture component).</summary>
    Diag,

    /// <summary>A full covariance matrix for each state (or mixture component).</summary>
    Full,

    /// <summary>One full covariance matrix shared by all states (for GMMHMM: shared by the mixture components of a state).</summary>
    Tied,
}
