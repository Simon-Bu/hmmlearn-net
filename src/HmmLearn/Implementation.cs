namespace HmmLearn;

/// <summary>How the forward-backward algorithm is computed (hmmlearn's <c>implementation</c>).</summary>
public enum Implementation
{
    /// <summary>With logarithms (the default of hmmlearn, robust against underflow).</summary>
    Log,

    /// <summary>With scaling (generally faster; fails with an underflow error on very unlikely data).</summary>
    Scaling,
}
