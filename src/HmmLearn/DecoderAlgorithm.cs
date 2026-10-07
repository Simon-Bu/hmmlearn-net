namespace HmmLearn;

/// <summary>The algorithm that finds the state sequence behind some observations (hmmlearn's decoder algorithm).</summary>
public enum DecoderAlgorithm
{
    /// <summary>Finds the most likely sequence of states, given all emissions.</summary>
    Viterbi,

    /// <summary>
    /// Finds the sequence of the individually most likely states (also known as smoothing or forward-backward): for each sample
    /// the state with the largest posterior probability.
    /// </summary>
    Map,
}
