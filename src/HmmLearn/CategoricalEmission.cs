namespace HmmLearn;

/// <summary>
/// The categorical (discrete) emission, shared by <see cref="CategoricalHmm"/> and, later, the variational categorical model
/// (port of BaseCategoricalHMM of hmmlearn/_emissions.py). Observations are a matrix with ONE column holding a symbol (an
/// integer-valued number, 0 .. n_features - 1) per sample; the emission probabilities are a matrix (n_components, n_features).
/// </summary>
internal static class CategoricalEmission
{
    /// <summary>
    /// Checks the symbols and records the number of features. Throws <see cref="ArgumentException"/> if x does not have exactly one
    /// column, a value is not an integer ("Symbols should be integers"), or a value is negative ("Symbols should be nonnegative").
    /// If <paramref name="nFeatures"/> has a value, a symbol larger than nFeatures - 1 throws ArgumentException ("Largest symbol is {max}
    /// but the model only emits symbols up to {nFeatures - 1}"); otherwise nFeatures becomes the largest symbol plus 1.
    /// </summary>
    internal static void CheckAndSetNFeatures(double[,] x, ref int? nFeatures) => throw new NotImplementedException();

    /// <summary>Likelihood of each sample under each state: result[i, c] = emissionProb[c, symbol of sample i], shape (n_samples, n_components).</summary>
    internal static double[,] Likelihood(double[,] emissionProb, double[,] x) => throw new NotImplementedException();

    /// <summary>Adds to obs[c, symbol of sample i] the posterior probability posteriors[i, c], for every sample and state (obs is (n_components, n_features)).</summary>
    internal static void AccumulateObs(double[,] obs, double[,] x, double[,] posteriors) => throw new NotImplementedException();

    /// <summary>
    /// A one-element observation (the drawn symbol) from the row <paramref name="state"/> of the emission probabilities: one uniform
    /// number u from <paramref name="random"/>, then the first symbol whose cumulative probability is greater than u (the last symbol if rounding leaves none).
    /// </summary>
    internal static double[] GenerateSample(double[,] emissionProb, int state, Random random) => throw new NotImplementedException();

    /// <summary>Free scalar parameters per letter: 's' = nc - 1, 't' = nc * (nc - 1), 'e' = nc * (nf - 1).</summary>
    internal static IReadOnlyDictionary<char, int> NFitScalars(int nComponents, int nFeatures) => throw new NotImplementedException();
}
