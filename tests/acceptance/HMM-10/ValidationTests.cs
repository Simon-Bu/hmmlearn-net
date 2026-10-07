using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm10;

/// <summary>
/// Acceptance tests for the validation of <see cref="CategoricalHmm"/> (ticket HMM-10, criterion C3): <c>Check</c> rejects a null
/// <see cref="CategoricalHmm.EmissionProb"/>, the wrong number of rows or columns, and rows that do not sum to 1 (ArgumentException);
/// <c>CheckAndSetNFeatures</c> (run by <c>Fit</c> before <c>Check</c>) rejects a fractional symbol, a negative symbol, a symbol above
/// NFeatures - 1, and observations with more than one column; the number of features is inferred as the largest symbol + 1 and respected
/// when given; <see cref="CategoricalStatistics"/> allocates Obs (n_components, n_features) of zeros.
/// </summary>
public class ValidationTests
{
    /// <summary>A model with a valid start probability and transition matrix, the given emission probabilities (or null) and the given
    /// number of features; init_params "" so that nothing is initialized and Check is the only gate.</summary>
    private static CategoricalHmm Model(int? nFeatures, double[,]? emissionProb)
    {
        var model = new CategoricalHmm(
            2, 1.0, 1.0, 1.0, nFeatures, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "ste", "", Implementation.Log);
        model.StartProb = new[] { 0.6, 0.4 };
        model.TransMat = new double[,] { { 0.7, 0.3 }, { 0.4, 0.6 } };
        model.EmissionProb = emissionProb;
        return model;
    }

    /// <summary>A fit-ready model (valid start, transition and emission) with n_features 3, used to exercise the symbol checks of
    /// CheckAndSetNFeatures, which Fit runs before Check.</summary>
    private static CategoricalHmm Fittable()
    {
        var model = new CategoricalHmm(
            2, 1.0, 1.0, 1.0, 3, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "ste", "", Implementation.Log);
        model.StartProb = new[] { 0.6, 0.4 };
        model.TransMat = new double[,] { { 0.7, 0.3 }, { 0.4, 0.6 } };
        model.EmissionProb = new double[,] { { 0.2, 0.3, 0.5 }, { 0.4, 0.3, 0.3 } };
        return model;
    }

    /// <summary>Check throws when EmissionProb is null.</summary>
    [Fact]
    public void a_null_emission_prob_throws()
    {
        var model = Model(3, null);

        Assert.Throws<ArgumentException>(() => model.Score(new double[,] { { 0 }, { 1 } }));
    }

    /// <summary>Check throws when EmissionProb has a different number of rows than n_components.</summary>
    [Fact]
    public void an_emission_prob_with_the_wrong_number_of_rows_throws()
    {
        var model = Model(3, new double[,] { { 0.4, 0.3, 0.3 } });

        Assert.Throws<ArgumentException>(() => model.Score(new double[,] { { 0 } }));
    }

    /// <summary>Check throws when EmissionProb has a different number of columns than n_features.</summary>
    [Fact]
    public void an_emission_prob_with_the_wrong_number_of_columns_throws()
    {
        var model = Model(3, new double[,] { { 0.3, 0.3, 0.2, 0.2 }, { 0.4, 0.4, 0.1, 0.1 } });

        Assert.Throws<ArgumentException>(() => model.Score(new double[,] { { 0 } }));
    }

    /// <summary>Check throws when a row of EmissionProb does not sum to 1.</summary>
    [Fact]
    public void an_emission_prob_whose_rows_do_not_sum_to_one_throws()
    {
        var model = Model(3, new double[,] { { 0.4, 0.3, 0.2 }, { 0.5, 0.3, 0.2 } });

        Assert.Throws<ArgumentException>(() => model.Score(new double[,] { { 0 } }));
    }

    /// <summary>When NFeatures is not given, Check sets it to the number of columns of EmissionProb.</summary>
    [Fact]
    public void check_sets_n_features_to_the_number_of_columns_of_the_emission_prob()
    {
        var emissionProb = new double[,] { { 0.25, 0.25, 0.25, 0.25 }, { 0.5, 0.2, 0.2, 0.1 } };
        var model = Model(null, emissionProb);

        model.Score(new double[,] { { 0 }, { 3 }, { 1 } });

        Assert.True(model.NFeatures == emissionProb.GetLength(1),
            $"n_features: expected {emissionProb.GetLength(1)}, got {model.NFeatures}");
    }

    /// <summary>CheckAndSetNFeatures (via Fit) throws when a symbol is not an integer.</summary>
    [Fact]
    public void a_fractional_symbol_throws()
    {
        var model = Fittable();

        Assert.Throws<ArgumentException>(() => model.Fit(new double[,] { { 0.5 }, { 1 } }));
    }

    /// <summary>CheckAndSetNFeatures (via Fit) throws when a symbol is negative.</summary>
    [Fact]
    public void a_negative_symbol_throws()
    {
        var model = Fittable();

        Assert.Throws<ArgumentException>(() => model.Fit(new double[,] { { 0 }, { -1 } }));
    }

    /// <summary>CheckAndSetNFeatures (via Fit) throws when a symbol is larger than n_features - 1.</summary>
    [Fact]
    public void a_symbol_above_n_features_minus_one_throws()
    {
        var model = Fittable();

        Assert.Throws<ArgumentException>(() => model.Fit(new double[,] { { 0 }, { 3 } }));
    }

    /// <summary>CheckAndSetNFeatures (via Fit) throws when the observations have more than one column.</summary>
    [Fact]
    public void observations_with_more_than_one_column_throw()
    {
        var model = Fittable();

        Assert.Throws<ArgumentException>(() => model.Fit(new double[,] { { 0, 1 }, { 1, 0 } }));
    }

    /// <summary>When n_features is not given, Fit infers it as the largest symbol + 1.</summary>
    [Fact]
    public void n_features_is_inferred_as_the_largest_symbol_plus_one()
    {
        var model = new CategoricalHmm(
            2, 1.0, 1.0, 1.0, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "ste", "", Implementation.Log);
        model.StartProb = new[] { 0.6, 0.4 };
        model.TransMat = new double[,] { { 0.7, 0.3 }, { 0.4, 0.6 } };
        model.EmissionProb = new double[,] { { 0.2, 0.3, 0.5 }, { 0.4, 0.3, 0.3 } };
        var x = new double[,] { { 0 }, { 2 }, { 1 }, { 0 } };
        var largest = 0.0;
        for (var i = 0; i < x.GetLength(0); i++)
            largest = Math.Max(largest, x[i, 0]);

        model.Fit(x);

        Assert.True(model.NFeatures == (int)largest + 1,
            $"n_features: expected {(int)largest + 1} (largest symbol + 1), got {model.NFeatures}");
    }

    /// <summary>When n_features is given (larger than the largest symbol), Fit respects it.</summary>
    [Fact]
    public void n_features_is_respected_when_given()
    {
        var model = new CategoricalHmm(
            2, 1.0, 1.0, 1.0, 5, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "ste", "", Implementation.Log);
        model.StartProb = new[] { 0.6, 0.4 };
        model.TransMat = new double[,] { { 0.7, 0.3 }, { 0.4, 0.6 } };
        model.EmissionProb = new double[,] { { 0.2, 0.2, 0.2, 0.2, 0.2 }, { 0.5, 0.2, 0.1, 0.1, 0.1 } };
        var x = new double[,] { { 0 }, { 2 }, { 1 } };

        model.Fit(x);

        Assert.True(model.NFeatures == 5, $"n_features: expected 5, got {model.NFeatures}");
    }

    /// <summary>CategoricalStatistics allocates Obs with the given shape, filled with zeros.</summary>
    [Fact]
    public void categorical_statistics_allocates_obs_with_the_given_shape_filled_with_zeros()
    {
        var stats = new CategoricalStatistics(3, 5);

        Assert.Equal(3, stats.Obs.GetLength(0));
        Assert.Equal(5, stats.Obs.GetLength(1));
        Assert.True(stats.Obs.Cast<double>().All(v => v == 0.0), "Obs must be allocated as zeros");
    }
}
