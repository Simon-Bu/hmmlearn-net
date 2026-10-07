using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm11;

/// <summary>
/// Acceptance tests for the validation of <see cref="MultinomialHmm"/> (ticket HMM-11, criterion C3): Check rejects a null
/// emissionprob, a wrong number of rows or of columns and rows that do not sum to 1, and sets NFeatures to the number of
/// columns of the emissionprob when it is null; CheckAndSetNFeatures rejects non-integer counts, negative counts and totals
/// different from NTrials, and leaves the totals free (and sets NFeatures) when NTrials is null. A model without NTrials
/// cannot sample: Sample throws InvalidOperationException.
/// </summary>
public class ValidationTests
{
    /// <summary>A fully valid 2-component model (init_params "" : nothing is initialized); NFeatures and EmissionProb come
    /// from the arguments.</summary>
    private static MultinomialHmm Model(int? nFeatures, double[,]? emissionProb, int? nTrials = 5)
    {
        var model = new MultinomialHmm(2, nTrials, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "ste", "", Implementation.Log);
        model.NFeatures = nFeatures;
        model.StartProb = new[] { 0.6, 0.4 };
        model.TransMat = new double[,] { { 0.7, 0.3 }, { 0.4, 0.6 } };
        model.EmissionProb = emissionProb;
        return model;
    }

    private static double[,] ValidEmissionProb(int features)
        => features == 3
            ? new double[,] { { 0.2, 0.3, 0.5 }, { 0.6, 0.3, 0.1 } }
            : new double[,] { { 0.1, 0.2, 0.3, 0.4 }, { 0.4, 0.3, 0.2, 0.1 } };

    /// <summary>A null emissionprob is rejected with an ArgumentException.</summary>
    [Fact]
    public void a_null_emission_prob_throws()
    {
        var model = Model(3, null);
        Assert.Throws<ArgumentException>(() => model.Score(new double[,] { { 1, 1, 3 } }));
    }

    /// <summary>An emissionprob with the wrong number of rows (states) is rejected.</summary>
    [Fact]
    public void an_emission_prob_with_the_wrong_number_of_rows_throws()
    {
        var model = Model(3, new double[,] { { 0.2, 0.3, 0.5 } });
        Assert.Throws<ArgumentException>(() => model.Score(new double[,] { { 1, 1, 3 } }));
    }

    /// <summary>An emissionprob with the wrong number of columns (features) is rejected.</summary>
    [Fact]
    public void an_emission_prob_with_the_wrong_number_of_columns_throws()
    {
        var model = Model(4, ValidEmissionProb(3));
        Assert.Throws<ArgumentException>(() => model.Score(new double[,] { { 1, 1, 3 } }));
    }

    /// <summary>An emissionprob whose rows do not sum to 1 is rejected.</summary>
    [Fact]
    public void an_emission_prob_whose_rows_do_not_sum_to_one_throws()
    {
        var model = Model(3, new double[,] { { 0.4, 0.3, 0.2 }, { 0.5, 0.3, 0.2 } });
        Assert.Throws<ArgumentException>(() => model.Score(new double[,] { { 1, 1, 3 } }));
    }

    /// <summary>When NFeatures is null, Check sets it to the number of columns of the emissionprob.</summary>
    [Fact]
    public void check_sets_n_features_to_the_number_of_columns_of_the_emission_prob()
    {
        var model = Model(null, ValidEmissionProb(4));

        model.Score(new double[,] { { 1, 1, 2, 1 } });

        Assert.True(model.NFeatures == 4, $"NFeatures: expected 4, got {model.NFeatures}");
    }

    /// <summary>A non-integer count is rejected, even when the total of the sample matches NTrials.</summary>
    [Fact]
    public void a_fractional_count_throws_even_when_the_total_matches()
    {
        var model = Model(3, ValidEmissionProb(3));
        var x = new double[,] { { 2.5, 2.5, 0 }, { 1, 1, 3 } };
        Assert.Throws<ArgumentException>(() => model.Fit(x));
    }

    /// <summary>A negative count is rejected (its row still totals NTrials, so the total check cannot be what fires).</summary>
    [Fact]
    public void a_negative_count_throws()
    {
        var model = Model(3, ValidEmissionProb(3));
        var x = new double[,] { { -1, 6, 0 }, { 2, 1, 2 } };
        Assert.Throws<ArgumentException>(() => model.Fit(x));
    }

    /// <summary>A sample whose total differs from NTrials is rejected (all counts are integers).</summary>
    [Fact]
    public void a_sample_whose_total_differs_from_n_trials_throws()
    {
        var model = Model(3, ValidEmissionProb(3));
        var x = new double[,] { { 1, 1, 1 }, { 2, 1, 2 } };
        Assert.Throws<ArgumentException>(() => model.Fit(x));
    }

    /// <summary>When NTrials is null, the totals may vary per sample: the fit succeeds and leaves valid distributions.</summary>
    [Fact]
    public void totals_may_vary_per_sample_when_n_trials_is_null()
    {
        var model = Model(3, ValidEmissionProb(3), nTrials: null);
        var x = new double[,] { { 1, 1, 1 }, { 2, 1, 2 } };

        model.Fit(x);

        Assert.True(model.NFeatures == 3, $"NFeatures: expected 3, got {model.NFeatures}");
        Assert.True(model.StartProb!.All(v => v >= 0.0) && Math.Abs(model.StartProb.Sum() - 1.0) <= 1e-8,
            "startprob is not a valid distribution");
        Assert.True(RowsSumToOne(model.TransMat!) && model.TransMat.Cast<double>().All(v => v >= 0.0),
            "transmat is not row-stochastic");
        Assert.True(RowsSumToOne(model.EmissionProb!) && model.EmissionProb.Cast<double>().All(v => v >= 0.0),
            "emissionprob is not row-stochastic");
    }

    /// <summary>A model without NTrials cannot sample: Sample throws InvalidOperationException.</summary>
    [Fact]
    public void a_sample_without_n_trials_cannot_be_drawn()
    {
        var model = Model(3, ValidEmissionProb(3), nTrials: null);
        Assert.Throws<InvalidOperationException>(() => model.Sample(10));
    }

    /// <summary>Fitting data with a different number of columns than a previous fit is rejected.</summary>
    [Fact]
    public void a_different_number_of_columns_than_a_previous_fit_throws()
    {
        var model = Model(null, ValidEmissionProb(4));
        model.Fit(new double[,] { { 1, 1, 1, 2 }, { 2, 1, 1, 1 } });

        Assert.Throws<ArgumentException>(() => model.Fit(new double[,] { { 1, 1, 1 }, { 2, 1, 2 } }));
    }

    /// <summary>True when every row of a sums to 1 (within 1e-8).</summary>
    private static bool RowsSumToOne(double[,] a)
    {
        for (var i = 0; i < a.GetLength(0); i++)
        {
            var sum = 0.0;
            for (var j = 0; j < a.GetLength(1); j++)
                sum += a[i, j];
            if (Math.Abs(sum - 1.0) > 1e-8)
                return false;
        }
        return true;
    }
}
