using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm15;

/// <summary>
/// Acceptance tests for the validation and the statistics of <see cref="PoissonHmm"/> (ticket HMM-15, criterion C3):
/// CheckAndSetNFeatures (non-negative integer counts, the number of features recorded once, and the column count pinned after
/// the first fit), the shape of Check (null or wrongly shaped Lambdas with the documented message; the columns of Lambdas set
/// NFeatures while it is unknown), and that a negative rate is an error while a zero rate is legal (a zero count under a zero
/// rate costs 0, a positive count scores -infinity). The shape and null checks of the base class (startprob, transmat, and the
/// column count of x against NFeatures) are inherited from AbstractHmm and were already accepted by the HMM-7 ticket.
/// </summary>
public class ValidationTests
{
    /// <summary>A fully specified 2-component, 3-feature model (InitParams "" so nothing is initialized), with the given
    /// NFeatures and Lambdas.</summary>
    private static PoissonHmm Model(int? nFeatures, double[,]? lambdas, Implementation implementation = Implementation.Log)
    {
        var model = new PoissonHmm(
            2, 1.0, 1.0, 0.0, 0.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "stl", "", implementation);
        model.NFeatures = nFeatures;
        model.StartProb = new[] { 0.6, 0.4 };
        model.TransMat = new double[,] { { 0.7, 0.3 }, { 0.4, 0.6 } };
        model.Lambdas = lambdas;
        return model;
    }

    private static double[,] ValidLambdas() => new double[,] { { 3.1, 1.4, 4.5 }, { 1.6, 5.3, 0.1 } };

    private static double[,] ValidX() => new double[,] { { 2.0, 1.0, 3.0 }, { 0.0, 2.0, 1.0 } };

    [Fact]
    public void a_null_lambdas_throws()
    {
        var model = Model(3, null);

        var ex = Assert.Throws<ArgumentException>(() => model.Score(ValidX()));
        Assert.Equal("lambdas_ must have shape (n_components, n_features)", ex.Message);
    }

    [Fact]
    public void a_lambdas_with_the_wrong_number_of_rows_throws()
    {
        var model = Model(3, new double[,] { { 3.1, 1.4, 4.5 } });

        var ex = Assert.Throws<ArgumentException>(() => model.Score(ValidX()));
        Assert.Equal("lambdas_ must have shape (n_components, n_features)", ex.Message);
    }

    [Fact]
    public void a_lambdas_with_the_wrong_number_of_columns_throws()
    {
        var model = Model(3, new double[,] { { 3.1, 1.4 }, { 1.6, 5.3 } });

        var ex = Assert.Throws<ArgumentException>(() => model.Score(ValidX()));
        Assert.Equal("lambdas_ must have shape (n_components, n_features)", ex.Message);
    }

    /// <summary>While NFeatures is unknown, Check records the number of columns of the given Lambdas (here 4) and accepts the
    /// inference; afterwards NFeatures is pinned.</summary>
    [Fact]
    public void check_sets_n_features_to_the_number_of_columns_of_lambdas_when_unknown()
    {
        var model = Model(null, new double[,] { { 1.0, 2.0, 3.0, 4.0 }, { 0.5, 1.5, 2.5, 3.5 } });
        var x = new double[,] { { 1.0, 1.0, 2.0, 1.0 } };

        var score = model.Score(x);

        Assert.True(model.NFeatures == 4, $"NFeatures after the first inference: expected 4, got {model.NFeatures}");
        Assert.True(double.IsFinite(score) && score < 0.0, $"the score must be finite and negative, got {score}");
    }

    /// <summary>A negative rate is invalid: Score with a model whose rates include a negative entry must throw an
    /// ArgumentException (a zero rate is NOT rejected, see the next test). Both implementations are covered.</summary>
    [Theory]
    [InlineData(Implementation.Log)]
    [InlineData(Implementation.Scaling)]
    public void a_negative_rate_throws(Implementation implementation)
    {
        var model = Model(3, new double[,] { { 3.1, 1.4, 4.5 }, { -1.6, 5.3, 0.1 } }, implementation);

        Assert.Throws<ArgumentException>(() => model.Score(ValidX()));
    }

    /// <summary>
    /// A zero rate is legal and the log likelihood is exact at the degenerate points: a zero count under a zero rate costs 0
    /// (0 * log(0) is 0, rate is 0, GammaLn(1) is 0) and a positive count under a zero rate scores exactly -infinity. Trivial
    /// input, no rounding involved. Both implementations are covered.
    /// </summary>
    [Theory]
    [InlineData(Implementation.Log)]
    [InlineData(Implementation.Scaling)]
    public void a_zero_count_under_a_zero_rate_scores_zero_and_a_positive_count_scores_minus_infinity(Implementation implementation)
    {
        var model = new PoissonHmm(
            1, 1.0, 1.0, 0.0, 0.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "stl", "", implementation);
        model.NFeatures = 1;
        model.StartProb = new[] { 1.0 };
        model.TransMat = new double[,] { { 1.0 } };
        model.Lambdas = new double[,] { { 0.0 } };

        Close.Equal(0.0, model.Score(new double[,] { { 0.0 } }), $"a zero count under a zero rate ({implementation})");
        if (implementation == Implementation.Log)
            Assert.True(double.IsNegativeInfinity(model.Score(new double[,] { { 1.0 } })),
                $"a positive count under a zero rate must score -infinity ({implementation})");
        else
            // With the scaling implementation an impossible observation (probability 0 under every state) is the underflow error of
            // the forward pass (Hmmc.ForwardScaling), not -infinity.
            Assert.Throws<ArithmeticException>(() => model.Score(new double[,] { { 1.0 } }));
    }

    /// <summary>Fit validates the counts of x: fractional counts are rejected by CheckAndSetNFeatures (Init) with an
    /// ArgumentException.</summary>
    [Fact]
    public void a_fractional_count_throws()
    {
        var model = Model(3, ValidLambdas());
        var x = new double[,] { { 2.5, 1.0, 3.0 }, { 0.0, 2.0, 1.0 } };

        Assert.Throws<ArgumentException>(() => model.Fit(x));
    }

    /// <summary>Fit validates the counts of x: negative counts are rejected by CheckAndSetNFeatures (Init) with an
    /// ArgumentException.</summary>
    [Fact]
    public void a_negative_count_throws()
    {
        var model = Model(3, ValidLambdas());
        var x = new double[,] { { -1.0, 2.0, 3.0 }, { 0.0, 2.0, 1.0 } };

        Assert.Throws<ArgumentException>(() => model.Fit(x));
    }

    /// <summary>Once NFeatures is known (3), a fit on data with a different number of columns is rejected by CheckObservations
    /// (inherited) with an ArgumentException.</summary>
    [Fact]
    public void a_different_number_of_columns_than_the_model_features_throws()
    {
        var model = Model(3, ValidLambdas());

        Assert.Throws<ArgumentException>(() => model.Fit(new double[,] { { 1.0, 1.0 }, { 2.0, 1.0 } }));
    }

    /// <summary>The statistics are allocated with NObs 0 and zeroed arrays of the documented shapes: Start (n_components),
    /// Trans (n_components x n_components), Post (n_components) and Obs (n_components x n_features).</summary>
    [Fact]
    public void statistics_are_allocated_zeroed_with_the_documented_shapes()
    {
        var stats = new PoissonStatistics(2, 3);

        Assert.Equal(0, stats.NObs);
        Close.Equal(new double[2], stats.Start, "Start");
        Close.Equal(new double[2, 2], stats.Trans, "Trans");
        Close.Equal(new double[2], stats.Post, "Post");
        Close.Equal(new double[2, 3], stats.Obs, "Obs");
    }
}
