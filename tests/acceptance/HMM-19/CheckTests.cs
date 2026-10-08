using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm19;

/// <summary>
/// Acceptance tests for the validation of <see cref="VariationalBaseHmm"/> (ticket HMM-19, criterion C1): Check accepts valid
/// Dirichlet priors and posteriors and, since the values are not otherwise checked, also negative and NaN values with the right
/// shapes; it throws ArgumentException with the documented message when StartProbPriorValues or StartProbPosterior is null or
/// not of length NComponents, or when TransMatPriorValues or TransMatPosterior is null or not of shape (NComponents,
/// NComponents). Fit and Score on a model with invalid parameters fail the same way, because both call Check with the invalid
/// parameters still in place (with init_params "" the initialization leaves the explicitly set arrays untouched).
/// </summary>
public class CheckTests
{
    /// <summary>A model with the given (possibly invalid) variational parameters; startprob and transmat stay unset.</summary>
    private static StubVariationalHmm Model(int nComponents, double[]? startProbPriorValues, double[]? startProbPosterior,
        double[,]? transMatPriorValues, double[,]? transMatPosterior) =>
        new(nComponents, null, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "", Implementation.Log)
        {
            StartProbPriorValues = startProbPriorValues,
            StartProbPosterior = startProbPosterior,
            TransMatPriorValues = transMatPriorValues,
            TransMatPosterior = transMatPosterior,
        };

    private static double[] PriorVec() => new[] { 0.5, 1.5, 2.0 };

    private static double[] PostVec() => new[] { 1.2, 0.9, 0.9 };

    private static double[,] PriorMat() => Matrix(new[] { 0.8, 1.2, 0.9 }, new[] { 1.1, 0.7, 1.4 }, new[] { 0.6, 1.9, 0.8 });

    private static double[,] PostMat() => Matrix(new[] { 2.5, 3.1, 0.4 }, new[] { 0.9, 4.2, 2.7 }, new[] { 3.6, 1.1, 0.7 });

    private static double[,] Matrix(params double[][] rows)
    {
        var m = new double[rows.Length, rows[0].Length];
        for (var i = 0; i < rows.Length; i++)
            for (var j = 0; j < rows[i].Length; j++)
                m[i, j] = rows[i][j];
        return m;
    }

    private static double[,] Data(int rows, int columns)
    {
        var x = new double[rows, columns];
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < columns; c++)
                x[r, c] = -0.5;
        return x;
    }

    /// <summary>Check accepts priors and posteriors that are set and have the right shapes.</summary>
    [Fact]
    public void check_accepts_valid_priors_and_posteriors() =>
        Model(3, PriorVec(), PostVec(), PriorMat(), PostMat()).RunCheck();

    /// <summary>Check does not look at the values: negative or NaN values with the right shapes are accepted.</summary>
    [Fact]
    public void check_accepts_negative_and_nan_values_as_long_as_the_shapes_are_right() =>
        Model(3,
                new[] { -0.5, 0.0, 2.0 },
                new[] { 1.0, -1.0, double.NaN },
                Matrix(new[] { -1.0, 0.5, 2.0 }, new[] { 0.0, double.NaN, -3.0 }, new[] { 7.0, -7.0, 0.1 }),
                Matrix(new[] { 0.2, -0.3, 4.0 }, new[] { -2.0, 1.1, 0.0 }, new[] { double.NaN, 5.0, -5.0 }))
            .RunCheck();

    [Fact]
    public void check_throws_when_startprob_prior_is_null()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Model(3, null, PostVec(), PriorMat(), PostMat()).RunCheck());
        Assert.Equal("startprob_prior_ must have length n_components", ex.Message);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void check_throws_when_startprob_prior_has_the_wrong_length(int length)
    {
        var wrong = new double[length];
        var ex = Assert.Throws<ArgumentException>(
            () => Model(3, wrong, PostVec(), PriorMat(), PostMat()).RunCheck());
        Assert.Equal("startprob_prior_ must have length n_components", ex.Message);
    }

    [Fact]
    public void check_throws_when_startprob_posterior_is_null()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Model(3, PriorVec(), null, PriorMat(), PostMat()).RunCheck());
        Assert.Equal("startprob_posterior_ must have length n_components", ex.Message);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void check_throws_when_startprob_posterior_has_the_wrong_length(int length)
    {
        var wrong = new double[length];
        var ex = Assert.Throws<ArgumentException>(
            () => Model(3, PriorVec(), wrong, PriorMat(), PostMat()).RunCheck());
        Assert.Equal("startprob_posterior_ must have length n_components", ex.Message);
    }

    [Fact]
    public void check_throws_when_transmat_prior_is_null()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Model(3, PriorVec(), PostVec(), null, PostMat()).RunCheck());
        Assert.Equal("transmat_prior_ must have shape (n_components, n_components)", ex.Message);
    }

    [Theory]
    [InlineData(2, 2)]
    [InlineData(2, 3)]
    [InlineData(3, 2)]
    public void check_throws_when_transmat_prior_has_the_wrong_shape(int rows, int columns)
    {
        var wrong = new double[rows, columns];
        var ex = Assert.Throws<ArgumentException>(
            () => Model(3, PriorVec(), PostVec(), wrong, PostMat()).RunCheck());
        Assert.Equal("transmat_prior_ must have shape (n_components, n_components)", ex.Message);
    }

    [Fact]
    public void check_throws_when_transmat_posterior_is_null()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Model(3, PriorVec(), PostVec(), PriorMat(), null).RunCheck());
        Assert.Equal("transmat_posterior_ must have shape (n_components, n_components)", ex.Message);
    }

    [Theory]
    [InlineData(2, 2)]
    [InlineData(2, 3)]
    [InlineData(3, 2)]
    public void check_throws_when_transmat_posterior_has_the_wrong_shape(int rows, int columns)
    {
        var wrong = new double[rows, columns];
        var ex = Assert.Throws<ArgumentException>(
            () => Model(3, PriorVec(), PostVec(), PriorMat(), wrong).RunCheck());
        Assert.Equal("transmat_posterior_ must have shape (n_components, n_components)", ex.Message);
    }

    /// <summary>Fit fails with the documented ArgumentException when a variational array is invalid (init_params "" leaves the
    /// arrays untouched, so Check sees them as they were set).</summary>
    [Fact]
    public void fit_with_invalid_parameters_fails_the_same_way()
    {
        var model = Model(3, PriorVec(), new[] { 0.5, 0.5 }, PriorMat(), PostMat());
        var ex = Assert.Throws<ArgumentException>(() => model.Fit(Data(4, 3), new[] { 2, 2 }));
        Assert.Equal("startprob_posterior_ must have length n_components", ex.Message);
    }

    /// <summary>Score fails with the documented ArgumentException when a variational array is invalid, after the fitted
    /// parameters have been checked.</summary>
    [Fact]
    public void score_with_invalid_parameters_fails_the_same_way()
    {
        var model = Model(3, PriorVec(), PostVec(), PriorMat(), new double[2, 2]);
        model.StartProb = new[] { 0.25, 0.25, 0.5 };
        model.TransMat = Matrix(new[] { 0.5, 0.3, 0.2 }, new[] { 0.2, 0.5, 0.3 }, new[] { 0.3, 0.2, 0.5 });
        var ex = Assert.Throws<ArgumentException>(() => model.Score(Data(2, 3)));
        Assert.Equal("transmat_posterior_ must have shape (n_components, n_components)", ex.Message);
    }
}
