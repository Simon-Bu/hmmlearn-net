using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm19;

/// <summary>
/// Acceptance tests for the initialization of <see cref="VariationalBaseHmm"/> (ticket HMM-19, criterion C3): Init fills the
/// prior values with 1/NComponents or the scalar prior, and draws Dirichlet posteriors with the documented scaling (the
/// startprob posterior sums to the number of sequences, each transmat row sums to the total length divided by NComponents),
/// only for the parameters that need initialization (posterior set and prior missing also initializes); fixed and missing
/// RandomState are respected; Init leaves everything untouched with init_params ""; a set parameter named in init_params
/// raises the overwrite warning; and fewer data points than free parameters raises the degenerate-solution warning.
/// </summary>
public class InitTests
{
    private static StubVariationalHmm Model(int nComponents, double? startProbPrior, double? transMatPrior, string @params,
        string initParams, int? randomState, double[]? startProbPriorValues = null, double[]? startProbPosterior = null,
        double[,]? transMatPriorValues = null, double[,]? transMatPosterior = null) =>
        new(nComponents, startProbPrior, transMatPrior, DecoderAlgorithm.Viterbi, randomState, 0, 1e-2, false, @params,
            initParams, Implementation.Log)
        {
            StartProbPriorValues = startProbPriorValues,
            StartProbPosterior = startProbPosterior,
            TransMatPriorValues = transMatPriorValues,
            TransMatPosterior = transMatPosterior,
        };

    private static double[,] Data(int rows, int columns)
    {
        var x = new double[rows, columns];
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < columns; c++)
                x[r, c] = -0.5;
        return x;
    }

    private static double[,] Matrix(params double[][] rows)
    {
        var m = new double[rows.Length, rows[0].Length];
        for (var i = 0; i < rows.Length; i++)
            for (var j = 0; j < rows[i].Length; j++)
                m[i, j] = rows[i][j];
        return m;
    }

    private static double[] UnitVector(int n)
    {
        var v = new double[n];
        for (var i = 0; i < n; i++)
            v[i] = 1.0;
        return v;
    }

    private static double[,] UnitMatrix(int n)
    {
        var m = new double[n, n];
        for (var r = 0; r < n; r++)
            for (var c = 0; c < n; c++)
                m[r, c] = 1.0;
        return m;
    }

    private static void AssertEach(double[] values, double expected, string what)
    {
        for (var i = 0; i < values.Length; i++)
            Close.Equal(expected, values[i], $"{what}[{i}]");
    }

    private static void AssertEach(double[,] values, double expected, string what)
    {
        for (var r = 0; r < values.GetLength(0); r++)
            for (var c = 0; c < values.GetLength(1); c++)
                Close.Equal(expected, values[r, c], $"{what}[{r}, {c}]");
    }

    /// <summary>A drawn startprob posterior: n_components positive values that sum to the documented scale.</summary>
    private static void AssertPosteriorVector(double[]? values, int nComponents, double expectedSum, string what)
    {
        Assert.NotNull(values);
        Assert.Equal(nComponents, values!.Length);
        for (var i = 0; i < values.Length; i++)
            Assert.True(values[i] > 0.0, $"{what}[{i}] = {values[i]} is not positive");
        Close.Equal(expectedSum, values.Sum(), what + " sum");
    }

    /// <summary>A drawn transmat posterior: an n_components x n_components matrix of positive values whose rows sum to the
    /// documented scale.</summary>
    private static void AssertPosteriorMatrix(double[,]? values, int nComponents, double expectedRowSum, string what)
    {
        Assert.NotNull(values);
        var m = values!;
        Assert.Equal(nComponents, m.GetLength(0));
        Assert.Equal(nComponents, m.GetLength(1));
        for (var r = 0; r < nComponents; r++)
        {
            var sum = 0.0;
            for (var c = 0; c < nComponents; c++)
            {
                Assert.True(m[r, c] > 0.0, $"{what}[{r}, {c}] = {m[r, c]} is not positive");
                sum += m[r, c];
            }
            Close.Equal(expectedRowSum, sum, $"{what} row {r} sum");
        }
    }

    private static bool VectorsAreClose(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            return false;
        for (var i = 0; i < a.Length; i++)
            if (!Close.IsClose(a[i], b[i]))
                return false;
        return true;
    }

    private static bool MatricesAreClose(double[,] a, double[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            return false;
        for (var r = 0; r < a.GetLength(0); r++)
            for (var c = 0; c < a.GetLength(1); c++)
                if (!Close.IsClose(a[r, c], b[r, c]))
                    return false;
        return true;
    }

    private static bool ParametersAreClose(StubVariationalHmm a, StubVariationalHmm b) =>
        VectorsAreClose(a.StartProbPriorValues!, b.StartProbPriorValues!)
        && VectorsAreClose(a.StartProbPosterior!, b.StartProbPosterior!)
        && MatricesAreClose(a.TransMatPriorValues!, b.TransMatPriorValues!)
        && MatricesAreClose(a.TransMatPosterior!, b.TransMatPosterior!);

    /// <summary>With nothing set, Init fills the prior values with 1/n_components and draws positive posteriors scaled by the
    /// number of sequences (startprob) and the total length divided by n_components (each transmat row).</summary>
    [Fact]
    public void init_fills_the_priors_and_draws_validly_scaled_posteriors()
    {
        const int nc = 3;
        var model = Model(nc, null, null, "st", "st", 42);
        var x = Data(5, nc);
        var lengths = new[] { 2, 3 };

        model.RunInit(x, lengths);

        Assert.Equal(nc, model.NFeatures);
        Assert.NotNull(model.StartProbPriorValues);
        Assert.Equal(nc, model.StartProbPriorValues!.Length);
        AssertEach(model.StartProbPriorValues, 1.0 / nc, "startprob_prior_");
        AssertPosteriorVector(model.StartProbPosterior, nc, lengths.Length, "startprob_posterior_");
        Assert.NotNull(model.TransMatPriorValues);
        Assert.Equal(nc, model.TransMatPriorValues!.GetLength(0));
        Assert.Equal(nc, model.TransMatPriorValues!.GetLength(1));
        AssertEach(model.TransMatPriorValues, 1.0 / nc, "transmat_prior_");
        AssertPosteriorMatrix(model.TransMatPosterior, nc, lengths.Sum() / (double)nc, "transmat_posterior_");
    }

    /// <summary>A scalar prior fills the prior values with that scalar (the posterior is still drawn with the documented
    /// scaling).</summary>
    [Fact]
    public void init_uses_the_scalar_priors_for_the_prior_values()
    {
        const int nc = 3;
        var model = Model(nc, 2.5, 0.7, "st", "st", 42);
        var x = Data(5, nc);
        var lengths = new[] { 2, 3 };

        model.RunInit(x, lengths);

        AssertNotNullEach(model.StartProbPriorValues, 2.5);
        AssertNotNullEachMatrix(model.TransMatPriorValues, 0.7);
        AssertPosteriorVector(model.StartProbPosterior, nc, lengths.Length, "startprob_posterior_");
        AssertPosteriorMatrix(model.TransMatPosterior, nc, lengths.Sum() / (double)nc, "transmat_posterior_");

        static void AssertNotNullEach(double[]? values, double expected)
        {
            Assert.NotNull(values);
            AssertEach(values!, expected, "prior values");
        }

        static void AssertNotNullEachMatrix(double[,]? values, double expected)
        {
            Assert.NotNull(values);
            AssertEach(values!, expected, "prior values");
        }
    }

    /// <summary>Two models with the same RandomState draw identical prior values and posteriors.</summary>
    [Fact]
    public void init_with_the_same_random_state_is_reproducible()
    {
        const int nc = 3;
        var x = Data(5, nc);
        var lengths = new[] { 2, 3 };
        var a = Model(nc, null, null, "st", "st", 42);
        var b = Model(nc, null, null, "st", "st", 42);
        a.RunInit(x, lengths);
        b.RunInit(x, lengths);

        Close.Equal(a.StartProbPriorValues!, b.StartProbPriorValues, "startprob_prior_");
        Close.Equal(a.StartProbPosterior!, b.StartProbPosterior, "startprob_posterior_");
        Close.Equal(a.TransMatPriorValues!, b.TransMatPriorValues, "transmat_prior_");
        Close.Equal(a.TransMatPosterior!, b.TransMatPosterior, "transmat_posterior_");
    }

    /// <summary>Two models with different RandomState values draw different posteriors.</summary>
    [Fact]
    public void init_with_different_seeds_draws_different_posteriors()
    {
        const int nc = 3;
        var x = Data(5, nc);
        var lengths = new[] { 2, 3 };
        var a = Model(nc, null, null, "st", "st", 42);
        var b = Model(nc, null, null, "st", "st", 1337);
        a.RunInit(x, lengths);
        b.RunInit(x, lengths);

        Assert.False(ParametersAreClose(a, b), "two different seeds drew identical parameters");
    }

    /// <summary>With no RandomState, Init still draws positive posteriors with the documented scaling and the uniform prior
    /// values (the actual values are random).</summary>
    [Fact]
    public void init_without_a_random_state_still_draws_valid_posteriors()
    {
        const int nc = 3;
        var model = Model(nc, null, null, "st", "st", null);
        var x = Data(5, nc);
        var lengths = new[] { 2, 3 };

        model.RunInit(x, lengths);

        AssertEach(model.StartProbPriorValues!, 1.0 / nc, "startprob_prior_");
        AssertPosteriorVector(model.StartProbPosterior, nc, lengths.Length, "startprob_posterior_");
        AssertEach(model.TransMatPriorValues!, 1.0 / nc, "transmat_prior_");
        AssertPosteriorMatrix(model.TransMatPosterior, nc, lengths.Sum() / (double)nc, "transmat_posterior_");
    }

    /// <summary>A second Init call on the same model with a fixed RandomState draws the same values again.</summary>
    [Fact]
    public void a_second_init_call_with_a_fixed_random_state_draws_the_same_values()
    {
        const int nc = 3;
        var model = Model(nc, null, null, "st", "st", 7);
        var x = Data(4, nc);
        var lengths = new[] { 2, 2 };

        model.RunInit(x, lengths);
        var firstPosterior = (double[])model.StartProbPosterior!.Clone();
        var firstTransPosterior = (double[,])model.TransMatPosterior!.Clone();

        model.RunInit(x, lengths);

        Close.Equal(firstPosterior, model.StartProbPosterior, "startprob_posterior_ on the second call");
        Close.Equal(firstTransPosterior, model.TransMatPosterior, "transmat_posterior_ on the second call");
    }

    /// <summary>With init_params "" and all four arrays set, Init leaves them exactly in place and warns about nothing.</summary>
    [Fact]
    public void init_leaves_the_explicit_parameters_alone_when_init_params_is_empty()
    {
        const int nc = 3;
        var startProbPriorValues = new[] { 0.5, 1.5, 2.0 };
        var startProbPosterior = new[] { 1.2, 0.9, 0.9 };
        var transMatPriorValues = Matrix(new[] { 0.8, 1.2, 0.9 }, new[] { 1.1, 0.7, 1.4 }, new[] { 0.6, 1.9, 0.8 });
        var transMatPosterior = Matrix(new[] { 2.5, 3.1, 0.4 }, new[] { 0.9, 4.2, 2.7 }, new[] { 3.6, 1.1, 0.7 });
        var model = Model(nc, null, null, "st", "", 42, startProbPriorValues, startProbPosterior, transMatPriorValues,
            transMatPosterior);
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.RunInit(Data(4, nc), new[] { 2, 2 });

        Assert.Same(startProbPriorValues, model.StartProbPriorValues);
        Assert.Same(startProbPosterior, model.StartProbPosterior);
        Assert.Same(transMatPriorValues, model.TransMatPriorValues);
        Assert.Same(transMatPosterior, model.TransMatPosterior);
        Assert.Empty(warnings);
    }

    /// <summary>With init_params "", a set posterior whose prior is missing still initializes that parameter (the short-circuited
    /// NeedsInit on the prior), fills the prior with 1/n_components, redraws the posterior with the documented scaling, and
    /// leaves the untouched parameter's arrays in place without warning.</summary>
    [Fact]
    public void init_initializes_when_the_posterior_is_set_but_the_prior_is_not()
    {
        const int nc = 3;
        var startProbPosterior = new[] { 1.2, 0.9, 0.9 };
        var transMatPriorValues = Matrix(new[] { 0.8, 1.2, 0.9 }, new[] { 1.1, 0.7, 1.4 }, new[] { 0.6, 1.9, 0.8 });
        var transMatPosterior = Matrix(new[] { 2.5, 3.1, 0.4 }, new[] { 0.9, 4.2, 2.7 }, new[] { 3.6, 1.1, 0.7 });
        var model = Model(nc, null, null, "st", "", 42, null, startProbPosterior, transMatPriorValues, transMatPosterior);
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.RunInit(Data(5, nc), new[] { 2, 3 });

        AssertEach(model.StartProbPriorValues!, 1.0 / nc, "startprob_prior_");
        Assert.NotSame(startProbPosterior, model.StartProbPosterior);
        AssertPosteriorVector(model.StartProbPosterior, nc, 2, "startprob_posterior_");
        Assert.Same(transMatPriorValues, model.TransMatPriorValues);
        Assert.Same(transMatPosterior, model.TransMatPosterior);
        Assert.Empty(warnings);
    }

    /// <summary>A parameter that is set and named in init_params raises the overwrite warning (once per named letter, the
    /// posterior one, since the prior check is only evaluated when the posterior check does not initialize) and the parameter
    /// is re-initialized.</summary>
    [Fact]
    public void init_warns_when_init_params_names_a_set_posterior()
    {
        const int nc = 3;
        var startProbPosterior = new[] { 1.2, 0.9, 0.9 };
        var transMatPosterior = Matrix(new[] { 2.5, 3.1, 0.4 }, new[] { 0.9, 4.2, 2.7 }, new[] { 3.6, 1.1, 0.7 });
        var model = Model(nc, null, null, "st", "st", 42, UnitVector(nc), startProbPosterior, UnitMatrix(nc),
            transMatPosterior);
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.RunInit(Data(5, nc), new[] { 2, 3 });

        Assert.Equal(new[]
        {
            "Even though the 'startprob_posterior_' attribute is set, it will be overwritten during initialization because 'init_params' contains 's'",
            "Even though the 'transmat_posterior_' attribute is set, it will be overwritten during initialization because 'init_params' contains 't'",
        }, warnings);
        Assert.NotSame(startProbPosterior, model.StartProbPosterior);
        AssertPosteriorVector(model.StartProbPosterior, nc, 2, "startprob_posterior_");
        Assert.NotSame(transMatPosterior, model.TransMatPosterior);
        AssertPosteriorMatrix(model.TransMatPosterior, nc, 5.0 / nc, "transmat_posterior_");
    }

    /// <summary>Fewer data points than free parameters (8 = 2 + 6 for a three-component "st" fit) raises the degenerate-solution
    /// warning with the exact BaseHmm message.</summary>
    [Fact]
    public void init_warns_about_a_degenerate_solution_when_there_are_fewer_data_points_than_free_parameters()
    {
        const int nc = 3;
        var model = Model(nc, null, null, "st", "", 42, UnitVector(nc), UnitVector(nc), UnitMatrix(nc), UnitMatrix(nc));
        var warnings = new List<string>();
        model.Warning += warnings.Add;
        var x = Data(2, nc);

        model.RunInit(x, new[] { 1, 1 });

        var freeScalars = (nc - 1) + nc * (nc - 1);
        var size = x.GetLength(0) * x.GetLength(1);
        Assert.Equal(new[]
        {
            $"Fitting a model with {freeScalars} free scalar parameters with only {size} data points will result in a degenerate solution.",
        }, warnings);
    }

    /// <summary>The free-scalar count only covers the letters of Params: with params "s" (2 free scalars) one data value (1 < 2) warns,
    /// and two data values (2 = 2) do not; the data have one column, so the size is the number of rows.</summary>
    [Fact]
    public void init_counts_the_free_parameters_of_the_letters_in_params_only()
    {
        const int nc = 3;
        var warnings = new List<string>();
        var smallModel = Model(nc, null, null, "s", "", 42, UnitVector(nc), UnitVector(nc), UnitMatrix(nc), UnitMatrix(nc));
        smallModel.Warning += warnings.Add;
        var small = Data(1, 1);
        smallModel.RunInit(small, new[] { 1 });
        var smallSize = small.GetLength(0) * small.GetLength(1);
        Assert.Equal(new[]
        {
            $"Fitting a model with {nc - 1} free scalar parameters with only {smallSize} data points will result in a degenerate solution.",
        }, warnings);

        var enoughWarnings = new List<string>();
        var enoughModel = Model(nc, null, null, "s", "", 42, UnitVector(nc), UnitVector(nc), UnitMatrix(nc), UnitMatrix(nc));
        enoughModel.Warning += enoughWarnings.Add;
        enoughModel.RunInit(Data(2, 1), new[] { 1, 1 });
        Assert.Empty(enoughWarnings);
    }

    /// <summary>Enough data points for the free parameters (9 >= 8) raise no degenerate-solution warning.</summary>
    [Fact]
    public void init_does_not_warn_about_degeneracy_with_enough_data_points()
    {
        const int nc = 3;
        var model = Model(nc, null, null, "st", "", 42, UnitVector(nc), UnitVector(nc), UnitMatrix(nc), UnitMatrix(nc));
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.RunInit(Data(3, nc), new[] { 1, 1, 1 });

        Assert.Empty(warnings);
    }
}
