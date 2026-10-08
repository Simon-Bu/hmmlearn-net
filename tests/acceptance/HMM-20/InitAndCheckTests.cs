using System.Collections.Generic;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm20;

/// <summary>
/// Acceptance tests for the initialization and validation of <see cref="VariationalCategoricalHmm"/> (ticket HMM-20, criterion
/// C2): Init fills the emission prior with the scalar prior or 1 / NFeatures and draws an emission posterior whose rows sum to
/// (sum of lengths) / NComponents, reproducibly for a given RandomState and only when 'e' must be initialized (with the
/// "will be overwritten" warning otherwise); Check rejects emission arrays that are null, of different shapes, or not
/// (NComponents x NFeatures); NFeatures is inferred from the largest symbol or respected when given; fractional, negative and
/// oversized symbols are rejected; and the three scalar constructor priors end up in the three prior arrays.
/// </summary>
public class InitAndCheckTests
{
    private static ExposedModel Model(int nComponents = 3, int? nFeatures = null, double? emissionProbPrior = null,
        string initParams = "ste", string @params = "ste", int? randomState = 42,
        double? startProbPrior = null, double? transMatPrior = null) =>
        new(nComponents, startProbPrior, transMatPrior, emissionProbPrior, nFeatures, DecoderAlgorithm.Viterbi,
            randomState, 10, 1e-6, false, @params, initParams, Implementation.Log);

    /// <summary>Symbols into an (n, 1) column of observations.</summary>
    private static double[,] Symbols(params double[] symbols)
    {
        var x = new double[symbols.Length, 1];
        for (var i = 0; i < symbols.Length; i++)
            x[i, 0] = symbols[i];
        return x;
    }

    private static double[,] Fill(int rows, int cols, double value)
    {
        var m = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                m[i, j] = value;
        return m;
    }

    private static double[] Fill1(int count, double value)
    {
        var v = new double[count];
        for (var i = 0; i < count; i++)
            v[i] = value;
        return v;
    }

    private static double RowSum(double[,] m, int row)
    {
        var sum = 0.0;
        for (var j = 0; j < m.GetLength(1); j++)
            sum += m[row, j];
        return sum;
    }

    /// <summary>A model whose start and transition arrays are Check-valid, so that <c>RunCheck</c> reaches the emission checks;
    /// the emission arrays are whatever the test gives.</summary>
    private static ExposedModel CheckReady(int nComponents, int? nFeatures, double[,]? emissionPrior, double[,]? emissionPosterior)
    {
        var model = Model(nComponents, nFeatures, null, "");
        model.StartProbPriorValues = Fill1(nComponents, 1.0);
        model.StartProbPosterior = Fill1(nComponents, 2.0);
        model.TransMatPriorValues = Fill(nComponents, nComponents, 1.0);
        model.TransMatPosterior = Fill(nComponents, nComponents, 2.0);
        model.EmissionProbPriorValues = emissionPrior;
        model.EmissionProbPosterior = emissionPosterior;
        return model;
    }

    /// <summary>Init fills EmissionProbPriorValues with the scalar prior given to the constructor (every entry, shape
    /// NComponents x NFeatures).</summary>
    [Fact]
    public void init_fills_the_emission_prior_with_the_scalar_prior()
    {
        var model = Model(2, 3, 0.25);
        model.RunInit(Symbols(0, 1, 2, 1, 0), new[] { 2, 3 });

        Assert.NotNull(model.EmissionProbPriorValues);
        Close.Equal(Fill(2, 3, 0.25), model.EmissionProbPriorValues!, "the emission prior from the scalar prior 0.25");
    }

    /// <summary>With no scalar prior, the emission prior is 1 / NFeatures, with NFeatures inferred from the largest symbol when it
    /// was not given to the constructor.</summary>
    [Fact]
    public void init_fills_the_emission_prior_with_one_over_nfeatures_when_the_scalar_prior_is_null()
    {
        var model = Model(2, null, null);
        model.RunInit(Symbols(0, 3, 1, 2), new[] { 2, 2 }); // largest symbol 3: NFeatures becomes 4

        Assert.Equal(4, model.NFeatures);
        Assert.NotNull(model.EmissionProbPriorValues);
        Assert.Equal(4, model.EmissionProbPriorValues!.GetLength(1));
        Close.Equal(Fill(2, 4, 1.0 / 4), model.EmissionProbPriorValues, "the emission prior 1 / NFeatures");
    }

    /// <summary>Init draws one Dirichlet row per state, multiplied by (sum of lengths) / NComponents: every row of the emission
    /// posterior sums to that scale, with positive finite entries.</summary>
    [Fact]
    public void init_emission_posterior_rows_sum_to_total_length_over_ncomponents()
    {
        var model = Model(3, 2, null, randomState: 7);
        model.RunInit(Symbols(0, 1, 0, 1, 0), new[] { 2, 3 });

        Assert.NotNull(model.EmissionProbPosterior);
        var posterior = model.EmissionProbPosterior!;
        Assert.Equal(3, posterior.GetLength(0));
        Assert.Equal(2, posterior.GetLength(1));
        var scale = 5.0 / 3.0; // (sum of lengths) / NComponents
        for (var c = 0; c < posterior.GetLength(0); c++)
        {
            Close.Equal(scale, RowSum(posterior, c), $"row {c} of the emission posterior must sum to (sum of lengths) / NComponents");
            for (var j = 0; j < posterior.GetLength(1); j++)
                Assert.True(posterior[c, j] > 0.0 && !double.IsNaN(posterior[c, j]),
                    $"the entry [{c}, {j}] of the emission posterior must be positive and finite");
        }
    }

    /// <summary>The random initialization is reproducible for a given RandomState (start, transition and emission posteriors are
    /// all the same) and different for another seed.</summary>
    [Fact]
    public void init_is_reproducible_for_a_given_random_state()
    {
        var x = Symbols(0, 1, 2, 1, 0, 2);
        var lengths = new[] { 3, 3 };

        var first = Model(2, 3, null, randomState: 123);
        first.RunInit(x, lengths);
        var second = Model(2, 3, null, randomState: 123);
        second.RunInit(x, lengths);
        var other = Model(2, 3, null, randomState: 456);
        other.RunInit(x, lengths);

        Assert.NotNull(first.EmissionProbPosterior);
        Close.Equal(first.EmissionProbPosterior!, second.EmissionProbPosterior!, "the emission posterior of two runs with seed 123");
        Close.Equal(first.StartProbPosterior!, second.StartProbPosterior!, "the startprob posterior of two runs with seed 123");
        Close.Equal(first.TransMatPosterior!, second.TransMatPosterior!, "the transmat posterior of two runs with seed 123");

        Assert.False(MatricesAreClose(first.EmissionProbPosterior!, other.EmissionProbPosterior!),
            "a different seed must change the emission posterior");
    }

    private static bool MatricesAreClose(double[,] a, double[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            return false;
        for (var i = 0; i < a.GetLength(0); i++)
            for (var j = 0; j < a.GetLength(1); j++)
                if (!Close.IsClose(a[i, j], b[i, j]))
                    return false;
        return true;
    }

    /// <summary>When 'e' is not in InitParams and the emission posterior is already set, Init leaves the emission arrays alone
    /// (same reference, prior values still null).</summary>
    [Fact]
    public void init_does_not_touch_the_emission_arrays_when_e_does_not_need_init()
    {
        var known = new double[,] { { 0.3, 0.4, 0.3 }, { 0.8, 0.1, 0.1 } };
        var model = Model(2, 3, null, initParams: "st");
        model.EmissionProbPosterior = known;
        model.RunInit(Symbols(0, 1, 0, 1), new[] { 2, 2 });

        Assert.Same(known, model.EmissionProbPosterior);
        Assert.Null(model.EmissionProbPriorValues);
    }

    /// <summary>When 'e' is in InitParams, Init overwrites a set emission posterior (new array) and raises the "will be
    /// overwritten" warning.</summary>
    [Fact]
    public void init_overwrites_a_set_emission_posterior_when_e_is_in_init_params()
    {
        var known = new double[,] { { 0.3, 0.4, 0.3 }, { 0.8, 0.1, 0.1 } };
        var model = Model(2, 3, null, initParams: "ste");
        model.EmissionProbPosterior = known;
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.RunInit(Symbols(0, 1, 0, 1), new[] { 2, 2 });

        Assert.NotSame(known, model.EmissionProbPosterior);
        Assert.Contains("Even though the 'emissionprob_posterior_' attribute is set, it will be overwritten during initialization because 'init_params' contains 'e'",
            warnings);
    }

    /// <summary>Check rejects a null emission prior or a null emission posterior with ArgumentException.</summary>
    [Fact]
    public void check_rejects_null_emission_prior_or_posterior()
    {
        Assert.Throws<ArgumentException>(() => CheckReady(2, 3, null, null).RunCheck());
        var model = CheckReady(2, 3, Fill(2, 3, 1.0), null);
        Assert.Throws<ArgumentException>(() => model.RunCheck());
    }

    /// <summary>Check rejects emission prior and posterior of different shapes with ArgumentException.</summary>
    [Fact]
    public void check_rejects_mismatched_prior_and_posterior_shapes()
    {
        Assert.Throws<ArgumentException>(() => CheckReady(2, 3, Fill(2, 3, 1.0), Fill(2, 4, 2.0)).RunCheck());
        Assert.Throws<ArgumentException>(() => CheckReady(2, 3, Fill(2, 4, 1.0), Fill(2, 3, 2.0)).RunCheck());
    }

    /// <summary>Check rejects an emission posterior that is not (NComponents x NFeatures): too many columns for the given
    /// NFeatures, or too many rows for the number of components.</summary>
    [Fact]
    public void check_rejects_an_emission_posterior_that_is_not_ncomponents_by_nfeatures()
    {
        // NFeatures 3 (constructor): a matching (2, 4) pair passes the shape-mismatch check but not the (n_components, n_features)
        // check.
        Assert.Throws<ArgumentException>(() => CheckReady(2, 3, Fill(2, 4, 1.0), Fill(2, 4, 2.0)).RunCheck());
        // 3 rows for 2 components.
        Assert.Throws<ArgumentException>(() => CheckReady(2, 3, Fill(3, 3, 1.0), Fill(3, 3, 2.0)).RunCheck());
    }

    /// <summary>With NFeatures null, Check sets it to the number of columns of the emission posterior.</summary>
    [Fact]
    public void check_sets_nfeatures_to_the_number_of_posterior_columns_when_it_is_null()
    {
        var model = CheckReady(2, null, Fill(2, 5, 1.0), Fill(2, 5, 2.0));
        model.RunCheck();

        Assert.Equal(5, model.NFeatures);
    }

    /// <summary>NFeatures is inferred from the largest symbol of the data when not given to the constructor.</summary>
    [Fact]
    public void nfeatures_is_inferred_from_the_largest_symbol()
    {
        var model = Model(2, null, null);
        model.RunInit(Symbols(0, 2, 4, 1), new[] { 2, 2 });

        Assert.Equal(5, model.NFeatures);
    }

    /// <summary>NFeatures given to the constructor is respected (not changed by a smaller data maximum).</summary>
    [Fact]
    public void nfeatures_is_respected_when_given()
    {
        var model = Model(2, 3, null);
        model.RunInit(Symbols(0, 1, 2, 1), new[] { 2, 2 });

        Assert.Equal(3, model.NFeatures);
        Assert.NotNull(model.EmissionProbPriorValues);
        Assert.Equal(3, model.EmissionProbPriorValues!.GetLength(1));
    }

    /// <summary>The symbols of the data are validated: fractional, negative and (when NFeatures is given) larger than
    /// NFeatures - 1 are rejected with ArgumentException.</summary>
    [Fact]
    public void check_and_set_rejects_fractional_negative_or_oversized_symbols()
    {
        var fractional = Assert.Throws<ArgumentException>(() => Model(2, null, null).CheckAndSet(Symbols(0, 1.5)));
        Assert.Contains("integers", fractional.Message);

        var negative = Assert.Throws<ArgumentException>(() => Model(2, null, null).CheckAndSet(Symbols(0, -1)));
        Assert.Contains("nonnegative", negative.Message);

        var oversized = Assert.Throws<ArgumentException>(() => Model(2, 3, null).CheckAndSet(Symbols(0, 3)));
        Assert.Contains("Largest symbol is 3 but the model only emits symbols up to 2", oversized.Message);
    }

    /// <summary>The scalar priors 1, 2 and 3 given to the constructor end up in the three prior arrays after Init (startprob of
    /// length NComponents, transmat NComponents x NComponents, emission NComponents x NFeatures).</summary>
    [Fact]
    public void constructor_scalar_priors_fill_the_three_prior_arrays()
    {
        var model = Model(3, 4, 3.0, startProbPrior: 1.0, transMatPrior: 2.0);
        model.RunInit(Symbols(0, 1, 2, 3, 1), new[] { 3, 2 });

        Assert.NotNull(model.StartProbPriorValues);
        Close.Equal(Fill1(3, 1.0), model.StartProbPriorValues!, "the startprob prior from the scalar 1");
        Close.Equal(Fill(3, 3, 2.0), model.TransMatPriorValues!, "the transmat prior from the scalar 2");
        Close.Equal(Fill(3, 4, 3.0), model.EmissionProbPriorValues!, "the emission prior from the scalar 3");
    }

    /// <summary>GetNFitScalarsPerParam is null until NFeatures is known, then the categorical scalars: s = NComponents - 1,
    /// t = NComponents x (NComponents - 1), e = NComponents x (NFeatures - 1).</summary>
    [Fact]
    public void nfit_scalars_are_null_before_nfeatures_is_known_and_exact_after()
    {
        var model = Model(3, null, null);
        Assert.Null(model.NFitScalars());

        model.RunInit(Symbols(0, 1, 2, 3, 4, 2), new[] { 3, 3 }); // largest symbol 4: NFeatures 5
        var scalars = model.NFitScalars();
        Assert.NotNull(scalars);
        Assert.Equal(2, scalars!['s']);
        Assert.Equal(6, scalars!['t']);
        Assert.Equal(12, scalars!['e']);
    }
}
