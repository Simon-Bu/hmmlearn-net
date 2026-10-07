using System.Globalization;
using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm8;

/// <summary>
/// Acceptance tests for the fit helpers of the HMM-8 ticket (criterion C3): <see cref="SufficientStatistics"/> allocates zeroed
/// storage of the right shape; <c>NeedsInit</c> answers true exactly when the param letter is in init_params (warning when the
/// attribute is set and will be overwritten) and otherwise when the attribute is not set; <c>CheckAndSetNFeatures</c> uses the
/// column count of the first fit and throws ArgumentException on a mismatch; <c>CheckSum1</c> accepts values whose sum is 1 up to
/// the documented 1e-8 + 1e-5 tolerance and throws ArgumentException with the documented message otherwise, for vectors and for
/// matrices (rows).
/// </summary>
public class FitHelperTests
{
    private static JsonElement InferenceCase(string name) =>
        Assert.Single(Golden.Load("base").GetProperty("inference_cases").Items(), c => c.GetProperty("name").AsString() == name);

    /// <summary>A minimal model used to reach the protected fit helpers; no fit is run in these tests.</summary>
    private sealed class HelperProbe : AbstractHmm
    {
        public HelperProbe(string initParams = "st")
            : base(2, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", initParams, Implementation.Scaling)
        {
        }

        public bool ProbeNeedsInit(char code, string name, bool isSet) => NeedsInit(code, name, isSet);

        public void ProbeCheckAndSetNFeatures(double[,] x) => CheckAndSetNFeatures(x);

        public static void ProbeCheckSum1(string name, double[] values) => CheckSum1(name, values);

        public static void ProbeCheckSum1(string name, double[,] values) => CheckSum1(name, values);

        protected override void Check()
        {
        }

        protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

        protected override double[] GenerateSampleFromState(int state, Random random) => new[] { (double)state };
    }

    /// <summary>SufficientStatistics allocates zeroed storage: NObs 0, Start of n zeros, Trans n x n zeros.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void sufficient_statistics_allocates_zeroed_storage(int n)
    {
        var stats = new SufficientStatistics(n);
        Assert.Equal(0, stats.NObs);
        Assert.Equal(n, stats.Start.Length);
        Assert.Equal(n, stats.Trans.GetLength(0));
        Assert.Equal(n, stats.Trans.GetLength(1));
        foreach (var value in stats.Start)
            Assert.Equal(0.0, value);
        for (var r = 0; r < n; r++)
            for (var c = 0; c < n; c++)
                Assert.Equal(0.0, stats.Trans[r, c]);
    }

    /// <summary>NeedsInit is true when the param letter is in init_params, and warns that the set attribute will be overwritten.</summary>
    [Fact]
    public void needs_init_is_true_when_the_param_is_in_init_params_even_if_the_attribute_is_set()
    {
        var warnings = new List<string>();
        var model = new HelperProbe("st");
        model.Warning += warnings.Add;
        var result = model.ProbeNeedsInit('s', "startprob_", true);
        Assert.True(result);
        Assert.Equal(
            new[] { "Even though the 'startprob_' attribute is set, it will be overwritten during initialization because 'init_params' contains 's'" },
            warnings);
    }

    /// <summary>NeedsInit is true when the param letter is in init_params and the attribute is not set (no warning).</summary>
    [Fact]
    public void needs_init_is_true_when_the_param_is_in_init_params_and_the_attribute_is_not_set()
    {
        var warnings = new List<string>();
        var model = new HelperProbe("st");
        model.Warning += warnings.Add;
        Assert.True(model.ProbeNeedsInit('s', "startprob_", false));
        Assert.Empty(warnings);
    }

    /// <summary>When the param letter is not in init_params, NeedsInit depends only on whether the attribute is set, with no
    /// warning.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void needs_init_depends_only_on_is_set_when_the_param_is_not_in_init_params(bool isSet)
    {
        var warnings = new List<string>();
        var model = new HelperProbe("st");
        model.Warning += warnings.Add;
        Assert.Equal(!isSet, model.ProbeNeedsInit('v', "transmat_", isSet));
        Assert.Empty(warnings);
    }

    /// <summary>CheckAndSetNFeatures sets NFeatures to the column count of the first fit and accepts later fits with the same
    /// column count.</summary>
    [Fact]
    public void check_and_set_n_features_uses_the_columns_of_the_first_fit_and_accepts_consistent_ones()
    {
        var model = new HelperProbe();
        model.ProbeCheckAndSetNFeatures(new double[3, 2]);
        Assert.Equal(2, (int?)model.NFeatures);
        model.ProbeCheckAndSetNFeatures(new double[4, 2]);
        Assert.Equal(2, (int?)model.NFeatures);
    }

    /// <summary>CheckAndSetNFeatures throws ArgumentException when the column count does not match, whether NFeatures was set by
    /// an earlier call or preset.</summary>
    [Fact]
    public void check_and_set_n_features_throws_when_the_column_count_does_not_match()
    {
        var model = new HelperProbe();
        model.ProbeCheckAndSetNFeatures(new double[3, 2]);
        var ex = Assert.Throws<ArgumentException>(() => model.ProbeCheckAndSetNFeatures(new double[3, 3]));
        Assert.Equal("Unexpected number of dimensions, got 3 but expected 2", ex.Message);

        var preset = new HelperProbe();
        preset.NFeatures = 2;
        Assert.Throws<ArgumentException>(() => preset.ProbeCheckAndSetNFeatures(new double[3, 3]));
    }

    /// <summary>CheckSum1 accepts vectors whose sum is 1, including a sum within the documented 1e-8 + 1e-5 tolerance.</summary>
    [Fact]
    public void check_sum1_accepts_vectors_that_sum_to_one()
    {
        var tol = 1e-8 + 1e-5;
        var cases = new[]
        {
            new[] { 0.5, 0.5 },
            new[] { 1.0 / 3, 1.0 / 3, 1.0 / 3 },
            new[] { 1.0, tol / 2 },
        };
        foreach (var values in cases)
            HelperProbe.ProbeCheckSum1("startprob", values);
    }

    /// <summary>CheckSum1 throws ArgumentException with the documented message for vectors whose sum is not 1 beyond the
    /// documented 1e-8 + 1e-5 tolerance.</summary>
    [Fact]
    public void check_sum1_rejects_vectors_that_do_not_sum_to_one()
    {
        var tol = 1e-8 + 1e-5;
        var ex = Assert.Throws<ArgumentException>(() => HelperProbe.ProbeCheckSum1("startprob", new[] { 0.5, 0.5, 0.5 }));
        // The stub documents the message as "{name} must sum to 1 (got {sum:F4})". The codebase formats user-facing messages in
        // the invariant culture (ConvergenceMonitor); either culture's decimal separator is accepted here.
        var sum = 0.5 + 0.5 + 0.5;
        var invariant = sum.ToString("F4", CultureInfo.InvariantCulture);
        var local = sum.ToString("F4");
        Assert.Contains(ex.Message, new[] { $"startprob must sum to 1 (got {invariant})", $"startprob must sum to 1 (got {local})" });

        Assert.Throws<ArgumentException>(() => HelperProbe.ProbeCheckSum1("startprob", new[] { 0.0, 0.0 }));
        Assert.Throws<ArgumentException>(() => HelperProbe.ProbeCheckSum1("startprob", new[] { 1.0, 2 * tol }));
    }

    /// <summary>CheckSum1 accepts matrices whose rows sum to 1.</summary>
    [Fact]
    public void check_sum1_accepts_matrices_whose_rows_sum_to_one()
    {
        var c = InferenceCase("wikipedia");
        HelperProbe.ProbeCheckSum1("transmat", c.GetProperty("transmat").AsMatrix());
        var identity = new double[2, 2]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 },
        };
        HelperProbe.ProbeCheckSum1("identity", identity);
    }

    /// <summary>CheckSum1 throws ArgumentException for matrices with a row that does not sum to 1.</summary>
    [Fact]
    public void check_sum1_rejects_matrices_with_a_row_that_does_not_sum_to_one()
    {
        var bad = new double[2, 2]
        {
            { 0.5, 0.5 },
            { 0.4, 0.4 },
        };
        var ex = Assert.Throws<ArgumentException>(() => HelperProbe.ProbeCheckSum1("transmat", bad));
        Assert.StartsWith("transmat rows must sum to 1 (got row sums of ", ex.Message);

        var zeroRow = new double[2, 2]
        {
            { 0.0, 0.0 },
            { 0.5, 0.5 },
        };
        Assert.Throws<ArgumentException>(() => HelperProbe.ProbeCheckSum1("transmat", zeroRow));
    }
}
