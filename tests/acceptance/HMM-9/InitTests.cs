using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm9;

/// <summary>
/// Acceptance tests for the initialization of <see cref="BaseHmm"/> (ticket HMM-9, criterion C3): Init draws the start
/// probabilities and each transition row from a Dirichlet distribution with every parameter 1 / n_components, only for the
/// parameters that must be initialized (not set yet, or named by InitParams), reproducibly for a given RandomState (a new
/// generator is created at each call, so a second Init call redrews the same values), leaves the explicit parameters alone
/// (same reference) when InitParams does not name them, records the number of features of x, raises the exact
/// "will be overwritten" warnings for the set parameters named by InitParams, and raises the degenerate-solution warning
/// when there are fewer data points than free scalar parameters of the letters of Params (not when there are as many, not
/// when the number of free scalars is unknown).
/// </summary>
public class InitTests
{
    /// <summary>A stub with the given initialization settings, ready for a CallInit call.</summary>
    private static StubBaseHmm Model(int nComponents, string initParams, string @params, int? randomState,
        double[]? startProb = null, double[,]? transMat = null) =>
        new(nComponents, 1.0, 1.0, DecoderAlgorithm.Viterbi, randomState, 0, 1e-2, false, @params, initParams, Implementation.Log)
        { StartProb = startProb, TransMat = transMat };

    /// <summary>Arbitrary log frame probability matrix of the given shape; Init only uses the number of columns of x.</summary>
    private static double[,] Data(int rows, int columns)
    {
        var x = new double[rows, columns];
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < columns; c++)
                x[r, c] = -0.5;
        return x;
    }

    private static void AssertValidDistribution(double[]? vector, string what)
    {
        Assert.NotNull(vector);
        Close.Equal(1.0, vector!.Sum(), what + " sum");
        for (var i = 0; i < vector.Length; i++)
            Assert.InRange(vector[i], 0.0, 1.0);
    }

    private static void AssertValidStochasticMatrix(double[,]? matrix, string what)
    {
        Assert.NotNull(matrix);
        var m = matrix!;
        for (var r = 0; r < m.GetLength(0); r++)
        {
            var sum = 0.0;
            for (var c = 0; c < m.GetLength(1); c++)
            {
                sum += m[r, c];
                Assert.InRange(m[r, c], 0.0, 1.0);
            }
            Close.Equal(1.0, sum, $"{what} row {r} sum");
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

    [Fact]
    public void init_draws_valid_distributions_and_records_the_features_when_nothing_is_set()
    {
        var model = Model(3, "", "st", 42);
        model.CallInit(Data(4, 3), new[] { 2, 2 });

        AssertValidDistribution(model.StartProb, "startprob");
        AssertValidStochasticMatrix(model.TransMat, "transmat");
        Assert.Equal(3, model.NFeatures);
    }

    [Fact]
    public void init_with_a_fixed_random_state_is_reproducible()
    {
        var first = Model(3, "st", "st", 42);
        var second = Model(3, "st", "st", 42);
        first.CallInit(Data(4, 3), new[] { 2, 2 });
        second.CallInit(Data(4, 3), new[] { 2, 2 });

        Close.Equal(first.StartProb!, second.StartProb!, "startprob");
        Close.Equal(first.TransMat!, second.TransMat!, "transmat");
    }

    [Fact]
    public void init_with_different_seeds_draws_different_parameters()
    {
        var first = Model(3, "st", "st", 42);
        var second = Model(3, "st", "st", 1337);
        first.CallInit(Data(4, 3), new[] { 2, 2 });
        second.CallInit(Data(4, 3), new[] { 2, 2 });

        Assert.False(VectorsAreClose(first.StartProb!, second.StartProb!) && MatricesAreClose(first.TransMat!, second.TransMat!),
            "two different seeds drew the same parameters");
    }

    [Fact]
    public void init_without_a_random_state_still_draws_valid_distributions()
    {
        var model = Model(3, "st", "st", null);
        model.CallInit(Data(4, 3), new[] { 2, 2 });

        AssertValidDistribution(model.StartProb, "startprob");
        AssertValidStochasticMatrix(model.TransMat, "transmat");
    }

    [Fact]
    public void init_leaves_explicit_parameters_alone_when_init_params_does_not_name_them()
    {
        var startProb = new[] { 0.7, 0.3 };
        var transMat = new double[,] { { 0.8, 0.2 }, { 0.4, 0.6 } };
        var model = Model(2, "", "st", 42, startProb, transMat);

        model.CallInit(Data(4, 2), new[] { 2, 2 });

        Assert.Same(startProb, model.StartProb);
        Assert.Same(transMat, model.TransMat);
    }

    [Fact]
    public void init_draws_only_the_parameters_named_by_init_params_when_the_others_are_set()
    {
        // 't' only: the start probabilities stay, the transition matrix is drawn
        var startProb = new[] { 0.9, 0.1 };
        var model = Model(2, "t", "st", 42, startProb, null);
        model.CallInit(Data(4, 2), new[] { 2, 2 });
        Assert.Same(startProb, model.StartProb);
        AssertValidStochasticMatrix(model.TransMat, "transmat");

        // 's' only: the transition matrix stays, the start probabilities are drawn
        var transMat = new double[,] { { 0.8, 0.2 }, { 0.4, 0.6 } };
        var other = Model(2, "s", "st", 42, null, transMat);
        other.CallInit(Data(4, 2), new[] { 2, 2 });
        Assert.Same(transMat, other.TransMat);
        AssertValidDistribution(other.StartProb, "startprob");
    }

    [Fact]
    public void init_warns_and_overwrites_the_set_parameters_named_by_init_params()
    {
        var startProb = new[] { 0.9, 0.1 };
        var transMat = new double[,] { { 0.8, 0.2 }, { 0.4, 0.6 } };
        var model = Model(2, "st", "st", 42, startProb, transMat);
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.CallInit(Data(4, 2), new[] { 2, 2 });

        Assert.Equal(new[]
        {
            "Even though the 'startprob_' attribute is set, it will be overwritten during initialization because 'init_params' contains 's'",
            "Even though the 'transmat_' attribute is set, it will be overwritten during initialization because 'init_params' contains 't'",
        }, warnings);
        AssertValidDistribution(model.StartProb, "startprob");
        AssertValidStochasticMatrix(model.TransMat, "transmat");
        Assert.False(VectorsAreClose(startProb, model.StartProb!), "the start probabilities were not overwritten");
    }

    [Fact]
    public void init_warns_about_a_degenerate_solution_when_there_are_fewer_data_points_than_free_parameters()
    {
        var nc = 3;
        var x = Data(2, 3);
        var model = Model(nc, "st", "st", 42);
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.CallInit(x, new[] { 1, 1 });

        var freeParameters = (nc - 1) + nc * (nc - 1); // the s and t entries of the stub
        var size = x.GetLength(0) * x.GetLength(1);
        Assert.Equal(new[]
        {
            $"Fitting a model with {freeParameters} free scalar parameters with only {size} data points will result in a degenerate solution.",
        }, warnings);
    }

    [Fact]
    public void init_counts_the_free_parameters_of_the_letters_in_params_only()
    {
        var nc = 3;
        // params = "s": only the start probabilities are free (nc - 1 parameters)
        var few = Data(1, 1);
        var modelFew = Model(nc, "s", "s", 42);
        var warningsFew = new List<string>();
        modelFew.Warning += warningsFew.Add;
        modelFew.CallInit(few, new[] { 1 });
        Assert.Equal(new[]
        {
            $"Fitting a model with {nc - 1} free scalar parameters with only {few.GetLength(0) * few.GetLength(1)} data points will result in a degenerate solution.",
        }, warningsFew);

        // as many data points as free parameters: no warning
        var exactly = Data(2, 1);
        var modelExactly = Model(nc, "s", "s", 42);
        var warningsExactly = new List<string>();
        modelExactly.Warning += warningsExactly.Add;
        modelExactly.CallInit(exactly, new[] { 1, 1 });
        Assert.Empty(warningsExactly);
    }

    [Fact]
    public void init_does_not_warn_about_degeneracy_when_the_free_scalars_are_unknown()
    {
        var model = new NullScalarsStub();
        var warnings = new List<string>();
        model.Warning += warnings.Add;

        model.CallInit(Data(1, 1), new[] { 1 });

        Assert.Empty(warnings);
    }

    [Fact]
    public void a_second_init_call_with_a_fixed_random_state_draws_the_same_values()
    {
        var model = Model(3, "st", "st", 7);
        var x = Data(4, 3);
        model.CallInit(x, new[] { 2, 2 });
        var startProb = (double[])model.StartProb!.Clone();
        var transMat = (double[,])model.TransMat!.Clone();

        model.CallInit(x, new[] { 2, 2 });

        Close.Equal(startProb, model.StartProb, "startprob");
        Close.Equal(transMat, model.TransMat, "transmat");
    }
}
