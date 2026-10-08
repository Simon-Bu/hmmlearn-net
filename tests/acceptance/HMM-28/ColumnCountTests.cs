using HmmLearn;

namespace HmmLearn.Tests.Acceptance.Hmm28;

/// <summary>
/// Acceptance tests for ticket HMM-28 (criterion C1): the inference entry points (Score, ScoreSamples, Decode, Predict,
/// PredictProba) reject an observation matrix whose number of columns differs from what the model expects with an
/// <see cref="ArgumentException"/> (exact type), and still accept the right number of columns. Hand-set models, InitParams "".
/// </summary>
public class ColumnCountTests
{
    private static readonly double[] Start = { 0.6, 0.4 };
    private static readonly double[,] Trans = { { 0.7, 0.3 }, { 0.4, 0.6 } };

    /// <summary>A matrix of two rows and <paramref name="columns"/> columns of small non-negative integers.</summary>
    private static double[,] X(int columns)
    {
        var x = new double[2, columns];
        for (var i = 0; i < 2; i++)
            for (var j = 0; j < columns; j++)
                x[i, j] = (i + j) % 2;
        return x;
    }

    private static T Set<T>(T model) where T : AbstractHmm
    {
        model.StartProb = Start;
        model.TransMat = Trans;
        return model;
    }

    private static PoissonHmm Poisson() => Set(new PoissonHmm(
        2, 1.0, 1.0, 0.0, 0.0, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "stl", "", Implementation.Log))
        .With(m => m.Lambdas = new double[,] { { 3.1, 1.4, 4.5 }, { 1.6, 5.3, 0.1 } });

    private static MultinomialHmm Multinomial() => Set(new MultinomialHmm(
        2, null, 1.0, 1.0, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "ste", "", Implementation.Log))
        .With(m => m.EmissionProb = new double[,] { { 0.5, 0.3, 0.2 }, { 0.1, 0.4, 0.5 } });

    private static CategoricalHmm Categorical() => Set(new CategoricalHmm(
        2, 1.0, 1.0, 1.0, 3, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "ste", "", Implementation.Log))
        .With(m => m.EmissionProb = new double[,] { { 0.5, 0.3, 0.2 }, { 0.1, 0.4, 0.5 } });

    private static VariationalCategoricalHmm VariationalCategorical() => Set(new VariationalCategoricalHmm(
        nComponents: 2, nFeatures: 3, initParams: ""))
        .With(m =>
        {
            m.StartProbPriorValues = new[] { 1.0, 1.0 };
            m.StartProbPosterior = new[] { 1.5, 2.5 };
            m.TransMatPriorValues = new double[,] { { 1.0, 1.0 }, { 1.0, 1.0 } };
            m.TransMatPosterior = new double[,] { { 2.0, 3.0 }, { 1.0, 4.0 } };
            m.EmissionProbPriorValues = new double[,] { { 1.0, 1.0, 1.0 }, { 2.0, 2.0, 2.0 } };
            m.EmissionProbPosterior = new double[,] { { 2.0, 1.0, 1.0 }, { 1.0, 3.0, 2.0 } };
            m.EmissionProb = new double[,] { { 0.5, 0.25, 0.25 }, { 1.0 / 6, 0.5, 1.0 / 3 } };
        });

    private static GaussianHmm Gaussian() => Set(new GaussianHmm(
        2, CovarianceType.Diag, 1e-3, 1.0, 1.0, 0, 0, 1e-2, 1,
        DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "stmc", "", Implementation.Log))
        .With(m =>
        {
            m.Means = new double[,] { { 0.0, 1.0, 2.0 }, { 1.0, 0.0, -1.0 } };
            m.Covars = new double[,,] { { { 1.0, 2.0, 1.5 } }, { { 0.5, 1.0, 1.0 } } };
        });

    private static GmmHmm Gmm() => Set(new GmmHmm(
        nComponents: 2, nMix: 2, covarianceType: CovarianceType.Diag, initParams: "", implementation: Implementation.Log))
        .With(m =>
        {
            m.Weights = new double[,] { { 0.5, 0.5 }, { 0.3, 0.7 } };
            m.Means = new double[,,] { { { 0, 1, 2 }, { 1, 1, 1 } }, { { 1, 0, -1 }, { 0, 0, 0 } } };
            m.Covars = new double[,,,] { { { { 1, 2, 1.5 } }, { { 1, 1, 1 } } }, { { { 0.5, 1, 1 } }, { { 2, 2, 2 } } } };
            m.NFeatures = 3;
        });

    /// <summary>The right number of columns gives a finite score; every wrong one throws from the five entry points.</summary>
    private static void AssertColumnCountChecked(AbstractHmm model, int rightColumns, params int[] wrongColumns)
    {
        var score = model.Score(X(rightColumns));
        Assert.True(double.IsFinite(score), $"control with {rightColumns} column(s): the score must be finite, got {score}");

        foreach (var columns in wrongColumns)
        {
            var x = X(columns);
            Assert.Throws<ArgumentException>(() => model.Score(x));
            Assert.Throws<ArgumentException>(() => model.ScoreSamples(x));
            Assert.Throws<ArgumentException>(() => model.Decode(x));
            Assert.Throws<ArgumentException>(() => model.Predict(x));
            Assert.Throws<ArgumentException>(() => model.PredictProba(x));
        }
    }

    [Fact]
    public void poisson_rejects_a_wrong_number_of_columns() => AssertColumnCountChecked(Poisson(), 3, 2, 4);

    [Fact]
    public void multinomial_rejects_a_wrong_number_of_columns() => AssertColumnCountChecked(Multinomial(), 3, 2, 4);

    [Fact]
    public void categorical_rejects_two_columns() => AssertColumnCountChecked(Categorical(), 1, 2);

    [Fact]
    public void variational_categorical_rejects_two_columns() => AssertColumnCountChecked(VariationalCategorical(), 1, 2);

    [Fact]
    public void gaussian_rejects_a_wrong_number_of_columns() => AssertColumnCountChecked(Gaussian(), 3, 2, 4);

    [Fact]
    public void gmm_rejects_a_wrong_number_of_columns() => AssertColumnCountChecked(Gmm(), 3, 2, 4);

    [Fact]
    public void the_message_gives_both_column_counts()
    {
        var ex = Assert.Throws<ArgumentException>(() => Poisson().Score(X(4)));

        Assert.Equal("x", ex.ParamName);
        Assert.Contains("4", ex.Message);
        Assert.Contains("3", ex.Message);
    }
}

internal static class ModelExtensions
{
    /// <summary>Applies <paramref name="setup"/> to the model and returns it.</summary>
    public static T With<T>(this T model, Action<T> setup)
    {
        setup(model);
        return model;
    }
}
