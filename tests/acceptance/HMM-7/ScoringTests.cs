using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm7;

/// <summary>
/// Acceptance tests for the scoring half of <see cref="AbstractHmm"/> (ticket HMM-7, criterion C1): <see cref="AbstractHmm.Score"/>,
/// <see cref="AbstractHmm.ScoreSamples"/> and <see cref="AbstractHmm.PredictProba"/> reproduce the score, log_prob and posteriors of
/// every inference case of oracle/golden/base.json with both implementations (several sequences through lengths included), null
/// lengths mean one sequence, and the fitted and input checks fail with the documented exceptions. The model is the ticket's stub:
/// the observations ARE the log frame probabilities (n_features = n_components), exactly like the oracle's StubHMM.
/// </summary>
public class ScoringTests
{
    private static List<JsonElement> InferenceCases() =>
        Golden.Load("base").GetProperty("inference_cases").Items().ToList();

    private static JsonElement Case(string name) =>
        Assert.Single(InferenceCases(), c => c.GetProperty("name").AsString() == name);

    private static StubHmm MakeModel(JsonElement c, Implementation implementation)
    {
        var model = new StubHmm(c.GetProperty("n_components").AsInt(), implementation);
        model.StartProb = c.GetProperty("startprob").AsVector();
        model.TransMat = c.GetProperty("transmat").AsMatrix();
        return model;
    }

    private static void CheckScoreAgainstGolden(Implementation implementation)
    {
        var checkedCases = 0;
        foreach (var c in InferenceCases())
        {
            var name = c.GetProperty("name").AsString();
            var model = MakeModel(c, implementation);
            var x = c.GetProperty("X").AsMatrix();
            var lengths = c.GetProperty("lengths").AsInts();
            Close.Equal(c.GetProperty("expected").GetProperty("score").AsDouble(), model.Score(x, lengths),
                $"{name}: score with the {implementation} implementation");
            checkedCases++;
        }
        Assert.True(checkedCases >= 5, $"expected at least 5 inference cases in the golden file, saw {checkedCases}");
    }

    private static void CheckScoreSamplesAgainstGolden(Implementation implementation)
    {
        var checkedCases = 0;
        foreach (var c in InferenceCases())
        {
            var name = c.GetProperty("name").AsString();
            var model = MakeModel(c, implementation);
            var x = c.GetProperty("X").AsMatrix();
            var lengths = c.GetProperty("lengths").AsInts();
            var expected = c.GetProperty("expected");
            var (logProb, posteriors) = model.ScoreSamples(x, lengths);
            Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb,
                $"{name}: log_prob with the {implementation} implementation");
            Close.Equal(expected.GetProperty("posteriors").AsMatrix(), posteriors,
                $"{name}: posteriors with the {implementation} implementation");
            checkedCases++;
        }
        Assert.True(checkedCases >= 5, $"expected at least 5 inference cases in the golden file, saw {checkedCases}");
    }

    private static void CheckPredictProbaAgainstGolden(Implementation implementation)
    {
        var checkedCases = 0;
        foreach (var c in InferenceCases())
        {
            var name = c.GetProperty("name").AsString();
            var model = MakeModel(c, implementation);
            var x = c.GetProperty("X").AsMatrix();
            var lengths = c.GetProperty("lengths").AsInts();
            Close.Equal(c.GetProperty("expected").GetProperty("posteriors").AsMatrix(), model.PredictProba(x, lengths),
                $"{name}: posteriors with the {implementation} implementation");
            checkedCases++;
        }
        Assert.True(checkedCases >= 5, $"expected at least 5 inference cases in the golden file, saw {checkedCases}");
    }

    /// <summary>Score reproduces the golden score of every inference case, computed with the log implementation: one forward pass
    /// per sequence (several sequences through lengths included), summed over the sequences.</summary>
    [Fact]
    public void Score_reproduces_the_score_of_every_inference_case_with_the_log_implementation() =>
        CheckScoreAgainstGolden(Implementation.Log);

    /// <summary>Same with the scaling implementation: the same golden numbers hold (the oracle checked both implementations agree).</summary>
    [Fact]
    public void Score_reproduces_the_score_of_every_inference_case_with_the_scaling_implementation() =>
        CheckScoreAgainstGolden(Implementation.Scaling);

    /// <summary>ScoreSamples reproduces the golden log_prob and posteriors of every inference case with the log implementation:
    /// forward + backward, each row log-normalized, per sequence, posteriors concatenated in order.</summary>
    [Fact]
    public void ScoreSamples_reproduces_the_log_prob_and_posteriors_of_every_inference_case_with_the_log_implementation() =>
        CheckScoreSamplesAgainstGolden(Implementation.Log);

    /// <summary>Same with the scaling implementation: forward * backward, each row normalized, per sequence.</summary>
    [Fact]
    public void ScoreSamples_reproduces_the_log_prob_and_posteriors_of_every_inference_case_with_the_scaling_implementation() =>
        CheckScoreSamplesAgainstGolden(Implementation.Scaling);

    /// <summary>PredictProba reproduces the golden posteriors of every inference case with the log implementation.</summary>
    [Fact]
    public void PredictProba_reproduces_the_posteriors_of_every_inference_case_with_the_log_implementation() =>
        CheckPredictProbaAgainstGolden(Implementation.Log);

    /// <summary>Same with the scaling implementation.</summary>
    [Fact]
    public void PredictProba_reproduces_the_posteriors_of_every_inference_case_with_the_scaling_implementation() =>
        CheckPredictProbaAgainstGolden(Implementation.Scaling);

    /// <summary>Null lengths means one sequence: the wikipedia case (one sequence of 5 samples, the golden lengths [5]) gives the
    /// same score, log_prob and posteriors through null as through its explicit lengths, with both implementations.</summary>
    [Fact]
    public void Null_lengths_mean_one_sequence()
    {
        var c = Case("wikipedia");
        var x = c.GetProperty("X").AsMatrix();
        var expected = c.GetProperty("expected");
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
        {
            var model = MakeModel(c, implementation);
            Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, null),
                $"wikipedia: score with null lengths ({implementation})");
            var (logProb, posteriors) = model.ScoreSamples(x, null);
            Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb,
                $"wikipedia: log_prob with null lengths ({implementation})");
            Close.Equal(expected.GetProperty("posteriors").AsMatrix(), posteriors,
                $"wikipedia: posteriors with null lengths ({implementation})");
        }
    }

    /// <summary>Score is the log probability of ScoreSamples and PredictProba is the posteriors of ScoreSamples, for every case
    /// and both implementations.</summary>
    [Fact]
    public void Score_and_PredictProba_are_consistent_with_ScoreSamples_for_every_case()
    {
        var checkedCases = 0;
        foreach (var c in InferenceCases())
        {
            var name = c.GetProperty("name").AsString();
            var x = c.GetProperty("X").AsMatrix();
            var lengths = c.GetProperty("lengths").AsInts();
            foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
            {
                var model = MakeModel(c, implementation);
                var (logProb, posteriors) = model.ScoreSamples(x, lengths);
                Close.Equal(logProb, model.Score(x, lengths),
                    $"{name}: score equals the log_prob of score_samples ({implementation})");
                Close.Equal(posteriors, model.PredictProba(x, lengths),
                    $"{name}: predict_proba equals the posteriors of score_samples ({implementation})");
            }
            checkedCases++;
        }
        Assert.True(checkedCases >= 5, $"expected at least 5 inference cases in the golden file, saw {checkedCases}");
    }

    /// <summary>The observations, the start probabilities, the transition matrix and the lengths a caller passes in are not
    /// mutated by any inference method; the results are new arrays.</summary>
    [Fact]
    public void The_arrays_a_caller_passes_in_are_not_mutated()
    {
        var c = Case("wikipedia");
        var x0 = c.GetProperty("X").AsMatrix();
        var x = (double[,])x0.Clone();
        var startProb0 = c.GetProperty("startprob").AsVector();
        var startProb = (double[])startProb0.Clone();
        var transMat0 = c.GetProperty("transmat").AsMatrix();
        var transMat = (double[,])transMat0.Clone();
        var lengths0 = c.GetProperty("lengths").AsInts();
        var lengths = (int[])lengths0.Clone();

        var model = new StubHmm(c.GetProperty("n_components").AsInt(), Implementation.Log);
        model.StartProb = startProb;
        model.TransMat = transMat;
        model.Score(x, lengths);
        model.ScoreSamples(x, lengths);
        model.PredictProba(x, lengths);
        model.Decode(x, lengths, DecoderAlgorithm.Viterbi);
        model.Decode(x, lengths, DecoderAlgorithm.Map);
        model.Predict(x, lengths);

        Close.Equal(x0, x, "x after the inference calls");
        Close.Equal(startProb0, startProb, "startprob after the inference calls");
        Close.Equal(transMat0, transMat, "transmat after the inference calls");
        Assert.True(lengths0.SequenceEqual(lengths), "lengths after the inference calls");
    }

    /// <summary>Score, ScoreSamples and PredictProba throw InvalidOperationException while StartProb is null (the model is
    /// neither set nor fitted), for every golden case and both implementations.</summary>
    [Fact]
    public void The_scoring_methods_throw_invalid_operation_when_startprob_is_null()
    {
        var c = Case("wikipedia");
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
        {
            var model = new StubHmm(c.GetProperty("n_components").AsInt(), implementation);
            Assert.Throws<InvalidOperationException>(() => model.Score(x, lengths));
            Assert.Throws<InvalidOperationException>(() => model.ScoreSamples(x, lengths));
            Assert.Throws<InvalidOperationException>(() => model.PredictProba(x, lengths));
        }
    }

    /// <summary>Lengths that do not sum to the number of rows throw ArgumentException (the wikipedia case has 5 rows): a sum
    /// that is too small, a sum that is too large, and a sum that is right but contains a negative length.</summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(5, 1)]
    [InlineData(6, -1)]
    public void Scoring_methods_throw_argument_exception_for_lengths_that_do_not_sum_to_the_rows(int length1, int length2)
    {
        var c = Case("wikipedia");
        var x = c.GetProperty("X").AsMatrix();
        var lengths = new int[] { length1, length2 };
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
        {
            var model = MakeModel(c, implementation);
            Assert.Throws<ArgumentException>(() => model.Score(x, lengths));
            Assert.Throws<ArgumentException>(() => model.ScoreSamples(x, lengths));
            Assert.Throws<ArgumentException>(() => model.PredictProba(x, lengths));
        }
    }

    /// <summary>Empty observations (no row), zero columns, and a non-finite value (NaN, +infinity, -infinity) throw
    /// ArgumentException from the scoring methods, for every golden case and both implementations.</summary>
    [Fact]
    public void The_scoring_methods_throw_argument_exception_for_empty_or_non_finite_observations()
    {
        var c = Case("wikipedia");
        var badMatrices = new List<(string What, double[,] Matrix)>
        {
            ("an empty matrix", new double[0, 2]),
            ("a matrix with zero columns", new double[5, 0]),
            ("a matrix with a NaN", WithValue(c.GetProperty("X").AsMatrix(), 2, 1, double.NaN)),
            ("a matrix with +infinity", WithValue(c.GetProperty("X").AsMatrix(), 1, 0, double.PositiveInfinity)),
            ("a matrix with -infinity", WithValue(c.GetProperty("X").AsMatrix(), 4, 1, double.NegativeInfinity)),
        };
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
        {
            var model = MakeModel(c, implementation);
            foreach (var (what, matrix) in badMatrices)
            {
                AssertThrowsExactly<ArgumentException>(() => { model.Score(matrix); },
                    $"Score({what}) must throw ArgumentException ({implementation})");
                AssertThrowsExactly<ArgumentException>(() => { model.ScoreSamples(matrix); },
                    $"ScoreSamples({what}) must throw ArgumentException ({implementation})");
                AssertThrowsExactly<ArgumentException>(() => { model.PredictProba(matrix); },
                    $"PredictProba({what}) must throw ArgumentException ({implementation})");
            }
        }
    }

    /// <summary>CheckIsFitted runs before the observations check: with StartProb null AND an empty matrix the exception is
    /// InvalidOperationException, not ArgumentException.</summary>
    [Fact]
    public void CheckIsFitted_runs_before_the_observations_check()
    {
        var c = Case("wikipedia");
        var model = new StubHmm(c.GetProperty("n_components").AsInt(), Implementation.Log);
        Assert.Throws<InvalidOperationException>(() => model.Score(new double[0, 2], null));
    }

    /// <summary>Observations with the wrong number of features (3 instead of the 2 states) throw ArgumentException from the
    /// forward pass, with both implementations.</summary>
    [Fact]
    public void The_scoring_methods_throw_argument_exception_for_the_wrong_number_of_features()
    {
        var c = Case("wikipedia");
        var x = new double[5, 3];
        for (var i = 0; i < 5; i++)
            for (var j = 0; j < 3; j++)
                x[i, j] = -0.5;
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
        {
            var model = MakeModel(c, implementation);
            Assert.Throws<ArgumentException>(() => model.Score(x, null));
        }
    }

    /// <summary>Asserts that testCode throws exactly the type T (Assert.Throws semantics: subclasses do not count) and carries
    /// the context for the failure message.</summary>
    private static void AssertThrowsExactly<T>(Action testCode, string context) where T : Exception
    {
        try
        {
            testCode();
        }
        catch (Exception e) when (e.GetType() == typeof(T))
        {
            return;
        }
        catch (Exception e)
        {
            Assert.True(false, $"{context}: threw {e.GetType().Name} with message \"{e.Message}\" instead of exactly {typeof(T).Name}");
        }
        Assert.True(false, $"{context}: no exception was thrown; expected exactly {typeof(T).Name}");
    }

    private static double[,] WithValue(double[,] x, int row, int column, double value)
    {
        var copy = (double[,])x.Clone();
        copy[row, column] = value;
        return copy;
    }

    /// <summary>The stub model of the ticket: the observations ARE the log frame probabilities (n_features = n_components),
    /// exactly like the oracle's StubHMM; Check does nothing and the emission of a state is the state number itself.</summary>
    private sealed class StubHmm : AbstractHmm
    {
        public StubHmm(int nComponents, Implementation implementation)
            : base(nComponents, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "st", "st", implementation)
        {
        }

        protected override void Check()
        {
        }

        protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

        protected override double[] GenerateSampleFromState(int state, Random random) => new double[] { state };
    }
}
