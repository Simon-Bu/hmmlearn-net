using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm7;

/// <summary>
/// Acceptance tests for the decoding half of <see cref="AbstractHmm"/> (ticket HMM-7, criterion C2): <see cref="AbstractHmm.Decode"/>
/// (Viterbi and MAP) and <see cref="AbstractHmm.Predict"/> reproduce the golden viterbi and map expectations of every inference case
/// of oracle/golden/base.json with both implementations: log_prob by formula (logsumexp over xi / sum of the maximum posteriors) and
/// states by argmax over the xi lattice (MAP: per sample, first index of the max). The model is the ticket's stub: the observations
/// ARE the log frame probabilities (n_features = n_components), exactly like the oracle's StubHMM.
/// </summary>
public class DecodingTests
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

    private static void CheckDecodeAgainstGolden(Implementation implementation, DecoderAlgorithm algorithm, string goldenKey)
    {
        var checkedCases = 0;
        foreach (var c in InferenceCases())
        {
            var name = c.GetProperty("name").AsString();
            var model = MakeModel(c, implementation);
            var x = c.GetProperty("X").AsMatrix();
            var lengths = c.GetProperty("lengths").AsInts();
            var expected = c.GetProperty("expected").GetProperty(goldenKey);
            var (logProb, states) = model.Decode(x, lengths, algorithm);
            Close.Equal(expected.GetProperty("log_prob").AsDouble(), logProb,
                $"{name}: {goldenKey} log_prob with the {implementation} implementation");
            AssertTrueStatesEqual(expected.GetProperty("states").AsInts(), states,
                $"{name}: {goldenKey} states with the {implementation} implementation");
            checkedCases++;
        }
        Assert.True(checkedCases >= 5, $"expected at least 5 inference cases in the golden file, saw {checkedCases}");
    }

    /// <summary>Decode with Viterbi reproduces the golden viterbi log_prob and states of every inference case, with the log
    /// implementation (several sequences through lengths included).</summary>
    [Fact]
    public void Decode_with_viterbi_reproduces_the_viterbi_expectation_of_every_inference_case_with_the_log_implementation() =>
        CheckDecodeAgainstGolden(Implementation.Log, DecoderAlgorithm.Viterbi, "viterbi");

    /// <summary>Same with the scaling implementation: the same golden numbers hold.</summary>
    [Fact]
    public void Decode_with_viterbi_reproduces_the_viterbi_expectation_of_every_inference_case_with_the_scaling_implementation() =>
        CheckDecodeAgainstGolden(Implementation.Scaling, DecoderAlgorithm.Viterbi, "viterbi");

    /// <summary>Decode with MAP reproduces the golden map log_prob (sum of the maximum posteriors per sample) and states of every
    /// inference case, with the log implementation.</summary>
    [Fact]
    public void Decode_with_map_reproduces_the_map_expectation_of_every_inference_case_with_the_log_implementation() =>
        CheckDecodeAgainstGolden(Implementation.Log, DecoderAlgorithm.Map, "map");

    /// <summary>Same with the scaling implementation.</summary>
    [Fact]
    public void Decode_with_map_reproduces_the_map_expectation_of_every_inference_case_with_the_scaling_implementation() =>
        CheckDecodeAgainstGolden(Implementation.Scaling, DecoderAlgorithm.Map, "map");

    /// <summary>Decode without an explicit algorithm uses the model's Algorithm property: with the constructor's default (Viterbi)
    /// the golden viterbi expectation holds, and after setting Algorithm to Map the golden map expectation holds, with both
    /// implementations.</summary>
    [Fact]
    public void Decode_without_an_algorithm_uses_the_Algorithm_property()
    {
        var checkedCases = 0;
        foreach (var c in InferenceCases())
        {
            var name = c.GetProperty("name").AsString();
            var x = c.GetProperty("X").AsMatrix();
            var lengths = c.GetProperty("lengths").AsInts();
            var expected = c.GetProperty("expected");
            foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
            {
                var model = MakeModel(c, implementation);
                model.Algorithm = DecoderAlgorithm.Viterbi;
                var (viterbiLogProb, viterbiStates) = model.Decode(x, lengths);
                Close.Equal(expected.GetProperty("viterbi").GetProperty("log_prob").AsDouble(), viterbiLogProb,
                    $"{name}: default (viterbi) log_prob with the {implementation} implementation");
                AssertTrueStatesEqual(expected.GetProperty("viterbi").GetProperty("states").AsInts(), viterbiStates,
                    $"{name}: default (viterbi) states with the {implementation} implementation");

                model.Algorithm = DecoderAlgorithm.Map;
                var (mapLogProb, mapStates) = model.Decode(x, lengths);
                Close.Equal(expected.GetProperty("map").GetProperty("log_prob").AsDouble(), mapLogProb,
                    $"{name}: default (map) log_prob with the {implementation} implementation");
                AssertTrueStatesEqual(expected.GetProperty("map").GetProperty("states").AsInts(), mapStates,
                    $"{name}: default (map) states with the {implementation} implementation");
            }
            checkedCases++;
        }
        Assert.True(checkedCases >= 5, $"expected at least 5 inference cases in the golden file, saw {checkedCases}");
    }

    /// <summary>Predict equals the states of Decode (no probabilities): with the constructor's default Algorithm (Viterbi) the
    /// golden viterbi states, and after setting Algorithm to Map the golden map states, with both implementations.</summary>
    [Fact]
    public void Predict_equals_the_states_of_Decode_and_of_the_golden_file()
    {
        var checkedCases = 0;
        foreach (var c in InferenceCases())
        {
            var name = c.GetProperty("name").AsString();
            var x = c.GetProperty("X").AsMatrix();
            var lengths = c.GetProperty("lengths").AsInts();
            var expected = c.GetProperty("expected");
            foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
            {
                var model = MakeModel(c, implementation);
                model.Algorithm = DecoderAlgorithm.Viterbi;
                var predicted = model.Predict(x, lengths);
                AssertTrueStatesEqual(model.Decode(x, lengths).States, predicted,
                    $"{name}: predict equals decode (viterbi) with the {implementation} implementation");
                AssertTrueStatesEqual(expected.GetProperty("viterbi").GetProperty("states").AsInts(), predicted,
                    $"{name}: predict (viterbi) with the {implementation} implementation");

                model.Algorithm = DecoderAlgorithm.Map;
                var mapPredicted = model.Predict(x, lengths);
                AssertTrueStatesEqual(model.Decode(x, lengths).States, mapPredicted,
                    $"{name}: predict equals decode (map) with the {implementation} implementation");
                AssertTrueStatesEqual(expected.GetProperty("map").GetProperty("states").AsInts(), mapPredicted,
                    $"{name}: predict (map) with the {implementation} implementation");
            }
            checkedCases++;
        }
        Assert.True(checkedCases >= 5, $"expected at least 5 inference cases in the golden file, saw {checkedCases}");
    }

    /// <summary>Decode and Predict throw InvalidOperationException while StartProb is null (the model is neither set nor
    /// fitted), with both implementations.</summary>
    [Fact]
    public void Decode_and_Predict_throw_invalid_operation_when_startprob_is_null()
    {
        var c = Case("wikipedia");
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
        {
            var model = new StubHmm(c.GetProperty("n_components").AsInt(), implementation);
            Assert.Throws<InvalidOperationException>(() => model.Decode(x, lengths));
            Assert.Throws<InvalidOperationException>(() => model.Decode(x, lengths, DecoderAlgorithm.Viterbi));
            Assert.Throws<InvalidOperationException>(() => model.Decode(x, lengths, DecoderAlgorithm.Map));
            Assert.Throws<InvalidOperationException>(() => model.Predict(x, lengths));
        }
    }

    /// <summary>Decode and Predict throw ArgumentException for lengths that do not sum to the number of rows (the wikipedia case
    /// has 5 rows), with both implementations.</summary>
    [Fact]
    public void Decode_and_Predict_throw_argument_exception_for_lengths_that_do_not_sum_to_the_rows()
    {
        var c = Case("wikipedia");
        var x = c.GetProperty("X").AsMatrix();
        var badLengths = new[] { new int[] { 2, 2 }, new int[] { 5, 1 }, new int[] { 6, -1 } };
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
        {
            var model = MakeModel(c, implementation);
            foreach (var lengths in badLengths)
            {
                Assert.Throws<ArgumentException>(() => model.Decode(x, lengths));
                Assert.Throws<ArgumentException>(() => model.Predict(x, lengths));
            }
        }
    }

    /// <summary>Decode and Predict throw ArgumentException for empty (no row), zero-column and non-finite observations, with both
    /// implementations.</summary>
    [Fact]
    public void Decode_and_Predict_throw_argument_exception_for_empty_or_non_finite_observations()
    {
        var c = Case("wikipedia");
        var x0 = c.GetProperty("X").AsMatrix();
        var badMatrices = new List<(string What, double[,] Matrix)>
        {
            ("an empty matrix", new double[0, 2]),
            ("a matrix with zero columns", new double[5, 0]),
            ("a matrix with a NaN", WithValue(x0, 2, 1, double.NaN)),
            ("a matrix with +infinity", WithValue(x0, 1, 0, double.PositiveInfinity)),
            ("a matrix with -infinity", WithValue(x0, 4, 1, double.NegativeInfinity)),
        };
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
        {
            var model = MakeModel(c, implementation);
            foreach (var (what, matrix) in badMatrices)
            {
                AssertThrowsExactly<ArgumentException>(() => { model.Decode(matrix, null); },
                    $"Decode({what}) must throw ArgumentException ({implementation})");
                AssertThrowsExactly<ArgumentException>(() => { model.Predict(matrix, null); },
                    $"Predict({what}) must throw ArgumentException ({implementation})");
            }
        }
    }

    private static void AssertTrueStatesEqual(int[] expected, int[] actual, string context) =>
        Assert.True(expected.SequenceEqual(actual),
            $"{context}: expected [{string.Join(", ", expected)}], got [{string.Join(", ", actual)}]");

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
