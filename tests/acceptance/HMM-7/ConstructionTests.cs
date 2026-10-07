using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests.Acceptance.Hmm7;

/// <summary>
/// Acceptance tests for the construction and hook surface of <see cref="AbstractHmm"/> (ticket HMM-7, criterion C4): the
/// constructor stores its arguments and creates the monitor with the same tolerances; the parameters are null until set;
/// <c>RaiseWarning</c> reaches every subscriber of the Warning event; the default <c>ComputeLikelihood</c> is the elementwise
/// exponential of <c>ComputeLogLikelihood</c>; the internal posterior helpers normalize forward/backward lattices into the golden
/// posteriors of the oracle and agree with each other; CheckIsFitted and CheckObservations enforce the fitted and input
/// contracts; and the checks run in the documented order (is-fitted, then Check, then observations).
/// </summary>
public class ConstructionTests
{
    private static List<JsonElement> InferenceCases() =>
        Golden.Load("base").GetProperty("inference_cases").Items().ToList();

    private static JsonElement Case(string name) =>
        Assert.Single(InferenceCases(), c => c.GetProperty("name").AsString() == name);

    /// <summary>The constructor stores all of its arguments in the properties.</summary>
    [Fact]
    public void The_constructor_stores_its_arguments_in_the_properties()
    {
        var model = new ProbeStub(5, DecoderAlgorithm.Map, 123, 9, 0.037, true, "ste", "ste", Implementation.Scaling);
        Assert.Equal(5, model.NComponents);
        Assert.Equal(DecoderAlgorithm.Map, model.Algorithm);
        Assert.Equal(123, (int?)model.RandomState);
        Assert.Equal(9, model.NIter);
        Assert.Equal(0.037, model.Tol);
        Assert.True(model.Verbose);
        Assert.Equal("ste", model.Params);
        Assert.Equal("ste", model.InitParams);
        Assert.Equal(Implementation.Scaling, model.Implementation);
    }

    /// <summary>The constructor creates the convergence monitor with the same tolerances it is given.</summary>
    [Fact]
    public void The_monitor_is_created_with_the_same_arguments()
    {
        var model = new ProbeStub(5, DecoderAlgorithm.Map, 123, 9, 0.037, true, "ste", "ste", Implementation.Scaling);
        Assert.NotNull(model.Monitor);
        Assert.Equal(0.037, model.Monitor.Tol);
        Assert.Equal(9, model.Monitor.NIter);
        Assert.True(model.Monitor.Verbose);
    }

    /// <summary>After construction the parameters are null: nothing is set or fitted yet.</summary>
    [Fact]
    public void The_parameters_are_null_after_construction()
    {
        var model = new ProbeStub(3, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "st", "st", Implementation.Log);
        Assert.Null(model.NFeatures);
        Assert.Null(model.StartProb);
        Assert.Null(model.TransMat);
    }

    /// <summary>RaiseWarning reaches every subscriber of the Warning event, and raising with no subscriber does not throw.</summary>
    [Fact]
    public void RaiseWarning_reaches_all_subscribers_and_needs_no_subscriber()
    {
        var model = new ProbeStub(3, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "st", "st", Implementation.Log);
        var received = new List<string>();
        var first = new Action<string>(m => received.Add("A:" + m));
        var second = new Action<string>(m => received.Add("B:" + m));
        model.Warning += first;
        model.Warning += second;
        model.RaiseWarningPublic("degenerate fit");
        Assert.Equal(new[] { "A:degenerate fit", "B:degenerate fit" }, received);
        model.Warning -= first;
        model.Warning -= second;
        // Raising with no subscriber must not throw.
        model.RaiseWarningPublic("nobody is listening");
    }

    /// <summary>The default ComputeLikelihood is the elementwise exponential of ComputeLogLikelihood (the stub returns a copy of
    /// the observations as the log likelihood), across a range of magnitudes including a large negative value.</summary>
    [Fact]
    public void ComputeLikelihood_defaults_to_the_exponential_of_compute_log_likelihood()
    {
        var model = new ProbeStub(2, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "st", "st", Implementation.Log);
        var x = new double[,]
        {
            { -0.5, 1.0 },
            { 2.0, -1.0 },
            { -20.0, 0.0 },
        };
        var likelihood = model.ComputeLikelihoodPublic(x);
        Assert.Equal(3, likelihood.GetLength(0));
        Assert.Equal(2, likelihood.GetLength(1));
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 2; j++)
                Close.Equal(Math.Exp(x[i, j]), likelihood[i, j], $"likelihood[{i}, {j}] must be the exponential of the log likelihood");
    }

    /// <summary>ComputePosteriorsLog normalizes the product of the log forward and backward lattices of the wikipedia sequence
    /// into the golden posteriors; each row sums to 1.</summary>
    [Fact]
    public void The_log_posterior_helper_gives_the_golden_posteriors_of_the_wikipedia_sequence()
    {
        var c = Case("wikipedia");
        var startProb = c.GetProperty("startprob").AsVector();
        var transMat = c.GetProperty("transmat").AsMatrix();
        var logX = c.GetProperty("X").AsMatrix();
        var golden = c.GetProperty("expected").GetProperty("posteriors").AsMatrix();

        var (_, fwdLog) = Hmmc.ForwardLog(startProb, transMat, logX);
        var bwdLog = Hmmc.BackwardLog(startProb, transMat, logX);
        var posteriors = AbstractHmm.ComputePosteriorsLog(fwdLog, bwdLog);
        Close.Equal(golden, posteriors, "wikipedia: the log posterior helper vs the golden posteriors");
        CheckRowSums(posteriors, "wikipedia: log posteriors");
    }

    /// <summary>ComputePosteriorsScaling normalizes the product of the scaled forward and backward lattices of the wikipedia
    /// sequence (frame probabilities are the exponentials of the log X) into the golden posteriors; each row sums to 1.</summary>
    [Fact]
    public void The_scaling_posterior_helper_gives_the_golden_posteriors_of_the_wikipedia_sequence()
    {
        var c = Case("wikipedia");
        var startProb = c.GetProperty("startprob").AsVector();
        var transMat = c.GetProperty("transmat").AsMatrix();
        var logX = c.GetProperty("X").AsMatrix();
        var golden = c.GetProperty("expected").GetProperty("posteriors").AsMatrix();

        var frameProb = Exp(logX);
        var (_, fwdScaling, scaling) = Hmmc.ForwardScaling(startProb, transMat, frameProb);
        var bwdScaling = Hmmc.BackwardScaling(startProb, transMat, frameProb, scaling);
        var posteriors = AbstractHmm.ComputePosteriorsScaling(fwdScaling, bwdScaling);
        Close.Equal(golden, posteriors, "wikipedia: the scaling posterior helper vs the golden posteriors");
        CheckRowSums(posteriors, "wikipedia: scaling posteriors");
    }

    /// <summary>On every sequence of the three-sequence case (lengths 7, 1, 5) the two posterior helpers give the golden
    /// posteriors of that sequence and agree with each other; every row sums to 1.</summary>
    [Fact]
    public void The_posterior_helpers_agree_on_every_sequence_of_the_three_sequence_case()
    {
        var c = Case("three-states-three-sequences");
        var startProb = c.GetProperty("startprob").AsVector();
        var transMat = c.GetProperty("transmat").AsMatrix();
        var x = c.GetProperty("X").AsMatrix();
        var lengths = c.GetProperty("lengths").AsInts();
        var golden = c.GetProperty("expected").GetProperty("posteriors").AsMatrix();

        var start = 0;
        for (var k = 0; k < lengths.Length; k++)
        {
            var length = lengths[k];
            var subX = Rows(x, start, length);
            var (_, fwdLog) = Hmmc.ForwardLog(startProb, transMat, subX);
            var bwdLog = Hmmc.BackwardLog(startProb, transMat, subX);
            var logPosteriors = AbstractHmm.ComputePosteriorsLog(fwdLog, bwdLog);

            var frameProb = Exp(subX);
            var (_, fwdScaling, scaling) = Hmmc.ForwardScaling(startProb, transMat, frameProb);
            var bwdScaling = Hmmc.BackwardScaling(startProb, transMat, frameProb, scaling);
            var scalingPosteriors = AbstractHmm.ComputePosteriorsScaling(fwdScaling, bwdScaling);

            var goldenSub = Rows(golden, start, length);
            Close.Equal(goldenSub, logPosteriors, $"sequence {k} (length {length}): log helper vs golden");
            Close.Equal(goldenSub, scalingPosteriors, $"sequence {k} (length {length}): scaling helper vs golden");
            Close.Equal(logPosteriors, scalingPosteriors, $"sequence {k} (length {length}): the helpers agree");
            CheckRowSums(logPosteriors, $"sequence {k}: log posteriors");
            CheckRowSums(scalingPosteriors, $"sequence {k}: scaling posteriors");
            start += length;
        }
    }

    /// <summary>The posterior helpers return new arrays and do not mutate the lattices they are given.</summary>
    [Fact]
    public void The_posterior_helpers_return_new_arrays_and_do_not_mutate_the_lattices()
    {
        var c = Case("wikipedia");
        var startProb = c.GetProperty("startprob").AsVector();
        var transMat = c.GetProperty("transmat").AsMatrix();
        var logX = c.GetProperty("X").AsMatrix();

        var (_, fwdLog) = Hmmc.ForwardLog(startProb, transMat, logX);
        var bwdLog = Hmmc.BackwardLog(startProb, transMat, logX);
        var fwdLog0 = (double[,])fwdLog.Clone();
        var bwdLog0 = (double[,])bwdLog.Clone();
        var logResult = AbstractHmm.ComputePosteriorsLog(fwdLog, bwdLog);
        Close.Equal(fwdLog0, fwdLog, "forward log lattice after ComputePosteriorsLog");
        Close.Equal(bwdLog0, bwdLog, "backward log lattice after ComputePosteriorsLog");
        Assert.False(ReferenceEquals(fwdLog, logResult), "ComputePosteriorsLog must return a new array");
        Assert.False(ReferenceEquals(bwdLog, logResult), "ComputePosteriorsLog must return a new array");

        var frameProb = Exp(logX);
        var (_, fwdScaling, scaling) = Hmmc.ForwardScaling(startProb, transMat, frameProb);
        var bwdScaling = Hmmc.BackwardScaling(startProb, transMat, frameProb, scaling);
        var fwdScaling0 = (double[,])fwdScaling.Clone();
        var bwdScaling0 = (double[,])bwdScaling.Clone();
        var scalingResult = AbstractHmm.ComputePosteriorsScaling(fwdScaling, bwdScaling);
        Close.Equal(fwdScaling0, fwdScaling, "forward scaling lattice after ComputePosteriorsScaling");
        Close.Equal(bwdScaling0, bwdScaling, "backward scaling lattice after ComputePosteriorsScaling");
        Assert.False(ReferenceEquals(fwdScaling, scalingResult), "ComputePosteriorsScaling must return a new array");
        Assert.False(ReferenceEquals(bwdScaling, scalingResult), "ComputePosteriorsScaling must return a new array");
    }

    /// <summary>CheckIsFitted throws InvalidOperationException while StartProb is null and does not throw once it is set.</summary>
    [Fact]
    public void CheckIsFitted_throws_invalid_operation_while_startprob_is_null()
    {
        var model = new ProbeStub(3, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "st", "st", Implementation.Log);
        Assert.Throws<InvalidOperationException>(model.CheckIsFittedPublic);
        model.StartProb = new double[] { 0.4, 0.35, 0.25 };
        // Once StartProb is set the check must not throw.
        model.CheckIsFittedPublic();
    }

    /// <summary>CheckObservations rejects a matrix with no rows, a matrix with no columns, and a matrix with a NaN, +infinity or
    /// -infinity; it accepts a finite matrix.</summary>
    [Fact]
    public void CheckObservations_rejects_empty_and_non_finite_matrices()
    {
        Assert.Throws<ArgumentException>(() => ProbeStub.CheckObservationsPublic(new double[0, 2]));
        Assert.Throws<ArgumentException>(() => ProbeStub.CheckObservationsPublic(new double[2, 0]));
        Assert.Throws<ArgumentException>(() => ProbeStub.CheckObservationsPublic(new double[0, 0]));
        Assert.Throws<ArgumentException>(() => ProbeStub.CheckObservationsPublic(WithValue(new double[2, 2], 0, 0, double.NaN)));
        Assert.Throws<ArgumentException>(() => ProbeStub.CheckObservationsPublic(WithValue(new double[2, 2], 1, 1, double.PositiveInfinity)));
        Assert.Throws<ArgumentException>(() => ProbeStub.CheckObservationsPublic(WithValue(new double[2, 2], 0, 1, double.NegativeInfinity)));
        // A finite matrix must pass the check.
        ProbeStub.CheckObservationsPublic(new double[,] { { -0.5, 1.0 }, { 0.0, -20.0 } });
    }

    /// <summary>The inference checks run in the documented order: CheckIsFitted before the model's Check, and the model's Check
    /// before the observations check (a fitted model with an empty matrix fails in Check, not in CheckObservations).</summary>
    [Fact]
    public void The_checks_run_in_the_order_is_fitted_then_check_then_observations()
    {
        var fitted = new CheckMarkerHmm();
        fitted.StartProb = new double[] { 0.5, 0.5 };
        fitted.TransMat = new double[,] { { 0.7, 0.3 }, { 0.3, 0.7 } };
        var empty = new double[0, 2];
        Assert.Throws<CheckMarkerException>(() => fitted.Score(empty, null));

        var unfitted = new CheckMarkerHmm();
        Assert.Throws<InvalidOperationException>(() => unfitted.Score(empty, null));
    }

    private static double[,] Exp(double[,] x)
    {
        var result = new double[x.GetLength(0), x.GetLength(1)];
        for (var i = 0; i < x.GetLength(0); i++)
            for (var j = 0; j < x.GetLength(1); j++)
                result[i, j] = Math.Exp(x[i, j]);
        return result;
    }

    private static double[,] Rows(double[,] x, int start, int count)
    {
        var result = new double[count, x.GetLength(1)];
        for (var i = 0; i < count; i++)
            for (var j = 0; j < x.GetLength(1); j++)
                result[i, j] = x[start + i, j];
        return result;
    }

    private static void CheckRowSums(double[,] posteriors, string what)
    {
        for (var i = 0; i < posteriors.GetLength(0); i++)
        {
            var sum = 0.0;
            for (var j = 0; j < posteriors.GetLength(1); j++)
                sum += posteriors[i, j];
            Assert.True(Math.Abs(sum - 1.0) <= 1e-9, $"{what}: row {i} sums to {sum}, expected 1");
        }
    }

    private static double[,] WithValue(double[,] x, int row, int column, double value)
    {
        var copy = (double[,])x.Clone();
        copy[row, column] = value;
        return copy;
    }

    /// <summary>The probe stub: Check does nothing and the emission of a state is the state number itself; it exposes the
    /// protected members the construction tests need.</summary>
    private sealed class ProbeStub : AbstractHmm
    {
        public ProbeStub(int nComponents, DecoderAlgorithm algorithm, int? randomState, int nIter, double tol,
            bool verbose, string @params, string initParams, Implementation implementation)
            : base(nComponents, algorithm, randomState, nIter, tol, verbose, @params, initParams, implementation)
        {
        }

        protected override void Check()
        {
        }

        protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

        protected override double[] GenerateSampleFromState(int state, Random random) => new double[] { state };

        public void RaiseWarningPublic(string message) => RaiseWarning(message);

        public double[,] ComputeLikelihoodPublic(double[,] x) => ComputeLikelihood(x);

        public void CheckIsFittedPublic() => CheckIsFitted();

        public static void CheckObservationsPublic(double[,] x) => CheckObservations(x);
    }

    /// <summary>A stub whose Check throws a marker exception, so the tests can tell whether Check ran before the observations
    /// check.</summary>
    private sealed class CheckMarkerHmm : AbstractHmm
    {
        public CheckMarkerHmm()
            : base(2, DecoderAlgorithm.Viterbi, null, 10, 1e-2, false, "st", "st", Implementation.Log)
        {
        }

        protected override void Check() => throw new CheckMarkerException();

        protected override double[,] ComputeLogLikelihood(double[,] x) => (double[,])x.Clone();

        protected override double[] GenerateSampleFromState(int state, Random random) => new double[] { state };
    }

    private sealed class CheckMarkerException : Exception
    {
        public CheckMarkerException() : base("the model's Check ran")
        {
        }
    }
}
