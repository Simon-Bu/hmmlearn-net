using HmmLearn;
using HmmLearn.Tests.Support;
using MathNet.Numerics;

namespace HmmLearn.Tests.Acceptance.Hmm20;

/// <summary>
/// Acceptance tests for the emission part of the variational machinery of <see cref="VariationalCategoricalHmm"/> (ticket HMM-20,
/// criterion C3): EstepBegin computes the subnormalized emission log likelihoods with the digamma function and the subnormalized
/// log likelihood of a sample is the entry of its symbol; DoMStep with 'e' sets the emission posterior to prior + Obs and the
/// emission probabilities to the row-normalization of that posterior (leaving them alone without 'e'); ComputeLowerBound adds the
/// Dirichlet KL terms of the emission rows to the base bound; Score, Decode and Sample use the normalized emission probabilities
/// after Fit; the subnormalized likelihoods of the E-step differ from the normalized ones after a fit; and the sufficient
/// statistics are a zeroed CategoricalStatistics to which the posteriors are accumulated into Obs only when 'e' is fitted.
/// </summary>
public class EmissionTests
{
    /// <summary>A model whose variational arrays are set explicitly (InitParams ""): fixed positive start and transition arrays,
    /// the emission prior (1,1,1)/(2,2,2) and the emission posterior given by the test.</summary>
    private static ExposedModel WithPosteriors(double[,] emissionPosterior, string @params = "ste")
    {
        var model = new ExposedModel(2, null, null, null, 3, DecoderAlgorithm.Viterbi, 1, 1, 1e-2, false, @params, "",
            Implementation.Log);
        model.StartProbPriorValues = new[] { 1.0, 1.0 };
        model.StartProbPosterior = new[] { 1.5, 2.5 };
        model.TransMatPriorValues = new double[2, 2] { { 1.0, 1.0 }, { 1.0, 1.0 } };
        model.TransMatPosterior = new double[2, 2] { { 2.0, 3.0 }, { 1.0, 4.0 } };
        model.EmissionProbPriorValues = new double[2, 3] { { 1.0, 1.0, 1.0 }, { 2.0, 2.0, 2.0 } };
        model.EmissionProbPosterior = emissionPosterior;
        return model;
    }

    private static double[,] Symbols(params double[] symbols)
    {
        var x = new double[symbols.Length, 1];
        for (var i = 0; i < symbols.Length; i++)
            x[i, 0] = symbols[i];
        return x;
    }

    private static double[] Row(double[,] m, int i)
    {
        var row = new double[m.GetLength(1)];
        for (var j = 0; j < row.Length; j++)
            row[j] = m[i, j];
        return row;
    }

    /// <summary>EstepBegin computes EmissionProbLogSubnorm entry by entry as digamma(posterior) - digamma(row sum of the
    /// posterior); it is null before the first E-step.</summary>
    [Fact]
    public void estep_begin_computes_the_emission_log_subnormalization_with_digamma()
    {
        var posterior = new double[2, 3] { { 1.5, 2.5, 0.7 }, { 0.4, 3.1, 5.2 } };
        var model = WithPosteriors(posterior);
        Assert.Null(model.LogSubnorm);

        model.RunEstepBegin();

        Assert.NotNull(model.LogSubnorm);
        var expected = new double[2, 3];
        for (var c = 0; c < posterior.GetLength(0); c++)
        {
            var rowSum = 0.0;
            for (var j = 0; j < posterior.GetLength(1); j++)
                rowSum += posterior[c, j];
            var digammaRowSum = SpecialFunctions.DiGamma(rowSum);
            for (var j = 0; j < posterior.GetLength(1); j++)
                expected[c, j] = SpecialFunctions.DiGamma(posterior[c, j]) - digammaRowSum;
        }
        Close.Equal(expected, model.LogSubnorm!, "the emission log subnormalization");
    }

    /// <summary>The subnormalized log likelihood of each sample is the entry of its symbol: result[i, c] =
    /// EmissionProbLogSubnorm[c, symbol of sample i].</summary>
    [Fact]
    public void subnormalized_log_likelihood_is_the_entry_of_the_symbol()
    {
        var x = Symbols(0, 2, 1, 0);
        var model = WithPosteriors(new double[2, 3] { { 1.5, 2.5, 0.7 }, { 0.4, 3.1, 5.2 } });
        model.RunEstepBegin();
        var logSubnorm = model.LogSubnorm!;

        var result = model.SubnormLogLikelihood(x);

        Assert.Equal(x.GetLength(0), result.GetLength(0));
        Assert.Equal(2, result.GetLength(1));
        for (var i = 0; i < x.GetLength(0); i++)
        {
            var symbol = (int)x[i, 0];
            for (var c = 0; c < 2; c++)
                Close.Equal(logSubnorm[c, symbol], result[i, c], $"result[{i}, {c}] (symbol {symbol})");
        }
    }

    /// <summary>DoMStep with 'e' sets EmissionProbPosterior = EmissionProbPriorValues + Obs (a new array) and EmissionProb to the
    /// row-normalization of that posterior.</summary>
    [Fact]
    public void mstep_with_e_sets_the_posterior_to_prior_plus_obs_and_emissionprob_to_the_row_normalization()
    {
        var prior = new double[2, 3] { { 1.0, 0.5, 2.0 }, { 0.2, 1.8, 0.3 } };
        var obs = new double[2, 3] { { 3.1, 0.0, 1.2 }, { 0.4, 2.2, 0.7 } };
        var model = WithPosteriors(new double[2, 3] { { 0.9, 0.1, 0.4 }, { 1.1, 2.2, 3.3 } });
        model.EmissionProbPriorValues = prior;
        var stats = new CategoricalStatistics(2, 3) { Obs = (double[,])obs.Clone() };

        model.RunMStep(stats);

        var expectedPosterior = new double[2, 3];
        var expectedProb = new double[2, 3];
        for (var c = 0; c < 2; c++)
        {
            var sum = 0.0;
            for (var j = 0; j < 3; j++)
            {
                expectedPosterior[c, j] = prior[c, j] + obs[c, j];
                sum += expectedPosterior[c, j];
            }
            for (var j = 0; j < 3; j++)
                expectedProb[c, j] = expectedPosterior[c, j] / sum;
        }
        Close.Equal(expectedPosterior, model.EmissionProbPosterior!, "the emission posterior = prior + Obs");
        Close.Equal(expectedProb, model.EmissionProb!, "the emission probabilities = the row-normalized posterior");
    }

    /// <summary>DoMStep without 'e' in Params leaves the emission posterior and the emission probabilities alone (same
    /// reference).</summary>
    [Fact]
    public void mstep_without_e_leaves_the_emission_arrays_alone()
    {
        var posterior = new double[2, 3] { { 0.9, 0.1, 0.4 }, { 1.1, 2.2, 3.3 } };
        var prob = new double[2, 3] { { 0.2, 0.3, 0.5 }, { 0.9, 0.05, 0.05 } };
        var model = WithPosteriors(posterior);
        model.Params = "st"; // no 'e'
        model.EmissionProb = prob;

        model.RunMStep(new CategoricalStatistics(2, 3));

        Assert.Same(posterior, model.EmissionProbPosterior);
        Assert.Same(prob, model.EmissionProb);
    }

    /// <summary>ComputeLowerBound is the base bound (the log probability minus the Dirichlet KL of the start and the transition
    /// posteriors from their priors) minus the sum over the states of the Dirichlet KL of the emission row posteriors from their
    /// priors.</summary>
    [Fact]
    public void lower_bound_adds_the_emission_kl_divergences_to_the_base_bound()
    {
        var startPrior = new[] { 1.0, 1.0, 1.0, 1.0 };
        var startPosterior = new[] { 1.3, 2.7, 0.8, 4.1 };
        var transPrior = new double[4, 4]
        {
            { 0.5, 1.5, 1.0, 2.0 }, { 2.5, 0.7, 1.3, 1.9 }, { 1.1, 2.2, 0.4, 3.0 }, { 0.9, 1.2, 2.8, 0.6 }
        };
        var transPosterior = new double[4, 4]
        {
            { 1.7, 3.1, 0.9, 2.4 }, { 0.6, 4.2, 1.8, 3.3 }, { 2.9, 1.4, 3.7, 1.1 }, { 3.5, 0.8, 2.2, 4.0 }
        };
        var emissionPrior = new double[4, 3]
        {
            { 0.8, 1.2, 2.0 }, { 1.5, 0.6, 1.1 }, { 0.4, 2.3, 1.7 }, { 1.9, 1.0, 0.5 }
        };
        var emissionPosterior = new double[4, 3]
        {
            { 2.1, 3.4, 5.0 }, { 4.7, 1.2, 2.6 }, { 0.9, 3.8, 4.4 }, { 3.2, 2.5, 6.1 }
        };
        var model = new ExposedModel(4, null, null, null, 3, DecoderAlgorithm.Viterbi, 1, 1, 1e-2, false, "ste", "",
            Implementation.Log);
        model.StartProbPriorValues = startPrior;
        model.StartProbPosterior = startPosterior;
        model.TransMatPriorValues = transPrior;
        model.TransMatPosterior = transPosterior;
        model.EmissionProbPriorValues = emissionPrior;
        model.EmissionProbPosterior = emissionPosterior;
        const double LogProb = -42.7;

        var bound = LogProb - KlDivergence.KlDirichlet(startPosterior, startPrior);
        for (var i = 0; i < transPosterior.GetLength(0); i++)
            bound -= KlDivergence.KlDirichlet(Row(transPosterior, i), Row(transPrior, i));
        for (var c = 0; c < emissionPosterior.GetLength(0); c++)
            bound -= KlDivergence.KlDirichlet(Row(emissionPosterior, c), Row(emissionPrior, c));

        Close.Equal(bound, model.Bound(LogProb), "the lower bound with the emission KL terms");
    }

    /// <summary>The data of the fitted-model tests: 10 categorical symbols in two sequences.</summary>
    private static double[,] FitData() => Symbols(0, 2, 1, 0, 2, 1, 1, 0, 2, 0);

    /// <summary>A small model fitted with 'e': the posterior is initialized at random with a fixed seed, so every test gets the
    /// same fitted parameters.</summary>
    private static ExposedModel FitSmallModel(Implementation implementation = Implementation.Log)
    {
        var model = new ExposedModel(2, null, null, null, null, DecoderAlgorithm.Viterbi, 2024, 5, 1e-12, false,
            "ste", "ste", implementation);
        model.Fit(FitData(), new[] { 6, 4 });
        return model;
    }

    /// <summary>After a Fit with 'e', Score uses the normalized emission probabilities: it equals the log probability computed in
    /// the test with the normalized start, transition and emission probabilities (not the subnormalized ones of the E-step).</summary>
    [Fact]
    public void score_uses_the_normalized_emissionprob_after_fit()
    {
        var model = FitSmallModel();
        Assert.NotNull(model.EmissionProb);
        var x = FitData();
        var lengths = new[] { 6, 4 };

        var expected = ForwardLogProb(model.StartProb!, model.TransMat!, model.EmissionProb!, x, lengths);

        Close.Equal(expected, model.Score(x, lengths), "the score after fit (normalized emission probabilities)");
    }

    /// <summary>After a Fit with 'e', Decode (Viterbi) uses the normalized emission probabilities: the log probability and the
    /// states equal the Viterbi computed in the test with the normalized parameters.</summary>
    [Fact]
    public void decode_uses_the_normalized_emissionprob_after_fit()
    {
        var model = FitSmallModel();
        var x = FitData();
        var lengths = new[] { 6, 4 };

        var (expectedLogProb, expectedStates) = ViterbiLogProb(model.StartProb!, model.TransMat!, model.EmissionProb!, x, lengths);

        var (logProb, states) = model.Decode(x, lengths);

        Close.Equal(expectedLogProb, logProb, "the decode log probability after fit");
        Assert.Equal(expectedStates, states);
    }

    /// <summary>After a Fit with 'e', Sample draws the observations from the normalized emission probabilities: with a fixed
    /// generator, the states and the symbols equal the draws computed in the test from the model's own StartProb, TransMat and
    /// EmissionProb (documented order: first state, its observation, then for each next sample the state and the observation).</summary>
    [Fact]
    public void sample_uses_the_normalized_emissionprob_after_fit()
    {
        const int N = 12;
        var model = FitSmallModel();
        var expectedSymbols = new double[N];
        var expectedStates = new int[N];
        var random = new Random(99);
        var firstState = Draw(CumulativeVector(model.StartProb!), random);
        expectedStates[0] = firstState;
        expectedSymbols[0] = (double)Draw(CumulativeRow(model.EmissionProb!, firstState), random);
        for (var t = 1; t < N; t++)
        {
            expectedStates[t] = Draw(CumulativeRow(model.TransMat!, expectedStates[t - 1]), random);
            expectedSymbols[t] = (double)Draw(CumulativeRow(model.EmissionProb!, expectedStates[t]), random);
        }

        var (x, states) = model.Sample(N, new Random(99));

        Assert.Equal(expectedStates, states);
        for (var i = 0; i < N; i++)
            Close.Equal(expectedSymbols[i], x[i, 0], $"the symbol {i} sampled after fit");
    }

    /// <summary>After a Fit with 'e', the subnormalized emission log likelihoods of the E-step (from the digamma) differ from the
    /// log of the normalized emission probabilities: the two are not the same matrix.</summary>
    [Fact]
    public void after_fit_the_subnormalized_emission_likelihood_differs_from_the_normalized_one()
    {
        var model = FitSmallModel();
        var subnorm = model.LogSubnorm!;
        var emissionProb = model.EmissionProb!;
        var maxDifference = 0.0;
        for (var c = 0; c < emissionProb.GetLength(0); c++)
            for (var j = 0; j < emissionProb.GetLength(1); j++)
                maxDifference = Math.Max(maxDifference, Math.Abs(subnorm[c, j] - Math.Log(emissionProb[c, j])));

        Assert.True(maxDifference > 1e-3,
            $"the subnormalized log likelihoods must differ from the normalized ones (max difference {maxDifference:R})");
    }

    /// <summary>InitializeSufficientStatistics returns a CategoricalStatistics with zero Obs of shape (NComponents, NFeatures),
    /// zero base statistics, and no sequence accumulated yet.</summary>
    [Fact]
    public void initialize_sufficient_statistics_returns_zeroed_categorical_statistics()
    {
        var model = new ExposedModel(2, null, null, null, 4, DecoderAlgorithm.Viterbi, 1, 1, 1e-2, false, "ste", "",
            Implementation.Log);

        var stats = model.InitializeStats();

        Assert.IsType<CategoricalStatistics>(stats);
        var categorical = (CategoricalStatistics)stats;
        Assert.Equal(2, categorical.Obs.GetLength(0));
        Assert.Equal(4, categorical.Obs.GetLength(1));
        for (var c = 0; c < 2; c++)
            for (var j = 0; j < 4; j++)
                Assert.Equal(0.0, categorical.Obs[c, j]);
        Assert.Equal(0, stats.NObs);
        Assert.Equal(2, stats.Start.Length);
        Assert.Equal(2, stats.Trans.GetLength(0));
        Assert.Equal(2, stats.Trans.GetLength(1));
    }

    /// <summary>AccumulateSufficientStatistics with 'e' adds the posteriors of each sample to Obs[state, symbol] (and the start
    /// posteriors to Start, as the base does); without 'e' the Obs stays zero.</summary>
    [Fact]
    public void accumulate_adds_the_posteriors_to_obs_only_when_e_is_fitted()
    {
        var x = Symbols(1, 0, 1);
        var posteriors = new double[3, 2] { { 0.2, 0.8 }, { 0.5, 0.5 }, { 0.1, 0.9 } };
        var lattice = new double[3, 2] { { 1.0, 1.0 }, { 1.0, 1.0 }, { 1.0, 1.0 } };

        var model = WithPosteriors(new double[2, 3] { { 1.0, 1.0, 1.0 }, { 2.0, 2.0, 2.0 } });
        model.Params = "se"; // 'e' but not 't': no transition statistics to worry about
        var stats = model.InitializeStats();
        model.Accumulate(stats, x, lattice, posteriors, lattice, lattice);

        var expectedObs = new double[2, 3];
        for (var i = 0; i < x.GetLength(0); i++)
            for (var c = 0; c < 2; c++)
                expectedObs[c, (int)x[i, 0]] += posteriors[i, c];
        Close.Equal(expectedObs, ((CategoricalStatistics)stats).Obs, "the accumulated observations");
        for (var c = 0; c < 2; c++)
            Close.Equal(posteriors[0, c], stats.Start[c], $"Start[{c}]");
        Assert.Equal(1, stats.NObs);

        var withoutE = WithPosteriors(new double[2, 3] { { 1.0, 1.0, 1.0 }, { 2.0, 2.0, 2.0 } });
        withoutE.Params = "s"; // no 'e'
        var statsWithoutE = withoutE.InitializeStats();
        withoutE.Accumulate(statsWithoutE, x, lattice, posteriors, lattice, lattice);
        for (var c = 0; c < 2; c++)
            for (var j = 0; j < 3; j++)
                Assert.Equal(0.0, ((CategoricalStatistics)statsWithoutE).Obs[c, j]);
    }

    /// <summary>GenerateSampleFromState draws the symbol from the row of EmissionProb of the state: with a one-hot row the symbol
    /// is deterministic (zero and one are trivial inputs).</summary>
    [Fact]
    public void generate_sample_from_state_draws_from_the_emissionprob_row()
    {
        var model = new ExposedModel(2, null, null, null, 3, DecoderAlgorithm.Viterbi, 1, 1, 1e-2, false, "ste", "",
            Implementation.Log);
        model.EmissionProb = new double[2, 3] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } };
        var random = new Random(7);

        Assert.Equal(new[] { 0.0 }, model.SampleFromState(0, random));
        Assert.Equal(new[] { 1.0 }, model.SampleFromState(1, random));
    }

    /// <summary>ComputeLikelihood is the row of EmissionProb of the symbol of each sample (result[i, c] = EmissionProb[c,
    /// symbol]); ComputeLogLikelihood is its natural log (the log of 0 is -infinity).</summary>
    [Fact]
    public void likelihood_and_log_likelihood_are_the_rows_of_emissionprob()
    {
        var model = new ExposedModel(2, null, null, null, 3, DecoderAlgorithm.Viterbi, 1, 1, 1e-2, false, "ste", "",
            Implementation.Log);
        var emissionProb = new double[2, 3] { { 0.2, 0.3, 1.0 }, { 0.9, 0.05, 0.0 } };
        model.EmissionProb = emissionProb;
        var x = Symbols(0, 1, 2);

        var likelihood = model.Likelihood(x);
        for (var i = 0; i < x.GetLength(0); i++)
            for (var c = 0; c < 2; c++)
                Assert.Equal(emissionProb[c, (int)x[i, 0]], likelihood[i, c]);

        var logLikelihood = model.LogLikelihood(x);
        for (var i = 0; i < x.GetLength(0); i++)
            for (var c = 0; c < 2; c++)
            {
                var expected = Math.Log(emissionProb[c, (int)x[i, 0]]);
                if (double.IsNegativeInfinity(expected))
                    Assert.True(double.IsNegativeInfinity(logLikelihood[i, c]), $"log likelihood [{i}, {c}] of 0 must be -infinity");
                else
                    Close.Equal(expected, logLikelihood[i, c], $"log likelihood [{i}, {c}]");
            }
    }

    // --- test-side forward / Viterbi / sampling helpers, computed from the model's own arrays ---

    /// <summary>The natural log probability of x under the normalized parameters, with the standard forward pass in log space.</summary>
    private static double ForwardLogProb(double[] startProb, double[,] transMat, double[,] emissionProb, double[,] x, int[] lengths)
    {
        var nc = startProb.Length;
        var offset = 0;
        var logProb = 0.0;
        foreach (var length in lengths)
        {
            var alpha = new double[nc];
            for (var c = 0; c < nc; c++)
                alpha[c] = Math.Log(startProb[c]) + Math.Log(emissionProb[c, (int)x[offset, 0]]);
            for (var t = 1; t < length; t++)
            {
                var next = new double[nc];
                for (var j = 0; j < nc; j++)
                {
                    var incoming = new double[nc];
                    for (var i = 0; i < nc; i++)
                        incoming[i] = alpha[i] + Math.Log(transMat[i, j]);
                    next[j] = LogSumExp(incoming) + Math.Log(emissionProb[j, (int)x[offset + t, 0]]);
                }
                alpha = next;
            }
            logProb += LogSumExp(alpha);
            offset += length;
        }
        return logProb;
    }

    private static double LogSumExp(double[] values)
    {
        var max = values[0];
        for (var i = 1; i < values.Length; i++)
            max = Math.Max(max, values[i]);
        var sum = 0.0;
        for (var i = 0; i < values.Length; i++)
            sum += Math.Exp(values[i] - max);
        return max + Math.Log(sum);
    }

    /// <summary>The Viterbi log probability and states, with the standard algorithm in log space.</summary>
    private static (double LogProb, int[] States) ViterbiLogProb(double[] startProb, double[,] transMat, double[,] emissionProb, double[,] x, int[] lengths)
    {
        var nc = startProb.Length;
        var states = new int[x.GetLength(0)];
        var offset = 0;
        var logProb = 0.0;
        foreach (var length in lengths)
        {
            var delta = new double[nc];
            var psi = new int[length, nc];
            for (var c = 0; c < nc; c++)
                delta[c] = Math.Log(startProb[c]) + Math.Log(emissionProb[c, (int)x[offset, 0]]);
            for (var t = 1; t < length; t++)
            {
                var next = new double[nc];
                for (var j = 0; j < nc; j++)
                {
                    var bestIncoming = delta[0] + Math.Log(transMat[0, j]);
                    var bestIncomingIndex = 0;
                    for (var i = 1; i < nc; i++)
                    {
                        var candidate = delta[i] + Math.Log(transMat[i, j]);
                        if (candidate > bestIncoming)
                        {
                            bestIncoming = candidate;
                            bestIncomingIndex = i;
                        }
                    }
                    psi[t, j] = bestIncomingIndex;
                    next[j] = bestIncoming + Math.Log(emissionProb[j, (int)x[offset + t, 0]]);
                }
                delta = next;
            }
            var best = delta[0];
            var bestIndex = 0;
            for (var i = 1; i < nc; i++)
            {
                if (delta[i] > best)
                {
                    best = delta[i];
                    bestIndex = i;
                }
            }
            states[offset + length - 1] = bestIndex;
            for (var t = length - 1; t > 0; t--)
                states[offset + t - 1] = psi[t, states[offset + t]];
            logProb += best;
            offset += length;
        }
        return (logProb, states);
    }

    private static double[] CumulativeVector(double[] values)
    {
        var cdf = new double[values.Length];
        var acc = 0.0;
        for (var i = 0; i < values.Length; i++)
        {
            acc += values[i];
            cdf[i] = acc;
        }
        return cdf;
    }

    private static double[] CumulativeRow(double[,] matrix, int row)
    {
        var cols = matrix.GetLength(1);
        var cdf = new double[cols];
        var acc = 0.0;
        for (var j = 0; j < cols; j++)
        {
            acc += matrix[row, j];
            cdf[j] = acc;
        }
        return cdf;
    }

    /// <summary>The documented draw: one uniform number u, the first index whose cumulative probability is greater than u, the last
    /// index if rounding leaves none.</summary>
    private static int Draw(double[] cdf, Random random)
    {
        var u = random.NextDouble();
        for (var i = 0; i < cdf.Length; i++)
            if (cdf[i] > u)
                return i;
        return cdf.Length - 1;
    }
}
