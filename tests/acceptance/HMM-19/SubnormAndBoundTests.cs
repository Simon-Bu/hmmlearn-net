using HmmLearn;
using HmmLearn.Tests.Support;
using MathNet.Numerics;

namespace HmmLearn.Tests.Acceptance.Hmm19;

/// <summary>
/// Acceptance tests for the variational machinery of <see cref="VariationalBaseHmm"/> (ticket HMM-19, criterion C4): EstepBegin
/// subnormalizes the startprobabilities and each transmat row with exp(DiGamma(p) - DiGamma(sum(p))), producing values in (0,
/// 1) that sum to less than 1 and refreshing them when the posteriors change; DoMStep sets the posteriors to prior plus
/// statistics (only the letters of Params are touched) and normalizes the parameters, replacing the caller's arrays without
/// mutating them; and ComputeLowerBound is the current log probability minus the Dirichlet KL terms between posteriors and
/// priors.
/// </summary>
public class SubnormAndBoundTests
{
    private static StubVariationalHmm Model(double[]? startProbPosterior, double[,]? transMatPosterior,
        double[]? startProbPriorValues = null, double[,]? transMatPriorValues = null, string @params = "st") =>
        new(3, null, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, @params, "", Implementation.Log)
        {
            StartProbPosterior = startProbPosterior,
            TransMatPosterior = transMatPosterior,
            StartProbPriorValues = startProbPriorValues,
            TransMatPriorValues = transMatPriorValues,
        };

    private static double[,] Matrix(params double[][] rows)
    {
        var m = new double[rows.Length, rows[0].Length];
        for (var i = 0; i < rows.Length; i++)
            for (var j = 0; j < rows[i].Length; j++)
                m[i, j] = rows[i][j];
        return m;
    }

    private static double[] Row(double[,] m, int i)
    {
        var cols = m.GetLength(1);
        var row = new double[cols];
        for (var j = 0; j < cols; j++)
            row[j] = m[i, j];
        return row;
    }

    private static double[] Add(double[] a, double[] b)
    {
        var r = new double[a.Length];
        for (var i = 0; i < a.Length; i++)
            r[i] = a[i] + b[i];
        return r;
    }

    private static double[] Normalize(double[] v)
    {
        var sum = v.Sum();
        var r = new double[v.Length];
        for (var i = 0; i < v.Length; i++)
            r[i] = v[i] / sum;
        return r;
    }

    private static double[,] AddMatrix(double[,] a, double[,] b)
    {
        var rows = a.GetLength(0);
        var cols = a.GetLength(1);
        var r = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                r[i, j] = a[i, j] + b[i, j];
        return r;
    }

    private static double[,] RowNormalize(double[,] m)
    {
        var rows = m.GetLength(0);
        var cols = m.GetLength(1);
        var r = new double[rows, cols];
        for (var i = 0; i < rows; i++)
        {
            var sum = 0.0;
            for (var j = 0; j < cols; j++)
                sum += m[i, j];
            for (var j = 0; j < cols; j++)
                r[i, j] = m[i, j] / sum;
        }
        return r;
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

    /// <summary>EstepBegin subnormalizes the startprobabilities with exp(DiGamma(p) - DiGamma(sum(p))): the documented value
    /// for each component, each in (0, 1), a total below 1, and the caller's posterior untouched.</summary>
    [Fact]
    public void estep_begin_subnormalizes_the_startprobabilities_with_digamma()
    {
        var posterior = new[] { 2.0, 5.0, 3.0 };
        var posterior0 = (double[])posterior.Clone();
        var model = Model(posterior, Matrix(new[] { 1.0, 1.0, 1.0 }, new[] { 2.0, 2.0, 2.0 }, new[] { 3.0, 3.0, 3.0 }));

        model.RunEstepBegin();

        var sub = model.Sub;
        Assert.NotNull(sub);
        var pSum = posterior.Sum();
        var expected = new double[posterior.Length];
        for (var i = 0; i < posterior.Length; i++)
            expected[i] = Math.Exp(SpecialFunctions.DiGamma(posterior[i]) - SpecialFunctions.DiGamma(pSum));
        Close.Equal(expected, sub!, "startprob_subnorm_");
        for (var i = 0; i < sub!.Length; i++)
            Assert.True(sub[i] > 0.0 && sub[i] < 1.0, $"startprob_subnorm_[{i}] = {sub[i]} is not in (0, 1)");
        Assert.True(sub.Sum() < 1.0, $"startprob_subnorm_ sum {sub.Sum()} is not below 1");
        Assert.NotSame(posterior, sub);
        Close.Equal(posterior0, posterior, "caller's startprob_posterior_");
        Assert.NotNull(model.SubTrans);
    }

    /// <summary>EstepBegin subnormalizes each transmat row the same way: the documented value for each entry, each in (0, 1),
    /// each row summing to less than 1, and the caller's posterior untouched.</summary>
    [Fact]
    public void estep_begin_subnormalizes_each_transition_row_with_digamma()
    {
        var posterior = Matrix(
            new[] { 2.0, 4.0, 1.5 },
            new[] { 0.8, 7.0, 2.2 },
            new[] { 3.3, 0.5, 6.1 });
        var posterior0 = (double[,])posterior.Clone();
        var model = Model(new[] { 1.0, 1.0, 1.0 }, posterior);

        model.RunEstepBegin();

        var sub = model.SubTrans;
        Assert.NotNull(sub);
        var nc = 3;
        for (var i = 0; i < nc; i++)
        {
            var rowSum = 0.0;
            for (var j = 0; j < nc; j++)
                rowSum += posterior[i, j];
            var rowSub = 0.0;
            for (var j = 0; j < nc; j++)
            {
                Close.Equal(Math.Exp(SpecialFunctions.DiGamma(posterior[i, j]) - SpecialFunctions.DiGamma(rowSum)),
                    sub![i, j], $"transmat_subnorm_[{i}, {j}]");
                Assert.True(sub[i, j] > 0.0 && sub[i, j] < 1.0,
                    $"transmat_subnorm_[{i}, {j}] = {sub[i, j]} is not in (0, 1)");
                rowSub += sub[i, j];
            }
            Assert.True(rowSub < 1.0, $"transmat_subnorm_ row {i} sum {rowSub} is not below 1");
        }
        Assert.NotSame(posterior, sub);
        Close.Equal(posterior0, posterior, "caller's transmat_posterior_");
        Assert.NotNull(model.Sub);
    }

    /// <summary>EstepBegin recomputes the subnorms on every call: after the posterior changes, the subnorm changes with it.</summary>
    [Fact]
    public void estep_begin_refreshes_the_subnorms_when_the_posteriors_change()
    {
        var model = new StubVariationalHmm(3, null, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "",
            Implementation.Log);
        model.StartProbPosterior = new[] { 2.0, 5.0, 3.0 };
        model.TransMatPosterior = Matrix(new[] { 1.0, 1.0, 1.0 }, new[] { 2.0, 2.0, 2.0 }, new[] { 3.0, 3.0, 3.0 });
        model.RunEstepBegin();
        var first = (double[])model.Sub!.Clone();

        model.StartProbPosterior = new[] { 9.0, 0.5, 0.5 };
        model.RunEstepBegin();

        Assert.False(VectorsAreClose(first, model.Sub!), "the subnormalized startprobabilities were not recomputed");
    }

    /// <summary>DoMStep adds the statistics to the priors (new arrays) and normalizes: startprob_posterior = prior + Start,
    /// startprob = that, normalized; each transmat row likewise.</summary>
    [Fact]
    public void m_step_adds_the_statistics_to_the_priors_and_normalizes()
    {
        var startPrior = new[] { 1.0, 1.0 };
        var transPrior = new double[,] { { 1.0, 1.0 }, { 1.0, 1.0 } };
        var model = new StubVariationalHmm(2, null, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "",
            Implementation.Log)
        {
            StartProbPriorValues = startPrior,
            TransMatPriorValues = transPrior,
        };
        var stats = new SufficientStatistics(2)
        {
            NObs = 2,
            Start = new[] { 1.0, 3.0 },
            Trans = new double[,] { { 1.0, 3.0 }, { 3.0, 1.0 } },
        };

        model.RunMStep(stats);

        var expectedStartPosterior = Add(startPrior, stats.Start);
        Close.Equal(expectedStartPosterior, model.StartProbPosterior, "startprob_posterior_");
        Close.Equal(Normalize(expectedStartPosterior), model.StartProb, "startprob_");
        var expectedTransPosterior = AddMatrix(transPrior, stats.Trans);
        Close.Equal(expectedTransPosterior, model.TransMatPosterior, "transmat_posterior_");
        Close.Equal(RowNormalize(expectedTransPosterior), model.TransMat, "transmat_");
    }

    /// <summary>DoMStep replaces the caller's arrays and the statistics' arrays rather than mutating them.</summary>
    [Fact]
    public void m_step_replaces_the_callers_arrays_and_does_not_mutate_them()
    {
        var startPrior = new[] { 1.0, 1.0 };
        var transPrior = new double[,] { { 1.0, 1.0 }, { 1.0, 1.0 } };
        var startPrior0 = (double[])startPrior.Clone();
        var transPrior0 = (double[,])transPrior.Clone();
        var model = new StubVariationalHmm(2, null, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "",
            Implementation.Log)
        {
            StartProbPriorValues = startPrior,
            TransMatPriorValues = transPrior,
        };
        var stats = new SufficientStatistics(2)
        {
            NObs = 2,
            Start = new[] { 1.0, 3.0 },
            Trans = new double[,] { { 1.0, 3.0 }, { 3.0, 1.0 } },
        };
        var startStats0 = (double[])stats.Start.Clone();
        var transStats0 = (double[,])stats.Trans.Clone();

        model.RunMStep(stats);

        Assert.NotSame(startPrior, model.StartProbPosterior);
        Assert.NotSame(stats.Start, model.StartProbPosterior);
        Assert.NotSame(transPrior, model.TransMatPosterior);
        Assert.NotSame(stats.Trans, model.TransMatPosterior);
        Close.Equal(startPrior0, startPrior, "caller's startprob_prior_");
        Close.Equal(transPrior0, transPrior, "caller's transmat_prior_");
        Close.Equal(startStats0, stats.Start, "stats.Start");
        Close.Equal(transStats0, stats.Trans, "stats.Trans");
    }

    /// <summary>DoMStep updates only the letters of Params: with "s" the transition arrays keep their exact references, and with
    /// "t" the startprob arrays stay unset.</summary>
    [Fact]
    public void m_step_only_updates_the_letters_of_params()
    {
        var startPrior = new[] { 1.0, 1.0 };
        var transPrior = new double[,] { { 1.0, 1.0 }, { 1.0, 1.0 } };
        var stats = new SufficientStatistics(2)
        {
            NObs = 2,
            Start = new[] { 1.0, 3.0 },
            Trans = new double[,] { { 1.0, 3.0 }, { 3.0, 1.0 } },
        };

        var transPosterior = new double[,] { { 1.0, 2.0 }, { 2.0, 1.0 } };
        var transMat = new double[,] { { 0.4, 0.6 }, { 0.6, 0.4 } };
        var sModel = new StubVariationalHmm(2, null, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "s", "",
            Implementation.Log)
        {
            StartProbPriorValues = startPrior,
            TransMatPosterior = transPosterior,
            TransMat = transMat,
        };
        sModel.RunMStep(stats);
        Close.Equal(Normalize(Add(startPrior, stats.Start)), sModel.StartProb, "startprob_");
        Assert.Same(transPosterior, sModel.TransMatPosterior);
        Assert.Same(transMat, sModel.TransMat);

        var tModel = new StubVariationalHmm(2, null, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "t", "",
            Implementation.Log)
        {
            StartProbPriorValues = startPrior,
            TransMatPriorValues = transPrior,
        };
        tModel.RunMStep(stats);
        Assert.Null(tModel.StartProb);
        Assert.Null(tModel.StartProbPosterior);
        Close.Equal(AddMatrix(transPrior, stats.Trans), tModel.TransMatPosterior, "transmat_posterior_");
        Close.Equal(RowNormalize(AddMatrix(transPrior, stats.Trans)), tModel.TransMat, "transmat_");
    }

    /// <summary>ComputeLowerBound is the current log probability minus the Dirichlet KL between the startprob posteriors and
    /// priors and between each transmat row of posteriors and priors.</summary>
    [Fact]
    public void lower_bound_is_the_log_probability_minus_the_dirichlet_kl_terms()
    {
        var startPrior = new[] { 0.5, 1.5, 1.0 };
        var startPosterior = new[] { 1.2, 2.8, 0.9 };
        var transPrior = Matrix(
            new[] { 0.8, 1.2, 0.9 },
            new[] { 1.1, 0.7, 1.4 },
            new[] { 0.6, 1.9, 0.8 });
        var transPosterior = Matrix(
            new[] { 2.5, 3.1, 0.4 },
            new[] { 0.9, 4.2, 2.7 },
            new[] { 3.6, 1.1, 0.7 });
        var model = new StubVariationalHmm(3, null, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "",
            Implementation.Log)
        {
            StartProbPriorValues = startPrior,
            StartProbPosterior = startPosterior,
            TransMatPriorValues = transPrior,
            TransMatPosterior = transPosterior,
        };
        var logProb = -12.37;

        var expected = logProb - KlDivergence.KlDirichlet(startPosterior, startPrior);
        for (var i = 0; i < 3; i++)
            expected -= KlDivergence.KlDirichlet(Row(transPosterior, i), Row(transPrior, i));

        Close.Equal(expected, model.Bound(logProb), "lower bound");
    }

    /// <summary>ComputeLowerBound tracks the current posteriors: it equals the log probability minus the KL terms for each
    /// posterior pair the model currently holds.</summary>
    [Fact]
    public void lower_bound_follows_the_current_posteriors()
    {
        var startPrior = new[] { 0.5, 1.5, 1.0 };
        var transPrior = Matrix(
            new[] { 0.8, 1.2, 0.9 },
            new[] { 1.1, 0.7, 1.4 },
            new[] { 0.6, 1.9, 0.8 });
        var model = new StubVariationalHmm(3, null, null, DecoderAlgorithm.Viterbi, null, 1, 1e-2, false, "st", "",
            Implementation.Log)
        {
            StartProbPriorValues = startPrior,
            TransMatPriorValues = transPrior,
        };
        var logProb = -7.25;

        double Expected(double[] startPosterior, double[,] transPosterior)
        {
            var bound = logProb - KlDivergence.KlDirichlet(startPosterior, startPrior);
            for (var i = 0; i < 3; i++)
                bound -= KlDivergence.KlDirichlet(Row(transPosterior, i), Row(transPrior, i));
            return bound;
        }

        var firstPosteriorPair = (new[] { 1.2, 2.8, 0.9 },
            Matrix(new[] { 2.5, 3.1, 0.4 }, new[] { 0.9, 4.2, 2.7 }, new[] { 3.6, 1.1, 0.7 }));
        model.StartProbPosterior = firstPosteriorPair.Item1;
        model.TransMatPosterior = firstPosteriorPair.Item2;
        Close.Equal(Expected(firstPosteriorPair.Item1, firstPosteriorPair.Item2), model.Bound(logProb),
            "lower bound, first posteriors");

        var secondPosteriorPair = (new[] { 4.0, 0.3, 2.2 },
            Matrix(new[] { 0.4, 5.5, 1.1 }, new[] { 3.7, 0.2, 6.6 }, new[] { 1.9, 2.4, 0.8 }));
        model.StartProbPosterior = secondPosteriorPair.Item1;
        model.TransMatPosterior = secondPosteriorPair.Item2;
        Close.Equal(Expected(secondPosteriorPair.Item1, secondPosteriorPair.Item2), model.Bound(logProb),
            "lower bound, second posteriors");
    }
}
