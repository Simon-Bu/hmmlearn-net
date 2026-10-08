using System;
using System.Collections.Generic;
using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;
using Xunit;

namespace HmmLearn.Tests.Acceptance.Hmm21;

/// <summary>
/// Fitting with only start and transition parameters (params "st", init_params "") must reproduce the oracle
/// "start-and-transitions-only" case of every variational_gaussian_*.json file, with both implementations; the emission
/// arrays the fit was given stay unchanged, and the caller's arrays are not mutated.
/// </summary>
public class StartAndTransitionsParityTests
{
    public static IEnumerable<object[]> FitCases()
    {
        foreach (var (name, covariance) in new[]
        {
            ("full", CovarianceType.Full),
            ("diag", CovarianceType.Diag),
            ("tied", CovarianceType.Tied),
            ("spherical", CovarianceType.Spherical),
        })
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
            yield return new object[] { name, covariance, implementation };
    }

    [Theory]
    [MemberData(nameof(FitCases))]
    public void FitWithOnlyStartAndTransitionsMatchesTheOracle(string covarianceName, CovarianceType covariance,
        Implementation implementation)
    {
        var root = Golden.Load("variational_gaussian_" + covarianceName);
        var fitCase = Assert.Single(root.GetProperty("fit_cases").Items(),
            x => x.GetProperty("name").GetString() == "start-and-transitions-only");
        var constructor = fitCase.GetProperty("constructor");

        var model = new VariationalGaussianHmm(
            constructor.GetProperty("n_components").AsInt(),
            covariance,
            nIter: constructor.GetProperty("n_iter").AsInt(),
            tol: constructor.GetProperty("tol").AsDouble(),
            @params: constructor.GetProperty("params").AsString(),
            initParams: "");
        model.Implementation = implementation;

        var x = fitCase.GetProperty("X").AsMatrix();
        var lengths = fitCase.GetProperty("lengths").AsInts();
        var init = fitCase.GetProperty("init");

        var startProbPrior = init.GetProperty("startprob_prior").AsVector();
        var startProbPosterior = init.GetProperty("startprob_posterior").AsVector();
        var transMatPrior = init.GetProperty("transmat_prior").AsMatrix();
        var transMatPosterior = init.GetProperty("transmat_posterior").AsMatrix();
        var meansPrior = init.GetProperty("means_prior").AsMatrix();
        var meansPosterior = init.GetProperty("means_posterior").AsMatrix();
        var betaPrior = init.GetProperty("beta_prior").AsVector();
        var betaPosterior = init.GetProperty("beta_posterior").AsVector();
        var dofPrior = init.GetProperty("dof_prior").AsVector();
        var dofPosterior = init.GetProperty("dof_posterior").AsVector();
        var scalePrior = init.GetProperty("scale_prior").AsTensor3();
        var scalePosterior = init.GetProperty("scale_posterior").AsTensor3();
        var covars = init.GetProperty("covars").AsTensor3();

        model.StartProbPriorValues = startProbPrior;
        model.StartProbPosterior = startProbPosterior;
        model.TransMatPriorValues = transMatPrior;
        model.TransMatPosterior = transMatPosterior;
        model.MeansPriorValues = meansPrior;
        model.MeansPosterior = meansPosterior;
        model.BetaPriorValues = betaPrior;
        model.BetaPosterior = betaPosterior;
        model.DofPriorValues = dofPrior;
        model.DofPosterior = dofPosterior;
        model.ScalePriorValues = scalePrior;
        model.ScalePosterior = scalePosterior;
        model.Covars = covars;

        // The fit must not mutate the arrays the caller passed in.
        var startProbPosteriorBefore = Clone(startProbPosterior);
        var transMatPosteriorBefore = Clone(transMatPosterior);
        var meansPosteriorBefore = Clone(meansPosterior);
        var betaPosteriorBefore = Clone(betaPosterior);
        var dofPosteriorBefore = Clone(dofPosterior);
        var scalePosteriorBefore = Clone(scalePosterior);
        var covarsBefore = Clone(covars);

        model.Fit(x, lengths);

        // The caller's arrays are untouched.
        Close.Equal(startProbPosteriorBefore, startProbPosterior, "caller's startprob_posterior");
        Close.Equal(transMatPosteriorBefore, transMatPosterior, "caller's transmat_posterior");
        Close.Equal(meansPosteriorBefore, meansPosterior, "caller's means_posterior");
        Close.Equal(betaPosteriorBefore, betaPosterior, "caller's beta_posterior");
        Close.Equal(dofPosteriorBefore, dofPosterior, "caller's dof_posterior");
        Close.Equal(scalePosteriorBefore, scalePosterior, "caller's scale_posterior");
        Close.Equal(covarsBefore, covars, "caller's covars");

        var expected = fitCase.GetProperty("expected");

        // The start and transition posteriors (and their point estimates) come from the oracle.
        if (expected.Has("startprob_posterior"))
            Close.Equal(expected.GetProperty("startprob_posterior").AsVector(), model.StartProbPosterior, "startprob_posterior");
        if (expected.Has("transmat_posterior"))
            Close.Equal(expected.GetProperty("transmat_posterior").AsMatrix(), model.TransMatPosterior, "transmat_posterior");
        if (expected.Has("startprob"))
            Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb, "startprob");
        if (expected.Has("transmat"))
            Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat, "transmat");

        // An "st" fit leaves the emission arrays exactly as they were given.
        Close.Equal(meansPosteriorBefore, model.MeansPosterior, "means_posterior unchanged by the fit");
        Close.Equal(betaPosteriorBefore, model.BetaPosterior, "beta_posterior unchanged by the fit");
        Close.Equal(dofPosteriorBefore, model.DofPosterior, "dof_posterior unchanged by the fit");
        Close.Equal(scalePosteriorBefore, model.ScalePosterior, "scale_posterior unchanged by the fit");
        Close.Equal(covarsBefore, model.Covars, "covars unchanged by the fit");

        // The monitored variational lower bounds, iteration count and convergence flag come from the oracle.
        if (expected.Has("history"))
        {
            var expectedHistory = expected.GetProperty("history").AsVector();
            Assert.Equal(expectedHistory.Length, model.Monitor.History.Count);
            for (var i = 0; i < expectedHistory.Length; i++)
                Close.Equal(expectedHistory[i], model.Monitor.History[i], $"history[{i}]");
        }
        if (expected.Has("n_iter_done"))
            Assert.Equal(expected.GetProperty("n_iter_done").AsInt(), model.Monitor.Iter);
        if (expected.Has("converged"))
            Assert.Equal(expected.GetProperty("converged").AsBool(), model.Monitor.Converged);
        if (expected.Has("score"))
            Close.Equal(expected.GetProperty("score").AsDouble(), model.Score(x, lengths), "score");
    }

    private static double[] Clone(double[] array)
    {
        var result = new double[array.Length];
        array.CopyTo(result, 0);
        return result;
    }

    private static double[,] Clone(double[,] array) => (double[,])array.Clone();

    private static double[,,] Clone(double[,,] array) => (double[,,])array.Clone();
}
