using System.Text.Json;
using HmmLearn;
using HmmLearn.Tests.Support;
using Xunit;

namespace HmmLearn.Tests.Acceptance.Hmm22;

/// <summary>
/// HMM-22: every fit case of the four variational Gaussian golden files, with both implementations. Fitting a model that is
/// given all its parameters (init_params "") must reproduce every expected field of the oracle - the six posterior arrays,
/// the startprob/transmat/means/covars point estimates, the monitored lower-bound history, the iteration count, the
/// convergence flag and the score - within the oracle tolerance, and the fit must not mutate the caller's arrays.
/// </summary>
public class FitParityTests
{
    private static readonly string[] CaseNames =
    {
        "ten-iterations",
        "defaults-100-iterations-max",
        "start-transitions-means-without-covars",
        "start-and-transitions-only",
    };

    public static IEnumerable<object[]> FitCases()
    {
        foreach (var (name, covariance) in new[]
        {
            ("full", CovarianceType.Full),
            ("spherical", CovarianceType.Spherical),
            ("diag", CovarianceType.Diag),
            ("tied", CovarianceType.Tied),
        })
        foreach (var caseName in CaseNames)
        foreach (var implementation in new[] { Implementation.Log, Implementation.Scaling })
            yield return new object[] { name, covariance, caseName, implementation };
    }

    [Theory]
    [MemberData(nameof(FitCases))]
    public void FitMatchesTheOracle(string covarianceName, CovarianceType covariance, string caseName, Implementation implementation)
    {
        var root = Golden.Load("variational_gaussian_" + covarianceName);
        var fitCase = Assert.Single(root.GetProperty("fit_cases").Items(),
            x => x.GetProperty("name").GetString() == caseName);
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
        var startProbPriorBefore = Clone(startProbPrior);
        var startProbPosteriorBefore = Clone(startProbPosterior);
        var transMatPriorBefore = Clone(transMatPrior);
        var transMatPosteriorBefore = Clone(transMatPosterior);
        var meansPriorBefore = Clone(meansPrior);
        var meansPosteriorBefore = Clone(meansPosterior);
        var betaPriorBefore = Clone(betaPrior);
        var betaPosteriorBefore = Clone(betaPosterior);
        var dofPriorBefore = Clone(dofPrior);
        var dofPosteriorBefore = Clone(dofPosterior);
        var scalePriorBefore = Clone(scalePrior);
        var scalePosteriorBefore = Clone(scalePosterior);
        var covarsBefore = Clone(covars);

        model.Fit(x, lengths);

        // The caller's arrays are untouched.
        Close.Equal(startProbPriorBefore, startProbPrior, "caller's startprob_prior");
        Close.Equal(startProbPosteriorBefore, startProbPosterior, "caller's startprob_posterior");
        Close.Equal(transMatPriorBefore, transMatPrior, "caller's transmat_prior");
        Close.Equal(transMatPosteriorBefore, transMatPosterior, "caller's transmat_posterior");
        Close.Equal(meansPriorBefore, meansPrior, "caller's means_prior");
        Close.Equal(meansPosteriorBefore, meansPosterior, "caller's means_posterior");
        Close.Equal(betaPriorBefore, betaPrior, "caller's beta_prior");
        Close.Equal(betaPosteriorBefore, betaPosterior, "caller's beta_posterior");
        Close.Equal(dofPriorBefore, dofPrior, "caller's dof_prior");
        Close.Equal(dofPosteriorBefore, dofPosterior, "caller's dof_posterior");
        Close.Equal(scalePriorBefore, scalePrior, "caller's scale_prior");
        Close.Equal(scalePosteriorBefore, scalePosterior, "caller's scale_posterior");
        Close.Equal(covarsBefore, covars, "caller's covars");

        var expected = fitCase.GetProperty("expected");

        if (expected.Has("startprob_posterior"))
            Close.Equal(expected.GetProperty("startprob_posterior").AsVector(), model.StartProbPosterior, "startprob_posterior");
        if (expected.Has("transmat_posterior"))
            Close.Equal(expected.GetProperty("transmat_posterior").AsMatrix(), model.TransMatPosterior, "transmat_posterior");
        if (expected.Has("means_posterior"))
            Close.Equal(expected.GetProperty("means_posterior").AsMatrix(), model.MeansPosterior, "means_posterior");
        if (expected.Has("beta_posterior"))
            Close.Equal(expected.GetProperty("beta_posterior").AsVector(), model.BetaPosterior, "beta_posterior");
        if (expected.Has("dof_posterior"))
            Close.Equal(expected.GetProperty("dof_posterior").AsVector(), model.DofPosterior, "dof_posterior");
        if (expected.Has("scale_posterior"))
            Close.Equal(expected.GetProperty("scale_posterior").AsTensor3(), model.ScalePosterior, "scale_posterior");
        if (expected.Has("covars"))
            Close.Equal(expected.GetProperty("covars").AsTensor3(), model.Covars, "covars");
        if (expected.Has("startprob"))
            Close.Equal(expected.GetProperty("startprob").AsVector(), model.StartProb, "startprob");
        if (expected.Has("transmat"))
            Close.Equal(expected.GetProperty("transmat").AsMatrix(), model.TransMat, "transmat");

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
