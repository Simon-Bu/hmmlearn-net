namespace HmmLearn;

/// <summary>The learning half of VariationalGaussianHmm: the M-step (port of VariationalGaussianHMM._do_mstep of hmmlearn/vhmm.py).</summary>
public partial class VariationalGaussianHmm
{
    /// <summary>
    /// M-step of the Variational Bayes EM. First the base update of the start and transition posteriors. Then, with <c>stats</c> the
    /// <see cref="GaussianStatistics"/> of the E-step (Post, Obs, Obs2, ObsObsT) and the NEW values of the means when 'm' was just updated:
    /// <list type="bullet">
    /// <item>If Params contains 'm': BetaPosterior[c] = BetaPriorValues[c] + Post[c]; MeansPosterior[c, f] =
    /// (BetaPriorValues[c] * MeansPriorValues[c, f] + Obs[c, f]) / BetaPosterior[c].</item>
    /// <item>If Params contains 'c' (BetaPosterior and MeansPosterior are those after the previous step when 'm' is not in Params), per type:
    /// <b>Full</b>: DofPosterior[c] = DofPriorValues[c] + Post[c]; ScalePosterior[c] = ScalePriorValues[c] + ObsObsT[c] +
    /// BetaPriorValues[c] * outer(MeansPriorValues[c], MeansPriorValues[c]) - BetaPosterior[c] * outer(MeansPosterior[c], MeansPosterior[c]);
    /// Covars[c] = ScalePosterior[c] / DofPosterior[c]. <b>Tied</b>: DofPosterior[0] = DofPriorValues[0] + sum over c of Post[c];
    /// ScalePosterior[0] = ScalePriorValues[0] + the sum over c of the full-type matrix above without ScalePriorValues; Covars[0] = ScalePosterior[0] /
    /// DofPosterior[0]. <b>Diag</b>: DofPosterior[c] = DofPriorValues[c] + Post[c]; ScalePosterior[c, f] = ScalePriorValues[c, f] + Obs2[c, f] +
    /// BetaPriorValues[c] * MeansPriorValues[c, f]^2 - BetaPosterior[c] * MeansPosterior[c, f]^2; Covars[c, f] = ScalePosterior[c, f] /
    /// DofPosterior[c]. <b>Spherical</b>: DofPosterior[c] = DofPriorValues[c] + Post[c]; ScalePosterior[c] = ScalePriorValues[c] + the mean over
    /// the features f of (Obs2[c, f] + BetaPriorValues[c] * MeansPriorValues[c, f]^2 - BetaPosterior[c] * MeansPosterior[c, f]^2);
    /// Covars[c] = ScalePosterior[c] / DofPosterior[c].</item>
    /// </list>
    /// Arrays are in the compact uniform shapes; new arrays replace the old ones; Covars is assigned through the validating property
    /// (a degenerate result throws <see cref="ArgumentException"/>, where hmmlearn stores it unchecked).
    /// </summary>
    protected override void DoMStep(SufficientStatistics stats)
    {
        base.DoMStep(stats);
        var gs = (GaussianStatistics)stats;
        if (Params.Contains('m'))
            UpdateMeans(gs);
        if (Params.Contains('c'))
            UpdateCovariances(gs);
    }

    /// <summary>The means update: BetaPosterior = BetaPriorValues + Post; MeansPosterior = (BetaPriorValues * MeansPriorValues + Obs) / BetaPosterior, in new arrays.</summary>
    private void UpdateMeans(GaussianStatistics gs)
    {
        var nc = NComponents;
        var nf = gs.Obs.GetLength(1);
        var betaPrior = BetaPriorValues!;
        var meansPrior = MeansPriorValues!;
        var betaPosterior = new double[nc];
        var meansPosterior = new double[nc, nf];
        for (var c = 0; c < nc; c++)
        {
            betaPosterior[c] = betaPrior[c] + gs.Post[c];
            for (var f = 0; f < nf; f++)
                meansPosterior[c, f] = (betaPrior[c] * meansPrior[c, f] + gs.Obs[c, f]) / betaPosterior[c];
        }
        BetaPosterior = betaPosterior;
        MeansPosterior = meansPosterior;
    }

    /// <summary>Dispatches the covariance update to the per-type helper.</summary>
    private void UpdateCovariances(GaussianStatistics gs)
    {
        switch (CovarianceType)
        {
            case CovarianceType.Full:
                UpdateFullCovariances(gs);
                break;
            case CovarianceType.Tied:
                UpdateTiedCovariances(gs);
                break;
            case CovarianceType.Diag:
                UpdateDiagCovariances(gs);
                break;
            case CovarianceType.Spherical:
                UpdateSphericalCovariances(gs);
                break;
        }
    }

    /// <summary>Full: per-state dof = dof_prior + post, scale = scale_prior + obs*obs.T + beta_prior * outer(means_prior) - beta_posterior * outer(means_posterior), covars = scale / dof.</summary>
    private void UpdateFullCovariances(GaussianStatistics gs)
    {
        var nc = NComponents;
        var nf = gs.Obs.GetLength(1);
        var dofPrior = DofPriorValues!;
        var scalePrior = ScalePriorValues!;
        var betaPrior = BetaPriorValues!;
        var meansPrior = MeansPriorValues!;
        var betaPosterior = BetaPosterior!;
        var meansPosterior = MeansPosterior!;
        var dofPosterior = new double[nc];
        var scalePosterior = new double[nc, nf, nf];
        var covars = new double[nc, nf, nf];
        for (var c = 0; c < nc; c++)
        {
            dofPosterior[c] = dofPrior[c] + gs.Post[c];
            for (var i = 0; i < nf; i++)
                for (var j = 0; j < nf; j++)
                {
                    scalePosterior[c, i, j] = scalePrior[c, i, j]
                        + gs.ObsObsT![c, i, j]
                        + betaPrior[c] * meansPrior[c, i] * meansPrior[c, j]
                        - betaPosterior[c] * meansPosterior[c, i] * meansPosterior[c, j];
                    covars[c, i, j] = scalePosterior[c, i, j] / dofPosterior[c];
                }
        }
        DofPosterior = dofPosterior;
        ScalePosterior = scalePosterior;
        Covars = covars;
    }

    /// <summary>Tied: one shared dof = dof_prior + sum of post, scale = scale_prior + the sum over states of (obs*obs.T + beta_prior * outer(means_prior) - beta_posterior * outer(means_posterior)), covars = scale / dof.</summary>
    private void UpdateTiedCovariances(GaussianStatistics gs)
    {
        var nc = NComponents;
        var nf = gs.Obs.GetLength(1);
        var dofPrior = DofPriorValues!;
        var scalePrior = ScalePriorValues!;
        var betaPrior = BetaPriorValues!;
        var meansPrior = MeansPriorValues!;
        var betaPosterior = BetaPosterior!;
        var meansPosterior = MeansPosterior!;
        var mass = 0.0;
        for (var c = 0; c < nc; c++)
            mass += gs.Post[c];
        var dofPosterior = new double[] { dofPrior[0] + mass };
        var scalePosterior = new double[1, nf, nf];
        var covars = new double[1, nf, nf];
        for (var i = 0; i < nf; i++)
            for (var j = 0; j < nf; j++)
            {
                var sum = 0.0;
                for (var c = 0; c < nc; c++)
                    sum += gs.ObsObsT![c, i, j]
                        + betaPrior[c] * meansPrior[c, i] * meansPrior[c, j]
                        - betaPosterior[c] * meansPosterior[c, i] * meansPosterior[c, j];
                scalePosterior[0, i, j] = scalePrior[0, i, j] + sum;
                covars[0, i, j] = scalePosterior[0, i, j] / dofPosterior[0];
            }
        DofPosterior = dofPosterior;
        ScalePosterior = scalePosterior;
        Covars = covars;
    }

    /// <summary>Diag: per-state dof = dof_prior + post, scale[c, f] = scale_prior[c, f] + obs2[c, f] + beta_prior[c] * means_prior[c, f]^2 - beta_posterior[c] * means_posterior[c, f]^2, covars = scale / dof.</summary>
    private void UpdateDiagCovariances(GaussianStatistics gs)
    {
        var nc = NComponents;
        var nf = gs.Obs.GetLength(1);
        var dofPrior = DofPriorValues!;
        var scalePrior = ScalePriorValues!;
        var betaPrior = BetaPriorValues!;
        var meansPrior = MeansPriorValues!;
        var betaPosterior = BetaPosterior!;
        var meansPosterior = MeansPosterior!;
        var dofPosterior = new double[nc];
        var scalePosterior = new double[nc, 1, nf];
        var covars = new double[nc, 1, nf];
        for (var c = 0; c < nc; c++)
        {
            dofPosterior[c] = dofPrior[c] + gs.Post[c];
            for (var f = 0; f < nf; f++)
            {
                scalePosterior[c, 0, f] = scalePrior[c, 0, f]
                    + gs.Obs2[c, f]
                    + betaPrior[c] * meansPrior[c, f] * meansPrior[c, f]
                    - betaPosterior[c] * meansPosterior[c, f] * meansPosterior[c, f];
                covars[c, 0, f] = scalePosterior[c, 0, f] / dofPosterior[c];
            }
        }
        DofPosterior = dofPosterior;
        ScalePosterior = scalePosterior;
        Covars = covars;
    }

    /// <summary>Spherical: per-state dof = dof_prior + post, scale[c] = scale_prior[c] + the mean over the features of (obs2[c, f] + beta_prior[c] * means_prior[c, f]^2 - beta_posterior[c] * means_posterior[c, f]^2), covars = scale / dof.</summary>
    private void UpdateSphericalCovariances(GaussianStatistics gs)
    {
        var nc = NComponents;
        var nf = gs.Obs.GetLength(1);
        var dofPrior = DofPriorValues!;
        var scalePrior = ScalePriorValues!;
        var betaPrior = BetaPriorValues!;
        var meansPrior = MeansPriorValues!;
        var betaPosterior = BetaPosterior!;
        var meansPosterior = MeansPosterior!;
        var dofPosterior = new double[nc];
        var scalePosterior = new double[nc, 1, 1];
        var covars = new double[nc, 1, 1];
        for (var c = 0; c < nc; c++)
        {
            dofPosterior[c] = dofPrior[c] + gs.Post[c];
            var sum = 0.0;
            for (var f = 0; f < nf; f++)
                sum += gs.Obs2[c, f]
                    + betaPrior[c] * meansPrior[c, f] * meansPrior[c, f]
                    - betaPosterior[c] * meansPosterior[c, f] * meansPosterior[c, f];
            scalePosterior[c, 0, 0] = scalePrior[c, 0, 0] + sum / nf;
            covars[c, 0, 0] = scalePosterior[c, 0, 0] / dofPosterior[c];
        }
        DofPosterior = dofPosterior;
        ScalePosterior = scalePosterior;
        Covars = covars;
    }
}
