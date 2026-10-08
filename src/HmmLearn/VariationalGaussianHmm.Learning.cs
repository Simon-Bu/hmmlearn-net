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
        if (Params.Contains('m') || Params.Contains('c'))
            throw new NotImplementedException();
    }
}
