# Oracle: numbers from hmmlearn 0.3.3

`oracle/golden/*.json` holds numbers computed by the original hmmlearn 0.3.3 (see `NOTICE.md`). They are the numerical
specification of the port: a test that compares the .NET result to them proves parity. Read-only for every agent.

## Regenerating (pilot only)

```
docker run --rm -v "<repo>:/work" -w /work python:3.12 sh -c "pip install -q -r oracle/requirements.txt && python oracle/gen_oracle.py"
```

Versions are pinned in `oracle/requirements.txt` (Python 3.12, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1).

## Conventions

- **Explicit inputs.** Every case lists its inputs (start probabilities, transition matrices, means, sequences...). Nothing
  depends on the random generator of numpy. Never try to reproduce it: sampling is tested statistically, not against numbers.
- **Tolerance.** `|actual - expected| <= atol + rtol * |expected|` with rtol = 1e-6 and atol = 1e-12 (numpy.allclose);
  NaN equals NaN; infinities must be equal. `Close.Equal(...)` in `tests/HmmLearn.Tests/Support/Golden.cs` does exactly this.
- **Both implementations.** Each model case was computed with `implementation="log"` and `"scaling"`; the generator checked
  they agree, so the same expected values hold for both.
- **Non-finite numbers** are the strings `"NaN"`, `"Infinity"`, `"-Infinity"` (JSON has no literal for them); `GoldenJson.AsDouble`
  decodes them.
- **Shapes** follow numpy (row-major): a matrix is a list of rows. `X` is `(n_samples, n_features)`; categorical symbols are
  `(n_samples, 1)` integers. `lengths` lists the lengths of the sequences concatenated in `X` (they sum to `n_samples`).
  A state sequence is a list of integers.
- **Compact covariances have a uniform shape**, whatever the covariance type (spherical, diag, full, tied):

  | Model | spherical | diag | full | tied |
  |---|---|---|---|---|
  | HMM models (rank 3) | `(nc, 1, 1)` | `(nc, 1, nf)` | `(nc, nf, nf)` | `(1, nf, nf)` |
  | `GMMHMM` (rank 4) | `(nc, nm, 1, 1)` | `(nc, nm, 1, nf)` | `(nc, nm, nf, nf)` | `(nc, 1, nf, nf)` |

  (`nc` states, `nm` mixture components, `nf` features.) hmmlearn itself uses a different shape per type (`(nc,)`, `(nc, nf)`,
  `(nc, nf, nf)`, `(nf, nf)`); the uniform shape is the .NET one. `scale_prior`, `scale_posterior` and `covars` of the
  variational Gaussian model use the HMM shapes; `dof_prior` and `dof_posterior` are vectors of `nc` numbers (one number for tied).
- **Cases.** Every file has a `module`, a `tolerance`, and one or more lists of cases (`*_cases`); each case has a unique `name`.
  Model files have `inference_cases` (parameters given, nothing fitted) and `fit_cases` (see below).
- **hmmlearn quirks kept on purpose** (the port must reproduce them; the cases that show them say so in their names or notes):
  Viterbi ties (`hmmc.json`, cases `viterbi-*`); `normalize` along axis 0 has no zero-sum guard in hmmlearn (the port guards both
  axes: no case covers it); the spherical initial covariance is the mean of ALL entries of the data covariance matrix
  (`internal_utils.json`); the Poisson gamma prior uses the total posterior mass of all states (`poisson.json`,
  `gamma-prior-uses-total-posterior-mass`); `params="c"` without `m` does not work in `GaussianHMM`; a variational model fitted
  without `t` in `params` fails.

## Files

| File | Content | Python source |
|---|---|---|
| `utils.json` | `normalize_cases` (axis null/0/1: `input`, `expected`), `log_normalize_cases`, `fill_covars_cases` (`covars` compact, `expected` full `(nc, nf, nf)`) | `utils.py` |
| `internal_utils.json` | `logdet_cases` (`matrix`, `expected`), `distribute_covar_cases` (`tied_covar`, `expected` compact), `validate_covars_cases` (`raises` true/false), `split_lengths_cases` (`X`, `lengths`, `expected` list of matrices, or `raises`) | `_utils.py` |
| `stats.json` | `cases`: `X`, `means`, `covars` (compact), `covariance_type`, `expected` `(n, nc)` log densities, or `raises` | `stats.py` |
| `hmmc.json` | `cases`: `startprob`, `transmat`, `frameprob`, `log_frameprob`, then per function the expected output: `forward_scaling` (`log_prob`, `fwdlattice`, `scaling`, or `raises`), `backward_scaling.bwdlattice`, `scaling_xi_sum`, `forward_log` (`log_prob`, `fwdlattice`), `backward_log.bwdlattice`, `log_xi_sum` (log space), `viterbi` (`log_prob`, `states`). Each function can be called with the oracle's own inputs, so no function depends on another | `ext/_hmmc.cpp` |
| `base.json` | `inference_cases` and `fit_cases` of a `BaseHMM` with a stub emission: the observations `X` ARE the log frame probabilities `(n_samples, n_components)` (the stub model returns `X` as its log likelihood and has no emission parameters). Use them for the inference and fit loop of `AbstractHmm` and `BaseHmm`; `fit_cases` use `params` among `s`, `t`, `st` and the priors `startprob_prior`, `transmat_prior` | `base.py` |
| `monitor.json` | `cases`: `tol`, `n_iter`, `steps` (`log_prob` reported, then `converged`, `iter`, `history` after it) and `stderr_lines` printed when verbose; `template` is the line format | `ConvergenceMonitor` |
| `categorical.json`, `multinomial.json`, `poisson.json`, `gaussian_<type>.json`, `gmm_<type>.json` | see below | `hmm.py`, `_emissions.py` |
| `variational_categorical.json`, `variational_gaussian_<type>.json` | `fit_cases` only, see below | `vhmm.py` |
| `variational_base.json` | `fit_cases` of a `VariationalBaseHMM` with a stub emission (observations are the log frame probabilities, the same for the plain and the subnormalized likelihood): `init` gives `startprob_prior`, `startprob_posterior`, `transmat_prior`, `transmat_posterior` (explicit, `init_params` empty); `expected` has the fitted posteriors, `startprob`, `transmat` (normalized posteriors), `history` (the lower bounds), `n_iter_done`, `converged`, `score` (null without `s`: hmmlearn only sets `startprob_` in the `s` update) | `base.py` (`VariationalBaseHMM`) |
| `kl.json` | `dirichlet_cases`, `normal_cases`, `multivariate_normal_cases`, `gamma_cases`, `wishart_cases` (arguments and `expected`) | `_kl_divergence.py` |
| `stationary.json` | `cases`: `transmat`, `expected` stationary distribution (unique); `non_unique_cases`: same fields for matrices with several closed classes, where hmmlearn returns the eigenvector that LAPACK finds first | `BaseHMM.get_stationary_distribution` |

### Model files

`inference_cases`: model parameters (`n_components`, `startprob`, `transmat`, then `emissionprob` / `lambdas` / `means` and `covars`
/ `weights`...), `X`, `lengths`, and `expected`:
`score`, `log_prob` and `posteriors` (those of `score_samples`: `(n_samples, nc)`), `viterbi` and `map` (each `log_prob` and `states`,
as returned by `decode(algorithm=...)`; the `map` log_prob is the sum of the largest posteriors, as in hmmlearn).

`fit_cases`: `constructor` (the hmmlearn constructor arguments, snake_case: `n_components`, `n_iter`, `tol`, `params`, priors,
`covariance_type`, `n_mix`, `n_features`, `n_trials`...), `init` (the explicit starting parameters: the model is built with
`init_params=""` and these values assigned, so no initialization runs), `X`, `lengths`, and `expected` after `fit(X, lengths)`:
the fitted parameters, `history` (the log probability, or for variational models the lower bound, reported at each iteration), `n_iter_done`
(`monitor_.iter`), `converged`, `score` (null when the model has no `startprob_`), `aic` and `bic` (EM models only).

Mapping of the JSON names to the .NET properties is given by the stubs of each ticket (`startprob` is `StartProb`, `transmat` is
`TransMat`, `emissionprob` is `EmissionProb`, `n_iter` is `NIter`, `init_params` is `InitParams`, and so on).

Notes: `multinomial.json` has `n_trials` null in a case where the totals of the samples vary (the port then sets no `NTrials`);
`CategoricalHMM` symbols are integers stored as numbers.
