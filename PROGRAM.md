# PROGRAM: port hmmlearn 0.3.3 to .NET 9

**Objective.** A .NET 9 library `HmmLearn` (namespace `HmmLearn`, one dependency: MathNet.Numerics 5.0.0) that reproduces hidden
Markov models of hmmlearn 0.3.3 (EM and variational Bayes), written by agents under the AnythingMachine factory, one ticket per item.

**done_when.** `dotnet test` is green on the whole solution, and the acceptance tests of every milestone reproduce the cases of
`oracle/golden` (numbers computed by hmmlearn 0.3.3, relative tolerance 1e-6, see `oracle/README.md`).

**Non-goals.** Plots, examples, documentation, scikit-learn compatibility (`get_params`...), serialization, array-valued priors
(priors are scalars), reproducing numpy's random generator (sampling is tested statistically), comparing the k-means with scikit-learn's.

**Reference.** `reference/hmmlearn-0.3.3/` (read-only copy of the original, tests included). Ticket `phase` = the milestone id.

## J1: foundations and discrete models
Oracle: utils, internal_utils, stats, hmmc, monitor, categorical, multinomial, stationary.
1. `Utils`: normalize, log-normalize, fill covariances (`utils.py`).
2. `InternalUtils`: log-determinant, split by lengths, covariance validation and distribution (`_utils.py`).
3. `Stats`: log multivariate normal density for the four covariance types (`stats.py`).
4. `Hmmc` forward and backward, scaling and log (`ext/_hmmc.cpp`).
5. `Hmmc` xi sums and Viterbi (`ext/_hmmc.cpp`).
6. `ConvergenceMonitor` (`base.py`).
7. `AbstractHmm` inference: score, decode, predict, sample (`base.py`, `_AbstractHMM`).
8. `AbstractHmm` fit loop, E-step and sufficient statistics (`base.py`, `_AbstractHMM`).
9. `BaseHmm`: priors, initialization, M-step, checks, stationary distribution, AIC and BIC (`base.py`, `BaseHMM`).
10. `CategoricalHmm` (`hmm.py`, `_emissions.py`).
11. `MultinomialHmm` (`hmm.py`, `_emissions.py`).

## J2: Gaussian emissions
Oracle: gaussian_spherical, gaussian_diag, gaussian_tied, gaussian_full.
12. `KMeans`: minimal deterministic Lloyd algorithm used by the default initializations.
13. `GaussianHmm`: likelihood, checks, sampling, sufficient statistics (`hmm.py`, `_emissions.py`).
14. `GaussianHmm`: initialization, M-step with priors, information criteria (`hmm.py`).

## J3: Poisson and mixtures
Oracle: poisson, gmm_spherical, gmm_diag, gmm_tied, gmm_full.
15. `PoissonHmm` (`hmm.py`, `_emissions.py`).
16. `GmmHmm`: likelihood, checks, sampling, sufficient statistics (`hmm.py`, `_emissions.py`).
17. `GmmHmm`: initialization with k-means, M-step with priors (`hmm.py`).

## J4: variational models
Oracle: kl, variational_categorical, variational_gaussian_spherical, variational_gaussian_diag, variational_gaussian_tied, variational_gaussian_full.
18. `KlDivergence`: Dirichlet, normal, multivariate normal, gamma, Wishart (`_kl_divergence.py`).
19. `VariationalBaseHmm` (`base.py`).
20. `VariationalCategoricalHmm` (`vhmm.py`).
21. `VariationalGaussianHmm`: initialization, likelihood, M-step (`vhmm.py`).
22. `VariationalGaussianHmm`: lower bound and checks (`vhmm.py`).

## J5: hardening of the follow-up findings
Origin: the open follow-up issues of the merged tickets (#6, #9, #14, #26, #32, #43). Oracle: stationary_non_unique (item 26).
23. `InternalUtils`: lengths sum without overflow, covariances without features (`_utils.py`).
24. `Stats`: lock the null, zero-row and zero-feature contracts (`stats.py`).
25. `Hmmc`: `ComputeLogXiSum` on zero samples (`ext/_hmmc.cpp`).
26. `BaseHmm`: stationary distribution when it is not unique (`base.py`).
27. `MultinomialHmm`: lock the free trial counts of `NTrials` (`hmm.py`, `_emissions.py`).
28. `PoissonHmm`: inference rejects a wrong number of columns (`_emissions.py`).
