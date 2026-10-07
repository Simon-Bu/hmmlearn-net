"""Writes oracle/golden/*.json: numbers computed by the original hmmlearn 0.3.3.

Run it in a container (see oracle/README.md); it only needs hmmlearn, numpy, scipy and scikit-learn.

Rules of the oracle:
* Every input is explicit (start probabilities, transition matrices, means, sequences...). Nothing depends on the
  random generator of numpy, so a port never has to reproduce it. Sequences are generated here with seeded generators
  and written out as data.
* Every model case is computed with both implementations ("log" and "scaling") and the script checks that they agree,
  so one expected value serves both.
* Compact covariance parameters are written in a uniform shape: rank 3 for the HMM models, rank 4 for GMMHMM
  (see oracle/README.md). Non-finite floats are written as the strings "NaN", "Infinity" and "-Infinity".
"""
import contextlib
import io
import json
import logging
import math
import os
import warnings

import numpy as np
from scipy import special

warnings.simplefilter("ignore")
logging.disable(logging.CRITICAL)
np.seterr(all="ignore")

from hmmlearn import _hmmc, _kl_divergence as kl, _utils, hmm, stats, utils, vhmm  # noqa: E402
from hmmlearn.base import ConvergenceMonitor  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
RTOL = 1e-6
ATOL = 1e-12
IMPLEMENTATIONS = ("log", "scaling")


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------

def enc(value):
    """Python / numpy value -> JSON-able value (non-finite floats become strings)."""
    if isinstance(value, dict):
        return {k: enc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [enc(v) for v in value]
    if isinstance(value, np.ndarray):
        return enc(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return f
    return value


def write(name, doc):
    """One line per case, so the files stay readable and greppable."""
    doc = enc(doc)
    lines = ["{"]
    items = list(doc.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if isinstance(value, list) and key.endswith("cases"):
            lines.append(f'"{key}": [')
            lines.append(",\n".join("  " + json.dumps(case, separators=(",", ":")) for case in value))
            lines.append("]" + comma)
        else:
            lines.append(f'"{key}": ' + json.dumps(value, separators=(",", ":")) + comma)
    lines.append("}")
    path = os.path.join(OUT, name + ".json")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    size = os.path.getsize(path)
    print(f"{name}.json: {size} bytes")


def close(a, b, path="value"):
    """Recursive allclose (NaN equals NaN); raises AssertionError with the path of the first difference."""
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) or isinstance(a, np.ndarray) and a.ndim > 0:
        a, b = list(a), list(b)
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{path}[{i}]")
    else:
        if isinstance(a, (bool, np.bool_, str)) or a is None:
            assert a == b, (path, a, b)
            return
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), (path, a, b)
        elif math.isinf(a) or math.isinf(b):
            assert a == b, (path, a, b)
        else:
            assert abs(a - b) <= 1e-9 + 1e-7 * abs(a), (path, a, b)


def both(make_result):
    """make_result(implementation) -> dict; checks both implementations agree and returns the "log" result."""
    results = [make_result(impl) for impl in IMPLEMENTATIONS]
    close(results[0], results[1])
    return results[0]


def prng(seed):
    return np.random.RandomState(seed)


def stochastic(r, rows, cols):
    a = r.rand(rows, cols) + 0.05
    return a / a.sum(axis=1, keepdims=True)


def distribution(r, n):
    a = r.rand(n) + 0.05
    return a / a.sum()


def spd(r, n, scale=1.0):
    a = r.randn(n, n)
    return scale * (a @ a.T / n + 0.5 * np.eye(n))


# Uniform compact covariance shapes -------------------------------------------------------------------------------------
# HMM models:  spherical (nc,1,1)      diag (nc,1,nf)      full (nc,nf,nf)      tied (1,nf,nf)
# GMMHMM:      spherical (nc,nm,1,1)   diag (nc,nm,1,nf)   full (nc,nm,nf,nf)   tied (nc,1,nf,nf)

def to_uniform(py, ct, gmm=False):
    py = np.asarray(py, dtype=float)
    if gmm:
        return {"spherical": lambda c: c[:, :, None, None], "diag": lambda c: c[:, :, None, :],
                "tied": lambda c: c[:, None, :, :], "full": lambda c: c}[ct](py)
    if ct == "spherical":
        return py.reshape(py.shape[0], -1)[:, 0].reshape(-1, 1, 1)
    return {"diag": lambda c: c[:, None, :], "tied": lambda c: c[None], "full": lambda c: c}[ct](py)


def to_python(u, ct, gmm=False):
    u = np.asarray(u, dtype=float)
    if gmm:
        return {"spherical": lambda c: c[:, :, 0, 0], "diag": lambda c: c[:, :, 0, :],
                "tied": lambda c: c[:, 0], "full": lambda c: c}[ct](u)
    return {"spherical": lambda c: c[:, 0, 0], "diag": lambda c: c[:, 0, :],
            "tied": lambda c: c[0], "full": lambda c: c}[ct](u)


# ---------------------------------------------------------------------------------------------------------------------
# utils.py, _utils.py, stats.py
# ---------------------------------------------------------------------------------------------------------------------

def gen_utils():
    r = prng(1)
    normalize_cases = []

    def add_normalize(name, a, axis):
        a = np.array(a, dtype=float)
        b = a.copy()
        utils.normalize(b, axis=axis)
        normalize_cases.append({"name": name, "axis": axis, "input": a, "expected": b})

    add_normalize("vector", [0.2, 3.0, 0.0, 1.8], None)
    add_normalize("matrix-whole", r.rand(3, 4) + 0.1, None)
    add_normalize("matrix-rows", r.rand(4, 3) + 0.1, 1)
    add_normalize("matrix-rows-with-a-zero-row", [[1.0, 3.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 4.0]], 1)
    add_normalize("matrix-columns", r.rand(3, 4) + 0.1, 0)

    log_normalize_cases = []

    def add_log_normalize(name, a, axis):
        a = np.array(a, dtype=float)
        b = a.copy()
        utils.log_normalize(b, axis=axis)
        log_normalize_cases.append({"name": name, "axis": axis, "input": a, "expected": b})

    add_log_normalize("vector", np.log([0.2, 3.0, 0.5, 1.8]), None)
    add_log_normalize("matrix-whole", np.log(r.rand(3, 4) + 0.1), None)
    add_log_normalize("matrix-rows", np.log(r.rand(4, 3) + 0.1), 1)
    add_log_normalize("matrix-columns", np.log(r.rand(3, 4) + 0.1), 0)
    add_log_normalize("rows-with-minus-infinity", [[-math.inf, 0.0, -1.0], [-2.0, -math.inf, -math.inf]], 1)
    add_log_normalize("row-of-minus-infinity-gives-nan", [[-math.inf, -math.inf], [0.0, -1.0]], 1)
    add_log_normalize("single-column-gives-zero", [[-3.0], [-0.5], [-math.inf]], 1)
    add_log_normalize("single-row-gives-zero", [[-3.0, -0.5, -1.0]], 0)

    fill_cases = []
    nc, nf = 3, 2
    compact = {
        "spherical": np.array([0.5, 1.5, 2.5]).reshape(nc, 1, 1),
        "diag": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).reshape(nc, 1, nf),
        "tied": np.array([[1.0, 0.3], [0.3, 2.0]]).reshape(1, nf, nf),
        "full": np.array([[[1.0, 0.2], [0.2, 2.0]], [[3.0, -0.5], [-0.5, 1.0]], [[2.0, 0.0], [0.0, 2.5]]]),
    }
    for ct, u in compact.items():
        expected = utils.fill_covars(to_python(u, ct), ct, nc, nf)
        fill_cases.append({"name": ct, "covariance_type": ct, "n_components": nc, "n_features": nf,
                           "covars": u, "expected": expected})
    write("utils", {"module": "utils.py", "tolerance": {"rtol": RTOL, "atol": ATOL},
                    "normalize_cases": normalize_cases, "log_normalize_cases": log_normalize_cases,
                    "fill_covars_cases": fill_cases})


def gen_internal_utils():
    r = prng(2)
    logdet_cases = []
    for name, m in [("pd-3x3", spd(r, 3)), ("pd-1x1", np.array([[2.5]])), ("pd-2x2", np.array([[2.0, 0.5], [0.5, 1.0]])),
                    ("negative-determinant-gives-nan", np.array([[0.0, 1.0], [1.0, 0.0]])),
                    ("singular-gives-minus-infinity", np.array([[1.0, 1.0], [1.0, 1.0]]))]:
        logdet_cases.append({"name": name, "matrix": m, "expected": float(_utils.logdet(m))})

    distribute_cases = []
    nf = 3
    tied = spd(r, nf) + np.array([[0, 0.2, 0], [0.2, 0, 0.1], [0, 0.1, 0]])
    for ct in ("spherical", "diag", "tied", "full"):
        for nc in (1, 3):
            expected = _utils.distribute_covar_matrix_to_match_covariance_type(tied, ct, nc)
            if ct == "spherical":
                # numpy returns (nc, nf) filled with the mean of ALL entries of the matrix.
                uniform = np.asarray(expected)[:, 0].reshape(nc, 1, 1)
            elif ct == "diag":
                uniform = np.asarray(expected)[:, None, :]
            elif ct == "tied":
                uniform = np.asarray(expected)[None]
            else:
                uniform = np.asarray(expected)
            distribute_cases.append({"name": f"{ct}-{nc}", "covariance_type": ct, "n_components": nc,
                                     "tied_covar": tied, "expected": uniform})

    validate_cases = []

    def add_validate(name, ct, nc, uniform):
        py = to_python(uniform, ct)
        try:
            _utils._validate_covars(np.array(py), ct, nc)
            raises = False
        except ValueError:
            raises = True
        validate_cases.append({"name": name, "covariance_type": ct, "n_components": nc, "covars": uniform, "raises": raises})

    add_validate("spherical-valid", "spherical", 3, np.array([0.5, 1.0, 2.0]).reshape(3, 1, 1))
    add_validate("spherical-zero", "spherical", 3, np.array([0.5, 0.0, 2.0]).reshape(3, 1, 1))
    add_validate("spherical-negative", "spherical", 3, np.array([0.5, -1.0, 2.0]).reshape(3, 1, 1))
    add_validate("spherical-wrong-count", "spherical", 3, np.array([0.5, 1.0]).reshape(2, 1, 1))
    add_validate("diag-valid", "diag", 3, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).reshape(3, 1, 2))
    add_validate("diag-zero", "diag", 3, np.array([[1.0, 2.0], [0.0, 4.0], [5.0, 6.0]]).reshape(3, 1, 2))
    add_validate("tied-valid", "tied", 3, np.array([[2.0, 0.5], [0.5, 1.0]]).reshape(1, 2, 2))
    add_validate("tied-not-symmetric", "tied", 3, np.array([[2.0, 0.5], [0.4, 1.0]]).reshape(1, 2, 2))
    add_validate("tied-not-positive-definite", "tied", 3, np.array([[1.0, 2.0], [2.0, 1.0]]).reshape(1, 2, 2))
    full_ok = np.array([[[1.0, 0.2], [0.2, 2.0]], [[3.0, -0.5], [-0.5, 1.0]], [[2.0, 0.0], [0.0, 2.5]]])
    add_validate("full-valid", "full", 3, full_ok)
    bad = full_ok.copy()
    bad[1] = [[3.0, -0.5], [-0.4, 1.0]]
    add_validate("full-one-not-symmetric", "full", 3, bad)
    bad = full_ok.copy()
    bad[2] = [[1.0, 3.0], [3.0, 1.0]]
    add_validate("full-one-not-positive-definite", "full", 3, bad)

    split_cases = []
    x = np.arange(12, dtype=float).reshape(6, 2)
    for name, lengths in [("three-sequences", [2, 3, 1]), ("one-sequence", [6]), ("no-lengths", None), ("unit-sequences", [1] * 6)]:
        parts = _utils.split_X_lengths(x, lengths)
        split_cases.append({"name": name, "X": x, "lengths": lengths, "expected": [p for p in parts]})
    try:
        _utils.split_X_lengths(x, [2, 2])
        raise AssertionError("expected ValueError")
    except ValueError:
        split_cases.append({"name": "lengths-do-not-sum-to-n-samples", "X": x, "lengths": [2, 2], "raises": True})
    write("internal_utils", {"module": "_utils.py", "tolerance": {"rtol": RTOL, "atol": ATOL},
                             "logdet_cases": logdet_cases, "distribute_covar_cases": distribute_cases,
                             "validate_covars_cases": validate_cases, "split_lengths_cases": split_cases})


def gen_stats():
    r = prng(3)
    cases = []

    def add(name, ct, x, means, uniform):
        py = to_python(uniform, ct)
        try:
            expected = stats.log_multivariate_normal_density(x, means, py, ct)
            cases.append({"name": name, "covariance_type": ct, "X": x, "means": means, "covars": uniform, "expected": expected})
        except ValueError:
            cases.append({"name": name, "covariance_type": ct, "X": x, "means": means, "covars": uniform, "raises": True})

    for nf in (1, 2, 3):
        nc = 3
        x = r.randn(7, nf) * 2
        means = r.randn(nc, nf) * 2
        add(f"spherical-nf{nf}", "spherical", x, means, (r.rand(nc) + 0.2).reshape(nc, 1, 1))
        add(f"diag-nf{nf}", "diag", x, means, (r.rand(nc, nf) + 0.2).reshape(nc, 1, nf))
        add(f"tied-nf{nf}", "tied", x, means, spd(r, nf).reshape(1, nf, nf))
        add(f"full-nf{nf}", "full", x, means, np.array([spd(r, nf) for _ in range(nc)]))
    x = np.array([[0.0, 1.0], [2.0, -1.0]])
    means = np.array([[0.0, 1.0], [1.0, 1.0]])
    add("diag-zero-variance-at-the-mean", "diag", x, means, np.array([[0.0, 1.0], [1.0, 1.0]]).reshape(2, 1, 2))
    add("full-singular-needs-the-min-covar-fallback", "full", x, means, np.array([[[1.0, 1.0], [1.0, 1.0]], np.eye(2)]))
    add("full-not-positive-definite-raises", "full", x, means, np.array([-np.eye(2), np.eye(2)]))
    write("stats", {"module": "stats.py", "tolerance": {"rtol": RTOL, "atol": ATOL}, "cases": cases})


# ---------------------------------------------------------------------------------------------------------------------
# _hmmc.cpp
# ---------------------------------------------------------------------------------------------------------------------

def gen_hmmc():
    cases = []

    def add(name, startprob, transmat, frameprob, note=None):
        sp, tm, fp = (np.asarray(v, dtype=float) for v in (startprob, transmat, frameprob))
        lfp = np.log(fp)
        case = {"name": name, "startprob": sp, "transmat": tm, "frameprob": fp, "log_frameprob": lfp}
        if note:
            case["note"] = note
        try:
            log_prob, fwd, scaling = _hmmc.forward_scaling(sp, tm, fp)
            bwd = _hmmc.backward_scaling(sp, tm, fp, scaling)
            case["forward_scaling"] = {"log_prob": log_prob, "fwdlattice": fwd, "scaling": scaling}
            case["backward_scaling"] = {"bwdlattice": bwd}
            case["scaling_xi_sum"] = _hmmc.compute_scaling_xi_sum(fwd, tm, bwd, fp)
        except Exception as error:  # noqa: BLE001 - pybind11 turns the C++ range_error into a Python exception
            case["forward_scaling"] = {"raises": True, "error": type(error).__name__}
        log_prob, fwd = _hmmc.forward_log(sp, tm, lfp)
        bwd = _hmmc.backward_log(sp, tm, lfp)
        case["forward_log"] = {"log_prob": log_prob, "fwdlattice": fwd}
        case["backward_log"] = {"bwdlattice": bwd}
        if np.isfinite(log_prob):
            case["log_xi_sum"] = _hmmc.compute_log_xi_sum(fwd, tm, bwd, lfp)
        v_log_prob, states = _hmmc.viterbi(sp, tm, lfp)
        case["viterbi"] = {"log_prob": v_log_prob, "states": np.asarray(states)}
        cases.append(case)

    # Wikipedia forward-backward example (hmmlearn tests/test_base.py).
    add("wikipedia", [0.5, 0.5], [[0.7, 0.3], [0.3, 0.7]], [[0.9, 0.2], [0.9, 0.2], [0.1, 0.8], [0.9, 0.2], [0.9, 0.2]])
    r = prng(4)
    add("random-3-states-12-samples", distribution(r, 3), stochastic(r, 3, 3), r.rand(12, 3) + 0.01)
    add("random-4-states-25-samples", distribution(r, 4), stochastic(r, 4, 4), r.rand(25, 4) + 0.01)
    left_right = [[0.5, 0.5, 0.0], [0.0, 0.6, 0.4], [0.0, 0.0, 1.0]]
    add("left-to-right-with-zeros", [1.0, 0.0, 0.0], left_right, r.rand(9, 3) + 0.05)
    add("one-sample", [0.3, 0.7], [[0.9, 0.1], [0.2, 0.8]], [[0.4, 0.6]])
    add("two-samples", [0.3, 0.7], [[0.9, 0.1], [0.2, 0.8]], [[0.4, 0.6], [0.5, 0.1]])
    add("zero-frame-probabilities", [0.4, 0.6], [[0.7, 0.3], [0.1, 0.9]], [[0.5, 0.0], [0.2, 0.3], [0.0, 0.4], [0.6, 0.1]])
    add("impossible-observation", [0.5, 0.5], [[0.7, 0.3], [0.3, 0.7]], [[0.5, 0.5], [0.0, 0.0], [0.5, 0.5]],
        "forward_scaling raises (sum < 1e-300); the log implementation returns log_prob = -inf")
    add("viterbi-ties", [1 / 3.0] * 3, [[1 / 3.0] * 3] * 3, [[0.5] * 3] * 4,
        "every path has the same probability: the final state is the first maximum (index 0); each earlier state is the LAST "
        "maximum (index n_components - 1), as in the C++ backtracking over (value, index) pairs")
    add("viterbi-partial-ties", [0.5, 0.5, 0.0], [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
        [[0.4, 0.4, 0.2], [0.3, 0.3, 0.4], [0.5, 0.5, 0.1]])
    write("hmmc", {"module": "ext/_hmmc.cpp", "tolerance": {"rtol": RTOL, "atol": ATOL}, "cases": cases})


# ---------------------------------------------------------------------------------------------------------------------
# ConvergenceMonitor
# ---------------------------------------------------------------------------------------------------------------------

def gen_monitor():
    cases = []

    def add(name, tol, n_iter, values):
        m = ConvergenceMonitor(tol, n_iter, False)
        steps = []
        for v in values:
            m.report(v)
            steps.append({"log_prob": v, "converged": bool(m.converged), "iter": m.iter, "history": list(m.history)})
        verbose = ConvergenceMonitor(tol, n_iter, True)
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            for v in values:
                verbose.report(v)
        cases.append({"name": name, "tol": tol, "n_iter": n_iter, "steps": steps, "stderr_lines": buffer.getvalue().splitlines()})

    add("converged-by-iterations", 1e-3, 2, [-0.01, -0.1])
    add("converged-by-log-prob", 1e-3, 10, [-0.03, -0.02, -0.01, -0.0101])
    add("decreasing-log-prob-is-converged-and-warned", 1e-3, 10, [-0.01, -0.02, -0.5])
    add("never-converged", 1e-6, 10, [-5.0, -4.0, -3.0, -2.5, -2.2])
    add("verbose-format", 1e-3, 10, [-12.5, -9.123456789, -9.1234, 1234.5, -0.000012345])
    write("monitor", {"module": "ConvergenceMonitor", "tolerance": {"rtol": RTOL, "atol": ATOL},
                      "template": "{iter:>10d} {log_prob:>16.8f} {delta:>+16.8f}", "cases": cases})


# ---------------------------------------------------------------------------------------------------------------------
# Emission models (EM)
# ---------------------------------------------------------------------------------------------------------------------

def infer(h, x, lengths):
    log_prob, posteriors = h.score_samples(x, lengths)
    v_log_prob, v_states = h.decode(x, lengths, algorithm="viterbi")
    m_log_prob, m_states = h.decode(x, lengths, algorithm="map")
    return {"score": h.score(x, lengths), "log_prob": log_prob, "posteriors": posteriors,
            "viterbi": {"log_prob": v_log_prob, "states": np.asarray(v_states)},
            "map": {"log_prob": m_log_prob, "states": np.asarray(m_states)}}


def safe(function):
    """The value, or None when the model has not got it (a variational model fitted without 's' has no startprob_)."""
    try:
        return function()
    except Exception:  # noqa: BLE001
        return None


def fit_summary(h, x, lengths, parameters):
    out = {name: safe(lambda getter=getter: getter(h)) for name, getter in parameters.items()}
    out["history"] = list(h.monitor_.history)
    out["n_iter_done"] = h.monitor_.iter
    out["converged"] = bool(h.monitor_.converged)
    out["score"] = safe(lambda: h.score(x, lengths))
    if hasattr(h, "aic"):  # the variational models have no information criteria
        out["aic"] = h.aic(x, lengths)
        out["bic"] = h.bic(x, lengths)
    return out


def sample_with(h, n, seed):
    return h.sample(n, random_state=prng(seed))


# ---- CategoricalHMM -------------------------------------------------------------------------------------------------

def gen_categorical():
    inference, fits = [], []

    def add_inference(name, sp, tm, em, x, lengths):
        nf = np.asarray(em).shape[1]

        def make(impl):
            h = hmm.CategoricalHMM(len(sp), init_params="", n_features=nf, implementation=impl)
            h.startprob_, h.transmat_, h.emissionprob_ = np.array(sp), np.array(tm), np.array(em)
            return infer(h, np.array(x), lengths)

        inference.append({"name": name, "n_components": len(sp), "n_features": nf, "startprob": sp, "transmat": tm,
                          "emissionprob": em, "X": x, "lengths": lengths, "expected": both(make)})

    add_inference("wikipedia-rainy-sunny", [0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
                  [[0], [1], [2]], [3])
    r = prng(11)
    sp, tm, em = distribution(r, 3), stochastic(r, 3, 3), stochastic(r, 3, 5)
    add_inference("three-states-three-sequences", sp, tm, em, r.randint(0, 5, size=(24, 1)), [10, 8, 6])
    add_inference("unit-sequence-and-others", sp, tm, em, r.randint(0, 5, size=(9, 1)), [1, 5, 3])
    add_inference("left-to-right-with-zeros", [1.0, 0.0, 0.0], [[0.5, 0.5, 0.0], [0.0, 0.6, 0.4], [0.0, 0.0, 1.0]],
                  [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], r.randint(0, 3, size=(14, 1)), [14])

    def add_fit(name, constructor, init, x, lengths):
        def make(impl):
            kw = dict(constructor)
            h = hmm.CategoricalHMM(init_params="", implementation=impl, **kw)
            h.startprob_, h.transmat_, h.emissionprob_ = (np.array(init[k]) for k in ("startprob", "transmat", "emissionprob"))
            h.fit(np.array(x), lengths)
            return fit_summary(h, np.array(x), lengths, {
                "startprob": lambda m: m.startprob_, "transmat": lambda m: m.transmat_,
                "emissionprob": lambda m: m.emissionprob_, "n_features": lambda m: m.n_features})

        fits.append({"name": name, "constructor": constructor, "init": init, "X": x, "lengths": lengths, "expected": both(make)})

    true = hmm.CategoricalHMM(2, init_params="", n_features=3)
    true.startprob_, true.transmat_ = np.array([0.6, 0.4]), np.array([[0.7, 0.3], [0.4, 0.6]])
    true.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    xs, _ = sample_with(true, 40, 21)
    init = {"startprob": [0.5, 0.5], "transmat": [[0.6, 0.4], [0.5, 0.5]], "emissionprob": [[0.3, 0.3, 0.4], [0.4, 0.3, 0.3]]}
    add_fit("two-states-defaults", {"n_components": 2, "n_features": 3, "n_iter": 10, "tol": 1e-2, "params": "ste"}, init, xs, [15, 15, 10])
    add_fit("two-states-tight-tolerance", {"n_components": 2, "n_features": 3, "n_iter": 25, "tol": 1e-9, "params": "ste"}, init, xs, [40])
    add_fit("emission-only", {"n_components": 2, "n_features": 3, "n_iter": 8, "tol": 1e-9, "params": "e"}, init, xs, [20, 20])
    add_fit("start-and-transitions-only", {"n_components": 2, "n_features": 3, "n_iter": 8, "tol": 1e-9, "params": "st"}, init, xs, [20, 20])
    add_fit("priors-below-one-clip-to-zero", {"n_components": 2, "n_features": 3, "n_iter": 6, "tol": 1e-9, "params": "ste",
                                              "startprob_prior": 0.5, "transmat_prior": 0.2, "emissionprob_prior": 0.1}, init, xs, [20, 20])
    add_fit("priors-above-one", {"n_components": 2, "n_features": 3, "n_iter": 6, "tol": 1e-9, "params": "ste",
                                 "startprob_prior": 3.0, "transmat_prior": 2.5, "emissionprob_prior": 4.0}, init, xs, [20, 20])
    r = prng(12)
    xs3 = r.randint(0, 4, size=(36, 1))
    add_fit("left-to-right-keeps-zeros", {"n_components": 3, "n_features": 4, "n_iter": 8, "tol": 1e-9, "params": "ste"},
            {"startprob": [1.0, 0.0, 0.0], "transmat": [[0.5, 0.5, 0.0], [0.0, 0.6, 0.4], [0.0, 0.0, 1.0]],
             "emissionprob": [[0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4]]}, xs3, [12, 12, 12])
    add_fit("unit-sequences-and-explicit-n-features", {"n_components": 2, "n_features": 5, "n_iter": 5, "tol": 1e-9, "params": "ste"},
            {"startprob": [0.5, 0.5], "transmat": [[0.6, 0.4], [0.5, 0.5]], "emissionprob": [[0.2] * 5, [0.1, 0.2, 0.3, 0.2, 0.2]]},
            xs[:7], [1, 1, 5])
    write("categorical", {"module": "CategoricalHMM", "tolerance": {"rtol": RTOL, "atol": ATOL},
                          "inference_cases": inference, "fit_cases": fits})


# ---- MultinomialHMM -------------------------------------------------------------------------------------------------

def gen_multinomial():
    inference, fits = [], []
    r = prng(31)
    sp, tm = [0.6, 0.4], [[0.8, 0.2], [0.2, 0.8]]
    em = [[0.5, 0.3, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]]
    x = np.array([[1, 1, 3, 0], [3, 1, 1, 0], [3, 0, 2, 0], [2, 2, 0, 1], [2, 2, 0, 1],
                  [0, 1, 1, 3], [1, 0, 3, 1], [2, 0, 1, 2], [0, 2, 1, 2], [1, 0, 1, 3]])

    def add_inference(name, sp, tm, em, x, lengths, n_trials):
        # hmmlearn refuses to score with n_trials=None; the row sums stand in for it. The oracle records null: the port
        # accepts totals that vary from one sample to the next when no n_trials is set.
        python_n_trials = np.asarray(x).sum(axis=1) if n_trials is None else n_trials

        def make(impl):
            h = hmm.MultinomialHMM(len(sp), n_trials=python_n_trials, init_params="", implementation=impl)
            h.startprob_, h.transmat_, h.emissionprob_ = np.array(sp), np.array(tm), np.array(em)
            return infer(h, np.array(x), lengths)

        inference.append({"name": name, "n_components": len(sp), "n_trials": n_trials, "startprob": sp, "transmat": tm,
                          "emissionprob": em, "X": x, "lengths": lengths, "expected": both(make)})

    add_inference("five-trials-two-sequences", sp, tm, em, x, [6, 4], 5)
    # Same data seen as the categorical special case (n_trials = 1): the Wikipedia example.
    add_inference("one-trial-is-categorical", [0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
                  [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [3], 1)
    varied = np.array([[1, 1, 3, 0], [6, 2, 2, 0], [3, 0, 2, 0], [4, 4, 0, 2], [0, 1, 1, 1], [1, 0, 3, 1]])
    add_inference("trial-counts-vary-per-sample", sp, tm, em, varied, [3, 3], None)

    def add_fit(name, constructor, init, x, lengths):
        def make(impl):
            h = hmm.MultinomialHMM(init_params="", implementation=impl, **constructor)
            h.startprob_, h.transmat_, h.emissionprob_ = (np.array(init[k]) for k in ("startprob", "transmat", "emissionprob"))
            h.fit(np.array(x), lengths)
            return fit_summary(h, np.array(x), lengths, {
                "startprob": lambda m: m.startprob_, "transmat": lambda m: m.transmat_,
                "emissionprob": lambda m: m.emissionprob_, "n_features": lambda m: m.n_features})

        fits.append({"name": name, "constructor": constructor, "init": init, "X": x, "lengths": lengths, "expected": both(make)})

    init = {"startprob": [0.5, 0.5], "transmat": [[0.6, 0.4], [0.5, 0.5]], "emissionprob": [[0.3, 0.3, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3]]}
    true = hmm.MultinomialHMM(2, n_trials=5, init_params="")
    true.startprob_, true.transmat_, true.emissionprob_ = np.array(sp), np.array(tm), np.array(em)
    xs, _ = sample_with(true, 30, 32)
    add_fit("five-trials-defaults", {"n_components": 2, "n_trials": 5, "n_iter": 10, "tol": 1e-2, "params": "ste"}, init, xs, [10, 10, 10])
    add_fit("five-trials-tight-tolerance", {"n_components": 2, "n_trials": 5, "n_iter": 20, "tol": 1e-9, "params": "ste"}, init, xs, [30])
    add_fit("emission-only", {"n_components": 2, "n_trials": 5, "n_iter": 6, "tol": 1e-9, "params": "e"}, init, xs, [15, 15])
    doubled = xs.copy()
    doubled[::2] *= 2
    add_fit("trial-counts-inferred-and-varying", {"n_components": 2, "n_trials": None, "n_iter": 6, "tol": 1e-9, "params": "ste"},
            init, doubled, [10, 10, 10])
    write("multinomial", {"module": "MultinomialHMM", "tolerance": {"rtol": RTOL, "atol": ATOL},
                          "inference_cases": inference, "fit_cases": fits})


# ---- PoissonHMM -----------------------------------------------------------------------------------------------------

def gen_poisson():
    inference, fits = [], []
    sp, tm = [0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]]
    lam = [[3.1, 1.4, 4.5], [1.6, 5.3, 0.1]]
    true = hmm.PoissonHMM(2, init_params="")
    true.startprob_, true.transmat_, true.lambdas_ = np.array(sp), np.array(tm), np.array(lam)
    true.n_features = 3
    x, _ = sample_with(true, 30, 41)

    def add_inference(name, x, lengths):
        def make(impl):
            h = hmm.PoissonHMM(2, init_params="", implementation=impl)
            h.startprob_, h.transmat_, h.lambdas_ = np.array(sp), np.array(tm), np.array(lam)
            return infer(h, np.array(x), lengths)

        inference.append({"name": name, "n_components": 2, "startprob": sp, "transmat": tm, "lambdas": lam, "X": x,
                          "lengths": lengths, "expected": both(make)})

    add_inference("two-states-three-features", x, [12, 10, 8])
    add_inference("zero-counts", [[0, 0, 0], [1, 0, 2], [0, 5, 0], [0, 0, 0]], [4])

    def add_fit(name, constructor, init, x, lengths):
        def make(impl):
            h = hmm.PoissonHMM(init_params="", implementation=impl, **constructor)
            h.startprob_, h.transmat_, h.lambdas_ = (np.array(init[k]) for k in ("startprob", "transmat", "lambdas"))
            h.fit(np.array(x), lengths)
            return fit_summary(h, np.array(x), lengths, {
                "startprob": lambda m: m.startprob_, "transmat": lambda m: m.transmat_, "lambdas": lambda m: m.lambdas_})

        fits.append({"name": name, "constructor": constructor, "init": init, "X": x, "lengths": lengths, "expected": both(make)})

    init = {"startprob": [0.5, 0.5], "transmat": [[0.6, 0.4], [0.5, 0.5]], "lambdas": [[2.0, 2.0, 3.0], [3.0, 3.0, 1.0]]}
    add_fit("defaults", {"n_components": 2, "n_iter": 10, "tol": 1e-2, "params": "stl"}, init, x, [10, 10, 10])
    add_fit("tight-tolerance", {"n_components": 2, "n_iter": 25, "tol": 1e-9, "params": "stl"}, init, x, [30])
    add_fit("lambdas-only", {"n_components": 2, "n_iter": 6, "tol": 1e-9, "params": "l"}, init, x, [15, 15])
    add_fit("gamma-prior-uses-total-posterior-mass", {"n_components": 2, "n_iter": 6, "tol": 1e-9, "params": "stl",
                                                      "lambdas_prior": 2.0, "lambdas_weight": 1.5}, init, x, [15, 15])
    write("poisson", {"module": "PoissonHMM", "tolerance": {"rtol": RTOL, "atol": ATOL},
                      "inference_cases": inference, "fit_cases": fits})


# ---- GaussianHMM ----------------------------------------------------------------------------------------------------

def gaussian_parameters(ct, nc, nf, seed):
    r = prng(seed)
    means = r.randn(nc, nf) * 4
    if ct == "spherical":
        covars = r.rand(nc) + 0.5
    elif ct == "diag":
        covars = r.rand(nc, nf) + 0.5
    elif ct == "tied":
        covars = spd(r, nf)
    else:
        covars = np.array([spd(r, nf) for _ in range(nc)])
    return distribution(r, nc), stochastic(r, nc, nc), means, covars


def gen_gaussian():
    for ct in ("spherical", "diag", "tied", "full"):
        inference, fits = [], []
        nc, nf = 3, 2
        sp, tm, means, covars = gaussian_parameters(ct, nc, nf, 50)

        def configured(impl, sp, tm, means, covars, **kw):
            h = hmm.GaussianHMM(nc, ct, init_params="", implementation=impl, **kw)
            h.startprob_, h.transmat_, h.means_ = np.array(sp), np.array(tm), np.array(means)
            h.covars_ = np.array(covars)
            h.n_features = np.asarray(means).shape[1]
            return h

        true = configured("log", sp, tm, means, covars)
        x, _ = sample_with(true, 36, 51)
        for name, lengths in [("three-sequences", [14, 12, 10]), ("with-a-unit-sequence", [1, 20, 15])]:
            def make(impl, lengths=lengths):
                return infer(configured(impl, sp, tm, means, covars), x, lengths)

            inference.append({"name": name, "n_components": nc, "n_features": nf, "covariance_type": ct, "startprob": sp,
                              "transmat": tm, "means": means, "covars": to_uniform(covars, ct), "X": x, "lengths": lengths,
                              "expected": both(make)})

        def add_fit(name, constructor, init, x, lengths):
            def make(impl):
                h = configured(impl, init["startprob"], init["transmat"], init["means"], to_python(init["covars"], ct),
                               **{k: v for k, v in constructor.items() if k not in ("n_components", "covariance_type")})
                h.fit(x, lengths)
                return fit_summary(h, x, lengths, {
                    "startprob": lambda m: m.startprob_, "transmat": lambda m: m.transmat_, "means": lambda m: m.means_,
                    "covars": lambda m: to_uniform(m._covars_, ct)})

            fits.append({"name": name, "constructor": dict(constructor, n_components=nc, covariance_type=ct), "init": init,
                         "X": x, "lengths": lengths, "expected": both(make)})

        init_sp, init_tm = np.full(nc, 1 / nc), np.full((nc, nc), 1 / nc)
        init_means = means + 1.5
        init_covars = to_uniform(covars, ct) * 1.5
        init = {"startprob": init_sp, "transmat": init_tm, "means": init_means, "covars": init_covars}
        add_fit("defaults", {"n_iter": 10, "tol": 1e-2, "params": "stmc"}, init, x, [14, 12, 10])
        add_fit("tight-tolerance", {"n_iter": 20, "tol": 1e-9, "params": "stmc"}, init, x, [36])
        add_fit("means-only", {"n_iter": 6, "tol": 1e-9, "params": "m"}, init, x, [18, 18])
        # params="c" alone is not a working configuration of hmmlearn: the statistics of the means are then not accumulated.
        add_fit("means-and-covars-only", {"n_iter": 6, "tol": 1e-9, "params": "mc"}, init, x, [18, 18])
        add_fit("start-and-transitions-only", {"n_iter": 6, "tol": 1e-9, "params": "st"}, init, x, [18, 18])
        add_fit("priors", {"n_iter": 6, "tol": 1e-9, "params": "stmc", "startprob_prior": 3.0, "transmat_prior": 2.0,
                           "means_prior": 1.0, "means_weight": 2.0, "covars_prior": 0.5, "covars_weight": 4.0}, init, x, [18, 18])
        add_fit("unit-sequences", {"n_iter": 5, "tol": 1e-9, "params": "stmc"}, init, x[:9], [1, 1, 7])
        lr_tm = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
        add_fit("left-to-right-keeps-zeros", {"n_iter": 6, "tol": 1e-9, "params": "stmc"},
                {"startprob": [1.0, 0.0, 0.0], "transmat": lr_tm, "means": init_means, "covars": init_covars}, x, [18, 18])
        write(f"gaussian_{ct}", {"module": "GaussianHMM", "covariance_type": ct, "tolerance": {"rtol": RTOL, "atol": ATOL},
                                "inference_cases": inference, "fit_cases": fits})


# ---- GMMHMM ---------------------------------------------------------------------------------------------------------

def gmm_parameters(ct, nc, nm, nf, seed):
    r = prng(seed)
    means = r.randn(nc, nm, nf) * 3
    if ct == "spherical":
        covars = r.rand(nc, nm) + 0.5
    elif ct == "diag":
        covars = r.rand(nc, nm, nf) + 0.5
    elif ct == "tied":
        covars = np.array([spd(r, nf) for _ in range(nc)])
    else:
        covars = np.array([[spd(r, nf) for _ in range(nm)] for _ in range(nc)])
    weights = np.array([distribution(r, nm) for _ in range(nc)])
    return distribution(r, nc), stochastic(r, nc, nc), weights, means, covars


def gen_gmm():
    for ct in ("spherical", "diag", "tied", "full"):
        inference, fits = [], []
        nc, nm, nf = 2, 2, 2
        sp, tm, weights, means, covars = gmm_parameters(ct, nc, nm, nf, 60)

        def configured(impl, sp, tm, weights, means, covars, **kw):
            h = hmm.GMMHMM(nc, nm, covariance_type=ct, init_params="", implementation=impl, **kw)
            h.startprob_, h.transmat_ = np.array(sp), np.array(tm)
            h.weights_, h.means_, h.covars_ = np.array(weights), np.array(means), np.array(covars)
            h.n_features = nf
            return h

        true = configured("log", sp, tm, weights, means, covars)
        x, _ = sample_with(true, 40, 61)
        for name, lengths in [("three-sequences", [15, 13, 12]), ("with-a-unit-sequence", [1, 24, 15])]:
            def make(impl, lengths=lengths):
                return infer(configured(impl, sp, tm, weights, means, covars), x, lengths)

            inference.append({"name": name, "n_components": nc, "n_mix": nm, "n_features": nf, "covariance_type": ct,
                              "startprob": sp, "transmat": tm, "weights": weights, "means": means,
                              "covars": to_uniform(covars, ct, gmm=True), "X": x, "lengths": lengths, "expected": both(make)})

        def add_fit(name, constructor, init, x, lengths):
            def make(impl):
                h = configured(impl, init["startprob"], init["transmat"], init["weights"], init["means"],
                               to_python(init["covars"], ct, gmm=True),
                               **{k: v for k, v in constructor.items() if k not in ("n_components", "n_mix", "covariance_type")})
                h.fit(x, lengths)
                return fit_summary(h, x, lengths, {
                    "startprob": lambda m: m.startprob_, "transmat": lambda m: m.transmat_, "weights": lambda m: m.weights_,
                    "means": lambda m: m.means_, "covars": lambda m: to_uniform(m.covars_, ct, gmm=True)})

            fits.append({"name": name, "constructor": dict(constructor, n_components=nc, n_mix=nm, covariance_type=ct),
                         "init": init, "X": x, "lengths": lengths, "expected": both(make)})

        init = {"startprob": np.full(nc, 1 / nc), "transmat": np.full((nc, nc), 1 / nc), "weights": np.full((nc, nm), 1 / nm),
                "means": means + 1.0, "covars": to_uniform(covars, ct, gmm=True) * 1.3}
        add_fit("defaults", {"n_iter": 10, "tol": 1e-2, "params": "stmcw"}, init, x, [15, 13, 12])
        add_fit("tight-tolerance", {"n_iter": 15, "tol": 1e-9, "params": "stmcw"}, init, x, [40])
        add_fit("weights-only", {"n_iter": 5, "tol": 1e-9, "params": "w"}, init, x, [20, 20])
        add_fit("means-only", {"n_iter": 5, "tol": 1e-9, "params": "m"}, init, x, [20, 20])
        add_fit("covars-only", {"n_iter": 5, "tol": 1e-9, "params": "c"}, init, x, [20, 20])
        add_fit("explicit-priors", {"n_iter": 5, "tol": 1e-9, "params": "stmcw", "startprob_prior": 2.0, "transmat_prior": 2.0,
                                    "weights_prior": 2.0, "means_prior": 0.5, "means_weight": 1.5, "covars_prior": 0.3,
                                    "covars_weight": 3.5}, init, x, [20, 20])
        write(f"gmm_{ct}", {"module": "GMMHMM", "covariance_type": ct, "tolerance": {"rtol": RTOL, "atol": ATOL},
                            "inference_cases": inference, "fit_cases": fits})


# ---------------------------------------------------------------------------------------------------------------------
# Variational models
# ---------------------------------------------------------------------------------------------------------------------

def gen_variational_categorical():
    fits = []
    true = hmm.CategoricalHMM(3, init_params="", n_features=3)
    true.startprob_ = np.array([1 / 3.0] * 3)
    true.transmat_ = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    true.emissionprob_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    x, _ = sample_with(true, 45, 71)

    def add_fit(name, constructor, init, x, lengths):
        def make(impl):
            m = vhmm.VariationalCategoricalHMM(init_params="", implementation=impl, **constructor)
            for key, value in init.items():
                setattr(m, key + "_", np.array(value))
            m.fit(x, lengths)
            return fit_summary(m, x, lengths, {
                "startprob_prior": lambda h: h.startprob_prior_, "startprob_posterior": lambda h: h.startprob_posterior_,
                "transmat_prior": lambda h: h.transmat_prior_, "transmat_posterior": lambda h: h.transmat_posterior_,
                "emissionprob_prior": lambda h: h.emissionprob_prior_, "emissionprob_posterior": lambda h: h.emissionprob_posterior_,
                "startprob": lambda h: h.startprob_, "transmat": lambda h: h.transmat_, "emissionprob": lambda h: h.emissionprob_})

        fits.append({"name": name, "constructor": constructor, "init": init, "X": x, "lengths": lengths, "expected": both(make)})

    nc, nf = 4, 3
    init = {"startprob_prior": np.full(nc, 0.25), "startprob_posterior": np.full(nc, 0.75),
            "transmat_prior": np.full((nc, nc), 0.25), "transmat_posterior": np.array(
                [[3.0, 2.0, 1.0, 2.0], [1.0, 4.0, 2.0, 1.0], [2.0, 1.0, 3.0, 2.0], [2.0, 2.0, 2.0, 2.0]]),
            "emissionprob_prior": np.full((nc, nf), 1 / 3.0),
            "emissionprob_posterior": np.array([[0.3, 0.4, 0.3], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6], [0.2, 0.6, 0.2]]) * 5}
    base = {"n_components": nc, "n_features": nf, "params": "ste"}
    add_fit("defaults-100-iterations-max", dict(base, n_iter=100, tol=1e-6), init, x, [15, 15, 15])
    add_fit("ten-iterations", dict(base, n_iter=10, tol=1e-9), init, x, [45])
    # A variational model fitted without "t" fails in hmmlearn (fit reads transmat_, which only the "t" update sets).
    add_fit("transitions-and-emissions-only", dict(base, n_iter=6, tol=1e-9, params="te"), init, x, [20, 25])
    add_fit("start-and-transitions-only", dict(base, n_iter=6, tol=1e-9, params="st"), init, x, [20, 25])
    add_fit("unit-sequences", dict(base, n_iter=5, tol=1e-9), init, x[:8], [1, 2, 5])
    write("variational_categorical", {"module": "VariationalCategoricalHMM", "tolerance": {"rtol": RTOL, "atol": ATOL},
                                      "fit_cases": fits})


def gen_variational_gaussian():
    for ct in ("spherical", "diag", "tied", "full"):
        fits = []
        nc, nf = 3, 2
        sp, tm, means, covars = gaussian_parameters(ct, nc, nf, 80)
        true = hmm.GaussianHMM(nc, ct, init_params="")
        true.startprob_, true.transmat_, true.means_ = sp, tm, means
        true.covars_ = covars
        true.n_features = nf
        x, _ = sample_with(true, 45, 81)
        x_mean = x.mean(axis=0)
        cv = np.cov(x.T) + 1e-3 * np.eye(nf)
        counts = np.array([20.0, 14.0, 11.0])

        init = {
            "startprob_prior": np.full(nc, 1 / nc), "startprob_posterior": np.array([1.5, 1.0, 0.5]),
            "transmat_prior": np.full((nc, nc), 1 / nc), "transmat_posterior": stochastic(prng(82), nc, nc) * 15,
            "means_prior": np.full((nc, nf), x_mean), "means_posterior": means + 0.5,
            "beta_prior": np.ones(nc), "beta_posterior": counts,
        }
        if ct == "tied":
            init.update(dof_prior=float(nf), dof_posterior=float(counts.sum()),
                        scale_prior=np.eye(nf) * 1e-3, scale_posterior=cv * counts.sum(), covars=cv)
        elif ct == "full":
            init.update(dof_prior=np.full(nc, float(nf)), dof_posterior=counts,
                        scale_prior=np.broadcast_to(np.eye(nf) * 1e-3, (nc, nf, nf)).copy(),
                        scale_posterior=np.array([cv * c for c in counts]), covars=np.array([cv] * nc))
        elif ct == "diag":
            init.update(dof_prior=np.full(nc, float(nf)), dof_posterior=counts, scale_prior=np.full((nc, nf), 1e-3),
                        scale_posterior=np.array([np.diag(cv) * c for c in counts]), covars=np.array([np.diag(cv)] * nc))
        else:
            init.update(dof_prior=np.full(nc, float(nf)), dof_posterior=counts, scale_prior=np.full(nc, 1e-3),
                        scale_posterior=np.array([cv.mean() * c for c in counts]), covars=np.array([cv.mean()] * nc))

        def add_fit(name, constructor, init, x, lengths):
            def make(impl):
                m = vhmm.VariationalGaussianHMM(nc, ct, init_params="", implementation=impl, **constructor)
                for key, value in init.items():
                    if key == "covars":
                        m.covars_ = np.array(value)
                    elif key in ("dof_prior", "dof_posterior") and ct == "tied":
                        setattr(m, key + "_", float(value))
                    else:
                        setattr(m, key + "_", np.array(value))
                m.n_features = nf
                m.fit(x, lengths)
                return fit_summary(m, x, lengths, {
                    "startprob_posterior": lambda h: h.startprob_posterior_, "transmat_posterior": lambda h: h.transmat_posterior_,
                    "means_posterior": lambda h: h.means_posterior_, "beta_posterior": lambda h: h.beta_posterior_,
                    "dof_posterior": lambda h: np.atleast_1d(h.dof_posterior_),
                    "scale_posterior": lambda h: to_uniform(h.scale_posterior_, ct),
                    "covars": lambda h: to_uniform(h._covars_, ct),
                    "startprob": lambda h: h.startprob_, "transmat": lambda h: h.transmat_})

            stored = dict(init)
            for key in ("scale_prior", "scale_posterior", "covars"):
                stored[key] = to_uniform(init[key], ct)
            stored["dof_prior"] = np.atleast_1d(init["dof_prior"])
            stored["dof_posterior"] = np.atleast_1d(init["dof_posterior"])
            fits.append({"name": name, "constructor": dict(constructor, n_components=nc, covariance_type=ct), "init": stored,
                         "X": x, "lengths": lengths, "expected": both(make)})

        base = {"params": "stmc"}
        add_fit("ten-iterations", dict(base, n_iter=10, tol=1e-9), init, x, [15, 15, 15])
        add_fit("defaults-100-iterations-max", dict(base, n_iter=100, tol=1e-6), init, x, [45])
        add_fit("start-transitions-means-without-covars", dict(base, n_iter=5, tol=1e-9, params="stm"), init, x, [20, 25])
        add_fit("start-and-transitions-only", dict(base, n_iter=5, tol=1e-9, params="st"), init, x, [20, 25])
        write(f"variational_gaussian_{ct}", {"module": "VariationalGaussianHMM", "covariance_type": ct,
                                             "tolerance": {"rtol": RTOL, "atol": ATOL}, "fit_cases": fits})


# ---------------------------------------------------------------------------------------------------------------------
# KL divergences and the stationary distribution
# ---------------------------------------------------------------------------------------------------------------------

def gen_kl():
    r = prng(90)
    dirichlet, normal, mvn, gamma, wishart = [], [], [], [], []
    for name, q, p in [("equal", [1, 2, 3, 4], [1, 2, 3, 4]), ("opposite", [1, 2, 3, 4], [4, 3, 2, 1]),
                       ("uniform-prior", [2.5, 1.5, 4.0], [1.0, 1.0, 1.0]), ("small-parameters", [0.5, 0.5, 0.5], [1.0, 2.0, 3.0]),
                       ("two-components", [7.5, 0.25], [1.5, 1.5])]:
        dirichlet.append({"name": name, "q": q, "p": p, "expected": kl.kl_dirichlet(q, p)})
    for name, args in [("equal", (0, 1, 0, 1)), ("shifted-mean", (0, 1, 1, 1)), ("other-variances", (0.5, 2.0, -1.0, 0.7))]:
        normal.append({"name": name, "mean_q": args[0], "variance_q": args[1], "mean_p": args[2], "variance_p": args[3],
                       "expected": kl.kl_normal_distribution(*args)})
    for name, d in [("one-dimension", 1), ("two-dimensions", 2), ("three-dimensions", 3)]:
        mq, mp = r.randn(d), r.randn(d)
        cq, cp = spd(r, d), spd(r, d)
        mvn.append({"name": name, "mean_q": mq, "covar_q": cq, "mean_p": mp, "covar_p": cp,
                    "expected": kl.kl_multivariate_normal_distribution(mq, cq, mp, cp)})
    mvn.append({"name": "equal", "mean_q": [0.0], "covar_q": [[1.0]], "mean_p": [0.0], "covar_p": [[1.0]],
                "expected": kl.kl_multivariate_normal_distribution([0.0], [[1.0]], [0.0], [[1.0]])})
    for name, args in [("equal", (1, 0.01, 1, 0.01)), ("other-shape", (1, 0.01, 2, 0.01)), ("other-rate", (1, 0.01, 1, 0.02)),
                       ("general", (3.5, 0.4, 2.0, 0.9))]:
        gamma.append({"name": name, "b_q": args[0], "c_q": args[1], "b_p": args[2], "c_p": args[3],
                      "expected": kl.kl_gamma_distribution(*args)})
    wishart.append({"name": "equal-one-dimension", "dof_q": 952.0, "scale_q": [[339.8474024737109]], "dof_p": 952.0,
                    "scale_p": [[339.8474024737109]], "expected": kl.kl_wishart_distribution(952.0, [[339.8474024737109]], 952.0, [[339.8474024737109]])})
    wishart.append({"name": "tests-example-one-dimension", "dof_q": 952.0, "scale_q": [[339.8474024737109]], "dof_p": 1.0,
                    "scale_p": [[0.001]], "expected": kl.kl_wishart_distribution(952.0, [[339.8474024737109]], 1.0, [[0.001]])})
    for d in (2, 3):
        sq, sp_ = spd(r, d) * 10, spd(r, d) * 0.1
        wishart.append({"name": f"{d}-dimensions", "dof_q": float(d + 8), "scale_q": sq, "dof_p": float(d + 1), "scale_p": sp_,
                        "expected": kl.kl_wishart_distribution(d + 8.0, sq, d + 1.0, sp_)})
    write("kl", {"module": "_kl_divergence.py", "tolerance": {"rtol": RTOL, "atol": ATOL}, "dirichlet_cases": dirichlet,
                 "normal_cases": normal, "multivariate_normal_cases": mvn, "gamma_cases": gamma, "wishart_cases": wishart})


def gen_stationary():
    r = prng(95)
    cases = []
    for name, tm in [("two-states", [[0.9, 0.1], [0.4, 0.6]]), ("random-4-states", stochastic(r, 4, 4)),
                     ("random-6-states", stochastic(r, 6, 6)), ("absorbing-chain-with-a-transient-state", [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.5]])]:
        h = hmm.CategoricalHMM(len(tm), init_params="")
        h.transmat_ = np.array(tm)
        cases.append({"name": name, "transmat": tm, "expected": h.get_stationary_distribution()})
    write("stationary", {"module": "BaseHMM.get_stationary_distribution", "tolerance": {"rtol": RTOL, "atol": ATOL}, "cases": cases})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gen_utils()
    gen_internal_utils()
    gen_stats()
    gen_hmmc()
    gen_monitor()
    gen_categorical()
    gen_multinomial()
    gen_poisson()
    gen_gaussian()
    gen_gmm()
    gen_variational_categorical()
    gen_variational_gaussian()
    gen_kl()
    gen_stationary()
