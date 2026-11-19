"""Evaluation metrics, paired bootstrap, recalibration, decision-curve analysis and operating thresholds.

Definitions used in the paper
  * AUROC, AUPRC (average precision): scikit-learn.
  * Brier score: mean (p - y)^2.
  * ECE: confidence ECE, 10 equal-width bins on max(p, 1 - p); |accuracy - confidence| weighted by bin share; predicted class =
    the model's class (LLM: argmax of the label probabilities; ML: p >= 0.5).
  * Calibration slope: b of logit(P(y=1)) = a + b * logit(p); calibration intercept: a with the slope fixed at 1 (offset).
  * Regression: MAE (parse failures excluded), % of predictions within +-20% of the actual dose (|pred - y| <= 0.2 y; failure = miss).
  * Paired bootstrap: simple (non-stratified) patient resampling, B = 1000, numpy default_rng(2026); one index matrix per task and
    split shared by all models (paired); percentile 95% CI; two-sided p = 2 min(P(d <= 0), P(d >= 0)), floored at 1/(B+1).
  * Seed summary: delta averaged over seeds inside every bootstrap replicate (same index matrix), then the percentile CI.
  * Recalibration (fitted on validation, applied unchanged to test): Platt p' = sigmoid(a + b logit p);
    temperature p' = sigmoid(logit(p) / T).
  * Decision curve: net benefit = TP/n - FP/n * pt / (1 - pt), treat if p >= pt; treat-all = prev - (1 - prev) pt / (1 - pt).
    Range summaries over a threshold grid (step 0.01); MACE 0.10-0.30, mortality 0.30-0.70 in the paper.
  * Operating thresholds chosen on validation and applied once to test: Youden J (primary), highest threshold with validation
    sensitivity >= 0.80, F1-maximising threshold, and 0.5 for reference; candidate thresholds = all distinct validation scores,
    ties -> highest threshold; positive = score >= threshold.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

SEED = 2026
B = 1000
N_BINS = 10


# ------------------------------------------------------------------------------------------------ point metrics
def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def conf_ece(y_pred, y_true, p, n_bins=N_BINS):
    """Confidence ECE. y_pred / y_true: 0/1 (or label strings), p: positive-class probability."""
    p = np.asarray(p, float)
    conf = np.maximum(p, 1.0 - p)
    correct = (np.asarray(y_pred).astype(str) == np.asarray(y_true).astype(str)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; N = len(conf)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += abs(correct[m].mean() - conf[m].mean()) * m.sum() / N
    return float(e)


def conf_parts(p, pred, true, n_bins=N_BINS):
    """Per-patient (bin, correct - conf) so that ECE under bootstrap weights is vectorised (same bins as conf_ece)."""
    p = np.asarray(p, float); conf = np.maximum(p, 1.0 - p)
    correct = (np.asarray(pred).astype(str) == np.asarray(true).astype(str)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    return b, correct - conf


def event_parts(y, p, n_bins=N_BINS):
    """Per-patient (bin, y - p) of the reliability diagram (Supplementary Figs. S1-S2): 10 uniform bins as
    sklearn.calibration_curve, bin k = [k/10, (k+1)/10), last bin closed."""
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.searchsorted(edges[1:-1], np.asarray(p, float), side="right"), 0, n_bins - 1)
    return b, np.asarray(y).astype(float) - np.asarray(p, float)


def ece_from_parts(b, g, W=None, n_bins=N_BINS):
    Z = np.zeros((len(b), n_bins)); Z[np.arange(len(b)), b] = 1.0
    if W is None:
        return float(np.abs(g @ Z).sum() / len(b))
    return np.abs(W @ (Z * g[:, None])).sum(1) / W.sum(1)


def calib_slope_intercept(y, p):
    from scipy.optimize import minimize
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))

    def nll(z):
        pr = np.clip(1 / (1 + np.exp(-np.clip(z, -30, 30))), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    a, slope = minimize(lambda ab: nll(ab[0] + ab[1] * lp), [0.0, 1.0], method="Nelder-Mead").x
    intercept = minimize(lambda a_: nll(a_[0] + lp), [0.0], method="Nelder-Mead").x[0]
    return float(slope), float(intercept)


def within20(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ae = np.abs(yhat - y)
    return (np.where(np.isnan(ae), np.inf, ae) <= 0.2 * y + 1e-9).astype(float)


# ------------------------------------------------------------------------------------------------ bootstrap
def boot_indices(n, B=B, seed=SEED):
    """(B, n) simple resampling indices (one matrix per task and split, shared by every model -> paired)."""
    return np.random.default_rng(seed).choice(np.arange(n), size=(B, n), replace=True)


def count_matrix(IDX):
    W = np.zeros(IDX.shape)
    for k in range(IDX.shape[0]):
        W[k] = np.bincount(IDX[k], minlength=IDX.shape[1])
    return W


def boot_metric(fn, IDX, *arrs):
    out = []
    for ix in IDX:
        try:
            out.append(fn(*[a[ix] for a in arrs]))
        except Exception:
            out.append(np.nan)
    return np.array(out, float)


def pct_ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def boot_p(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return float(max(min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0), 1 / (len(d) + 1)))


def summarize(est, reps):
    lo, hi = pct_ci(reps)
    return dict(est=float(est), lo=lo, hi=hi)


def paired(est_a, est_b, reps_a, reps_b):
    """Delta a - b with paired replicates."""
    d = np.asarray(reps_a) - np.asarray(reps_b); lo, hi = pct_ci(d)
    return dict(est=float(est_a - est_b), lo=lo, hi=hi, p=boot_p(d), excludes_0=bool(lo > 0 or hi < 0))


def seed_mean_delta(ests_a, ests_b, reps_a, reps_b):
    """Seed-averaged delta: mean over seeds of (a_k - b_k), averaged inside each replicate (lists over seeds, same index matrix)."""
    d = np.mean([np.asarray(ra) - np.asarray(rb) for ra, rb in zip(reps_a, reps_b)], axis=0)
    est = float(np.mean([a - b for a, b in zip(ests_a, ests_b)])); lo, hi = pct_ci(d)
    return dict(est=est, lo=lo, hi=hi, p=boot_p(d), excludes_0=bool(lo > 0 or hi < 0))


# ------------------------------------------------------------------------------------------------ model-level evaluation
def cls_eval(y, p, pred=None, IDX=None, W=None, slope=True):
    """Point estimates (+ bootstrap replicates if IDX/W given) of AUROC, AUPRC, ECE, Brier (+ slope, intercept)."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    pred = (p >= 0.5).astype(int) if pred is None else np.asarray(pred).astype(int)
    b, g = conf_parts(p, pred, y)
    est = dict(AUROC=float(roc_auc_score(y, p)), AUPRC=float(average_precision_score(y, p)), ECE=ece_from_parts(b, g), Brier=brier(y, p))
    if slope:
        est["slope"], est["intercept"] = calib_slope_intercept(y, p)
    reps = {}
    if IDX is not None:
        W = count_matrix(IDX) if W is None else W
        reps = dict(AUROC=boot_metric(roc_auc_score, IDX, y, p), AUPRC=boot_metric(average_precision_score, IDX, y, p),
                    ECE=ece_from_parts(b, g, W), Brier=(W @ ((p - y) ** 2)) / W.sum(1))
    return est, reps


def reg_eval(y, yhat, W=None):
    """MAE (parse failures excluded) and % within +-20% (failures = miss)."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ae = np.abs(yhat - y); ok = ~np.isnan(ae); hit = within20(y, yhat)
    est = dict(MAE=float(np.nanmean(ae)), within20=float(100 * hit.mean()), n_parsed=int(ok.sum()))
    reps = {}
    if W is not None:
        reps = dict(MAE=(W @ np.where(ok, ae, 0.0)) / (W @ ok.astype(float)), within20=100 * (W @ hit) / W.sum(1))
    return est, reps


# ------------------------------------------------------------------------------------------------ recalibration
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


def fit_platt(y_valid, p_valid):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000).fit(_logit(p_valid)[:, None], y_valid)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    return dict(a=a, b=b), (lambda q: _sig(a + b * _logit(q)))


def fit_temperature(y_valid, p_valid):
    from scipy.optimize import minimize_scalar
    z = _logit(p_valid); y = np.asarray(y_valid, float)

    def nll(logT):
        q = np.clip(_sig(z / np.exp(logT)), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    T = float(np.exp(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x))
    return dict(T=T), (lambda q: _sig(_logit(q) / T))


# ------------------------------------------------------------------------------------------------ decision curves
def threshold_grid(a, b, step=0.01):
    return np.round(np.arange(a, b + 1e-9, step), 2)


def net_benefit(y, p, ts, W=None):
    """Net benefit at thresholds ts (array). With W (bootstrap count matrix) also returns the (B, len(ts)) replicates."""
    y = np.asarray(y, float); n = len(y); ts = np.asarray(ts, float); odds = ts / (1 - ts)
    P = (np.asarray(p, float)[:, None] >= ts[None, :]).astype(float)
    nb = (y[:, None] * P).sum(0) / n - ((1 - y)[:, None] * P).sum(0) / n * odds
    if W is None:
        return nb
    nW = W.sum(1)[:, None]
    nbb = (W @ (y[:, None] * P)) / nW - (W @ ((1 - y)[:, None] * P)) / nW * odds[None, :]
    return nb, nbb


def treat_all(y, ts, W=None):
    y = np.asarray(y, float); ts = np.asarray(ts, float); odds = ts / (1 - ts); prev = y.mean()
    ta = prev - (1 - prev) * odds
    if W is None:
        return ta
    pb = ((W @ y) / W.sum(1))[:, None]
    return ta, pb - (1 - pb) * odds[None, :]


def dca_range_summary(y, p_by_model, ref, a, b, IDX):
    """Range summary over pt in [a, b] (step 0.01): % thresholds beating treat-all and treat-none, range-mean delta NB vs the reference
    model and vs treat-all, each with a paired bootstrap CI (mean taken inside each replicate)."""
    W = count_matrix(IDX); ts = threshold_grid(a, b); ta, tab = treat_all(y, ts, W)
    R = {m: net_benefit(y, p, ts, W) for m, p in p_by_model.items()}
    out = {}
    for m, (nb, nbb) in R.items():
        e = dict(share_beating_both_pct=100 * float(((nb > ta) & (nb > 0)).mean()),
                 mean_dNB_vs_treat_all=paired(nb.mean(), ta.mean(), nbb.mean(1), tab.mean(1)))
        if m != ref:
            e["mean_dNB_vs_ref"] = paired(nb.mean(), R[ref][0].mean(), nbb.mean(1), R[ref][1].mean(1))
        out[m] = e
    return out


# ------------------------------------------------------------------------------------------------ operating thresholds
def choose_thresholds(y_valid, p_valid):
    """Youden J, sensitivity >= 0.80 (highest such threshold), F1-max, on validation scores; ties -> highest threshold."""
    fpr, tpr, thr = roc_curve(y_valid, p_valid, drop_intermediate=False)
    thr = np.minimum(thr, 1.0); j = tpr - fpr
    ky = int(np.argmax(j)); ks = int(np.argmax(tpr >= 0.80))
    pr, rc, th2 = precision_recall_curve(y_valid, p_valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(pr[:-1] + rc[:-1] > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0.0)
    kf = len(f1) - 1 - int(np.argmax(f1[::-1]))
    return dict(youden=float(thr[ky]), sens80=float(thr[ks]), f1max=float(th2[kf]), fixed_0_5=0.5)


def confusion_stats(y, pred, W=None):
    """Sensitivity, specificity, PPV, NPV, F1, MCC, J, accuracy (point estimates, and bootstrap replicates if W is given)."""
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    tp = ((pred == 1) & (y == 1)).astype(float); fn = ((pred == 0) & (y == 1)).astype(float)
    tn = ((pred == 0) & (y == 0)).astype(float); fp = ((pred == 1) & (y == 0)).astype(float)

    def f(w):
        TP, FN, TN, FP = w @ tp, w @ fn, w @ tn, w @ fp
        with np.errstate(invalid="ignore", divide="ignore"):
            se = TP / (TP + FN); sp = TN / (TN + FP); ppv = TP / (TP + FP); npv = TN / (TN + FN)
            f1 = 2 * TP / (2 * TP + FP + FN)
            mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            acc = (TP + TN) / (TP + TN + FP + FN)
        return dict(sens=se, spec=sp, PPV=ppv, NPV=npv, F1=f1, MCC=mcc, J=se + sp - 1, acc=acc)
    est = {k: float(v) for k, v in f(np.ones(len(y))).items()}
    return (est, f(W)) if W is not None else (est, {})
