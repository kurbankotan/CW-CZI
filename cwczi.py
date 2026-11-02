# -*- coding: utf-8 -*-
"""Correlation-Weighted Cyclical Z-Score Imputation (CW-CZI).

A model-free imputer for numerical tabular data. Each missing cell is filled with
an offset from its column mean, scaled by that column's standard deviation, where
the offset is a correlation-weighted combination of the standardised values
observed in the same row. No predictive model is fitted at any stage.

Three aggregation rules, in increasing order of accuracy on wide matrices:

    uniform   z_hat_j = mean of the row's observed z-scores
    marginal  z_hat_j = sum_k r_jk z_k / sum_k |r_jk|
    ridge     z_hat_N = R_NO (R_OO + lam I)^-1 z_O            (recommended)

The ridge rule is the regularised conditional mean on the correlation scale. It
is what keeps the estimator accurate when columns are redundant: two columns
carrying the same information are discounted by the inverse instead of both
being added to a denominator. Rows are grouped by missingness pattern, so one
Cholesky factorisation serves every row that shares a pattern.

Usage
-----
    from cwczi import CWCZImputer

    imp = CWCZImputer(weighting='ridge', lam=0.1, max_iter=30)
    X_train_filled = imp.fit_transform(X_train)
    X_test_filled  = imp.transform(X_test)          # no leakage: stored moments

`fit` records mu, sigma and R of the converged matrix; `transform` runs a single
sweep with those stored parameters, so a row is imputed identically whether it arrives alone
or inside a batch. Input is a 2-D float array (or anything array-like) with NaN
for missing. Standardise or scale beforehand if your columns differ wildly in
range; the estimator works on the correlation scale either way.

Defaults: weighting='ridge', lam=0.1, max_iter=30, stopping earlier once the
change per cycle falls below tol. The article compares the rule at five cycles,
the setting of its first version: five are enough when cells are missing
completely at random, but when missingness depends on the observed values the
column means of the observed cells are biased and the refresh needs many more
cycles to remove that bias (Section III.F of the article). Thirty cost little
when they are not needed.

Kurban Kotan, Manisa Celal Bayar University.
Reference implementation for the CW-CZI article; see README.md.
"""
import inspect

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = ['CWCZImputer', 'CWCZAuto', 'linearity_diagnostic']

LAM_GRID  = [0.01, 0.03, 0.10, 0.30, 1.00, 3.00]
ITER_GRID = [1, 2, 5]
SEED = 42


# scikit-learn renamed force_all_finite to ensure_all_finite in 1.6. Resolve the
# name once from the signature: a try/except around the call would also swallow
# the TypeError that check_array raises for a non-numeric dtype and re-raise it
# with the wrong message.
_FINITE_KW = ('ensure_all_finite'
              if 'ensure_all_finite' in inspect.signature(check_array).parameters
              else 'force_all_finite')

def _check(X, copy=True):
    return check_array(X, dtype=float, copy=copy, **{_FINITE_KW: 'allow-nan'})

def _mu(m): m = np.asarray(m, float).copy(); m[~np.isfinite(m)] = 0.;  return m
def _sd(s): s = np.asarray(s, float).copy(); s[~np.isfinite(s)|(s==0)] = 1.; return s


class CWCZImputer(TransformerMixin, BaseEstimator):
    """weighting: 'uniform' | 'marginal' | 'ridge'."""

    # An imputer must accept NaN; inf is still rejected by check_array. Declared
    # for both the >=1.6 tag API and the older one so check_estimator agrees.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        return {'allow_nan': True}

    def __init__(self, weighting='ridge', lam=0.10, max_iter=30, tol=1e-6,
                 min_abs_r=0.0, copy=True):
        self.weighting, self.lam = weighting, lam
        self.max_iter, self.tol = max_iter, tol
        self.min_abs_r, self.copy = min_abs_r, copy

    def _R(self, Xw):
        p = Xw.shape[1]
        if self.weighting == 'uniform':
            R = np.ones((p, p))
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                R = np.corrcoef(Xw, rowvar=False)
            R = np.nan_to_num(np.atleast_2d(R), nan=0., posinf=0., neginf=0.)
            if self.min_abs_r > 0:
                R = np.where(np.abs(R) < self.min_abs_r, 0., R)
        np.fill_diagonal(R, 1.0)
        return R

    def _pass(self, X, mask, mu, sd, R):
        # Satırlar bir geçiş içinde etkileşmez. Aynı eksiklik desenini paylaşan
        # satırlar tek bir çarpanlara ayırma ile doldurulur.
        X = X.copy(); Z = (X - mu) / sd
        rows = np.where(mask.any(1))[0]
        if rows.size == 0: return X
        packed = np.packbits(mask[rows], axis=1).tobytes()
        stride = (mask.shape[1] + 7) // 8
        groups = {}
        for t, i in enumerate(rows):
            groups.setdefault(packed[t*stride:(t+1)*stride], []).append(i)

        for idx in groups.values():
            o = np.where(~mask[idx[0]])[0]; m = np.where(mask[idx[0]])[0]
            if o.size == 0:
                X[np.ix_(idx, m)] = mu[m]; continue
            Zi = Z[np.ix_(idx, o)]
            if self.weighting == 'ridge':
                Roo = R[np.ix_(o, o)] + self.lam * np.eye(o.size)
                try:
                    L = np.linalg.cholesky(Roo)
                    S = np.linalg.solve(L.T, np.linalg.solve(L, Zi.T))
                except np.linalg.LinAlgError:
                    S = np.linalg.lstsq(Roo, Zi.T, rcond=None)[0]
                Zhat = (R[np.ix_(m, o)] @ S).T
            else:
                W = np.ones((m.size, o.size)) if self.weighting == 'uniform' \
                    else R[np.ix_(m, o)]
                den = np.abs(W).sum(1); den[den < 1e-12] = 1.
                Zhat = (Zi @ W.T) / den
            X[np.ix_(idx, m)] = mu[m] + sd[m] * Zhat
        return X

    def _cycle(self, X):
        mask = np.isnan(X)
        mu, sd = _mu(np.nanmean(X, 0)), _sd(np.nanstd(X, 0))
        Xw = X.copy(); Xw[mask] = np.take(mu, np.where(mask)[1])
        n = max(int(mask.sum()), 1); self.deltas_, self.n_iter_ = [], 0
        for t in range(1, int(self.max_iter) + 1):
            mu, sd = _mu(Xw.mean(0)), _sd(Xw.std(0))
            R  = self._R(Xw)
            Xn = self._pass(Xw, mask, mu, sd, R)
            d  = float(np.linalg.norm(Xn - Xw) / np.sqrt(n))
            self.deltas_.append(d); Xw, self.n_iter_ = Xn, t
            if d < self.tol: break
        # mu, sd and R must all describe the SAME matrix: the converged one. Taking
        # the moments from the state before the last sweep and R from after it left
        # transform() combining parameters from two different iterations.
        return Xw, _mu(Xw.mean(0)), _sd(Xw.std(0)), self._R(Xw)

    def fit(self, X, y=None):
        X = _check(X, self.copy)
        _, self.mu_, self.sd_, self.R_ = self._cycle(X)
        self.n_features_in_ = X.shape[1]; return self

    def fit_transform(self, X, y=None, **kw):
        X = _check(X, self.copy)
        Xw, self.mu_, self.sd_, self.R_ = self._cycle(X)
        self.n_features_in_ = X.shape[1]; return Xw

    def transform(self, X, y=None):
        check_is_fitted(self, ['mu_','sd_','R_']); X = _check(X, self.copy)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but CWCZImputer is '
                             f'expecting {self.n_features_in_} features as input.')
        return self._pass(X, np.isnan(X), self.mu_, self.sd_, self.R_)


class CWCZAuto(TransformerMixin, BaseEstimator):
    """Kuralı, lam'ı ve döngü sayısını eğitim matrisinin içinde seçer."""

    def __init__(self, rules=('ridge','marginal'), lam_grid=None, iter_grid=None,
                 val_frac=0.2, n_val=2, seed=SEED, copy=True):
        # Stored exactly as given: scikit-learn's clone() requires __init__ to
        # record its arguments unchanged. Defaults are resolved in _cand().
        self.rules, self.lam_grid, self.iter_grid = rules, lam_grid, iter_grid
        self.val_frac, self.n_val, self.seed, self.copy = val_frac, n_val, seed, copy

    # An imputer must accept NaN; inf is still rejected by check_array. Declared
    # for both the >=1.6 tag API and the older one so check_estimator agrees.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        return {'allow_nan': True}

    def _cand(self):
        rules = list(self.rules)
        lam   = list(LAM_GRID  if self.lam_grid  is None else self.lam_grid)
        iters = list(ITER_GRID if self.iter_grid is None else self.iter_grid)
        return [(w, l, t) for w in rules for t in iters
                for l in (lam if w == 'ridge' else [0.0])]

    def _select(self, X):
        cand = self._cand()
        obs = np.argwhere(~np.isnan(X))
        if len(obs) < 60: return cand[0], {}
        err = {c: [] for c in cand}
        for v in range(self.n_val):
            g = np.random.default_rng(self.seed + 7919*v)
            hold = obs[g.choice(len(obs), max(int(self.val_frac*len(obs)), 30),
                                replace=False)]
            Xv = X.copy()
            Xv[hold[:,0], hold[:,1]] = np.nan
            # A row can lose every observed cell to the hold-out, leaving the
            # imputer nothing to condition on. Put ONE cell back — the same
            # column on both sides, and dropped from the hold-out set, so the
            # restored cell is neither wrong nor scored.
            keepmask = np.ones(len(hold), bool)
            for i in np.where(np.isnan(Xv).all(1))[0]:
                back = np.where(~np.isnan(X[i]))[0]
                if not back.size:
                    continue
                j = int(g.choice(back))
                Xv[i, j] = X[i, j]
                keepmask &= ~((hold[:,0] == i) & (hold[:,1] == j))
            hold = hold[keepmask]
            truth = X[hold[:,0], hold[:,1]].copy()
            for c in cand:
                w, l, t = c
                o = CWCZImputer(weighting=w, lam=l, max_iter=t,
                                tol=0.).fit_transform(Xv.copy())
                err[c].append(float(np.sqrt(np.mean(
                    (o[hold[:,0], hold[:,1]] - truth)**2))))
        mean = {k: float(np.mean(v)) for k, v in err.items()}
        return min(mean, key=mean.get), mean

    def _build(self, X):
        (w, l, t), curve = self._select(X)
        self.rule_, self.lam_, self.max_iter_, self.val_curve_ = w, l, t, curve
        self._imp = CWCZImputer(weighting=w, lam=l, max_iter=t, tol=0.)
        return self._imp

    def fit(self, X, y=None):
        X = _check(X, self.copy); self._build(X).fit(X)
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = self.max_iter_; return self

    def fit_transform(self, X, y=None, **kw):
        X = _check(X, self.copy); out = self._build(X).fit_transform(X)
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = self.max_iter_; return out

    def transform(self, X, y=None):
        check_is_fitted(self, ['_imp'])
        X = _check(X, self.copy)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but CWCZAuto is '
                             f'expecting {self.n_features_in_} features as input.')
        return self._imp.transform(X)


# ─── Screening for non-linear dependence ─────────────────────────────────────
def _rank(v):
    """Average ranks (ties share their mean rank), as Spearman's rho uses."""
    order = np.argsort(v, kind='mergesort')
    r = np.empty(v.size); r[order] = np.arange(1, v.size + 1)
    s = v[order]; start = 0
    for k in range(1, v.size + 1):                  # average the tied blocks
        if k == v.size or s[k] != s[start]:
            if k - start > 1:
                r[order[start:k]] = (start + 1 + k) / 2.0
            start = k
    return r


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def _nonlinear_share(y, x, n_bins):
    """Share of the variance of y explained by its conditional mean given x
    beyond the best straight line (the classical lack-of-linearity component).

    x is cut into quantile bins. The between-bin sum of squares of y splits into
    the part a weighted straight line through the bin means captures and the
    deviation from that line; the deviation is corrected for the B - 2 degrees
    of freedom it has under exact linearity, so a linear relation of any
    strength scores close to zero and only curvature counts."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return np.nan
    b = np.searchsorted(edges[1:-1], x, side='right')
    cnt = np.bincount(b, minlength=edges.size - 1).astype(float)
    nz = cnt > 0; B = int(nz.sum()); m = y.size
    if B < 3 or m <= B:
        return np.nan
    sy = np.bincount(b, weights=y, minlength=cnt.size)[nz]
    sx = np.bincount(b, weights=x, minlength=cnt.size)[nz]
    c = cnt[nz]; yb, xb = sy / c, sx / c
    ybar, xbar = y.mean(), x.mean()
    sst = float(((y - ybar) ** 2).sum())
    if sst <= 0:
        return np.nan
    ssb = float((c * (yb - ybar) ** 2).sum())
    sxx = float((c * (xb - xbar) ** 2).sum())
    sxy = float((c * (xb - xbar) * (yb - ybar)).sum())
    sslin = sxy * sxy / sxx if sxx > 0 else 0.0
    msw = (sst - ssb) / (m - B)
    return (ssb - sslin - (B - 2) * msw) / sst


def linearity_diagnostic(X, n_bins=10, min_obs=30, threshold=0.10):
    """Screen an incomplete numerical matrix for dependence that a correlation
    matrix cannot represent, BEFORE choosing an imputer.

    Every rule of CW-CZI reads the data through Pearson correlations, so it can
    only exploit the linear part of the dependence between columns. Two pairwise
    discrepancies flag what it misses; both are computed on the rows where the
    two columns are observed, so the input may contain NaN:

    sp_gap   |Spearman rho| - |Pearson r| for every pair of columns. Positive
             when the relation is monotone but not linear (curvature, outliers,
             heavy tails). It cannot see a relation that changes direction.
    eta_gap  for every ordered pair, the share of the variance of column j that
             its conditional mean given column k explains beyond the best
             straight line: the correlation ratio eta^2 of j on `n_bins` quantile
             bins of k, minus the part of it a straight line through the bin
             means already captures (the classical lack-of-linearity term),
             corrected for degrees of freedom. A linear relation of any strength
             and a pair of independent columns both score close to zero; it is
             positive whenever E[x_j | x_k] bends, including U-shaped and other
             non-monotone relations that the Spearman gap cannot see.

    Returns a dict with the matrix-level summaries
        sp_gap_mean, eta_gap_mean        mean discrepancy over the pairs
        sp_share, eta_share              share of pairs whose discrepancy
                                         exceeds `threshold`
    and the full pairwise matrices `sp_gap` (p x p, symmetric) and `eta_gap`
    (p x p, row j = target, column k = conditioning column), NaN where a pair had
    fewer than `min_obs` complete rows or too few distinct values to bin.

    No threshold on these numbers is universal; the article calibrates them on
    synthetic data with a known amount of non-linearity and checks them on the
    benchmark datasets. As a rule of thumb, an eta_share well above zero means
    a model-based imputer is worth its cost on that matrix.
    """
    X = _check(X)
    n, p = X.shape
    obs = ~np.isnan(X)
    sp = np.full((p, p), np.nan); eg = np.full((p, p), np.nan)
    for j in range(p):
        for k in range(j + 1, p):
            rows = obs[:, j] & obs[:, k]
            if rows.sum() < min_obs:
                continue
            a, b = X[rows, j], X[rows, k]
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                continue
            r = _corr(a, b); rho = _corr(_rank(a), _rank(b))
            sp[j, k] = sp[k, j] = abs(rho) - abs(r)
            for t, c, y, x in ((j, k, a, b), (k, j, b, a)):
                eg[t, c] = _nonlinear_share(y, x, n_bins)
    iu = np.triu_indices(p, 1)
    spv = sp[iu]; spv = spv[np.isfinite(spv)]
    off = ~np.eye(p, dtype=bool)
    egv = eg[off]; egv = egv[np.isfinite(egv)]
    return dict(
        sp_gap_mean=float(spv.mean()) if spv.size else np.nan,
        sp_share=float((spv > threshold).mean()) if spv.size else np.nan,
        eta_gap_mean=float(egv.mean()) if egv.size else np.nan,
        eta_share=float((egv > threshold).mean()) if egv.size else np.nan,
        n_pairs=int(spv.size), sp_gap=sp, eta_gap=eg)
