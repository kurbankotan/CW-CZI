# -*- coding: utf-8 -*-
"""CW-CZI benchmark, version 7 — the library behind run_benchmark.py.

Everything an experiment needs lives here: the dataset registry and loader, the
three missingness mechanisms, the error metrics, the baseline imputers (with and
without internal tuning) and one function per protocol. run_benchmark.py only
schedules these functions over blocks, in parallel and resumably, and writes
their records; analyze_results.py turns the records into tables and figures.

Changes against the version first submitted (v6), in the order a reader of the
article meets them:

  * Eleven datasets instead of three. The eight new ones come from the PMLB
    collection, which redistributes them from the UCI repository; the files are
    committed byte for byte (their SHA-256 equals the Git LFS object id).
  * Exact duplicate rows are removed before anything else, so no masked cell can
    be read off an identical copy of its own row and no test row has a twin in
    the training rows.
  * Masks are drawn column by column: every masked column loses the stated share
    of its observed cells. v6 ranked all cells of the matrix together, so under
    MNAR a few heavy-tailed or discrete columns absorbed most of the mask.
  * MAR is now MAR. A random quarter of the columns is never masked and serves
    as the conditioning set; each masked column depends on one of them. In v6 the
    conditioning cell could itself be masked, which made part of "MAR" MNAR.
  * Errors are scaled by each column's standard deviation over all observed
    cells BEFORE masking (Eq. 6 of the revised article). v6 scaled by the moments
    left after masking, which under MNAR shrink and inflate the error. The v6
    scaling is still recorded (nrmse_post) as a robustness check.
  * KNN, iterative SVD and the chained-equation imputer are also run with their
    hyperparameters chosen by the same internal validation CW-CZI (A) uses.
  * The Gaussian EM baseline groups rows by missingness pattern, as CW-CZI does,
    computes its covariance with the maximum-likelihood divisor, and reports its
    iteration count.
"""
import os
import time
import platform
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from cwczi import CWCZImputer, CWCZAuto, linearity_diagnostic, _check, _mu

# ─── Settings ────────────────────────────────────────────────────────────────
SEED       = 42
SEED_D     = SEED + 10_000        # Protocol D draws masks independent of Protocol A
MAX_ROWS   = 600
TEST_SIZE  = 0.20
N_REPEATS  = 5
MASK_RATES = [0.10, 0.20, 0.30, 0.40]
MECHANISMS = ['MCAR', 'MAR', 'MNAR']
COND_FRAC  = 0.25                 # MAR: share of columns kept fully observed
MNAR_NOISE = 0.15
LAM_GRID   = [0.01, 0.03, 0.10, 0.30, 1.00, 3.00]
ITER_GRID  = [1, 2, 5]
NA_STRINGS = ['?', 'NA', 'na', 'N/A', '', ' ', '-', '--', 'nan', 'None', 'null']
ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.environ.get('CWCZI_DATA', os.path.join(ROOT, 'datasets'))

# name -> file, target column, columns to drop, provenance
DATASETS = {
    'CKD':        dict(file='kidney_disease.csv', target='classification', drop=['id'],
                       title='Chronic Kidney Disease', doi='10.24432/C5G020'),
    'Heart':      dict(file='heart_cleveland.csv', target='num', drop=[],
                       title='Heart Disease (Cleveland)', doi='10.24432/C52P4X'),
    'Mice':       dict(file='Data_Cortex_Nuclear.csv', target='class', drop=['MouseID'],
                       title='Mice Protein Expression', doi='10.24432/C50S3Z'),
    'Wine':       dict(file='pmlb/wine_quality_red.tsv.gz', target='target',
                       title='Wine Quality (red)', doi='10.24432/C56S3T'),
    'Vehicle':    dict(file='pmlb/vehicle.tsv.gz', target='target',
                       title='Statlog (Vehicle Silhouettes)', doi='10.24432/C5HG6N'),
    'WDBC':       dict(file='pmlb/breast_cancer_wisconsin_diagnostic.tsv.gz', target='target',
                       title='Breast Cancer Wisconsin (Diagnostic)', doi='10.24432/C5DW2B'),
    'Ionosphere': dict(file='pmlb/ionosphere.tsv.gz', target='target',
                       title='Ionosphere', doi='10.24432/C5W01B'),
    'Satimage':   dict(file='pmlb/satimage.tsv.gz', target='target',
                       title='Statlog (Landsat Satellite)', doi='10.24432/C55887'),
    'Sonar':      dict(file='pmlb/sonar.tsv.gz', target='target',
                       title='Connectionist Bench (Sonar)', doi='10.24432/C5T01Q'),
    'Optdigits':  dict(file='pmlb/optdigits.tsv.gz', target='target',
                       title='Optical Recognition of Handwritten Digits', doi='10.24432/C50P49'),
    'Libras':     dict(file='pmlb/movement_libras.tsv.gz', target='target',
                       title='Libras Movement', doi='10.24432/C5GC82'),
}
ORIGINAL_DATASETS = ['CKD', 'Heart', 'Mice']

# Method labels used in every record, table and figure. "-T" marks a method whose
# hyperparameters are chosen by internal validation; CW-CZI (A) is the tuned
# CW-CZI. Iterative-BR / Iterative-RF are scikit-learn's IterativeImputer with a
# Bayesian-ridge / random-forest estimator: single deterministic imputations that
# approximate, but are not, MICE and MissForest.
METHODS = ['Mean', 'CW-CZI (U)', 'CW-CZI (M)', 'CW-CZI (R)', 'CW-CZI (A)',
           'KNN', 'KNN-T', 'SVD', 'SVD-T', 'EM',
           'Iterative-BR', 'Iterative-T', 'Iterative-RF']
# Sensitivity variants, reported apart from the main comparison: the forest at
# 100 trees and ten rounds, and the fixed ridge rule run for 30 cycles instead of 5.
SENSITIVITY_METHODS = ['Iterative-RF100', 'CW-CZI (R30)']
HEAVY = {'Iterative-RF', 'Iterative-RF100'}          # what the user's machine runs
PROPOSED = 'CW-CZI (R)'
TUNED_REF = 'CW-CZI (A)'
DEFAULT_SET = ['Mean', 'CW-CZI (U)', 'CW-CZI (M)', 'CW-CZI (R)', 'KNN', 'SVD', 'EM',
               'Iterative-BR', 'Iterative-RF']
TUNED_SET = ['CW-CZI (A)', 'KNN-T', 'SVD-T', 'Iterative-T', 'EM', 'Iterative-RF']


def slug(method):
    return (method.replace(' ', '').replace('(', '_').replace(')', '')
                  .replace('-', '_').lower())


# ─── Data ────────────────────────────────────────────────────────────────────
TARGET_NAMES = ['classification', 'class', 'target', 'label', 'diagnosis', 'outcome',
                'num', 'status', 'quality', 'severity', 'y']
ID_NAMES = ['id', 'index', 'unnamed: 0', 'no', 'sample', 'mouseid', 'patient_id']


def _texty(col):
    """True for a column holding text (pandas 2 'object' or pandas 3 'str')."""
    return not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(name):
    """Load one dataset exactly as every protocol sees it.

    Returns a dict with the float matrix `arr` (numerical and label-encoded
    categorical columns), the class labels `y`, the indices of the numerical
    block `num_idx` (the only block that is masked and imputed), the indices of
    the categorical block `cat_idx` with its raw values `raw_cat` (Protocol B
    re-derives their coding from the training rows), and bookkeeping."""
    cfg = DATASETS[name]
    path = os.path.join(DATA_DIR, cfg['file'])
    if path.endswith('.tsv.gz'):
        df = pd.read_csv(path, sep='\t')
    else:
        df = pd.read_csv(path, na_values=NA_STRINGS, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:                              # strip text, map NA spellings
        if _texty(df[c]):
            df[c] = df[c].astype(str).str.strip().replace({s: np.nan for s in NA_STRINGS})
    for c in df.columns:                              # text that is really numbers
        if _texty(df[c]):
            conv = pd.to_numeric(df[c], errors='coerce'); ok = df[c].notna().sum()
            if ok and conv.notna().sum() >= .9 * ok:
                df[c] = conv
    df = df.drop(columns=[c for c in cfg.get('drop', []) if c in df.columns])
    df = df.drop(columns=[c for c in df.columns if c.lower() in ID_NAMES])
    tgt = cfg['target']
    df = df[df[tgt].notna()]
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    y_raw = df[tgt].astype(str).str.strip()
    X = df.drop(columns=[tgt])

    def numeric_cols(F):
        # ordinal integer codes stay numerical; binary flags are categorical
        return [c for c in F.columns if pd.api.types.is_numeric_dtype(F[c])
                and F[c].notna().sum() > .3 * len(F) and F[c].nunique(dropna=True) > 2]

    num_cols = numeric_cols(X)
    keep = ~X[num_cols].isna().all(axis=1)            # a row needs one numerical value
    X, y_raw = X[keep], y_raw[keep]
    cnt = y_raw.value_counts(); keep = y_raw.isin(cnt[cnt >= 10].index)
    X, y_raw = X[keep], y_raw[keep]
    n_before_sub = len(X)
    if len(X) > MAX_ROWS:
        sel = np.random.default_rng(SEED).choice(len(X), MAX_ROWS, replace=False)
        X, y_raw = X.iloc[sel], y_raw.iloc[sel]
    cnt = y_raw.value_counts(); keep = y_raw.isin(cnt[cnt >= 10].index)
    X, y_raw = X[keep].reset_index(drop=True), y_raw[keep].reset_index(drop=True)
    # a column the subsample left with two values or fewer leaves the numerical block
    num_cols = [c for c in num_cols if c in numeric_cols(X)]
    cat_cols = [c for c in X.columns if c not in num_cols]
    raw_cat = X[cat_cols].astype(object).copy() if cat_cols else None
    if cat_cols:
        # Used by Protocol A only, where the categorical block is neither masked
        # nor scored. Protocol B re-derives it from the training rows alone.
        X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(
            X[cat_cols].astype(object))
        for c in cat_cols:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    arr = X.values.astype(float)
    num_idx = [X.columns.get_loc(c) for c in num_cols]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    assert not np.isnan(np.delete(arr, num_idx, axis=1)).any()
    y = LabelEncoder().fit_transform(y_raw)
    return dict(name=name, arr=arr, y=y, num_idx=num_idx, cat_idx=cat_idx,
                raw_cat=raw_cat, num_cols=num_cols, target=tgt, n_dup=n_dup,
                n_before_subsample=n_before_sub, path=path)


def describe(D):
    """One row of Table 1 and one row of Table 2."""
    sub = D['arr'][:, D['num_idx']]; inc = np.isnan(sub).any(1)
    row = dict(Dataset=D['name'], Title=DATASETS[D['name']]['title'],
               Instances=sub.shape[0], Numeric=sub.shape[1],
               Categorical=D['arr'].shape[1] - sub.shape[1],
               Classes=int(len(np.unique(D['y']))),
               Missing_cells_pct=round(100 * np.isnan(sub).mean(), 2),
               Missing_cols=int((np.isnan(sub).sum(0) > 0).sum()),
               Incomplete_rows_pct=round(100 * inc.mean(), 1),
               Duplicates_removed=D['n_dup'], Rows_before_subsample=D['n_before_subsample'],
               File=DATASETS[D['name']]['file'], SHA256=sha256(D['path']))
    test = dict(Dataset=D['name'])
    if inc.any() and (~inc).any():
        tab = pd.crosstab(D['y'], ~inc)
        chi2, p, dof, expected = chi2_contingency(tab)
        test.update(chi2=float(chi2), dof=int(dof), p=float(p),
                    min_expected=float(expected.min()))
        if tab.shape == (2, 2):
            test['p_fisher'] = float(fisher_exact(tab.values)[1])
    return row, test


# ─── Missingness mechanisms ──────────────────────────────────────────────────
def make_mask(X, rate, mech, rng, cond_frac=COND_FRAC):
    """Hide `rate` of the observed cells of each masked column.

    MCAR  every column is masked; cells are drawn uniformly at random.
    MAR   a random `cond_frac` of the columns (at least one) is never masked and
          forms the conditioning set. Every other column is assigned one
          conditioning column at random and loses preferentially the cells of
          the rows where that column's standardised value is high. Where the
          conditioning value is itself naturally missing it counts as its column
          mean, so removal depends on observed information only.
    MNAR  every column is masked and loses preferentially its own highest values.

    MAR and MNAR rank cells by sigmoid(-z) + 0.15 u, u ~ U(0, 1), and remove the
    lowest-ranked share of each column. Cells already missing are never chosen,
    and a row that would lose its last observed cell gets one cell back.
    Returns the boolean mask and the indices of the conditioning columns."""
    obs = ~np.isnan(X); n, p = X.shape
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0); sd[sd == 0] = 1
    Z = np.nan_to_num((X - mu) / sd, nan=0.0)
    masked_cols = np.arange(p); cond = np.array([], dtype=int)
    if mech == 'MCAR':
        sc = rng.random((n, p))
    elif mech == 'MAR':
        n_cond = max(1, int(round(cond_frac * p)))
        cond = np.sort(rng.choice(p, n_cond, replace=False))
        masked_cols = np.setdiff1d(np.arange(p), cond)
        driver = rng.choice(cond, size=p)
        sc = 1 / (1 + np.exp(Z[:, driver])) + MNAR_NOISE * rng.random((n, p))
    elif mech == 'MNAR':
        sc = 1 / (1 + np.exp(Z)) + MNAR_NOISE * rng.random((n, p))
    else:
        raise ValueError(mech)
    mask = np.zeros((n, p), bool)
    for j in masked_cols:
        o = np.flatnonzero(obs[:, j])
        k = int(round(rate * o.size))
        if k:
            mask[o[np.argsort(sc[o, j], kind='stable')[:k]], j] = True
    empty = ~(obs & ~mask).any(1)
    for i in np.flatnonzero(empty):
        j = rng.choice(np.flatnonzero(mask[i])); mask[i, j] = False
    return mask, cond


# ─── Error metrics ───────────────────────────────────────────────────────────
def score(out, Xn, mask, mu_post, sd_post, sd_pre):
    """Errors on the masked cells.

    `out` is the imputer's output on the standardised input (X - mu_post)/sd_post;
    it is mapped back to the original units first. Then

      nrmse_pre   sqrt(mean(((x_hat - x) / s_j)^2)), s_j the standard deviation of
                  column j over every cell observed before masking  (primary, Eq. 6)
      nrmse_post  the same with s_j taken from the cells left after masking (v6)
      nrmse_col   per-column RMSE / s_j, averaged over the masked columns
      mae_pre     mean(|x_hat - x| / s_j)
      var_ratio   per column std(x_hat) / std(x) over its masked cells, averaged
                  over the columns with at least three masked cells whose true
                  values are not all equal (a column whose highest values tie,
                  common under MNAR in discrete columns, has std(x) = 0 up to
                  round-off and no defined ratio)"""
    xhat = out * sd_post + mu_post
    ii, jj = np.nonzero(mask)
    e = xhat[ii, jj] - Xn[ii, jj]
    e_pre, e_post = e / sd_pre[jj], e / sd_post[jj]
    per_col, ratios = [], []
    for j in np.unique(jj):
        s = jj == j
        per_col.append(np.sqrt(np.mean(e_pre[s] ** 2)))
        if s.sum() >= 3:
            t = Xn[ii[s], j].std()
            if t > 1e-9 * sd_pre[j]:
                ratios.append(xhat[ii[s], j].std() / t)
    return dict(nrmse_pre=float(np.sqrt(np.mean(e_pre ** 2))),
                nrmse_post=float(np.sqrt(np.mean(e_post ** 2))),
                nrmse_col=float(np.mean(per_col)),
                mae_pre=float(np.mean(np.abs(e_pre))),
                var_ratio=float(np.mean(ratios)) if ratios else np.nan,
                n_scored=int(mask.sum()), n_cols_masked=int(np.unique(jj).size))


# ─── Baselines ───────────────────────────────────────────────────────────────
def pattern_groups(mask):
    """Rows grouped by missingness pattern: [(rows, observed cols, missing cols)]."""
    rows = np.flatnonzero(mask.any(1))
    if rows.size == 0:
        return []
    packed = np.packbits(mask[rows], axis=1)
    groups = {}
    for t, i in enumerate(rows):
        groups.setdefault(packed[t].tobytes(), []).append(i)
    out = []
    for idx in groups.values():
        idx = np.array(idx)
        out.append((idx, np.flatnonzero(~mask[idx[0]]), np.flatnonzero(mask[idx[0]])))
    return out


def _chol_solve(A, B):
    """A^{-1} B for symmetric positive-definite A, as CW-CZI computes it."""
    try:
        L = np.linalg.cholesky(A)
        return np.linalg.solve(L.T, np.linalg.solve(L, B))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, B, rcond=None)[0]


class GaussianEMImputer(BaseEstimator, TransformerMixin):
    """Conditional mean under a multivariate normal whose mean and covariance are
    estimated by EM. Rows sharing a missingness pattern share one factorisation,
    exactly as in CW-CZI, so the two differ in what they estimate and not in how
    the linear algebra is organised."""

    def __init__(self, max_iter=30, tol=1e-6, ridge=1e-4, copy=True):
        self.max_iter, self.tol, self.ridge, self.copy = max_iter, tol, ridge, copy

    def _estep(self, Xw, groups, mu, S, want_cov):
        p = S.shape[0]; acc = np.zeros((p, p)) if want_cov else None
        for idx, o, m in groups:
            if o.size == 0:
                Xw[np.ix_(idx, m)] = mu[m]
                if want_cov:
                    acc[np.ix_(m, m)] += len(idx) * S[np.ix_(m, m)]
                continue
            Soo = S[np.ix_(o, o)] + self.ridge * np.eye(o.size)
            Smo = S[np.ix_(m, o)]
            Bt = _chol_solve(Soo, Smo.T)                        # (|o|, |m|)
            Xw[np.ix_(idx, m)] = mu[m] + (Xw[np.ix_(idx, o)] - mu[o]) @ Bt
            if want_cov:
                acc[np.ix_(m, m)] += len(idx) * (S[np.ix_(m, m)] - Smo @ Bt)
        return acc

    def _em(self, X):
        mask = np.isnan(X); n, p = X.shape
        mu = _mu(np.nanmean(X, 0))
        Xw = X.copy(); Xw[mask] = np.take(mu, np.nonzero(mask)[1])
        S = np.cov(Xw, rowvar=False, bias=True).reshape(p, p) + self.ridge * np.eye(p)
        groups = pattern_groups(mask); m_cnt = max(int(mask.sum()), 1)
        self.n_iter_ = 0
        for it in range(int(self.max_iter)):
            prev = Xw.copy()
            acc = self._estep(Xw, groups, mu, S, True)
            mu = Xw.mean(0)
            S = (np.cov(Xw, rowvar=False, bias=True).reshape(p, p) + acc / n
                 + self.ridge * np.eye(p))
            self.n_iter_ = it + 1
            if np.linalg.norm(Xw - prev) / np.sqrt(m_cnt) < self.tol:
                break
        return Xw, mu, S

    def fit(self, X, y=None):
        X = _check(X, self.copy); _, self.mu_, self.S_ = self._em(X); return self

    def fit_transform(self, X, y=None, **kw):
        X = _check(X, self.copy); Xw, self.mu_, self.S_ = self._em(X); return Xw

    def transform(self, X, y=None):
        check_is_fitted(self, ['mu_', 'S_']); X = _check(X, self.copy)
        mask = np.isnan(X); Xw = X.copy()
        self._estep(Xw, pattern_groups(mask), self.mu_, self.S_, False)
        return Xw


class IterativeSVDImputer(BaseEstimator, TransformerMixin):
    """Low-rank completion by iterated truncated SVD (Troyanskaya et al., 2001)."""

    def __init__(self, rank=5, max_iter=50, tol=1e-6, copy=True):
        self.rank, self.max_iter, self.tol, self.copy = rank, max_iter, tol, copy

    def _k(self, shape):
        return int(max(1, min(self.rank, min(shape) - 1)))

    def _fill(self, X):
        mask = np.isnan(X); mu = _mu(np.nanmean(X, 0))
        Xw = X.copy(); Xw[mask] = np.take(mu, np.nonzero(mask)[1])
        k = self._k(Xw.shape); self.n_iter_ = 0
        for it in range(int(self.max_iter)):
            prev = Xw.copy()
            U, s, Vt = np.linalg.svd(Xw - mu, full_matrices=False)
            Xw[mask] = ((U[:, :k] * s[:k]) @ Vt[:k] + mu)[mask]
            self.n_iter_ = it + 1
            if np.linalg.norm(Xw - prev) / np.sqrt(max(mask.sum(), 1)) < self.tol:
                break
        return Xw, mu

    def fit(self, X, y=None):
        X = _check(X, self.copy); self.X_, self.mu_ = self._fill(X); return self

    def fit_transform(self, X, y=None, **kw):
        X = _check(X, self.copy); self.X_, self.mu_ = self._fill(X); return self.X_

    def transform(self, X, y=None):
        check_is_fitted(self, ['X_', 'mu_']); X = _check(X, self.copy)
        k = self._k(self.X_.shape)
        V = np.linalg.svd(self.X_ - self.mu_, full_matrices=False)[2][:k].T
        mask = np.isnan(X); Xw = X.copy()
        for i in np.flatnonzero(mask.any(1)):
            o, m = np.flatnonzero(~mask[i]), np.flatnonzero(mask[i])
            if o.size < k:
                Xw[i, m] = self.mu_[m]; continue
            c = np.linalg.lstsq(V[o], X[i, o] - self.mu_[o], rcond=None)[0]
            Xw[i, m] = self.mu_[m] + V[m] @ c
        return Xw


class LeanIterativeImputer(IterativeImputer):
    """IterativeImputer that drops each fitted estimator as soon as it has filled
    its column. fit_transform returns exactly what IterativeImputer returns, but
    memory no longer grows with columns x rounds x trees (a 100-tree forest over
    90 columns and 10 rounds would otherwise keep 90,000 trees alive). With no
    estimators kept there is nothing for transform() to apply, so this class is
    used only where the protocol calls fit_transform alone (A, C and E)."""

    def _impute_one_feature(self, *args, **kwargs):
        Xt, _ = super()._impute_one_feature(*args, **kwargs)
        return Xt, None

    def transform(self, X):
        raise RuntimeError('LeanIterativeImputer keeps no estimators; '
                           'use IterativeImputer when transform() is needed')


def iterative_br():
    return IterativeImputer(max_iter=20, random_state=SEED)


def iterative_rf(n_estimators=30, max_iter=5, n_jobs=1, lean=False):
    cls = LeanIterativeImputer if lean else IterativeImputer
    return cls(estimator=RandomForestRegressor(
        n_estimators=n_estimators, random_state=SEED, n_jobs=n_jobs),
        max_iter=max_iter, random_state=SEED)


# Base estimators for the tuned chained-equation imputer: the variants of
# IterativeImputer that scikit-learn's own documentation compares, minus the
# random forest, which is Iterative-RF itself.
ITERATIVE_BASES = {
    'BayesianRidge': lambda: iterative_br(),
    'KNN-15':        lambda: IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=15),
                                              max_iter=20, random_state=SEED),
    'Nystroem-Ridge': lambda: IterativeImputer(estimator=make_pipeline(
                          Nystroem(kernel='polynomial', degree=2, random_state=0),
                          Ridge(alpha=1e3)), max_iter=20, random_state=SEED),
}
KNN_GRID = [(k, w) for k in (1, 3, 5, 10, 20, 40) for w in ('uniform', 'distance')
            if not (k == 1 and w == 'distance')]
SVD_GRID = [1, 2, 3, 5, 8, 12, 20, 30]


def holdout_sets(X, val_frac=0.2, n_val=2, seed=SEED):
    """The validation splits of CWCZAuto._select, reproduced draw for draw, so that
    every tuned method is selected on exactly the same held-out cells."""
    obs = np.argwhere(~np.isnan(X))
    out = []
    for v in range(n_val):
        g = np.random.default_rng(seed + 7919 * v)
        hold = obs[g.choice(len(obs), max(int(val_frac * len(obs)), 30), replace=False)]
        Xv = X.copy(); Xv[hold[:, 0], hold[:, 1]] = np.nan
        keepmask = np.ones(len(hold), bool)
        for i in np.where(np.isnan(Xv).all(1))[0]:
            back = np.where(~np.isnan(X[i]))[0]
            if not back.size:
                continue
            j = int(g.choice(back)); Xv[i, j] = X[i, j]
            keepmask &= ~((hold[:, 0] == i) & (hold[:, 1] == j))
        hold = hold[keepmask]
        out.append((Xv, hold, X[hold[:, 0], hold[:, 1]].copy()))
    return out


class TunedImputer(TransformerMixin, BaseEstimator):
    """A baseline whose hyperparameter is chosen inside the training matrix by the
    protocol CW-CZI (A) uses: a fifth of the observed cells is hidden, every
    candidate reconstructs them, twice over, and the candidate with the lowest
    mean RMSE is refitted on all the observed data. family: 'knn' | 'svd' |
    'iterative'."""

    def __init__(self, family='knn', val_frac=0.2, n_val=2, seed=SEED, copy=True):
        self.family, self.val_frac, self.n_val, self.seed, self.copy = \
            family, val_frac, n_val, seed, copy

    def _candidates(self, X):
        n, p = X.shape
        if self.family == 'knn':
            return [('k=%d,%s' % (k, w), (lambda k=k, w=w: KNNImputer(n_neighbors=k, weights=w)))
                    for k, w in KNN_GRID if k < n]
        if self.family == 'svd':
            ranks = sorted({min(r, min(n, p) - 1) for r in SVD_GRID if r >= 1})
            return [('rank=%d' % r, (lambda r=r: IterativeSVDImputer(rank=r))) for r in ranks]
        if self.family == 'iterative':
            return list(ITERATIVE_BASES.items())
        raise ValueError(self.family)

    def _select(self, X):
        cand = self._candidates(X)
        if np.count_nonzero(~np.isnan(X)) < 60:
            return cand[0], {}
        err = {name: [] for name, _ in cand}
        for Xv, hold, truth in holdout_sets(X, self.val_frac, self.n_val, self.seed):
            for name, make in cand:
                o = make().fit_transform(Xv.copy())
                err[name].append(float(np.sqrt(np.mean((o[hold[:, 0], hold[:, 1]] - truth) ** 2))))
        mean = {k: float(np.mean(v)) for k, v in err.items()}
        best = min(mean, key=mean.get)
        return (best, dict(cand)[best]), mean

    def _build(self, X):
        (name, make), curve = self._select(X)
        self.choice_, self.val_curve_ = name, curve
        self._imp = make()
        return self._imp

    def fit(self, X, y=None):
        X = _check(X, self.copy); self._build(X).fit(X)
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = getattr(self._imp, 'n_iter_', 1); return self

    def fit_transform(self, X, y=None, **kw):
        X = _check(X, self.copy); out = self._build(X).fit_transform(X)
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = getattr(self._imp, 'n_iter_', 1); return out

    def transform(self, X, y=None):
        check_is_fitted(self, ['_imp']); return self._imp.transform(_check(X, self.copy))


def make_imputer(method, rf_jobs=1, lean=False):
    """`lean` lets the forest imputers discard their fitted trees; only valid
    where fit_transform is the sole call (Protocols A, C and E)."""
    if method == 'Mean':         return SimpleImputer(strategy='mean')
    if method == 'CW-CZI (U)':   return CWCZImputer(weighting='uniform', max_iter=5, tol=1e-6)
    if method == 'CW-CZI (M)':   return CWCZImputer(weighting='marginal', max_iter=5, tol=1e-6)
    if method == 'CW-CZI (R)':   return CWCZImputer(weighting='ridge', lam=0.10, max_iter=5, tol=1e-6)
    if method == 'CW-CZI (R30)': return CWCZImputer(weighting='ridge', lam=0.10, max_iter=30, tol=1e-6)
    if method == 'CW-CZI (A)':   return CWCZAuto()
    if method == 'KNN':          return KNNImputer(n_neighbors=5)
    if method == 'KNN-T':        return TunedImputer('knn')
    if method == 'SVD':          return IterativeSVDImputer(rank=5)
    if method == 'SVD-T':        return TunedImputer('svd')
    if method == 'EM':           return GaussianEMImputer()
    if method == 'Iterative-BR': return iterative_br()
    if method == 'Iterative-T':  return TunedImputer('iterative')
    if method == 'Iterative-RF': return iterative_rf(30, 5, rf_jobs, lean)
    if method == 'Iterative-RF100': return iterative_rf(100, 10, rf_jobs, lean)
    raise ValueError(method)


def choice_of(imp):
    if isinstance(imp, CWCZAuto):
        return '%s/lam=%g/T=%d' % (imp.rule_, imp.lam_, imp.max_iter_)
    return getattr(imp, 'choice_', '')


def run_imputer(method, X, rf_jobs=1):
    imp = make_imputer(method, rf_jobs, lean=True)          # fit_transform only
    t = time.perf_counter(); out = imp.fit_transform(X.copy())
    return imp, out, time.perf_counter() - t


# ─── Protocol A: reconstruction against ground truth ─────────────────────────
def block_A(D, mech, rate, rep, methods, rf_jobs=1):
    Xn = D['arr'][:, D['num_idx']].copy()
    mask, cond = make_mask(Xn, rate, mech, np.random.default_rng(SEED + 977 * rep))
    Xm = Xn.copy(); Xm[mask] = np.nan
    mu, sd = np.nanmean(Xm, 0), np.nanstd(Xm, 0); sd[sd == 0] = 1
    sd_pre = np.nanstd(Xn, 0); sd_pre[sd_pre == 0] = 1
    Xs = (Xm - mu) / sd
    recs = []
    for m in methods:
        imp, out, dt = run_imputer(m, Xs, rf_jobs)
        rec = dict(dataset=D['name'], mechanism=mech, rate=rate, repeat=rep, method=m,
                   seconds=dt, n_iter=int(getattr(imp, 'n_iter_', 1)),
                   choice=choice_of(imp), n_cond=int(cond.size))
        rec.update(score(out, Xn, mask, mu, sd, sd_pre))
        recs.append(rec)
    return recs


# ─── Protocol B: downstream classification ───────────────────────────────────
def classifiers():
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LR':  LogisticRegression(max_iter=1000, random_state=SEED),
        'DT':  DecisionTreeClassifier(random_state=SEED),
        'RF':  RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
        'GNB': GaussianNB(),
        'SVC': SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=SEED),
        'GB':  GradientBoostingClassifier(n_estimators=50, random_state=SEED),
    }


def fit_categoricals(raw, tr, te):
    """Mode-fill and integer-code the categorical block from the TRAINING rows only;
    a category first seen in the test rows gets a code of its own."""
    out_tr = np.empty((len(tr), raw.shape[1])); out_te = np.empty((len(te), raw.shape[1]))
    for k, col in enumerate(raw.columns):
        a = raw[col].iloc[tr]; b = raw[col].iloc[te]
        fill = a.mode(dropna=True); fill = fill.iloc[0] if len(fill) else 'NA'
        a = a.fillna(fill).astype(str); b = b.fillna(fill).astype(str)
        codes = {v: i for i, v in enumerate(sorted(a.unique()))}; unseen = len(codes)
        out_tr[:, k] = [codes[v] for v in a]
        out_te[:, k] = [codes.get(v, unseen) for v in b]
    return out_tr, out_te


def protocol_B_matrix(D):
    """The matrix Protocol B splits: the natural gaps where the numerical block has
    at least 1% of them, otherwise 20% MCAR injected before the split."""
    Xw = D['arr'].copy(); num_idx = D['num_idx']
    nat = np.isnan(Xw[:, num_idx]).mean()
    if nat < 0.01:
        sub = Xw[:, num_idx]
        sub[make_mask(sub, .20, 'MCAR', np.random.default_rng(SEED))[0]] = np.nan
        Xw[:, num_idx] = sub
        return Xw, 'injected 20% MCAR'
    return Xw, 'natural %.2f%%' % (100 * nat)


def split_B(D, rep, methods, rf_jobs=1):
    Xw, pattern = protocol_B_matrix(D)
    y, num_idx, cat_idx = D['y'], D['num_idx'], D['cat_idx']
    sss = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=SEED)
    tr, te = list(sss.split(Xw, y))[rep]
    Xtr, Xte, ytr, yte = Xw[tr].copy(), Xw[te].copy(), y[tr], y[te]
    if D['raw_cat'] is not None and cat_idx:
        ctr, cte = fit_categoricals(D['raw_cat'], tr, te)
        Xtr[:, cat_idx], Xte[:, cat_idx] = ctr, cte
    sc = StandardScaler()
    Xtr[:, num_idx] = sc.fit_transform(Xtr[:, num_idx])
    Xte[:, num_idx] = sc.transform(Xte[:, num_idx])
    avg = 'binary' if len(np.unique(y)) == 2 else 'macro'
    recs = []
    for mn in methods:
        imp = make_imputer(mn, rf_jobs)
        A_, B_ = Xtr.copy(), Xte.copy(); t = time.perf_counter()
        A_[:, num_idx] = imp.fit_transform(Xtr[:, num_idx])
        B_[:, num_idx] = imp.transform(Xte[:, num_idx])
        dt = time.perf_counter() - t
        assert not (np.isnan(A_).any() or np.isnan(B_).any()), f'{D["name"]}/{mn}: NaN left'
        for cn, clf in classifiers().items():
            clf.fit(A_, ytr); pr = clf.predict(B_)
            recs.append(dict(dataset=D['name'], repeat=rep, method=mn, classifier=cn,
                             pattern=pattern, accuracy=accuracy_score(yte, pr),
                             precision=precision_score(yte, pr, average=avg, zero_division=0),
                             recall=recall_score(yte, pr, average=avg, zero_division=0),
                             f1=f1_score(yte, pr, average=avg, zero_division=0),
                             impute_seconds=dt, choice=choice_of(imp)))
    return recs


# ─── Protocol C: wall-clock scaling ──────────────────────────────────────────
GRID_N = [200, 500, 1000, 2000]
GRID_P = [10, 20, 40, 80]


def synth(n, p, seed=SEED, rho=.8, block=5):
    """Gaussian matrix with a block correlation structure."""
    g = np.random.default_rng(seed); R = np.zeros((p, p))
    for b in range(0, p, block):
        R[b:b + block, b:b + block] = rho
    np.fill_diagonal(R, 1.)
    w, V = np.linalg.eigh(R); R = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(R)); R /= np.outer(d, d)
    return g.standard_normal((n, p)) @ np.linalg.cholesky(R).T


def cell_C(n, p, methods, rf_jobs=1):
    Xt = synth(n, p)
    M, _ = make_mask(Xt, .30, 'MCAR', np.random.default_rng(SEED))
    Xm = Xt.copy(); Xm[M] = np.nan
    mu, sd = np.nanmean(Xm, 0), np.nanstd(Xm, 0); sd[sd == 0] = 1
    sd_pre = Xt.std(0)
    recs = []
    for m in methods:
        imp, out, dt = run_imputer(m, (Xm - mu) / sd, rf_jobs)
        s = score(out, Xt, M, mu, sd, sd_pre)
        recs.append(dict(n=n, p=p, method=m, seconds=dt, nrmse_pre=s['nrmse_pre'],
                         n_iter=int(getattr(imp, 'n_iter_', 1)), choice=choice_of(imp),
                         n_patterns=len(pattern_groups(M))))
    return recs


# ─── Protocol D: convergence and sensitivity of CW-CZI ───────────────────────
D_CYCLES = [1, 2, 5, 10, 30]


def _masked_D(D, mech, rate, seed):
    Xn = D['arr'][:, D['num_idx']].copy()
    M, _ = make_mask(Xn, rate, mech, np.random.default_rng(seed))
    Xm = Xn.copy(); Xm[M] = np.nan
    mu, sd = np.nanmean(Xm, 0), np.nanstd(Xm, 0); sd[sd == 0] = 1
    sd_pre = np.nanstd(Xn, 0); sd_pre[sd_pre == 0] = 1
    return Xn, M, (Xm - mu) / sd, mu, sd, sd_pre


def convergence_D(D):
    Xn, M, Xs, mu, sd, sd_pre = _masked_D(D, 'MCAR', .30, SEED_D)
    recs = []
    for w in ['uniform', 'marginal', 'ridge']:
        imp = CWCZImputer(weighting=w, max_iter=40, tol=1e-10).fit(Xs.copy())
        recs += [dict(dataset=D['name'], weighting=w, iteration=t, delta=d)
                 for t, d in enumerate(imp.deltas_, 1)]
    return recs


def sensitivity_D(D, mech, rep):
    Xn, M, Xs, mu, sd, sd_pre = _masked_D(D, mech, .30, SEED_D + 977 * rep)
    recs = []
    for w in ['uniform', 'marginal', 'ridge']:
        for lam in (LAM_GRID if w == 'ridge' else [0.0]):
            for it in D_CYCLES:
                o = CWCZImputer(weighting=w, lam=lam, max_iter=it, tol=0.).fit_transform(Xs.copy())
                s = score(o, Xn, M, mu, sd, sd_pre)
                recs.append(dict(dataset=D['name'], mechanism=mech, repeat=rep, rule=w,
                                 lam=lam, max_iter=it, nrmse_pre=s['nrmse_pre'],
                                 nrmse_post=s['nrmse_post']))
    return recs


# ─── Protocol E: screening for non-linear dependence ─────────────────────────
E_METHODS = ['Mean', 'CW-CZI (R)', 'EM', 'Iterative-BR', 'KNN', 'Iterative-RF']
E_LEVELS = {'monotone': [0.0, 0.5, 1.0, 1.5, 2.0],
            'non-monotone': [0.0, 0.25, 0.5, 0.75, 1.0]}


def synth_nonlinear(kind, a, n=500, p=12, seed=SEED):
    """Latent Gaussian with blocks of four columns correlated at 0.8, observed
    through a non-linearity of strength a (a = 0 is the linear case).
    monotone      x = sinh(a z) / a on every column: ranks, hence Spearman, are
                  unchanged, while Pearson correlations fall as a grows.
    non-monotone  every second column becomes (1 - a) z + a (z^2 - 1)/sqrt(2):
                  at a = 1 those columns depend on the others only through z^2,
                  a relation with no linear and no monotone component."""
    Z = synth(n, p, seed=seed, rho=.8, block=4)
    X = Z.copy()
    if kind == 'monotone' and a > 0:
        X = np.sinh(a * Z) / a
    elif kind == 'non-monotone':
        X[:, 1::2] = (1 - a) * Z[:, 1::2] + a * (Z[:, 1::2] ** 2 - 1) / np.sqrt(2)
    return X


def block_E(kind, a, rep, methods, rf_jobs=1):
    Xt = synth_nonlinear(kind, a, seed=SEED + 977 * rep)
    M, _ = make_mask(Xt, .30, 'MCAR', np.random.default_rng(SEED + 977 * rep))
    Xm = Xt.copy(); Xm[M] = np.nan
    mu, sd = np.nanmean(Xm, 0), np.nanstd(Xm, 0); sd[sd == 0] = 1
    sd_pre = Xt.std(0)
    diag = linearity_diagnostic(Xm)
    # under the non-monotone kind only every second column is transformed; its
    # error is reported apart from that of the columns left linear
    odd = np.zeros_like(M); odd[:, 1::2] = M[:, 1::2]
    even = np.zeros_like(M); even[:, 0::2] = M[:, 0::2]
    recs = []
    for m in methods:
        imp, out, dt = run_imputer(m, (Xm - mu) / sd, rf_jobs)
        s = score(out, Xt, M, mu, sd, sd_pre)
        rec = dict(kind=kind, level=a, repeat=rep, method=m, seconds=dt,
                   nrmse_pre=s['nrmse_pre'], sp_gap=diag['sp_gap_mean'],
                   sp_share=diag['sp_share'], eta_gap=diag['eta_gap_mean'],
                   eta_share=diag['eta_share'],
                   nrmse_odd=score(out, Xt, odd, mu, sd, sd_pre)['nrmse_pre'],
                   nrmse_even=score(out, Xt, even, mu, sd, sd_pre)['nrmse_pre'])
        recs.append(rec)
    return recs


def diagnostic_real(D):
    Xn = D['arr'][:, D['num_idx']]
    d = linearity_diagnostic(Xn)
    return [dict(dataset=D['name'], p=Xn.shape[1], sp_gap=d['sp_gap_mean'],
                 sp_share=d['sp_share'], eta_gap=d['eta_gap_mean'],
                 eta_share=d['eta_share'], n_pairs=d['n_pairs'],
                 mean_abs_r=float(np.nanmean(np.abs(_pairwise_r(Xn)))),
                 redundancy=redundancy(Xn))]


def _pairwise_r(X):
    R = pd.DataFrame(X).corr().values
    return R[np.triu_indices(R.shape[0], 1)]


def redundancy(X):
    """1 - p_eff / p, with p_eff = (sum lambda)^2 / sum lambda^2 the effective
    number of columns of the pairwise correlation matrix: 0 for uncorrelated
    columns, close to 1 when a few directions carry everything."""
    R = pd.DataFrame(X).corr().fillna(0).values
    np.fill_diagonal(R, 1.0)
    lam = np.clip(np.linalg.eigvalsh(R), 0, None)
    return float(1 - (lam.sum() ** 2 / (lam ** 2).sum()) / R.shape[0])


# ─── Environment record ──────────────────────────────────────────────────────
def environment(extra=None):
    import scipy, sklearn
    env = dict(python=platform.python_version(), platform=platform.platform(),
               machine=platform.node(), processor=platform.processor() or 'n/a',
               cpu_count=os.cpu_count(), numpy=np.__version__, scipy=scipy.__version__,
               sklearn=sklearn.__version__, pandas=pd.__version__,
               threads={v: os.environ.get(v) for v in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
               time=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    if extra:
        env.update(extra)
    return env
