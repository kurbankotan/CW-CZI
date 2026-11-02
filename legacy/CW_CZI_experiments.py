# -*- coding: utf-8 -*-
"""
Correlation-Weighted Cyclical Z-Score Imputation (CW-CZI) — v4
Kurban Kotan, Manisa Celal Bayar University

v3'e göre iki değişiklik:
  1. MAR artık gerçek MAR. v3'te koşullayıcı sütunlar rng.permutation(p) ile
     seçiliyordu; permütasyon sabit nokta içerebildiği için (20 sütunda ortalama 1,
     çekilişlerin %62'sinde en az bir tane) bazı sütunlar kendi değerine koşullanıp
     MNAR'a dönüşüyordu. Artık sabit noktası olmayan bir permütasyon (derangement)
     kullanılıyor, yani hiçbir sütun kendi değerine koşullanmıyor.
  2. Çalışma ortamı (CPU, BLAS, kütüphane sürümleri, tekrar sayısı) sonuçlara
     yazılıyor; süre iddiaları böyle doğrulanabiliyor.
MCAR ve MNAR maskeleri değişmiyor: rng her (mekanizma, oran, tekrar) için yeniden
tohumlanıyor ve yalnız MAR dalı değişti. Dolayısıyla yalnız MAR satırları yenilenir.

v2'ye göre üç değişiklik:
  1. Maskeleme artık "tam blok" üzerinde değil, GÖZLENEN HÜCRELER üzerinde yapılıyor.
     Doğal boşluğu olan satırlar da kullanılıyor, hiçbir satır atılmıyor. Yer gerçeği
     yine var çünkü maskelenen hücreler gözlenmişti. Tam blok yaklaşımı Mice'ta
     satırların yarısını atıyordu ve atılan yarı sınıf açısından yanlıydı.
  2. Heart artık UCI'nin 303 satırlık Cleveland orijinali (ca/thal'de gerçek eksik),
     1025 satırlık çoğaltılmış sürüm değil.
  3. Tablo 1 sütun ve satır düzeyi eksiklik oranlarını da veriyor; ayrı bir tablo
     eksikliğin sınıfa bağlı olup olmadığını ki-kare ile test ediyor.
"""
import warnings; warnings.filterwarnings("ignore")

try:                                   # Colab: mount Drive; elsewhere: carry on
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except (ImportError, ModuleNotFoundError):
    IN_COLAB = False
    print('Colab dışında çalışıyor — DATA_DIR yerel bir klasöre işaret etmeli.')

import inspect, os, time, json, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, chi2_contingency

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ─── Ayarlar ─────────────────────────────────────────────────────────────────
DATA_DIR = "/content/drive/MyDrive/Datasets"
DATASETS = {
    'CKD'  : dict(file='kidney_disease.csv',      target='classification', drop=['id']),
    'Heart': dict(uci=45, file='heart_cleveland.csv', target=None, drop=[]),
    'Mice' : dict(file='Data_Cortex_Nuclear.csv', target='class',          drop=['MouseID']),
}

QUICK      = False                     # True -> ~5 dk'lik deneme koşusu
SEED       = 42
OUT        = 'results'
PROPOSED   = 'CW-CZI (R)'
MAX_ROWS   = 600
TEST_SIZE  = 0.20
N_REPEATS  = 2       if QUICK else 5
MASK_RATES = [0.20]  if QUICK else [0.10, 0.20, 0.30, 0.40]
MECHANISMS = ['MCAR'] if QUICK else ['MCAR', 'MAR', 'MNAR']
NA_STRINGS = ['?', 'NA', 'na', 'N/A', '', ' ', '-', '--', 'nan', 'None', 'null']

LAM_GRID  = [0.01, 0.03, 0.10, 0.30, 1.00, 3.00]
ITER_GRID = [1, 2, 5]
os.makedirs(OUT, exist_ok=True)

PALETTE = ['#0072B2','#D55E00','#009E73','#E69F00','#9467BD',
           '#56B4E9','#CC79A7','#6A9F00','#A0522D','#7F3F98']
MARKERS = ['o','s','^','D','v','P','X','*','h','<']
DASHES  = [(None,None),(5,2),(1,1.5),(7,2,1,2),(3,1),
           (6,2,1,2,1,2),(2,2),(8,3),(4,1,1,1),(9,2,2,2)]
plt.rcParams.update({'figure.dpi':140,'savefig.dpi':400,'font.size':8,
                     'font.family':'serif','axes.grid':True,'grid.alpha':.25,
                     'grid.linewidth':.4,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':.6,
                     'legend.frameon':False,'lines.linewidth':1.4})

# ─── Yardımcılar ─────────────────────────────────────────────────────────────
# scikit-learn renamed force_all_finite to ensure_all_finite in 1.6. Resolve the
# name once from the signature: a try/except around the call would also swallow
# the TypeError that check_array raises for a non-numeric dtype and re-raise it
# with the wrong message.
_FINITE_KW = ('ensure_all_finite'
              if 'ensure_all_finite' in inspect.signature(check_array).parameters
              else 'force_all_finite')

def _check(X, copy=True):
    return check_array(X, dtype=float, copy=copy, **{_FINITE_KW: 'allow-nan'})

def _texty(col):
    """True for a column holding text. pandas >= 3 gives string columns the `str`
    dtype rather than `object`, so testing `dtype == object` silently skips the
    strip-and-coerce step in load() and leaves tab-prefixed numeric columns
    (CKD's pcv, wc, rc) in the categorical block. Test for 'not numeric'."""
    return not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col))

def _mu(m): m = np.asarray(m, float).copy(); m[~np.isfinite(m)] = 0.;  return m
def _sd(s): s = np.asarray(s, float).copy(); s[~np.isfinite(s)|(s==0)] = 1.; return s

# ─── CW-CZI ──────────────────────────────────────────────────────────────────
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

    def __init__(self, weighting='ridge', lam=0.10, max_iter=5, tol=1e-6,
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
        self.rules = list(rules)
        self.lam_grid  = list(LAM_GRID  if lam_grid  is None else lam_grid)
        self.iter_grid = list(ITER_GRID if iter_grid is None else iter_grid)
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
        return [(w, l, t) for w in self.rules for t in self.iter_grid
                for l in (self.lam_grid if w == 'ridge' else [0.0])]

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
        self.n_iter_ = self.max_iter_; return self

    def fit_transform(self, X, y=None, **kw):
        X = _check(X, self.copy); out = self._build(X).fit_transform(X)
        self.n_iter_ = self.max_iter_; return out

    def transform(self, X, y=None):
        check_is_fitted(self, ['_imp']); return self._imp.transform(_check(X, self.copy))

# ─── Karşılaştırma temelleri ─────────────────────────────────────────────────
class GaussianEMImputer(BaseEstimator, TransformerMixin):
    """Çok değişkenli normal altında koşullu ortalama, momentler EM ile."""

    def __init__(self, max_iter=30, tol=1e-6, ridge=1e-4, copy=True):
        self.max_iter, self.tol, self.ridge, self.copy = max_iter, tol, ridge, copy

    def _em(self, X):
        mask = np.isnan(X); p = X.shape[1]; mu = _mu(np.nanmean(X, 0))
        Xw = X.copy(); Xw[mask] = np.take(mu, np.where(mask)[1])
        S = np.cov(Xw, rowvar=False) + self.ridge*np.eye(p)
        for _ in range(int(self.max_iter)):
            prev, bias = Xw.copy(), np.zeros((p, p))
            for i in np.where(mask.any(1))[0]:
                o, m = np.where(~mask[i])[0], np.where(mask[i])[0]
                if o.size == 0: Xw[i, m] = mu[m]; continue
                Soo = S[np.ix_(o,o)] + self.ridge*np.eye(o.size); Smo = S[np.ix_(m,o)]
                try:
                    Xw[i, m] = mu[m] + Smo @ np.linalg.solve(Soo, Xw[i, o] - mu[o])
                    C = S[np.ix_(m,m)] - Smo @ np.linalg.solve(Soo, Smo.T)
                except np.linalg.LinAlgError:
                    Xw[i, m] = mu[m]; C = S[np.ix_(m,m)]
                bias[np.ix_(m,m)] += C
            mu = Xw.mean(0)
            S  = np.cov(Xw, rowvar=False) + bias/X.shape[0] + self.ridge*np.eye(p)
            if np.linalg.norm(Xw-prev)/np.sqrt(max(mask.sum(),1)) < self.tol: break
        return Xw, mu, S

    def fit(self, X, y=None):
        X=_check(X,self.copy); _, self.mu_, self.S_ = self._em(X); return self

    def fit_transform(self, X, y=None, **kw):
        X=_check(X,self.copy); Xw, self.mu_, self.S_ = self._em(X); return Xw

    def transform(self, X, y=None):
        check_is_fitted(self, ['mu_','S_']); X=_check(X,self.copy)
        mask = np.isnan(X); Xw = X.copy()
        for i in np.where(mask.any(1))[0]:
            o, m = np.where(~mask[i])[0], np.where(mask[i])[0]
            if o.size == 0: Xw[i, m] = self.mu_[m]; continue
            Soo = self.S_[np.ix_(o,o)] + self.ridge*np.eye(o.size)
            try:
                Xw[i,m] = self.mu_[m] + self.S_[np.ix_(m,o)] @ np.linalg.solve(
                    Soo, Xw[i,o] - self.mu_[o])
            except np.linalg.LinAlgError:
                Xw[i,m] = self.mu_[m]
        return Xw


class IterativeSVDImputer(BaseEstimator, TransformerMixin):
    """Kesik SVD ile düşük ranklı matris tamamlama."""

    def __init__(self, rank=5, max_iter=50, tol=1e-6, copy=True):
        self.rank, self.max_iter, self.tol, self.copy = rank, max_iter, tol, copy

    def _fill(self, X):
        mask = np.isnan(X); mu = _mu(np.nanmean(X, 0))
        Xw = X.copy(); Xw[mask] = np.take(mu, np.where(mask)[1])
        k = int(max(1, min(self.rank, min(Xw.shape)-1)))
        for _ in range(int(self.max_iter)):
            prev = Xw.copy()
            U, s, Vt = np.linalg.svd(Xw - mu, full_matrices=False)
            Xw[mask] = ((U[:,:k]*s[:k]) @ Vt[:k] + mu)[mask]
            if np.linalg.norm(Xw-prev)/np.sqrt(max(mask.sum(),1)) < self.tol: break
        return Xw, mu

    def fit(self, X, y=None):
        X=_check(X,self.copy); self.X_, self.mu_ = self._fill(X); return self

    def fit_transform(self, X, y=None, **kw):
        X=_check(X,self.copy); self.X_, self.mu_ = self._fill(X); return self.X_

    def transform(self, X, y=None):
        check_is_fitted(self, ['X_','mu_']); X=_check(X,self.copy)
        k = int(max(1, min(self.rank, min(self.X_.shape)-1)))
        V = np.linalg.svd(self.X_ - self.mu_, full_matrices=False)[2][:k].T
        mask = np.isnan(X); Xw = X.copy()
        for i in np.where(mask.any(1))[0]:
            o, m = np.where(~mask[i])[0], np.where(mask[i])[0]
            if o.size < k: Xw[i,m] = self.mu_[m]; continue
            c = np.linalg.lstsq(V[o], X[i,o]-self.mu_[o], rcond=None)[0]
            Xw[i,m] = self.mu_[m] + V[m] @ c
        return Xw

# ─── Yöntem kümesi ───────────────────────────────────────────────────────────
def make_imputers():
    return {
        'Mean'       : SimpleImputer(strategy='mean'),
        'CW-CZI (U)' : CWCZImputer(weighting='uniform',  max_iter=5, tol=1e-6),
        'CW-CZI (M)' : CWCZImputer(weighting='marginal', max_iter=5, tol=1e-6),
        PROPOSED     : CWCZImputer(weighting='ridge', lam=0.10, max_iter=5, tol=1e-6),
        'CW-CZI (A)' : CWCZAuto(),
        'KNN'        : KNNImputer(n_neighbors=5),
        'SVD'        : IterativeSVDImputer(rank=5),
        'EM'         : GaussianEMImputer(),
        'MICE'       : IterativeImputer(max_iter=20, random_state=SEED),
        'MissForest' : IterativeImputer(estimator=RandomForestRegressor(
                           n_estimators=30, random_state=SEED, n_jobs=-1),
                           max_iter=5, random_state=SEED),
    }

def make_classifiers():
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LR' : LogisticRegression(max_iter=1000, random_state=SEED),
        'DT' : DecisionTreeClassifier(random_state=SEED),
        'RF' : RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        'GNB': GaussianNB(),
        'SVC': SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300, random_state=SEED),
        'GB' : GradientBoostingClassifier(n_estimators=50, random_state=SEED),
    }

METHODS = list(make_imputers())
assert len(METHODS) <= len(PALETTE)
COLOR  = dict(zip(METHODS, PALETTE))
MARKER = dict(zip(METHODS, MARKERS))
DASH   = dict(zip(METHODS, DASHES))

# ─── Doğrulama ───────────────────────────────────────────────────────────────
_r = np.random.default_rng(SEED); _p = 40
_R = np.zeros((_p,_p))
for _b in range(0,_p,5): _R[_b:_b+5,_b:_b+5] = .85
np.fill_diagonal(_R, 1.)
_w,_V = np.linalg.eigh(_R); _R = _V @ np.diag(np.clip(_w,1e-6,None)) @ _V.T
_d = np.sqrt(np.diag(_R)); _R /= np.outer(_d,_d)
_T = _r.standard_normal((400,_p)) @ np.linalg.cholesky(_R).T
_M = _r.random(_T.shape) < .30; _M[_M.all(1)] = False
_X = _T.copy(); _X[_M] = np.nan
print('\nSanity check — RMSE (gruplu korelasyon, %30 MCAR):')
for _n, _i in [('CW-CZI (U)', CWCZImputer(weighting='uniform')),
               ('CW-CZI (M)', CWCZImputer(weighting='marginal')),
               ('CW-CZI (R)', CWCZImputer(weighting='ridge')),
               ('CW-CZI (A)', CWCZAuto()),
               ('EM        ', GaussianEMImputer())]:
    _o = _i.fit_transform(_X)
    print(f'  {_n}  {np.sqrt(np.mean((_o[_M]-_T[_M])**2)):.4f}')
_imp = CWCZImputer().fit(_X[:300]); _te = _X[300:]
_a = _imp.transform(_te); _b = np.vstack([_imp.transform(_te[[i]]) for i in range(len(_te))])
assert np.abs(_a-_b).max() < 1e-10 and not np.isnan(_a).any()
print(f'  transform satır bağımsız: max fark = {np.abs(_a-_b).max():.1e}   OK')

# ─── Veri yükleme ────────────────────────────────────────────────────────────
TARGET_NAMES = ['classification','class','target','label','diagnosis','outcome',
                'num','status','quality','severity','y']
ID_NAMES     = ['id','index','unnamed: 0','no','sample','mouseid','patient_id']

def fetch_uci_heart(path):
    """UCI Cleveland orijinali (303 satır, ca/thal'de gerçek '?'). Bir kez indirilip
    Drive'a yazılır; sonraki koşularda oradan okunur."""
    if os.path.exists(path):
        return pd.read_csv(path, na_values=NA_STRINGS)
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        os.system('pip install -q ucimlrepo')
        from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=45)
    df = ds.data.features.copy()
    df['num'] = (ds.data.targets.iloc[:, 0] > 0).astype(int)
    df.to_csv(path, index=False)
    print(f'    UCI Cleveland indirildi ve kaydedildi: {path}  {df.shape}')
    return df

RAW_CAT = {}


def load(cfg):
    path = os.path.join(DATA_DIR, cfg['file'])
    if 'uci' in cfg:
        df = fetch_uci_heart(path)
    else:
        df = pd.read_csv(path, na_values=NA_STRINGS, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if _texty(df[c]):
            df[c] = df[c].astype(str).str.strip().replace({s: np.nan for s in NA_STRINGS})
    for c in df.columns:
        if _texty(df[c]):
            conv = pd.to_numeric(df[c], errors='coerce'); ok = df[c].notna().sum()
            if ok and conv.notna().sum() >= .9*ok: df[c] = conv
    df = df.drop(columns=[c for c in cfg.get('drop', []) if c in df.columns])
    df = df.drop(columns=[c for c in df.columns if c.lower() in ID_NAMES])
    tgt = cfg.get('target')
    if tgt is None:
        hit = [c for c in df.columns if c.lower() in TARGET_NAMES]
        tgt = hit[-1] if hit else df.columns[-1]
    y_raw, X = df[tgt], df.drop(columns=[tgt])
    # ordinal tam sayı kodları sayısal blokta kalır; ikili bayraklar kategoriktir
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
                and X[c].notna().sum() > .3*len(X) and X[c].nunique(dropna=True) > 2]
    cat_cols = [c for c in X.columns if c not in num_cols]
    raw_cat = X[cat_cols].astype(object).copy() if cat_cols else None
    if cat_cols:
        # This whole-frame fill is used by Protocol A only, where the categorical
        # block is never imputed and never scored. Protocol B re-derives it from
        # the training rows alone (see fit_categoricals below).
        X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(
            X[cat_cols].astype(object))
        for c in cat_cols: X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    arr, num_idx = X.values.astype(float), [X.columns.get_loc(c) for c in num_cols]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    keep = (~pd.isna(y_raw).values) & (~np.isnan(arr[:, num_idx]).all(1))
    arr, y_raw = arr[keep], y_raw[keep]
    if raw_cat is not None: raw_cat = raw_cat[keep]
    y = LabelEncoder().fit_transform(y_raw.astype(str).str.strip())
    cnt = pd.Series(y).value_counts(); keep = np.isin(y, cnt[cnt >= 10].index)
    arr, y = arr[keep], LabelEncoder().fit_transform(y[keep])
    if raw_cat is not None: raw_cat = raw_cat[keep]
    if len(arr) > MAX_ROWS:
        sel = np.random.default_rng(SEED).choice(len(arr), MAX_ROWS, replace=False)
        arr, y = arr[sel], y[sel]
        if raw_cat is not None: raw_cat = raw_cat.iloc[sel]
    assert not np.isnan(np.delete(arr, num_idx, axis=1)).any()
    RAW_CAT[cfg['file']] = (raw_cat, cat_idx)
    return arr, y, num_idx, tgt


def fit_categoricals(raw, tr, te):
    """Mode-fill and integer-code the categorical block from the TRAINING rows
    only. A category seen for the first time in the test rows gets its own code
    rather than borrowing one, so nothing about the test rows reaches the fit."""
    out_tr = np.empty((len(tr), raw.shape[1]))
    out_te = np.empty((len(te), raw.shape[1]))
    for k, col in enumerate(raw.columns):
        a = raw[col].iloc[tr]; b = raw[col].iloc[te]
        fill = a.mode(dropna=True)
        fill = fill.iloc[0] if len(fill) else 'NA'
        a = a.fillna(fill).astype(str); b = b.fillna(fill).astype(str)
        codes = {v: i for i, v in enumerate(sorted(a.unique()))}
        unseen = len(codes)
        out_tr[:, k] = [codes[v] for v in a]
        out_te[:, k] = [codes.get(v, unseen) for v in b]
    return out_tr, out_te

# Dimensions the article reports, per dataset: (rows, numeric, categorical,
# missing % of the numeric block). load() is sensitive to how pandas types the
# raw CSV columns, so a version change can silently move a column between the
# numeric and the categorical block and quietly redefine the experiment. Check
# it instead of hoping.
EXPECTED = {'CKD':   dict(n=400, num=14, cat=10, miss=13.89),
            'Heart': dict(n=303, num=10, cat=3,  miss=0.20),
            'Mice':  dict(n=600, num=77, cat=3,  miss=1.65)}
STRICT_DIMS = not QUICK          # QUICK subsamples, so only the full run is checked

def check_dims(name, arr, num_idx, miss_pct):
    e = EXPECTED.get(name)
    if e is None: return
    got = dict(n=arr.shape[0], num=len(num_idx),
               cat=arr.shape[1]-len(num_idx), miss=round(miss_pct, 2))
    if got == e: return
    msg = (f'{name}: dataset does not match the published run.\n'
           f'    expected {e}\n    got      {got}\n'
           f'    Installed pandas {pd.__version__}, numpy {np.__version__}. '
           f'Install requirements-lock.txt and re-run; results produced from a '
           f'different column split are not comparable with the article.')
    if STRICT_DIMS: raise AssertionError(msg)
    print('  UYARI ' + msg)

print('\nVeri setleri:')
DATA, rows, struct = {}, [], []
for name, cfg in DATASETS.items():
    try:
        arr, y, num_idx, tgt = load(cfg)
        DATA[name] = (arr, y, num_idx); sub = arr[:, num_idx]
        inc = np.isnan(sub).any(1)                     # eksikli satırlar
        rows.append(dict(Dataset=name, Target=tgt, Instances=arr.shape[0],
                         Numeric=len(num_idx), Categorical=arr.shape[1]-len(num_idx),
                         Classes=len(np.unique(y)),
                         Missing_cells_pct=round(100*np.isnan(sub).mean(), 2),
                         Missing_cols=int((np.isnan(sub).sum(0) > 0).sum()),
                         Incomplete_rows_pct=round(100*inc.mean(), 1)))
        check_dims(name, arr, num_idx, 100*np.isnan(sub).mean())
        # eksiklik sınıfa bağlı mı?
        if inc.any() and (~inc).any():
            c2, pv, dof, _ = chi2_contingency(pd.crosstab(y, ~inc))
            struct.append(dict(Dataset=name, chi2=round(float(c2),1), dof=int(dof),
                               p=float(pv), MCAR_rejected=bool(pv < .05)))
        else:
            struct.append(dict(Dataset=name, chi2=np.nan, dof=0, p=np.nan,
                               MCAR_rejected=False))
        print(f"  {name}: {arr.shape[0]}x{arr.shape[1]}, {len(num_idx)} sayısal, "
              f"target='{tgt}', %{100*np.isnan(sub).mean():.2f} hücre eksik, "
              f"{int((np.isnan(sub).sum(0)>0).sum())} sütun etkilenmiş, "
              f"%{100*inc.mean():.0f} satır eksikli")
    except Exception as e:
        print(f'  {name} HATA: {type(e).__name__}: {e}')
pd.DataFrame(rows).to_csv(f'{OUT}/table1_datasets.csv', index=False)
tabS = pd.DataFrame(struct); tabS.to_csv(f'{OUT}/table2_missingness_structure.csv', index=False)
print('\nEksikliğin sınıfa bağlılığı (ki-kare):')
print(tabS.to_string(index=False))

# ─── Eksiklik mekanizmaları ──────────────────────────────────────────────────
def _derangement(p, rng):
    """Sabit noktası olmayan permütasyon. MAR'da bir sütunun eksikliği asla kendi
    değerine koşullanmasın diye; sabit nokta o sütunu sessizce MNAR yapardı."""
    if p < 2:
        return np.arange(p)
    for _ in range(100):                       # kabul oranı ~1/e, 100 deneme fazlasıyla yeter
        q = rng.permutation(p)
        if not (q == np.arange(p)).any():
            return q
    q = np.arange(p)                           # Sattolo: tek çevrim, sabit noktası yok
    for i in range(p - 1, 0, -1):
        j = int(rng.integers(0, i))
        q[i], q[j] = q[j], q[i]
    return q


_fp = [int((_derangement(20, np.random.default_rng(_s)) == np.arange(20)).sum())
       for _s in range(500)]
assert max(_fp) == 0, 'derangement sabit nokta üretti'
print(f'MAR derangement kontrolü: 500 çekilişte sabit nokta = {max(_fp)}   OK')


def make_mask(X, rate, mech, rng):
    """GÖZLENEN hücreleri maskeler. Doğal boşluklar olduğu gibi kalır ve puanlamaya
    girmez. Hiçbir satır tamamen boşaltılmaz, hiçbir satır atılmaz."""
    obs = ~np.isnan(X); n, p = X.shape
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0); sd[sd == 0] = 1
    Z = np.nan_to_num((X - mu) / sd, nan=0.0)
    if   mech == 'MCAR': sc = rng.random((n, p))
    elif mech == 'MAR' : sc = 1/(1+np.exp(Z[:, _derangement(p, rng)])) + .15*rng.random((n,p))
    else               : sc = 1/(1+np.exp(Z)) + .15*rng.random((n,p))
    sc = np.where(obs, sc, np.inf)                 # zaten eksik olan maskelenemez
    k = int(round(rate * obs.sum()))
    mask = np.zeros(n*p, bool); mask[np.argsort(sc, axis=None)[:k]] = True
    mask = mask.reshape(n, p) & obs
    empty = ~(obs & ~mask).any(1)                  # hiç gözlenen hücre kalmayan satır
    for i in np.where(empty)[0]:
        j = rng.choice(np.where(mask[i])[0]); mask[i, j] = False
    return mask

# ─── Protokol A — yer gerçeğine karşı yeniden oluşturma ──────────────────────
print('\n─── Protokol A ───')
recA, t0 = [], time.time()
for name, (X, y, num_idx) in DATA.items():
    Xn = X[:, num_idx].copy()
    print(f'  {name}: {Xn.shape[0]}x{Xn.shape[1]} (tüm satırlar)', flush=True)
    for mech in MECHANISMS:
        for rate in MASK_RATES:
            for rep in range(N_REPEATS):
                rng  = np.random.default_rng(SEED + 977*rep)
                mask = make_mask(Xn, rate, mech, rng)
                Xm = Xn.copy(); Xm[mask] = np.nan          # doğal + yapay boşluk
                mu, sd = np.nanmean(Xm,0), np.nanstd(Xm,0); sd[sd==0] = 1
                Xs, Xts = (Xm-mu)/sd, (Xn-mu)/sd
                for m, imp in make_imputers().items():
                    t = time.perf_counter(); out = imp.fit_transform(Xs.copy())
                    dt = time.perf_counter() - t
                    e = out[mask] - Xts[mask]              # sadece maskelenen hücreler
                    rec = dict(dataset=name, mechanism=mech, rate=rate, repeat=rep,
                        method=m, nrmse=float(np.sqrt(np.mean(e**2))),
                        mae=float(np.mean(np.abs(e))),
                        var_ratio=float(out[mask].std()/(Xts[mask].std()+1e-12)),
                        n_iter=int(getattr(imp,'n_iter_',1)), seconds=dt,
                        n_scored=int(mask.sum()))
                    if m == 'CW-CZI (A)':
                        rec.update(chosen_rule=imp.rule_, chosen_lam=imp.lam_,
                                   chosen_iter=imp.max_iter_)
                    recA.append(rec)
        print(f'     {mech} bitti ({time.time()-t0:.0f}s)', flush=True)

A = pd.DataFrame(recA); A.to_csv(f'{OUT}/protocolA_raw.csv', index=False)
tabA1 = A.pivot_table(index=['mechanism','rate'], columns='method', values='nrmse',
                      aggfunc='mean')[METHODS].round(4)
tabA2 = A.pivot_table(index='dataset', columns='method', values='nrmse',
                      aggfunc='mean')[METHODS].round(4); tabA2.loc['MEAN']=tabA2.mean().round(4)
tabA3 = A.pivot_table(index='mechanism', columns='method', values='var_ratio',
                      aggfunc='mean')[METHODS].round(3)
tabA1.to_csv(f'{OUT}/tableA1_nrmse_by_mechanism.csv')
tabA2.to_csv(f'{OUT}/tableA2_nrmse_by_dataset.csv')
tabA3.to_csv(f'{OUT}/tableA3_variance_ratio.csv')
print('\nTablo A1 — mekanizma x oran NRMSE\n', tabA1.to_string())
print('\nTablo A2 — veri seti NRMSE\n', tabA2.to_string())
print('\nTablo A3 — std(imputed)/std(gerçek)\n', tabA3.to_string())

sel = A[A.method == 'CW-CZI (A)']
if len(sel):
    tabA4 = (sel.groupby(['dataset','chosen_rule'])
               .agg(n=('nrmse','size'), lam=('chosen_lam','median'),
                    iters=('chosen_iter','median')).reset_index())
    tabA4.to_csv(f'{OUT}/tableA4_auto_selection.csv', index=False)
    print('\nTablo A4 — otomatik seçim\n', tabA4.to_string(index=False))

# ─── Protokol B — sınıflandırma ──────────────────────────────────────────────
print('\n─── Protokol B ───')
recB, t0 = [], time.time()
for name, (X, y, num_idx) in DATA.items():
    Xw = X.copy(); nat = np.isnan(Xw[:, num_idx]).mean()
    if nat < 0.01:
        sub = Xw[:, num_idx]
        sub[make_mask(sub, .20, 'MCAR', np.random.default_rng(SEED))] = np.nan
        Xw[:, num_idx] = sub; pat = 'enjekte %20 MCAR'
    else:
        pat = f'doğal %{100*nat:.2f}'
    print(f'  {name} ({pat})', flush=True)
    sss = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=SEED)
    raw_cat, cat_idx = RAW_CAT.get(DATASETS[name]['file'], (None, []))
    for rep, (tr, te) in enumerate(sss.split(Xw, y)):
        Xtr, Xte, ytr, yte = Xw[tr].copy(), Xw[te].copy(), y[tr], y[te]
        # the categorical block is re-derived from the training rows alone, so no
        # information crosses the split anywhere in the pipeline
        if raw_cat is not None and cat_idx:
            ctr, cte = fit_categoricals(raw_cat, tr, te)
            Xtr[:, cat_idx], Xte[:, cat_idx] = ctr, cte
        sc = StandardScaler()
        Xtr[:, num_idx] = sc.fit_transform(Xtr[:, num_idx])
        Xte[:, num_idx] = sc.transform(Xte[:, num_idx])
        for mn, imp in make_imputers().items():
            A_, B_ = Xtr.copy(), Xte.copy(); t = time.perf_counter()
            A_[:, num_idx] = imp.fit_transform(Xtr[:, num_idx])
            B_[:, num_idx] = imp.transform(Xte[:, num_idx])
            dt = time.perf_counter() - t
            assert not (np.isnan(A_).any() or np.isnan(B_).any()), f'{name}/{mn}: NaN kaldı'
            avg = 'binary' if len(np.unique(y)) == 2 else 'macro'
            for cn, clf in make_classifiers().items():
                clf.fit(A_, ytr); pr = clf.predict(B_)
                recB.append(dict(dataset=name, repeat=rep, method=mn, classifier=cn,
                    accuracy=accuracy_score(yte, pr),
                    precision=precision_score(yte, pr, average=avg, zero_division=0),
                    recall=recall_score(yte, pr, average=avg, zero_division=0),
                    f1=f1_score(yte, pr, average=avg, zero_division=0),
                    impute_seconds=dt))
    print(f'     bitti ({time.time()-t0:.0f}s)', flush=True)

B = pd.DataFrame(recB); B.to_csv(f'{OUT}/protocolB_raw.csv', index=False)
# Every aggregate below is derived from unrounded values and rounded once, at the
# end. Rounding first and then averaging or subtracting shifts the last printed
# digit, which is what put these files out of step with the article.
_rawB1 = B.pivot_table(index='dataset', columns='method', values='accuracy',
                       aggfunc='mean')[METHODS]
_rawB1.loc['MEAN'] = _rawB1.mean()
tabB1 = _rawB1.round(4)
tabB1.to_csv(f'{OUT}/tableB1_accuracy_by_dataset.csv')
_b = _rawB1.drop(index='MEAN')
_a = A.pivot_table(index='dataset', columns='method', values='nrmse',
                   aggfunc='mean')[METHODS]
_spr = pd.DataFrame({'accuracy_spread': _b.max(1)-_b.min(1),
                     'nrmse_spread'   : _a.max(1)-_a.min(1)}).dropna()
_spr.loc['MEAN'] = _spr.mean()
_spr['ratio'] = _spr.nrmse_spread / _spr.accuracy_spread
tabB2 = _spr.round({'accuracy_spread': 4, 'nrmse_spread': 4, 'ratio': 1})
tabB2.to_csv(f'{OUT}/tableB2_protocol_discrimination.csv')
print('\nTablo B1 — doğruluk\n', tabB1.to_string())
print('\nTablo B2 — protokol ayırt ediciliği\n', tabB2.to_string())

# ─── Protokol C — çalışma süresi ölçeklemesi ─────────────────────────────────
print('\n─── Protokol C ───')
def synth(n, p, seed=0):
    g = np.random.default_rng(seed); R = np.zeros((p,p))
    for b in range(0,p,5): R[b:b+5,b:b+5] = .8
    np.fill_diagonal(R, 1.)
    w,V = np.linalg.eigh(R); R = V @ np.diag(np.clip(w,1e-6,None)) @ V.T
    d = np.sqrt(np.diag(R)); R /= np.outer(d,d)
    return g.standard_normal((n,p)) @ np.linalg.cholesky(R).T

grid_n = [200,500] if QUICK else [200,500,1000,2000]
grid_p = [10,20]   if QUICK else [10,20,40,80]
rec = []
for n in grid_n:
    for p in grid_p:
        Xt = synth(n, p, seed=SEED)
        M  = make_mask(Xt, .30, 'MCAR', np.random.default_rng(SEED))
        Xm = Xt.copy(); Xm[M] = np.nan
        mu, sd = np.nanmean(Xm,0), np.nanstd(Xm,0); sd[sd==0] = 1
        Xs, Xts = (Xm-mu)/sd, (Xt-mu)/sd
        for m, imp in make_imputers().items():
            t = time.perf_counter(); out = imp.fit_transform(Xs.copy())
            rec.append(dict(n=n, p=p, method=m, seconds=time.perf_counter()-t,
                            nrmse=float(np.sqrt(np.mean((out[M]-Xts[M])**2)))))
    print(f'  n={n} bitti', flush=True)
S = pd.DataFrame(rec); S.to_csv(f'{OUT}/protocolC_scaling.csv', index=False)
tabC = S.pivot_table(index=['n','p'], columns='method', values='seconds')[METHODS].round(3)
tabC.to_csv(f'{OUT}/tableC1_runtime.csv')
print('\nTablo C1 — saniye\n', tabC.to_string())

# ─── Protokol D — yakınsama ve duyarlılık ────────────────────────────────────
print('\n─── Protokol D ───')
conv, sens = [], []
for name, (X, y, num_idx) in DATA.items():
    Xn = X[:, num_idx].copy()
    M  = make_mask(Xn, .30, 'MCAR', np.random.default_rng(SEED))
    Xm = Xn.copy(); Xm[M] = np.nan
    mu, sd = np.nanmean(Xm,0), np.nanstd(Xm,0); sd[sd==0] = 1
    Xs, Ts = (Xm-mu)/sd, (Xn-mu)/sd
    for w in ['uniform','marginal','ridge']:
        imp = CWCZImputer(weighting=w, max_iter=40, tol=1e-10).fit(Xs.copy())
        for t, d in enumerate(imp.deltas_, 1):
            conv.append(dict(dataset=name, weighting=w, iteration=t, delta=d))
    for rep in range(N_REPEATS):
        M2 = make_mask(Xn, .30, 'MCAR', np.random.default_rng(SEED + 977*rep))
        X2 = Xn.copy(); X2[M2] = np.nan
        m2, s2 = np.nanmean(X2,0), np.nanstd(X2,0); s2[s2==0] = 1
        Z2, T2 = (X2-m2)/s2, (Xn-m2)/s2
        for w in ['uniform','marginal','ridge']:
            for lam in (LAM_GRID if w == 'ridge' else [0.0]):
                for it in [1,2,5,10,30]:
                    o = CWCZImputer(weighting=w, lam=lam, max_iter=it,
                                    tol=0.).fit_transform(Z2.copy())
                    sens.append(dict(dataset=name, repeat=rep, rule=w, lam=lam,
                        max_iter=it,
                        nrmse=float(np.sqrt(np.mean((o[M2]-T2[M2])**2)))))
C  = pd.DataFrame(conv); C.to_csv(f'{OUT}/protocolD_convergence.csv', index=False)
Sn = pd.DataFrame(sens); Sn.to_csv(f'{OUT}/protocolD_sensitivity.csv', index=False)
tabD1 = C.groupby(['dataset','weighting'])['iteration'].max().unstack().round(1)
tabD2 = Sn[Sn.rule=='ridge'].pivot_table(index='lam', columns='max_iter',
                                         values='nrmse').round(4)
tabD3 = Sn.pivot_table(index='rule', columns='max_iter', values='nrmse').round(4)
tabD1.to_csv(f'{OUT}/tableD1_cycles.csv')
tabD2.to_csv(f'{OUT}/tableD2_lambda_x_iter.csv')
tabD3.to_csv(f'{OUT}/tableD3_rule_x_iter.csv')
print('\nTablo D1 — 1e-10 eşiğine döngü\n', tabD1.to_string())
print('\nTablo D2 — ridge: lam x döngü\n', tabD2.to_string())
print('\nTablo D3 — kural x döngü\n', tabD3.to_string())

# ─── İstatistiksel karşılaştırma ─────────────────────────────────────────────
print('\n─── İstatistik ───')
def holm(p):
    o = np.argsort(p); adj = np.empty(len(p)); run = 0.
    for k, i in enumerate(o):
        run = max(run, (len(p)-k)*p[i]); adj[i] = min(run, 1.)
    return adj

# Two methods can land on the same completion and therefore on the same NRMSE.
# Reached by different arithmetic paths those two numbers can differ in the last
# bit, and a Wilcoxon test would then treat a 1e-16 gap as a signed observation
# and give it the smallest rank. Round to TIE_DP before every comparison so an
# equality is an equality, and the ranks, win counts and p-values do not depend
# on the BLAS or the library version.
TIE_DP = 12


def compare(df, val, keys, lower_better=True, ref=PROPOSED):
    wide = df.pivot_table(index=keys, columns='method',
                          values=val)[METHODS].dropna().round(TIE_DP)
    sign = 1 if lower_better else -1
    ranks = np.apply_along_axis(lambda r: rankdata(sign*r), 1, wide.values)
    avg = pd.Series(ranks.mean(0), index=wide.columns, name='avg_rank').sort_values()
    chi, pfr = friedmanchisquare(*[wide[m].values for m in wide.columns])
    out, ps = [], []
    for m in wide.columns:
        if m == ref: continue
        a, b = wide[ref].values, wide[m].values
        try:    p = wilcoxon(a, b)[1]
        except ValueError: p = 1.
        out.append(dict(method=m, mean_proposed=a.mean(), mean_other=b.mean(),
                        mean_diff=(b-a).mean(),
                        proposed_wins=((b>a).mean() if lower_better else (b<a).mean()),
                        p_raw=p)); ps.append(p)
    out = pd.DataFrame(out); out['p_holm'] = holm(np.array(ps))
    out['sig_005'] = out.p_holm < .05
    return avg, out.sort_values('p_holm'), (chi, pfr, len(wide))

rankA, sigA, (chiA, pA, nA) = compare(A, 'nrmse', ['dataset','mechanism','rate','repeat'], True)
rankB, sigB, (chiB, pB, nB) = compare(B, 'accuracy', ['dataset','repeat','classifier'], False)
rankA.round(3).to_frame().to_csv(f'{OUT}/tableE1_avg_ranks_nrmse.csv')
sigA.round(5).to_csv(f'{OUT}/tableE2_wilcoxon_nrmse.csv', index=False)
rankB.round(3).to_frame().to_csv(f'{OUT}/tableE3_avg_ranks_accuracy.csv')
sigB.round(5).to_csv(f'{OUT}/tableE4_wilcoxon_accuracy.csv', index=False)
json.dump({'friedman_nrmse': dict(chi2=float(chiA), p=float(pA), blocks=int(nA)),
           'friedman_accuracy': dict(chi2=float(chiB), p=float(pB), blocks=int(nB))},
          open(f'{OUT}/friedman.json','w'), indent=2)
print(f'Friedman NRMSE: chi2={chiA:.1f}, p={pA:.3e}, blok={nA}')
print('\nTablo E1 — NRMSE ortalama sıra\n', rankA.round(3).to_string())
print(f'\nTablo E2 — Wilcoxon vs {PROPOSED} (Holm)\n', sigA.round(5).to_string(index=False))
print(f'\nFriedman doğruluk: chi2={chiB:.1f}, p={pB:.3e}, blok={nB}')
print('\nTablo E3 — doğruluk ortalama sıra\n', rankB.round(3).to_string())
print(f'\nTablo E4 — Wilcoxon vs {PROPOSED} (Holm)\n', sigB.round(5).to_string(index=False))

# aynı testler otomatik kural için de: hangisi öne çıkarsa çıksın rakam hazır olsun
ALT = 'CW-CZI (A)'
_, sigA2, _ = compare(A, 'nrmse', ['dataset','mechanism','rate','repeat'], True, ALT)
_, sigB2, _ = compare(B, 'accuracy', ['dataset','repeat','classifier'], False, ALT)
sigA2.round(5).to_csv(f'{OUT}/tableE5_wilcoxon_nrmse_auto.csv', index=False)
sigB2.round(5).to_csv(f'{OUT}/tableE6_wilcoxon_accuracy_auto.csv', index=False)
print(f'\nTablo E5 — Wilcoxon vs {ALT}, NRMSE (Holm)\n', sigA2.round(5).to_string(index=False))
print(f'\nTablo E6 — Wilcoxon vs {ALT}, doğruluk (Holm)\n', sigB2.round(5).to_string(index=False))

# ─── Şekiller ────────────────────────────────────────────────────────────────
def st(m): return dict(color=COLOR[m], marker=MARKER[m], dashes=DASH[m],
                       markersize=3.4, markeredgewidth=0)

fig, axes = plt.subplots(1, len(MECHANISMS), figsize=(2.6*len(MECHANISMS)+.9, 2.5),
                         sharey=True); axes = np.atleast_1d(axes)
for ax, mech in zip(axes, MECHANISMS):
    sub = A[A.mechanism==mech].groupby(['method','rate'])['nrmse'].mean().unstack()
    for m in METHODS: ax.plot(sub.columns*100, sub.loc[m].values, label=m, **st(m))
    ax.set_title(mech, fontsize=8); ax.set_xlabel('masked cells (% of observed)')
axes[0].set_ylabel('NRMSE')
axes[-1].legend(fontsize=5.6, loc='center left', bbox_to_anchor=(1.02,.5))
fig.tight_layout(); fig.savefig(f'{OUT}/figure1_nrmse_vs_rate.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4.4, 2.5)); r = rankA.sort_values()
ax.barh(range(len(r)), r.values, height=.62, color=[COLOR[m] for m in r.index],
        edgecolor='none')
ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index, fontsize=7); ax.invert_yaxis()
ax.set_xlabel('average rank on NRMSE (1 = best)'); ax.grid(axis='y', visible=False)
ax.set_xlim(0, max(r.values)*1.16)
for i, v in enumerate(r.values): ax.text(v+.08, i, f'{v:.2f}', va='center', fontsize=6.5, color='#444')
fig.tight_layout(); fig.savefig(f'{OUT}/figure2_average_rank.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4.2, 2.4))
for w, c, lb in [('uniform',PALETTE[1],'uniform (U)'),('marginal',PALETTE[2],'marginal (M)'),
                 ('ridge',PALETTE[3],'ridge (R)')]:
    g = C[C.weighting==w].groupby('iteration')['delta']
    med, lo, hi = g.median(), g.min(), g.max()
    ax.plot(med.index, med.values, color=c, lw=1.6, label=lb)
    ax.fill_between(med.index, lo.reindex(med.index).values,
                    hi.reindex(med.index).values, color=c, alpha=.15, lw=0)
ax.set_yscale('log'); ax.set_xlabel('cycle'); ax.set_ylabel(r'normalised $\|\Delta X\|_F$')
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/figure3_convergence.png', bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5)); bp, bn = max(grid_p), max(grid_n)
sub = S[S.p==bp].groupby(['method','n'])['seconds'].mean().unstack()
for m in METHODS: axes[0].plot(sub.columns, sub.loc[m].values, label=m, **st(m))
axes[0].set_xlabel(f'rows n (p = {bp})'); axes[0].set_ylabel('seconds')
sub = S[S.n==bn].groupby(['method','p'])['seconds'].mean().unstack()
for m in METHODS: axes[1].plot(sub.columns, sub.loc[m].values, label=m, **st(m))
axes[1].set_xlabel(f'columns p (n = {bn})')
axes[1].legend(fontsize=5.6, loc='center left', bbox_to_anchor=(1.02,.5))
for a_, tk in zip(axes, [grid_n, grid_p]):
    a_.set_yscale('log'); a_.set_xscale('log'); a_.set_xticks(tk)
    a_.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    a_.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    a_.tick_params(axis='x', which='minor', length=0)
fig.tight_layout(); fig.savefig(f'{OUT}/figure4_runtime.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4.2, 2.4))
for w, c, lb in [('uniform',PALETTE[1],'uniform (U)'),('marginal',PALETTE[2],'marginal (M)'),
                 ('ridge',PALETTE[3],'ridge (R)')]:
    g = Sn[Sn.rule==w].groupby('max_iter')['nrmse'].mean()
    ax.plot(g.index, g.values, color=c, lw=1.6, marker='o', markersize=3.4, label=lb)
ax.set_xscale('log'); ax.set_xticks([1,2,5,10,30])
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
ax.set_xlabel('cycle budget'); ax.set_ylabel('NRMSE'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/figure5_cycle_by_rule.png', bbox_inches='tight')
plt.show()
print('şekiller yazıldı')

# ─── Paketle ve indir ────────────────────────────────────────────────────────
import platform, sklearn, scipy
env = dict(python=platform.python_version(), platform=platform.platform(),
           processor=platform.processor() or 'n/a', cpu_count=os.cpu_count(),
           numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
           pandas=pd.__version__, blas=str(np.__config__.show(mode='dicts'))[:400]
           if hasattr(np.__config__, 'show') else 'n/a',
           repeats=N_REPEATS, mask_rates=MASK_RATES, mechanisms=MECHANISMS, seed=SEED)
json.dump(env, open(f'{OUT}/environment.json', 'w'), indent=2, default=str)
print('\nÇalışma ortamı environment.json dosyasına yazıldı')

files_out = sorted(f for f in os.listdir(OUT) if f != 'manifest.txt')
open(f'{OUT}/manifest.txt','w').write('\n'.join(files_out))
files_out = sorted(os.listdir(OUT))

zip_path = 'cwczi_v4_results.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(OUT)): z.write(os.path.join(OUT,fn), os.path.join(OUT,fn))
print(f'\n{len(files_out)} dosya -> {zip_path}')
for fn in files_out: print('  ', fn)
if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
else:
    print(f'Sonuçlar burada: {os.path.abspath(zip_path)}')
