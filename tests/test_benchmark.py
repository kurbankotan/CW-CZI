# -*- coding: utf-8 -*-
"""Tests for the v7 benchmark: the masks mean what the article says they mean,
the error metric is the one Eq. 6 defines, and the baselines and their tuning
behave as described."""
import numpy as np
import pytest

import benchmark as bm
from cwczi import CWCZAuto, CWCZImputer, linearity_diagnostic


def gaussian(n=300, p=12, seed=0, miss=0.0):
    X = bm.synth(n, p, seed=seed)
    if miss:
        g = np.random.default_rng(seed + 1)
        X[g.random(X.shape) < miss] = np.nan
        X[np.isnan(X).all(1), 0] = 0.0
    return X


# ─── masks ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('mech', bm.MECHANISMS)
@pytest.mark.parametrize('rate', [0.1, 0.4])
def test_each_masked_column_loses_the_stated_share(mech, rate):
    X = gaussian(miss=0.05)
    obs = ~np.isnan(X)
    mask, cond = bm.make_mask(X, rate, mech, np.random.default_rng(1))
    assert not (mask & ~obs).any(), 'a naturally missing cell was masked'
    cols = np.setdiff1d(np.arange(X.shape[1]), cond)
    for j in cols:
        want = int(round(rate * obs[:, j].sum()))
        assert want - 3 <= mask[:, j].sum() <= want    # a few cells may be restored
    assert not mask[:, cond].any(), 'a conditioning column was masked'
    assert (obs & ~mask).any(1).all(), 'a row lost its last observed cell'


def test_mar_conditioning_set_is_a_quarter_of_the_columns_and_never_masked():
    X = gaussian(p=20)
    mask, cond = bm.make_mask(X, 0.3, 'MAR', np.random.default_rng(2))
    assert cond.size == 5 and not mask[:, cond].any()
    assert mask[:, np.setdiff1d(np.arange(20), cond)].any(0).all()


def test_mcar_and_mnar_have_no_conditioning_set():
    X = gaussian()
    for mech in ('MCAR', 'MNAR'):
        _, cond = bm.make_mask(X, 0.3, mech, np.random.default_rng(3))
        assert cond.size == 0


def test_mnar_removes_high_values_and_mar_follows_its_driver():
    X = gaussian(n=2000, p=8)
    m, _ = bm.make_mask(X, 0.3, 'MNAR', np.random.default_rng(4))
    for j in range(8):
        assert X[m[:, j], j].mean() > X[~m[:, j], j].mean() + 1.0
    # MAR: a masked column depends on a conditioning column, not on itself
    rng = np.random.default_rng(5)
    m, cond = bm.make_mask(X, 0.3, 'MAR', rng)
    j = np.setdiff1d(np.arange(8), cond)[0]
    gaps = [X[m[:, j], c].mean() - X[~m[:, j], c].mean() for c in cond]
    assert max(gaps) > 1.0                      # its driver is high where it is masked


def test_masks_are_nested_across_rates_for_a_fixed_seed():
    X = gaussian(miss=0.05)
    for mech in bm.MECHANISMS:
        a, _ = bm.make_mask(X, 0.2, mech, np.random.default_rng(7))
        b, _ = bm.make_mask(X, 0.4, mech, np.random.default_rng(7))
        assert (a & ~b).sum() <= 3              # up to the restored cells


# ─── error metric ────────────────────────────────────────────────────────────
def _setup(mech='MNAR', rate=0.3):
    Xn = gaussian(n=400, p=10)
    mask, _ = bm.make_mask(Xn, rate, mech, np.random.default_rng(8))
    Xm = Xn.copy(); Xm[mask] = np.nan
    mu, sd = np.nanmean(Xm, 0), np.nanstd(Xm, 0)
    sd_pre = Xn.std(0)
    return Xn, mask, Xm, mu, sd, sd_pre


def test_perfect_imputation_scores_zero_and_mean_scores_zero_variance():
    Xn, mask, Xm, mu, sd, sd_pre = _setup()
    perfect = (Xn - mu) / sd
    s = bm.score(perfect, Xn, mask, mu, sd, sd_pre)
    assert s['nrmse_pre'] < 1e-12 and s['nrmse_post'] < 1e-12
    s = bm.score(np.zeros_like(Xn), Xn, mask, mu, sd, sd_pre)   # column means
    assert s['var_ratio'] < 1e-12


def test_nrmse_pre_is_eq6():
    Xn, mask, Xm, mu, sd, sd_pre = _setup()
    out = np.nan_to_num((Xm - mu) / sd)                          # mean imputation
    s = bm.score(out, Xn, mask, mu, sd, sd_pre)
    ii, jj = np.nonzero(mask)
    xhat = mu[jj]
    want = np.sqrt(np.mean(((xhat - Xn[ii, jj]) / sd_pre[jj]) ** 2))
    assert abs(s['nrmse_pre'] - want) < 1e-12
    # under MNAR the post-mask scaling inflates the error, which is the point of Eq. 6
    assert s['nrmse_post'] > s['nrmse_pre']


# ─── baselines ───────────────────────────────────────────────────────────────
def _em_rowwise(X, max_iter=30, tol=1e-6, ridge=1e-4):
    """Reference EM, one row at a time."""
    mask = np.isnan(X); n, p = X.shape
    mu = np.nanmean(X, 0); Xw = X.copy(); Xw[mask] = np.take(mu, np.nonzero(mask)[1])
    S = np.cov(Xw, rowvar=False, bias=True) + ridge * np.eye(p)
    for _ in range(max_iter):
        prev = Xw.copy(); acc = np.zeros((p, p))
        for i in np.flatnonzero(mask.any(1)):
            o, m = np.flatnonzero(~mask[i]), np.flatnonzero(mask[i])
            Soo = S[np.ix_(o, o)] + ridge * np.eye(o.size); Smo = S[np.ix_(m, o)]
            B = np.linalg.solve(Soo, Smo.T)
            Xw[i, m] = mu[m] + (Xw[i, o] - mu[o]) @ B
            acc[np.ix_(m, m)] += S[np.ix_(m, m)] - Smo @ B
        mu = Xw.mean(0)
        S = np.cov(Xw, rowvar=False, bias=True) + acc / n + ridge * np.eye(p)
        if np.linalg.norm(Xw - prev) / np.sqrt(mask.sum()) < tol:
            break
    return Xw


def test_grouped_em_equals_rowwise_em():
    X = gaussian(n=150, p=6, miss=0.25)
    X[:30] = X[0]                                   # many rows share a pattern
    X[:30, 0] = np.linspace(-1, 1, 30)
    X[:30, 2] = np.nan
    a = bm.GaussianEMImputer().fit_transform(X.copy())
    b = _em_rowwise(X.copy())
    assert np.abs(a - b).max() < 1e-8


def test_em_transform_is_row_independent_and_counts_iterations():
    X = gaussian(n=200, p=8, miss=0.2)
    em = bm.GaussianEMImputer().fit(X[:150])
    assert 1 <= em.n_iter_ <= 30
    batch = em.transform(X[150:])
    one = np.vstack([em.transform(X[150 + i:151 + i]) for i in range(50)])
    assert np.allclose(batch, one)


def test_holdout_sets_reproduce_cwczauto_validation():
    """Every tuned method must be scored on the cells CW-CZI (A) is scored on."""
    X = gaussian(n=120, p=6, miss=0.2)
    auto = CWCZAuto(lam_grid=[0.1, 1.0], iter_grid=[1, 2]).fit(X.copy())
    for (w, lam, t), want in auto.val_curve_.items():
        errs = []
        for Xv, hold, truth in bm.holdout_sets(X):
            o = CWCZImputer(weighting=w, lam=lam, max_iter=t, tol=0.).fit_transform(Xv.copy())
            errs.append(np.sqrt(np.mean((o[hold[:, 0], hold[:, 1]] - truth) ** 2)))
        assert abs(np.mean(errs) - want) < 1e-12


@pytest.mark.parametrize('family', ['knn', 'svd', 'iterative'])
def test_tuned_imputer_picks_from_its_grid_and_is_deterministic(family):
    X = gaussian(n=150, p=8, miss=0.2)
    a = bm.TunedImputer(family); oa = a.fit_transform(X.copy())
    b = bm.TunedImputer(family); ob = b.fit_transform(X.copy())
    assert a.choice_ == b.choice_ and a.choice_ in a.val_curve_
    assert np.array_equal(oa, ob) and not np.isnan(oa).any()
    obs = ~np.isnan(X)
    assert np.allclose(oa[obs], X[obs])
    assert a.transform(X[:5]).shape == (5, 8)


def test_every_method_runs_and_leaves_observed_cells_alone():
    X = gaussian(n=120, p=6, miss=0.2)
    obs = ~np.isnan(X)
    for m in bm.METHODS:
        out = bm.make_imputer(m).fit_transform(X.copy())
        assert not np.isnan(out).any(), m
        assert np.allclose(out[obs], X[obs]), m


# ─── screening diagnostic ────────────────────────────────────────────────────
def test_diagnostic_is_near_zero_for_linear_data_and_grows_with_curvature():
    lin = bm.synth_nonlinear('monotone', 0.0)
    d0 = linearity_diagnostic(lin)
    assert abs(d0['sp_gap_mean']) < 0.02 and abs(d0['eta_gap_mean']) < 0.02
    mono = linearity_diagnostic(bm.synth_nonlinear('monotone', 2.0))
    assert mono['sp_gap_mean'] > d0['sp_gap_mean'] + 0.05
    um = linearity_diagnostic(bm.synth_nonlinear('non-monotone', 1.0))
    assert um['eta_gap_mean'] > d0['eta_gap_mean'] + 0.02
    assert um['sp_gap_mean'] < 0.05                 # Spearman cannot see a U-shape


def test_diagnostic_accepts_missing_values():
    X = bm.synth_nonlinear('non-monotone', 0.5)
    X[np.random.default_rng(0).random(X.shape) < 0.3] = np.nan
    d = linearity_diagnostic(X)
    assert np.isfinite(d['sp_gap_mean']) and np.isfinite(d['eta_gap_mean'])


def test_lean_forest_returns_exactly_what_the_full_one_returns():
    """The memory-lean IterativeImputer may drop its trees, never change a value."""
    X = gaussian(n=150, p=6, miss=0.25)
    full = bm.iterative_rf(10, 3, 1, lean=False).fit_transform(X.copy())
    lean = bm.iterative_rf(10, 3, 1, lean=True)
    out = lean.fit_transform(X.copy())
    assert np.array_equal(full, out)
    assert all(t.estimator is None for t in lean.imputation_sequence_)
    with pytest.raises(RuntimeError):
        lean.transform(X)


def test_variance_ratio_skips_a_column_whose_masked_values_tie():
    """Under MNAR a discrete column can lose only its tied maximum; std(x) is then
    zero up to round-off (seven copies of 1.025 have a std of 2.2e-16) and the
    ratio must not be computed from it."""
    Xn = gaussian(n=200, p=4)
    Xn[:7, 1] = 1.025
    assert 0 < Xn[:7, 1].std() < 1e-12             # the round-off this guards against
    mask = np.zeros_like(Xn, bool); mask[:7, 1] = True; mask[20:40, 0] = True
    Xm = np.where(mask, np.nan, Xn)
    mu, sd = np.nanmean(Xm, 0), np.nanstd(Xm, 0)
    out = np.random.default_rng(0).normal(size=Xn.shape)
    s = bm.score(out, Xn, mask, mu, sd, Xn.std(0))
    assert np.isfinite(s['var_ratio']) and s['var_ratio'] < 10


def test_screening_block_reports_transformed_and_linear_columns_apart():
    """Protocol E scores the transformed columns and the linear ones separately;
    the error over the whole matrix lies between the two."""
    recs = bm.block_E('non-monotone', 1.0, 0, ['Mean', 'CW-CZI (R)'])
    for r in recs:
        lo, hi = sorted([r['nrmse_odd'], r['nrmse_even']])
        assert lo - 1e-12 <= r['nrmse_pre'] <= hi + 1e-12
    r = {x['method']: x for x in recs}['CW-CZI (R)']
    assert r['eta_gap'] > r['sp_gap']        # a squared relation moves the non-linear share


def test_sensitivity_variants_are_runnable_and_kept_out_of_the_main_set():
    assert 'CW-CZI (R30)' in bm.SENSITIVITY_METHODS and 'CW-CZI (R30)' not in bm.METHODS
    imp = bm.make_imputer('CW-CZI (R30)')
    assert imp.max_iter == 30 and imp.lam == 0.10
    X = gaussian(n=120, p=5, miss=0.2)
    out = imp.fit_transform(X.copy())
    assert not np.isnan(out).any() and np.array_equal(out[~np.isnan(X)], X[~np.isnan(X)])


def test_estimator_default_cycle_budget_is_the_recommended_one():
    assert CWCZImputer().max_iter == 30
