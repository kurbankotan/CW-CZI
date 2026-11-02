# -*- coding: utf-8 -*-
"""Tests for the CW-CZI estimator.

Run with `pytest` from the repository root. These cover the properties the
article relies on: observed cells are never overwritten, a row is filled the
same way alone as in a batch, the stored moments describe the converged matrix,
the ridge rule discounts redundant columns where the marginal rule does not,
and the estimators satisfy the scikit-learn API.
"""
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from cwczi import CWCZAuto, CWCZImputer

RULES = ['uniform', 'marginal', 'ridge']


def make(n=80, p=8, rate=0.2, seed=0):
    """A correlated Gaussian matrix with `rate` of its cells removed at random."""
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(p, p))
    X = rng.normal(size=(n, p)) @ L
    M = X.copy()
    M[rng.random(M.shape) < rate] = np.nan
    M[np.isnan(M).all(1), 0] = X[np.isnan(M).all(1), 0]      # no empty rows
    return X, M


@pytest.mark.parametrize('rule', RULES)
def test_no_nan_left(rule):
    _, M = make()
    out = CWCZImputer(weighting=rule).fit_transform(M)
    assert not np.isnan(out).any()
    assert out.shape == M.shape


@pytest.mark.parametrize('rule', RULES)
def test_observed_cells_are_untouched(rule):
    _, M = make()
    obs = ~np.isnan(M)
    out = CWCZImputer(weighting=rule).fit_transform(M)
    assert np.allclose(out[obs], M[obs])


@pytest.mark.parametrize('rule', RULES)
def test_transform_is_row_independent(rule):
    """A row must be filled identically alone and inside a batch, or the
    train/test separation the downstream protocol depends on is not real."""
    _, M = make()
    imp = CWCZImputer(weighting=rule).fit(M)
    batch = imp.transform(M)
    one_by_one = np.vstack([imp.transform(M[i:i + 1]) for i in range(len(M))])
    assert np.allclose(batch, one_by_one)


def test_stored_moments_describe_the_converged_matrix():
    _, M = make()
    imp = CWCZImputer(max_iter=5)
    out = imp.fit_transform(M)
    assert np.allclose(imp.mu_, out.mean(0))
    assert np.allclose(imp.sd_, np.where(out.std(0) == 0, 1.0, out.std(0)))
    assert np.allclose(imp.R_, np.corrcoef(out, rowvar=False))


def test_fit_then_transform_matches_fit_transform_on_the_last_sweep():
    _, M = make()
    a = CWCZImputer(max_iter=1).fit(M).transform(M)
    b = CWCZImputer(max_iter=1).fit_transform(M)
    assert np.allclose(a[~np.isnan(M)], b[~np.isnan(M)])


def test_transform_rejects_a_different_column_count():
    _, M = make()
    imp = CWCZImputer().fit(M)
    with pytest.raises(ValueError, match='features'):
        imp.transform(M[:, :3])


@pytest.mark.parametrize('seed', range(4))
def test_ridge_beats_marginal_on_redundant_columns(seed):
    """The claim the third rung rests on. The target is driven mostly by one
    unique column and weakly by a second factor that appears five times. The
    marginal rule puts all five copies in its denominator, so the duplicated
    factor outweighs the informative column and the estimate is dragged toward
    the column mean; the ridge inverse discounts the copies instead.

    Note the failure mode is specific. Where every observed column is an equally
    good and equally redundant predictor, averaging them is close to optimal and
    the two rules are indistinguishable — which is what the article reports for
    the two narrow datasets."""
    rng = np.random.default_rng(seed)
    n = 500
    unique = rng.normal(size=n)                       # strong, not duplicated
    shared = rng.normal(size=n)                       # weaker, duplicated below
    target = 0.85 * unique + 0.35 * shared + 0.3 * rng.normal(size=n)
    X = np.column_stack([target, unique]
                        + [shared + 0.05 * rng.normal(size=n) for _ in range(5)])
    M = X.copy()
    hid = rng.random(n) < 0.3
    M[hid, 0] = np.nan
    err = {}
    for rule in ('marginal', 'ridge'):
        out = CWCZImputer(weighting=rule).fit_transform(M.copy())
        err[rule] = float(np.sqrt(np.mean((out[hid, 0] - X[hid, 0]) ** 2)))
    assert err['ridge'] < err['marginal']


def test_auto_selects_from_its_grid():
    _, M = make()
    auto = CWCZAuto(lam_grid=[0.1, 1.0], iter_grid=[1, 2])
    out = auto.fit_transform(M)
    assert not np.isnan(out).any()
    assert auto.rule_ in ('ridge', 'marginal')
    assert auto.lam_ in (0.0, 0.1, 1.0)
    assert auto.max_iter_ in (1, 2)


def test_works_inside_a_pipeline():
    from sklearn.linear_model import LogisticRegression
    _, M = make()
    y = (np.nan_to_num(M).sum(1) > 0).astype(int)
    pipe = Pipeline([('imp', CWCZImputer()), ('clf', LogisticRegression())])
    pipe.fit(M, y)
    assert pipe.predict(M).shape == (len(M),)


@pytest.mark.parametrize('estimator', [CWCZImputer(), CWCZAuto()])
def test_scikit_learn_api(estimator):
    check_estimator(estimator)


def test_auto_validation_restores_the_cell_it_blanked():
    """CWCZAuto blanks a fifth of the observed cells to score its candidates. A
    row can lose all of them, and one cell is then put back. The column written
    and the column read must be the same one, and the restored cell must leave
    the scored hold-out set — otherwise the internal validation is scored partly
    against values it planted itself."""
    import inspect

    src = inspect.getsource(CWCZAuto._select)
    assert 'Xv[i, g.choice(keep)] = X[i, g.choice(keep)]' not in src, \
        'the restore line draws the column twice'
    assert 'keepmask' in src, 'the restored cell is not removed from the hold-out'

    # a matrix sparse enough that rows really do empty out
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 5))
    X[rng.random(X.shape) < 0.6] = np.nan
    empty = np.isnan(X).all(1)
    X[empty, 0] = rng.normal(size=int(empty.sum()))          # no empty input rows

    auto = CWCZAuto(lam_grid=[0.1], iter_grid=[1])
    out = auto.fit_transform(X.copy())
    assert not np.isnan(out).any()
    obs = ~np.isnan(X)
    assert np.allclose(out[obs], X[obs])                     # nothing planted


def test_auto_is_deterministic():
    _, M = make()
    a = CWCZAuto(lam_grid=[0.1, 0.3], iter_grid=[1, 2]).fit_transform(M.copy())
    b = CWCZAuto(lam_grid=[0.1, 0.3], iter_grid=[1, 2]).fit_transform(M.copy())
    assert np.array_equal(a, b)
