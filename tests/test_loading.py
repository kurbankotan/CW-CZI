# -*- coding: utf-8 -*-
"""Regression tests for the loader in benchmark.py.

Three CKD columns (pcv, wc, rc) are stored in the source CSV with a leading tab,
so they arrive as text and have to be stripped and coerced before they can join
the numerical block. That step used to be guarded by `dtype == object`. From
pandas 3 a text column has the dtype `str`, so the guard was false, the coercion
never ran, and CKD loaded with 11 numerical columns at 10.68% missing instead of
14 at 13.89% — the same code and the same data measuring a different experiment.

These tests pin the property that the fix depends on, under whichever string
backend the installed pandas uses, and check that the script still spells the
test the safe way.
"""
import io
import pathlib
import re

import pandas as pd
import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'benchmark.py'
NA_STRINGS = ['?', 'NA', 'na', 'N/A', '', ' ', '-', '--', 'nan', 'None', 'null']

# A CKD-shaped fragment. Every value in pcv, wc and rc carries a leading tab, and
# the missing marker is '\t?' rather than '?' — which is why na_values does not
# catch it and the column arrives as text instead of as numbers.
def _cell(i, val):
    return '\t?' if i % 3 == 0 else f'\t{val}'


CSV = 'id,age,pcv,wc,rc,rbc\n' + '\n'.join(
    f'{i},{40 + i},{_cell(i, 40 + i)},{_cell(i, 7000 + 100 * i)},'
    f'{_cell(i, f"{4.0 + i / 10:.1f}")},normal' for i in range(12)) + '\n'


def texty(col):
    """The predicate the loader uses."""
    return not (pd.api.types.is_numeric_dtype(col)
                or pd.api.types.is_bool_dtype(col))


@pytest.fixture(params=[False, True], ids=['object-strings', 'str-dtype'])
def frame(request):
    """The same CSV under both of pandas' string backends."""
    if request.param:
        # pandas 2.x implements the str dtype on pyarrow; without it the option
        # raises. Skip rather than fail, and let CI install it (see
        # requirements.txt) so the backend that broke the loader is covered.
        pytest.importorskip('pyarrow',
                            reason='the str backend needs pyarrow on pandas 2.x')
    try:
        pd.options.future.infer_string = request.param
    except Exception:                       # option gone: only one backend
        if request.param:
            pytest.skip('pandas has a single string backend')
    try:
        yield pd.read_csv(io.StringIO(CSV), na_values=NA_STRINGS,
                          skipinitialspace=True)
    finally:
        try:
            pd.options.future.infer_string = False
        except Exception:
            pass


def test_text_columns_are_detected_under_either_backend(frame):
    for c in ('pcv', 'wc', 'rc', 'rbc'):
        assert texty(frame[c]), f'{c} not seen as text ({frame[c].dtype})'
    assert not texty(frame['age'])
    assert not texty(frame['id'])


def test_tab_prefixed_columns_become_numeric(frame):
    df = frame.copy()
    for c in df.columns:
        if texty(df[c]):
            df[c] = (df[c].astype(str).str.strip()
                     .replace({s: pd.NA for s in NA_STRINGS}))
    for c in df.columns:
        if texty(df[c]):
            conv = pd.to_numeric(df[c], errors='coerce')
            ok = df[c].notna().sum()
            if ok and conv.notna().sum() >= .9 * ok:
                df[c] = conv
    for c in ('pcv', 'wc', 'rc'):
        assert pd.api.types.is_numeric_dtype(df[c]), f'{c} stayed text'
    assert not pd.api.types.is_numeric_dtype(df['rbc'])   # genuinely categorical
    assert df['rc'].isna().sum() == 4                     # the '?' cells


def test_the_dtype_equals_object_guard_would_have_failed(frame):
    """Documents the bug: under the str backend the old guard is simply false."""
    old = [c for c in frame.columns if frame[c].dtype == object]
    new = [c for c in frame.columns if texty(frame[c])]
    assert set(old) <= set(new)
    if str(frame['pcv'].dtype) != 'object':
        assert 'pcv' not in old and 'pcv' in new


def test_script_does_not_reintroduce_the_object_guard():
    src = SCRIPT.read_text(encoding='utf-8')
    assert 'def _texty(' in src, 'the loader lost its dtype-agnostic predicate'
    body = src[src.index('def load(name):'):src.index('def describe(D):')]
    assert "dtype == object" not in body, \
        'load() tests dtype == object again; it breaks under pandas >= 3'
    assert '_texty(df[c])' in body


# Table 1 of the revised article: rows, numerical and categorical columns, missing
# share of the numerical block, and the SHA-256 of the committed source file.
PUBLISHED = {
    'Heart':      (303, 10, 3, 0.20, '0e709332aaae8ca38e2ea66655e556a7c9fc59f83f0644f9edf660ad00f1e239'),
    'Wine':       (586, 11, 0, 0.00, '6402396156d04f7269074f4e58fa40ad1147f9298179bf388962d6ef9f916824'),
    'CKD':        (400, 14, 10, 13.89, '835c57ec100db2d3751165fae615fec425caf444cca23bab6a853d4c08bf5556'),
    'Vehicle':    (600, 18, 0, 0.00, 'dad03273414ae7127e30b6e4e66df90319a148b37ead5dd376dd140be5baedc4'),
    'WDBC':       (569, 30, 0, 0.00, 'b3845ddf3afff808cd6449daa546caf4d8b18f109b7ebb2344990e4c37fea291'),
    'Ionosphere': (350, 32, 2, 0.00, 'adb6fc877404f2b55f6d59fc7322265c733f5cf0ae6d7c2e52c3468733630023'),
    'Satimage':   (600, 36, 0, 0.00, '1ee8cf9693351db7afe68f6fc32942845caae3e1030c688efa6c5d0b24229f46'),
    'Optdigits':  (600, 57, 7, 0.00, 'b70b7ac68a54d1ccd41bd24f11324c3af96b67541791fb9d8684cbdf151533c3'),
    'Sonar':      (208, 60, 0, 0.00, '64378cabaecdb50e2e1e644c1d6bc359f0627ebe313870ad09e1e41ed972529b'),
    'Mice':       (600, 77, 3, 1.65, '1d6722b089db85dccfcb84d62e7299dcffd17b41223c3da54321890b63fff7ad'),
    'Libras':     (330, 90, 0, 0.00, '77c9c61118e72c3f49f52fd5eda8ec95ce9a573cb2f1a51b15480966f320fc85'),
}


@pytest.mark.parametrize('name', sorted(PUBLISHED))
def test_datasets_load_with_the_published_dimensions(name):
    """The guard that stops a run whose data do not match the article: a pandas
    upgrade that retypes a column, or a substituted file, changes these."""
    import numpy as np
    import benchmark as bm
    n, num, cat, miss, digest = PUBLISHED[name]
    D = bm.load(name)
    sub = D['arr'][:, D['num_idx']]
    assert bm.sha256(D['path']) == digest, f'{name}: the source file is not the committed one'
    assert (sub.shape[0], sub.shape[1], D['arr'].shape[1] - sub.shape[1]) == (n, num, cat)
    assert round(100 * np.isnan(sub).mean(), 2) == miss
