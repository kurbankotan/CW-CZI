# -*- coding: utf-8 -*-
"""Check that every derived file in results/ is what the raw records produce.

The raw records under results/raw/ are the record of what was run; every table,
figure and numbers.json beside them is derived by analyze_results.py. This
script re-derives them in a scratch directory and compares each table and every
number with the committed one, so a stale derived file cannot survive into the
article.

    python check_results.py [results_dir]
"""
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent


def same_json(a, b, path='', tol=1e-9):
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                bad.append(f'{path}/{k}: present on one side only')
            else:
                bad += same_json(a[k], b[k], f'{path}/{k}', tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append(f'{path}: length {len(a)} != {len(b)}')
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                bad += same_json(x, y, f'{path}[{i}]', tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (math.isclose(a, b, rel_tol=tol, abs_tol=tol) or (a != a and b != b)):
            bad.append(f'{path}: {a} != {b}')
    elif a != b:
        bad.append(f'{path}: {a!r} != {b!r}')
    return bad


def main(results):
    results = pathlib.Path(results).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        shutil.copytree(results / 'raw', tmp / 'raw')
        subprocess.run([sys.executable, str(ROOT / 'analyze_results.py'), str(tmp)], check=True,
                       stdout=subprocess.DEVNULL)
        bad = []
        for f in sorted(results.glob('*.csv')):
            g = tmp / f.name
            if not g.exists():
                bad.append(f'{f.name}: not produced by the raw records'); continue
            a, b = pd.read_csv(f), pd.read_csv(g)
            try:
                pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)
            except AssertionError as e:
                bad.append(f'{f.name}: {str(e).splitlines()[0]}')
        bad += same_json(json.loads((results / 'numbers.json').read_text(encoding='utf-8')),
                         json.loads((tmp / 'numbers.json').read_text(encoding='utf-8')), 'numbers.json')
    for line in bad:
        print('MISMATCH', line)
    print(f'{len(bad)} mismatch(es) in {results}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'results'))
