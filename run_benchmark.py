# -*- coding: utf-8 -*-
"""Run the CW-CZI benchmark (v7), resumably and in parallel.

    python run_benchmark.py A            # reconstruction, every dataset and method
    python run_benchmark.py B            # downstream classification
    python run_benchmark.py C            # wall-clock scaling (run it alone: --workers 1)
    python run_benchmark.py D            # CW-CZI convergence and sensitivity
    python run_benchmark.py E            # non-linearity screening

Useful options

    --datasets CKD,Mice        only these datasets
    --methods Iterative-RF     only these methods (comma-separated, exact labels)
    --exclude Iterative-RF     every method except these
    --mechanisms MCAR --rates 0.3 --reps 0      a slice of Protocol A
    --workers 4                blocks run in parallel, one process each
    --rf-jobs 1                threads per random forest (default: 1)
    --out results/raw          where the records go
    --list                     print what would run and stop

Every record is appended to results/raw/<protocol>/<dataset>__<method>.csv the
moment its block finishes. A run that is interrupted — a closed laptop, a Colab
time-out — picks up where it stopped when it is started again with the same
command: blocks already on disk are skipped. Runs on different machines can
therefore share the work and their raw folders can simply be merged.

Each process is pinned to one BLAS thread so that blocks running side by side
do not compete for cores and timings stay comparable across methods.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import csv
import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import benchmark as bm

# ─── one worker call per block ───────────────────────────────────────────────
@lru_cache(maxsize=None)
def _data(name):
    return bm.load(name)


def _work(job):
    warnings.filterwarnings('ignore')
    proto, key, methods, rf_jobs = job
    t0 = time.time()
    try:
        if proto == 'A':
            name, mech, rate, rep = key
            recs = bm.block_A(_data(name), mech, rate, rep, methods, rf_jobs)
        elif proto == 'B':
            name, rep = key
            recs = bm.split_B(_data(name), rep, methods, rf_jobs)
        elif proto == 'C':
            n, p = key
            recs = bm.cell_C(n, p, methods, rf_jobs)
        elif proto == 'Dconv':
            recs = bm.convergence_D(_data(key[0]))
        elif proto == 'Dsens':
            name, mech, rep = key
            recs = bm.sensitivity_D(_data(name), mech, rep)
        elif proto == 'Ediag':
            recs = bm.diagnostic_real(_data(key[0]))
        elif proto == 'Esyn':
            kind, a, rep = key
            recs = bm.block_E(kind, a, rep, methods, rf_jobs)
        else:
            raise ValueError(proto)
        return proto, key, recs, None, time.time() - t0
    except Exception:                                    # keep going, report later
        return proto, key, [], traceback.format_exc(), time.time() - t0


# ─── where records go and what is already done ───────────────────────────────
def _file(out, proto, rec):
    sub = {'A': 'A', 'B': 'B', 'C': 'C', 'Dconv': 'D', 'Dsens': 'D',
           'Ediag': 'E', 'Esyn': 'E'}[proto]
    os.makedirs(os.path.join(out, sub), exist_ok=True)
    if proto in ('A', 'B'):
        name = f"{rec['dataset']}__{bm.slug(rec['method'])}.csv"
    elif proto == 'C':
        name = f"scaling__{bm.slug(rec['method'])}.csv"
    elif proto == 'Dconv':
        name = 'convergence.csv'
    elif proto == 'Dsens':
        name = f"sensitivity__{rec['dataset']}.csv"
    elif proto == 'Ediag':
        name = 'diagnostic_real.csv'
    else:
        name = f"synthetic__{bm.slug(rec['method'])}.csv"
    return os.path.join(out, sub, name)


def _append(path, recs):
    """One write per block: a crash can lose a block, never corrupt a file."""
    new = not os.path.exists(path)
    cols = list(recs[0].keys())
    if not new:
        with open(path, newline='', encoding='utf-8') as f:
            cols = next(csv.reader(f))
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction='ignore', lineterminator='\n')
    if new:
        w.writeheader()
    w.writerows(recs)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        f.write(buf.getvalue())


def _done(path, keycols):
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    return set(map(tuple, df[keycols].astype(str).values.tolist()))


def _k(*vals):
    return tuple(str(v) for v in vals)


# ─── job lists ───────────────────────────────────────────────────────────────
COST = {'Mice': 10, 'Libras': 7, 'Optdigits': 7, 'Sonar': 4, 'Satimage': 4,
        'WDBC': 3, 'Ionosphere': 2, 'CKD': 2, 'Vehicle': 1.5, 'Wine': 1, 'Heart': 1}


def jobs_for(args):
    out = args.out; js = []
    if args.protocol == 'A':
        for name in args.datasets:
            done = {m: _done(_file(out, 'A', dict(dataset=name, method=m)),
                             ['mechanism', 'rate', 'repeat']) for m in args.methods}
            for mech in args.mechanisms:
                for rate in args.rates:
                    for rep in args.reps:
                        todo = [m for m in args.methods if _k(mech, rate, rep) not in done[m]]
                        if todo:
                            js.append(('A', (name, mech, rate, rep), todo))
    elif args.protocol == 'B':
        for name in args.datasets:
            done = {m: _done(_file(out, 'B', dict(dataset=name, method=m)), ['repeat'])
                    for m in args.methods}
            for rep in args.reps:
                todo = [m for m in args.methods if _k(rep) not in done[m]]
                if todo:
                    js.append(('B', (name, rep), todo))
    elif args.protocol == 'C':
        done = {m: _done(_file(out, 'C', dict(method=m)), ['n', 'p']) for m in args.methods}
        for n in bm.GRID_N:
            for p in bm.GRID_P:
                todo = [m for m in args.methods if _k(n, p) not in done[m]]
                if todo:
                    js.append(('C', (n, p), todo))
    elif args.protocol == 'D':
        conv = _done(_file(out, 'Dconv', {}), ['dataset'])
        for name in args.datasets:
            if _k(name) not in conv:
                js.append(('Dconv', (name,), []))
            sens = _done(_file(out, 'Dsens', dict(dataset=name)), ['mechanism', 'repeat'])
            for mech in args.mechanisms:
                for rep in args.reps:
                    if _k(mech, rep) not in sens:
                        js.append(('Dsens', (name, mech, rep), []))
    elif args.protocol == 'E':
        diag = _done(_file(out, 'Ediag', {}), ['dataset'])
        for name in args.datasets:
            if _k(name) not in diag:
                js.append(('Ediag', (name,), []))
        methods = [m for m in args.methods if m in bm.E_METHODS] if args.methods_given \
            else list(bm.E_METHODS)
        done = {m: _done(_file(out, 'Esyn', dict(method=m)), ['kind', 'level', 'repeat'])
                for m in methods}
        for kind, levels in bm.E_LEVELS.items():
            for a in levels:
                for rep in args.reps:
                    todo = [m for m in methods if _k(kind, a, rep) not in done[m]]
                    if todo:
                        js.append(('Esyn', (kind, a, rep), todo))
    # heaviest first, so the pool finishes evenly
    js.sort(key=lambda j: -COST.get(j[1][0], 1) if isinstance(j[1][0], str) else 0)
    return [(p, k, m, args.rf_jobs) for p, k, m in js]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('protocol', choices=['A', 'B', 'C', 'D', 'E'])
    ap.add_argument('--datasets', default=','.join(bm.DATASETS))
    ap.add_argument('--methods', default=None)
    ap.add_argument('--exclude', default='')
    ap.add_argument('--mechanisms', default=','.join(bm.MECHANISMS))
    ap.add_argument('--rates', default=','.join(map(str, bm.MASK_RATES)))
    ap.add_argument('--reps', default=None)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument('--rf-jobs', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(bm.ROOT, 'results', 'raw'))
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    a.datasets = [d for d in a.datasets.split(',') if d]
    unknown = set(a.datasets) - set(bm.DATASETS)
    if unknown:
        ap.error(f'unknown datasets: {sorted(unknown)}')
    a.methods_given = a.methods is not None
    allowed = bm.METHODS + bm.SENSITIVITY_METHODS
    a.methods = [m for m in (a.methods.split(',') if a.methods else bm.METHODS) if m]
    bad = set(a.methods) - set(allowed)
    if bad:
        ap.error(f'unknown methods: {sorted(bad)}; choose from {allowed}')
    excl = set(m for m in a.exclude.split(',') if m)
    a.methods = [m for m in a.methods if m not in excl]
    a.mechanisms = [m for m in a.mechanisms.split(',') if m]
    a.rates = [float(r) for r in a.rates.split(',') if r]
    if a.reps is None:
        a.reps = list(range(bm.N_REPEATS))
    else:
        a.reps = [int(r) for r in a.reps.split(',') if r]

    js = jobs_for(a)
    print(f'Protocol {a.protocol}: {len(js)} block(s) to run, {a.workers} worker(s), '
          f'random-forest threads {a.rf_jobs}', flush=True)
    if a.list or not js:
        for j in js[:10]:
            print('  ', j[0], j[1], j[2])
        if len(js) > 10:
            print(f'   ... and {len(js) - 10} more')
        return 0
    os.makedirs(a.out, exist_ok=True)
    env = bm.environment(dict(protocol=a.protocol, workers=a.workers, rf_jobs=a.rf_jobs,
                              argv=sys.argv[1:]))
    stamp = time.strftime('%Y%m%d-%H%M%S')
    with open(os.path.join(a.out, f'environment_{a.protocol}_{env["machine"]}_{stamp}.json'),
              'w', encoding='utf-8') as f:
        json.dump(env, f, indent=2)

    t0 = time.time(); errors = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_work, j) for j in js]
        for k, fut in enumerate(as_completed(futs), 1):
            proto, key, recs, err, dt = fut.result()
            if err:
                errors += 1
                with open(os.path.join(a.out, 'errors.log'), 'a', encoding='utf-8') as f:
                    f.write(f'--- {proto} {key}\n{err}\n')
                print(f'  [{k}/{len(js)}] {proto} {key} FAILED (see errors.log)', flush=True)
                continue
            groups = {}
            for r in recs:
                groups.setdefault(_file(a.out, proto, r), []).append(r)
            for path, rs in groups.items():
                _append(path, rs)
            el = time.time() - t0
            eta = el / k * (len(js) - k)
            print(f'  [{k}/{len(js)}] {proto} {key} {dt:.0f}s   elapsed {el/60:.1f} min, '
                  f'about {eta/60:.0f} min left', flush=True)
    print(f'done in {(time.time() - t0)/60:.1f} min, {errors} failed block(s)', flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BrokenPipeError:                      # output piped into `head`
        sys.exit(0)
