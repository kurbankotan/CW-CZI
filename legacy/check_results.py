# -*- coding: utf-8 -*-
"""Recompute every aggregate in results/ from the raw protocol files.

The raw files — protocolA_raw.csv, protocolB_raw.csv, protocolC_scaling.csv and
the two protocolD files — are the record of what was run. Everything else in
results/ is derived from them, and a partial rerun can leave a derived file
behind. This script rebuilds each one and reports any cell that disagrees, so a
stale aggregate cannot survive into the article.

    python check_results.py [results_dir] [--write]

--write replaces the derived files with the recomputed ones.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

METHODS = ['Mean', 'CW-CZI (U)', 'CW-CZI (M)', 'CW-CZI (R)', 'CW-CZI (A)',
           'KNN', 'SVD', 'EM', 'MICE', 'MissForest']
PROPOSED, ALT = 'CW-CZI (R)', 'CW-CZI (A)'

# Two methods can land on the same completion and therefore on the same NRMSE.
# Reached by different arithmetic paths those two numbers can differ in the last
# bit, and a Wilcoxon test would then treat a 1e-16 gap as a signed observation
# and give it the smallest rank. Round to TIE_DP before every comparison so an
# equality is an equality, and the ranks, win counts and p-values do not depend
# on the BLAS or the library version.
TIE_DP = 12


def holm(p):
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(run, 1.)
    return adj


def compare(df, val, keys, lower_better=True, ref=PROPOSED):
    wide = df.pivot_table(index=keys, columns='method',
                          values=val)[METHODS].dropna().round(TIE_DP)
    sign = 1 if lower_better else -1
    ranks = np.apply_along_axis(lambda r: rankdata(sign * r), 1, wide.values)
    avg = pd.Series(ranks.mean(0), index=wide.columns, name='avg_rank').sort_values()
    chi, pfr = friedmanchisquare(*[wide[m].values for m in wide.columns])
    out, ps = [], []
    for m in wide.columns:
        if m == ref:
            continue
        a, b = wide[ref].values, wide[m].values
        try:
            p = wilcoxon(a, b)[1]
        except ValueError:
            p = 1.
        out.append(dict(method=m, mean_proposed=a.mean(), mean_other=b.mean(),
                        mean_diff=(b - a).mean(),
                        proposed_wins=((b > a).mean() if lower_better else (b < a).mean()),
                        p_raw=p))
        ps.append(p)
    out = pd.DataFrame(out)
    out['p_holm'] = holm(np.array(ps))
    out['sig_005'] = out.p_holm < .05
    return avg, out.sort_values('p_holm'), (chi, pfr, len(wide))


def build(R):
    """Every derived file, rebuilt from the raw ones."""
    A = pd.read_csv(R / 'protocolA_raw.csv')
    B = pd.read_csv(R / 'protocolB_raw.csv')
    S = pd.read_csv(R / 'protocolC_scaling.csv')
    C = pd.read_csv(R / 'protocolD_convergence.csv')
    D = pd.read_csv(R / 'protocolD_sensitivity.csv')
    out = {}

    out['tableA1_nrmse_by_mechanism.csv'] = A.pivot_table(
        index=['mechanism', 'rate'], columns='method', values='nrmse',
        aggfunc='mean')[METHODS].round(4)
    a2 = A.pivot_table(index='dataset', columns='method', values='nrmse',
                       aggfunc='mean')[METHODS]
    t2 = a2.round(4)
    t2.loc['MEAN'] = a2.mean().round(4)
    out['tableA2_nrmse_by_dataset.csv'] = t2
    out['tableA3_variance_ratio.csv'] = A.pivot_table(
        index='mechanism', columns='method', values='var_ratio',
        aggfunc='mean')[METHODS].round(3)
    sel = A[A.method == ALT]
    out['tableA4_auto_selection.csv'] = (
        sel.groupby(['dataset', 'chosen_rule'])
           .agg(n=('nrmse', 'size'), lam=('chosen_lam', 'median'),
                iters=('chosen_iter', 'median')).reset_index())

    rawB1 = B.pivot_table(index='dataset', columns='method', values='accuracy',
                          aggfunc='mean')[METHODS]
    rawB1.loc['MEAN'] = rawB1.mean()
    out['tableB1_accuracy_by_dataset.csv'] = rawB1.round(4)
    b = rawB1.drop(index='MEAN')
    spr = pd.DataFrame({'accuracy_spread': b.max(axis=1) - b.min(axis=1),
                        'nrmse_spread': a2.max(axis=1) - a2.min(axis=1)}).dropna()
    spr.loc['MEAN'] = spr.mean()
    spr['ratio'] = spr.nrmse_spread / spr.accuracy_spread
    out['tableB2_protocol_discrimination.csv'] = spr.round(
        {'accuracy_spread': 4, 'nrmse_spread': 4, 'ratio': 1})

    out['tableC1_runtime.csv'] = S.pivot_table(
        index=['n', 'p'], columns='method', values='seconds')[METHODS].round(3)
    out['tableD1_cycles.csv'] = C.groupby(
        ['dataset', 'weighting'])['iteration'].max().unstack().round(1)
    out['tableD2_lambda_x_iter.csv'] = D[D.rule == 'ridge'].pivot_table(
        index='lam', columns='max_iter', values='nrmse').round(4)
    out['tableD3_rule_x_iter.csv'] = D.pivot_table(
        index='rule', columns='max_iter', values='nrmse').round(4)

    rankA, sigA, fa = compare(A, 'nrmse', ['dataset', 'mechanism', 'rate', 'repeat'], True)
    rankB, sigB, fb = compare(B, 'accuracy', ['dataset', 'repeat', 'classifier'], False)
    _, sigA2, _ = compare(A, 'nrmse', ['dataset', 'mechanism', 'rate', 'repeat'], True, ALT)
    _, sigB2, _ = compare(B, 'accuracy', ['dataset', 'repeat', 'classifier'], False, ALT)
    out['tableE1_avg_ranks_nrmse.csv'] = rankA.round(3).to_frame()
    out['tableE2_wilcoxon_nrmse.csv'] = sigA.round(5)
    out['tableE3_avg_ranks_accuracy.csv'] = rankB.round(3).to_frame()
    out['tableE4_wilcoxon_accuracy.csv'] = sigB.round(5)
    out['tableE5_wilcoxon_nrmse_auto.csv'] = sigA2.round(5)
    out['tableE6_wilcoxon_accuracy_auto.csv'] = sigB2.round(5)
    out['friedman.json'] = {
        'friedman_nrmse': dict(chi2=float(fa[0]), p=float(fa[1]), blocks=int(fa[2])),
        'friedman_accuracy': dict(chi2=float(fb[0]), p=float(fb[1]), blocks=int(fb[2]))}
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    R = pathlib.Path(args[0] if args else 'results')
    write = '--write' in sys.argv
    want = build(R)
    bad = []
    for name, val in want.items():
        path = R / name
        if not path.exists():
            bad.append(f'{name}: missing')
            continue
        if name.endswith('.json'):
            fresh = json.dumps(val, indent=2)
            same = json.load(open(path)) == val
        else:
            # serialise exactly as the experiment script does, then compare text:
            # this sidesteps guessing how many index columns a file was written with
            fresh = val.to_csv(index=not isinstance(val.index, pd.RangeIndex))
            same = path.read_text(encoding='utf-8').replace('\r\n', '\n') == fresh
        if not same:
            bad.append(name)
        if write:
            path.write_text(fresh, encoding='utf-8')

    print(f'{len(want)} derived files checked against the raw protocol files')
    for line in bad:
        print('  STALE  ' + line)
    if write and bad:
        print('\nrewritten from the raw files')
    elif bad:
        print('\nrun again with --write to rebuild them')
    else:
        print('  all consistent')
    return 1 if (bad and not write) else 0


if __name__ == '__main__':
    sys.exit(main())
