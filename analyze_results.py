# -*- coding: utf-8 -*-
"""Turn results/raw into every table, figure and statistic of the revised article.

    python analyze_results.py [results_dir]

Reads the per-block records that run_benchmark.py wrote under results/raw/ and
writes, next to them in results/:

  table*.csv        one file per table of the article (unrounded values are kept
                    in the *_raw.csv files; tables round once, at the end)
  figure*.png       the figures
  numbers.json      every number the text quotes, so the manuscript can be
                    checked against the run
  completeness.txt  which (dataset, method, block) records exist and which do not

Statistics follow the revised Section II.I. The unit of inference is the
dataset: for each mechanism a method's error on a dataset is its mean over the
twenty (rate, repetition) blocks, the methods are ranked within each dataset,
and the Friedman and Holm-corrected Wilcoxon tests run over the eleven datasets.
Block-level ranks and win rates are reported beside them as descriptions, not
as tests, because blocks drawn from one dataset are not independent.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kendalltau, rankdata, spearmanr, wilcoxon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import benchmark as bm

warnings.filterwarnings('ignore')
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(bm.ROOT, 'results')
RAW = os.path.join(R, 'raw')
TIE_DP = 12
NUM = {}                                     # numbers quoted in the text

# Figures: eight methods at most per chart, each with a fixed colour, marker and
# dash, so that a method looks the same in every figure and identity never rests
# on colour alone (print, greyscale, colour-vision deficiency).
FIG_METHODS = ['CW-CZI (R)', 'EM', 'Iterative-RF', 'CW-CZI (A)', 'Iterative-BR',
               'KNN', 'CW-CZI (M)', 'Mean']
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
           '#6250d6', '#e34948']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
DASHES = [(None, None), (5, 2), (1, 1.5), (7, 2, 1, 2), (3, 1), (6, 2, 1, 2, 1, 2),
          (2, 2), (8, 3)]
STYLE = {m: dict(color=c, marker=k, dashes=d) for m, c, k, d in
         zip(FIG_METHODS, PALETTE, MARKERS, DASHES)}
INK, MUTED = '#0b0b0b', '#52514e'
plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 400, 'font.size': 8,
                     'font.family': 'serif', 'axes.grid': True, 'grid.alpha': .25,
                     'grid.linewidth': .4, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .6,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'legend.frameon': False, 'lines.linewidth': 1.4})


def line(ax, x, y, m, **kw):
    s = STYLE[m]
    ax.plot(x, y, color=s['color'], marker=s['marker'], dashes=s['dashes'],
            markersize=3.6, markeredgewidth=0.6, markeredgecolor='white',
            label=m, **kw)


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(R, name), bbox_inches='tight')
    plt.close(fig)


# ─── reading ─────────────────────────────────────────────────────────────────
def read(sub, prefix=''):
    d = os.path.join(RAW, sub)
    if not os.path.isdir(d):
        return pd.DataFrame()
    fs = sorted(f for f in os.listdir(d) if f.endswith('.csv') and f.startswith(prefix))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(os.path.join(d, f)) for f in fs], ignore_index=True)


def present(methods, df):
    have = set(df.method.unique())
    return [m for m in methods if m in have]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(run, 1.)
    return adj


def order_datasets(A):
    p = {d: len(bm.load(d)['num_idx']) for d in A.dataset.unique()}
    return sorted(p, key=lambda d: (p[d], d)), p


def place_labels(ax, xs, ys, names, size=5.2):
    """Direct labels that collide neither with each other nor with the points.
    Each label tries, in order, the right, left, upper and lower side of its
    point at growing distances, and keeps the first spot that is free; a thin
    leader line joins it to its point when it ended up away from it."""
    fig = ax.figure; fig.canvas.draw()
    to_disp = ax.transData.transform
    k = fig.dpi / 72.0                                     # display pixels per point
    lh = size * 1.25 * k
    P = [to_disp((x, y)) for x, y in zip(xs, ys)]
    r = 4.2 * k                                            # point radius incl. margin
    obstacles = [(x - r, x + r, y - r, y + r) for x, y in P]
    x_lo, y_lo = to_disp((ax.get_xlim()[0], ax.get_ylim()[0]))
    x_hi, y_hi = to_disp((ax.get_xlim()[1], ax.get_ylim()[1]))

    def free(b, own):
        if b[0] < x_lo or b[1] > x_hi + 40 * k or b[2] < y_lo or b[3] > y_hi:
            return False
        return all(b[0] >= o[1] or b[1] <= o[0] or b[2] >= o[3] or b[3] <= o[2]
                   for j, o in enumerate(obstacles) if j != own)

    order = sorted(range(len(xs)), key=lambda i: -P[i][1])
    for i in order:
        x, y = P[i]; w = len(names[i]) * size * 0.60 * k
        best = None
        for d in [1, 2, 3, 4, 5, 6, 8]:
            g = 2.5 * k * d
            for bx, by in [(x + g, y - lh / 2), (x - g - w, y - lh / 2), (x + g * .3, y + g),
                           (x + g * .3, y - g - lh), (x - w - g * .3, y + g), (x - w - g * .3, y - g - lh)]:
                box = (bx, bx + w, by, by + lh)
                if free(box, i):
                    best = box; break
            if best:
                break
        if best is None:
            best = (x + 2.5 * k, x + 2.5 * k + w, y - lh / 2, y + lh / 2)
        obstacles.append(best)
        moved = abs((best[0] + best[1]) / 2 - x) > w / 2 + 6 * k or abs((best[2] + best[3]) / 2 - y) > lh
        ax.annotate(names[i], (xs[i], ys[i]), xytext=((best[0] - x) / k, (best[2] - y) / k),
                    textcoords='offset points', fontsize=size, color=MUTED, va='bottom', ha='left',
                    arrowprops=dict(arrowstyle='-', lw=.35, color='#a3a29b', shrinkA=0, shrinkB=2)
                    if moved else None)


# ─── statistics ──────────────────────────────────────────────────────────────
def block_ranks(df, val, methods, keys):
    methods = present(methods, df)
    wide = df[df.method.isin(methods)].pivot_table(index=keys, columns='method',
                                                   values=val)[methods].dropna().round(TIE_DP)
    rk = np.apply_along_axis(rankdata, 1, wide.values)
    return pd.Series(rk.mean(0), index=methods), wide


def dataset_level(df, val, methods, ref, lower_better=True):
    """Methods ranked within each dataset on their mean over the dataset's blocks;
    Friedman over datasets; Wilcoxon of `ref` against each other method over the
    datasets, Holm-corrected; effect sizes beside every p-value."""
    sign = 1 if lower_better else -1
    methods = present(methods, df)
    means = df[df.method.isin(methods)].groupby(['dataset', 'method'])[val].mean() \
        .unstack()[methods].dropna()
    rk = np.apply_along_axis(lambda r: rankdata(sign * r), 1, means.round(TIE_DP).values)
    avg = pd.Series(rk.mean(0), index=methods)
    chi, pf = friedmanchisquare(*[means[m].values for m in methods]) if len(means) > 2 \
        else (np.nan, np.nan)
    keys = [c for c in ['dataset', 'mechanism', 'rate', 'repeat', 'classifier']
            if c in df.columns]
    _, bw = block_ranks(df, val, methods, keys) if lower_better else (None, None)
    rows = []
    for m in methods:
        if m == ref:
            continue
        a, b = means[ref].values, means[m].values
        try:
            p = wilcoxon(a, b)[1]
        except ValueError:
            p = 1.0
        d = (b - a) if lower_better else (a - b)            # > 0: ref is better
        rel = 100 * d / np.where(np.abs(b) > 0, np.abs(b), np.nan)
        row = dict(method=m, ref_mean=a.mean(), other_mean=b.mean(), mean_diff=d.mean(),
                   median_rel_diff_pct=float(np.nanmedian(rel)),
                   datasets_ref_better=int((d > 0).sum()), datasets=len(d), p_raw=p)
        if bw is not None:
            row['blocks_ref_better_pct'] = 100 * float((bw[m].values > bw[ref].values).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out['p_holm'] = holm(out.p_raw.values)
    return avg.sort_values(), out.sort_values('mean_diff'), dict(chi2=chi, p=pf, n=len(means)), means


def sensitivity_r30(Aall):
    """The fixed ridge rule run for 30 cycles instead of 5 (Section III.F): its gain
    over the default, and its standing in the untuned comparison in place of R."""
    out, rows = {}, []
    for mech in bm.MECHANISMS:
        sub = Aall[Aall.mechanism == mech]
        need = ['CW-CZI (R)', 'CW-CZI (R30)', 'EM']
        w = sub.pivot_table(index=['dataset', 'rate', 'repeat'], columns='method', values='nrmse_pre')
        if not set(need) <= set(w.columns):
            continue
        w = w.dropna(subset=need)
        d = w.groupby(level='dataset').mean()
        gain = 100 * (d['CW-CZI (R)'] - d['CW-CZI (R30)']) / d['CW-CZI (R)']
        methods = present([m if m != 'CW-CZI (R)' else 'CW-CZI (R30)' for m in bm.DEFAULT_SET], sub)
        avg, pair, fr, _ = dataset_level(sub[sub.dataset.isin(d.index)], 'nrmse_pre', methods,
                                         'CW-CZI (R30)')
        em = pair.set_index('method').loc['EM']
        r5, r30 = w['CW-CZI (R)'].mean(), w['CW-CZI (R30)'].mean()
        out[mech] = dict(datasets=len(d), R=r5, R30=r30, gain_pct=100 * (r5 - r30) / r5,
                         datasets_r30_better=int((gain > 1e-9).sum()),
                         largest_gain_dataset=gain.idxmax(), largest_gain_pct=float(gain.max()),
                         largest_loss_dataset=gain.idxmin(), largest_loss_pct=float(gain.min()),
                         rank_r30=float(avg['CW-CZI (R30)']), rank_em=float(avg['EM']),
                         ranks={k: float(v) for k, v in avg.items()},
                         vs_em_mean_diff=float(em.mean_diff), vs_em_r30_better=int(em.datasets_ref_better),
                         vs_em_p_holm=float(em.p_holm))
        for ds in d.index:
            rows.append(dict(mechanism=mech, dataset=ds, R=d.loc[ds, 'CW-CZI (R)'],
                             R30=d.loc[ds, 'CW-CZI (R30)'], EM=d.loc[ds, 'EM'], gain_pct=gain[ds]))
    if rows:
        pd.DataFrame(rows).round(6).to_csv(os.path.join(R, 'tableA8_r30_sensitivity.csv'), index=False)
    return out


# ─── Protocol A ──────────────────────────────────────────────────────────────
def protocol_A(A, order, width):
    A = A[A.method.isin(bm.METHODS)].copy()
    MET = present(bm.METHODS, A)
    A.to_csv(os.path.join(R, 'protocolA_raw.csv'), index=False)
    mechs = bm.MECHANISMS
    # Table: mechanism x rate (mean over datasets and repetitions)
    t = A.pivot_table(index=['mechanism', 'rate'], columns='method', values='nrmse_pre',
                      aggfunc='mean')[MET]
    t.loc[mechs].round(6).to_csv(os.path.join(R, 'tableA1_nrmse_by_mechanism_rate.csv'))
    # Table: dataset x mechanism
    t = A.pivot_table(index=['mechanism', 'dataset'], columns='method', values='nrmse_pre',
                      aggfunc='mean')[MET]
    t = t.reindex(pd.MultiIndex.from_product([mechs, order]))
    t.round(6).to_csv(os.path.join(R, 'tableA2_nrmse_by_dataset.csv'))
    # variance ratio: the median over blocks is reported, because under MNAR the
    # hidden cells are the upper tail of each column and their own spread can be
    # tiny (Ionosphere's columns pile up at their maximum), which makes the
    # block-level ratio, and therefore its mean, unstable
    vr = {stat: A.pivot_table(index='mechanism', columns='method', values='var_ratio',
                              aggfunc=stat).reindex(columns=MET).loc[mechs] for stat in ('median', 'mean')}
    pd.concat(vr, names=['statistic']).round(3).to_csv(os.path.join(R, 'tableA3_variance_ratio.csv'))
    NUM['var_ratio'] = vr['median'].round(3).to_dict()
    NUM['var_ratio_mean'] = vr['mean'].round(3).to_dict()
    vd = A[A.mechanism == 'MNAR'].pivot_table(index='dataset', columns='method', values='var_ratio',
                                               aggfunc='mean').reindex(columns=MET)
    NUM['var_ratio_mnar_by_dataset'] = vd.round(3).to_dict()
    NUM['var_ratio_mnar_above_one'] = {m: int((vd[m] > 1).sum()) for m in MET}

    # ── the ladder, by dataset and mechanism, against width and redundancy
    lad = A[A.method.isin(['CW-CZI (U)', 'CW-CZI (M)', 'CW-CZI (R)'])] \
        .pivot_table(index=['mechanism', 'dataset'], columns='method', values='nrmse_pre')
    lad.columns = ['M', 'R', 'U']; lad = lad[['U', 'M', 'R']]
    lad['U_to_M_pct'] = 100 * (lad.U - lad.M) / lad.U
    lad['M_to_R_pct'] = 100 * (lad.M - lad.R) / lad.M
    red = {d: bm.redundancy(bm.load(d)['arr'][:, bm.load(d)['num_idx']]) for d in order}
    lad = lad.reset_index()
    lad['p'] = lad.dataset.map(width); lad['redundancy'] = lad.dataset.map(red)
    lad['n_over_p'] = lad.dataset.map({d: bm.load(d)['arr'].shape[0] / width[d] for d in order})
    lad = lad.set_index(['mechanism', 'dataset']).reindex(
        pd.MultiIndex.from_product([mechs, order])).reset_index() \
        .rename(columns={'level_0': 'mechanism', 'level_1': 'dataset'})
    lad.round(6).to_csv(os.path.join(R, 'table4_ladder_by_dataset.csv'), index=False)
    pooled = A[A.method.isin(['CW-CZI (U)', 'CW-CZI (M)', 'CW-CZI (R)'])] \
        .pivot_table(index='mechanism', columns='method', values='nrmse_pre')
    NUM['ladder_pooled'] = {mech: dict(
        U=pooled.loc[mech, 'CW-CZI (U)'], M=pooled.loc[mech, 'CW-CZI (M)'],
        R=pooled.loc[mech, 'CW-CZI (R)'],
        U_to_M=100 * (1 - pooled.loc[mech, 'CW-CZI (M)'] / pooled.loc[mech, 'CW-CZI (U)']),
        M_to_R=100 * (1 - pooled.loc[mech, 'CW-CZI (R)'] / pooled.loc[mech, 'CW-CZI (M)']))
        for mech in mechs}
    cor = []
    for mech in mechs:
        s = lad[lad.mechanism == mech]
        for x in ['p', 'redundancy', 'n_over_p']:
            rho, pv = spearmanr(s[x], s.M_to_R_pct)
            cor.append(dict(mechanism=mech, predictor=x, spearman_rho=rho, p=pv, n=len(s)))
    cor = pd.DataFrame(cor); cor.round(6).to_csv(os.path.join(R, 'table4b_width_correlation.csv'),
                                                  index=False)
    NUM['width_correlation'] = cor.round(6).to_dict('records')
    NUM['ladder_by_dataset'] = lad.round(6).to_dict('records')

    # ── accuracy rankings: untuned set (reference R) and tuned set (reference A)
    for tag, methods, ref in [('untuned', present(bm.DEFAULT_SET, A), bm.PROPOSED),
                              ('tuned', present(['CW-CZI (R)'] + bm.TUNED_SET, A), bm.TUNED_REF),
                              ('all', MET, bm.PROPOSED)]:
        res = []
        for mech in mechs:
            sub = A[A.mechanism == mech]
            avg, pair, fr, _ = dataset_level(sub, 'nrmse_pre', methods, ref)
            brk, _ = block_ranks(sub, 'nrmse_pre', methods, ['dataset', 'rate', 'repeat'])
            for m in avg.index:
                res.append(dict(mechanism=mech, method=m, dataset_avg_rank=avg[m],
                                block_avg_rank=brk[m],
                                mean_nrmse=sub[sub.method == m].nrmse_pre.mean()))
            pair.insert(0, 'mechanism', mech)
            pair.round(5).to_csv(os.path.join(R, f'tableE_{tag}_{mech}_pairwise.csv'), index=False)
            NUM.setdefault('friedman', {})[f'{tag}_{mech}'] = {k: float(v) for k, v in fr.items()}
            NUM.setdefault('pairwise', {})[f'{tag}_{mech}'] = pair.round(5).to_dict('records')
        res = pd.DataFrame(res)
        res.round(6).to_csv(os.path.join(R, f'tableE_{tag}_ranks.csv'), index=False)
        NUM.setdefault('ranks', {})[tag] = res.round(6).to_dict('records')
        # overall: rank within each (dataset, mechanism), averaged
        allm = A[A.method.isin(methods)].groupby(['mechanism', 'dataset', 'method']) \
            .nrmse_pre.mean().unstack()[methods].dropna(axis=1, how='all').dropna().round(TIE_DP)
        ov = pd.Series(np.apply_along_axis(rankdata, 1, allm.values).mean(0), index=allm.columns)
        NUM.setdefault('overall_rank', {})[tag] = ov.sort_values().round(3).to_dict()

    # ── robustness of the ordering to the normalisation (Reviewer 1, point 4)
    rob = []
    for mech in mechs:
        sub = A[A.mechanism == mech]; ranks = {}
        for val in ['nrmse_pre', 'nrmse_post', 'nrmse_col']:
            ranks[val], _ = block_ranks(sub, val, MET, ['dataset', 'rate', 'repeat'])
            for m in ranks[val].index:
                rob.append(dict(mechanism=mech, metric=val, method=m, block_avg_rank=ranks[val][m],
                                mean=sub[sub.method == m][val].mean()))
        for val in ['nrmse_post', 'nrmse_col']:
            tau = kendalltau(ranks['nrmse_pre'].values, ranks[val].values)[0]
            NUM.setdefault('normalisation_tau', {})[f'{mech}_{val}'] = float(tau)
    pd.DataFrame(rob).round(6).to_csv(os.path.join(R, 'tableA5_normalisation_robustness.csv'),
                                      index=False)

    # how much the post-masking scale inflates the error, by mechanism
    NUM['post_over_pre'] = {mech: float(A[A.mechanism == mech].nrmse_post.mean()
                                        / A[A.mechanism == mech].nrmse_pre.mean()) for mech in mechs}
    NUM['post_over_pre_R'] = {mech: float(A[(A.mechanism == mech) & (A.method == 'CW-CZI (R)')].nrmse_post.mean()
                                          / A[(A.mechanism == mech) & (A.method == 'CW-CZI (R)')].nrmse_pre.mean())
                              for mech in mechs}
    # does the error rise with the rate? (share of method x mechanism curves that are monotone)
    t = A.groupby(['mechanism', 'method', 'rate']).nrmse_pre.mean().unstack()
    NUM['monotone_in_rate'] = {mech: float((t.loc[mech].diff(axis=1).iloc[:, 1:] >= 0).all(axis=1).mean())
                               for mech in mechs}
    NUM['mnar_over_mcar'] = float(A[A.mechanism == 'MNAR'].groupby('method').nrmse_pre.mean().median()
                                  / A[A.mechanism == 'MCAR'].groupby('method').nrmse_pre.mean().median())
    # the default of the ridge rule, judged only on the eight datasets added in the revision
    new8 = [d for d in order if d not in bm.ORIGINAL_DATASETS]
    for tag, sub_ds in [('new8', new8), ('orig3', bm.ORIGINAL_DATASETS)]:
        sub = A[A.dataset.isin(sub_ds)]
        rk = {}
        for mech in mechs:
            avg, _, _, _ = dataset_level(sub[sub.mechanism == mech], 'nrmse_pre',
                                         present(bm.DEFAULT_SET, sub), bm.PROPOSED)
            rk[mech] = avg.round(3).to_dict()
        NUM.setdefault('ranks_subset', {})[tag] = rk
    # per dataset: which untuned method is most accurate under each mechanism
    best = A[A.method.isin(present(bm.DEFAULT_SET, A))].groupby(['mechanism', 'dataset', 'method']) \
        .nrmse_pre.mean().unstack()
    NUM['best_untuned_by_dataset'] = {f'{m}|{d}': best.loc[(m, d)].idxmin() for m, d in best.index}
    # the ridge rule against the best other untuned method, dataset by dataset
    rows = []
    for (m, d) in best.index:
        r = best.loc[(m, d)]; others = r.drop('CW-CZI (R)')
        rows.append(dict(mechanism=m, dataset=d, best_other=others.idxmin(),
                         best_other_nrmse=others.min(), R=r['CW-CZI (R)'],
                         rel_pct=100 * (r['CW-CZI (R)'] - others.min()) / others.min()))
    bo = pd.DataFrame(rows)
    bo.round(6).to_csv(os.path.join(R, 'tableA9_r_vs_best_other.csv'), index=False)
    NUM['r_vs_best_other'] = {m: dict(r_best=int((g.rel_pct < 0).sum()),
                                      within_5=int(((g.rel_pct >= 0) & (g.rel_pct <= 5)).sum()),
                                      beyond_10=int((g.rel_pct > 10).sum()),
                                      worst=g.loc[g.rel_pct.idxmax(), ['dataset', 'best_other', 'rel_pct']]
                                      .to_dict())
                              for m, g in bo.groupby('mechanism')}

    # ── default versus tuned (Reviewer 2, point 2)
    pairs = [('CW-CZI (R)', 'CW-CZI (A)'), ('KNN', 'KNN-T'), ('SVD', 'SVD-T'),
             ('Iterative-BR', 'Iterative-T')]
    dt = []
    for mech in mechs:
        sub = A[A.mechanism == mech]
        for a, b in pairs:
            x = sub[sub.method == a].set_index(['dataset', 'rate', 'repeat']).nrmse_pre
            y = sub[sub.method == b].set_index(['dataset', 'rate', 'repeat']).nrmse_pre
            j = x.index.intersection(y.index); x, y = x[j], y[j]
            dx = x.groupby(level='dataset').mean(); dy = y.groupby(level='dataset').mean()
            dt.append(dict(mechanism=mech, default=a, tuned=b, default_mean=x.mean(),
                           tuned_mean=y.mean(), gain_pct=100 * (x.mean() - y.mean()) / x.mean(),
                           tuned_better_blocks_pct=100 * float((y < x - 1e-12).mean()),
                           tied_blocks_pct=100 * float((abs(y - x) <= 1e-12).mean()),
                           tuned_worse_blocks_pct=100 * float((y > x + 1e-12).mean()),
                           datasets_tuned_better=int((dy < dx - 1e-12).sum()),
                           datasets_tuned_worse=int((dy > dx + 1e-12).sum()),
                           largest_gain_dataset=(100 * (dx - dy) / dx).idxmax(),
                           largest_gain_pct=float((100 * (dx - dy) / dx).max()),
                           largest_loss_dataset=(100 * (dx - dy) / dx).idxmin(),
                           largest_loss_pct=float((100 * (dx - dy) / dx).min())))
    dt = pd.DataFrame(dt); dt.round(6).to_csv(os.path.join(R, 'tableA6_default_vs_tuned.csv'),
                                               index=False)
    NUM['default_vs_tuned'] = dt.round(6).to_dict('records')
    ch = A[A.choice.notna() & (A.choice != '')]
    if len(ch):
        cc = ch.groupby(['method', 'choice']).size().rename('n').reset_index() \
            .sort_values(['method', 'n'], ascending=[True, False])
        cc.to_csv(os.path.join(R, 'tableA4_tuning_choices.csv'), index=False)
        NUM['tuning_choices'] = {m: g.set_index('choice').n.head(6).to_dict()
                                 for m, g in cc.groupby('method')}
        # what CW-CZI (A) chose, and whether its ridge parameter follows redundancy
        ca = ch[ch.method == 'CW-CZI (A)'].copy()
        if len(ca):
            ca['rule'] = ca.choice.str.split('/').str[0]
            ca['lam'] = ca.choice.str.extract(r'lam=([0-9.]+)')[0].astype(float)
            ca['T'] = ca.choice.str.extract(r'T=(\d+)')[0].astype(int)
            rid = ca[ca.rule == 'ridge']
            geo = rid.groupby('dataset').lam.apply(lambda v: float(np.exp(np.log(v).mean())))
            ds = [d for d in order if d in geo.index]
            rho, pv = spearmanr([geo[d] for d in ds], [red[d] for d in ds])
            by_mech = {}
            for mech in mechs:
                g = rid[rid.mechanism == mech].groupby('dataset').lam \
                    .apply(lambda v: float(np.exp(np.log(v).mean())))
                dm = [d for d in order if d in g.index]
                r_, p_ = spearmanr([g[d] for d in dm], [red[d] for d in dm])
                by_mech[mech] = dict(spearman=float(r_), p=float(p_))
            NUM['auto_choices'] = dict(
                runs=len(ca), rule_counts=ca.rule.value_counts().to_dict(),
                T_counts={int(k): int(v) for k, v in ca['T'].value_counts().items()},
                lam_counts={float(k): int(v) for k, v in rid.lam.value_counts().items()},
                lam_geo_mean_by_dataset=geo.round(6).to_dict(),
                lam_vs_redundancy=dict(spearman=float(rho), p=float(pv)),
                lam_vs_redundancy_by_mechanism=by_mech,
                redundancy={d: float(red[d]) for d in order})
    sec = A.groupby('method').seconds.mean()
    NUM['seconds_per_fit_A'] = sec.round(6).to_dict()
    return lad


def figures_A(A, lad, order, width):
    mechs = bm.MECHANISMS
    # Figure 2: NRMSE against the masking rate, by mechanism
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ax, mech in zip(axes, mechs):
        sub = A[A.mechanism == mech].groupby(['method', 'rate']).nrmse_pre.mean().unstack()
        for m in FIG_METHODS:
            if m in sub.index:
                line(ax, sub.columns * 100, sub.loc[m].values, m)
        ax.set_title(mech, fontsize=8); ax.set_xlabel('masked cells (% of each column)')
        ax.set_xticks([10, 20, 30, 40])
    axes[0].set_ylabel('NRMSE (Eq. 6)')
    axes[-1].legend(fontsize=5.8, loc='center left', bbox_to_anchor=(1.02, .5))
    save(fig, 'figure3_nrmse_vs_rate.png')

    # Figure 3: dataset-level average rank, by mechanism
    rk = pd.read_csv(os.path.join(R, 'tableE_all_ranks.csv'))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharex=True)
    for ax, mech in zip(axes, mechs):
        s = rk[rk.mechanism == mech].sort_values('dataset_avg_rank')
        cols = [STYLE[m]['color'] if m in STYLE else '#a3a29b' for m in s.method]
        ax.barh(range(len(s)), s.dataset_avg_rank, height=.62, color=cols, edgecolor='white',
                linewidth=.8)
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s.method, fontsize=6.2)
        ax.invert_yaxis(); ax.grid(axis='y', visible=False); ax.set_title(mech, fontsize=8)
        for i, v in enumerate(s.dataset_avg_rank):
            ax.text(v + .15, i, f'{v:.2f}', va='center', fontsize=5.6, color=MUTED)
        ax.set_xlim(0, len(s) + 1.2)
        ax.set_xlabel('average rank over datasets')
    save(fig, 'figure4_average_rank.png')

    # Figure 7: the ladder's second step against width and against redundancy
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharey='row')
    for col, mech in enumerate(mechs):
        s = lad[lad.mechanism == mech]
        for row, (xcol, xlab) in enumerate([('p', 'numerical columns p'),
                                            ('redundancy', 'redundancy 1 − p_eff / p')]):
            ax = axes[row, col]
            ax.axhline(0, color=MUTED, lw=.6)
            ax.plot(s[xcol], s.M_to_R_pct, 'o', color=PALETTE[0], markersize=4.5,
                    markeredgecolor='white', markeredgewidth=.6)
            rho = spearmanr(s[xcol], s.M_to_R_pct)[0]
            ax.set_title(f'{mech}: Spearman ρ = {rho:.2f}', fontsize=7.5)
            ax.set_xlabel(xlab, fontsize=7)
            place_labels(ax, s[xcol].values, s.M_to_R_pct.values, list(s.dataset))
        axes[0, 0].set_ylabel('M → R improvement (%)'); axes[1, 0].set_ylabel('M → R improvement (%)')
    save(fig, 'figure2_ladder_vs_redundancy.png')


# ─── Protocol B ──────────────────────────────────────────────────────────────
def protocol_B(B, A, order):
    B = B[B.method.isin(bm.METHODS)].copy()
    MET = present(bm.METHODS, B)
    B.to_csv(os.path.join(R, 'protocolB_raw.csv'), index=False)
    t = B.pivot_table(index='dataset', columns='method', values='accuracy',
                      aggfunc='mean')[MET].reindex(order)
    t.loc['MEAN'] = t.mean()
    t.round(6).to_csv(os.path.join(R, 'tableB1_accuracy_by_dataset.csv'))
    avg, pair, fr, means = dataset_level(B, 'accuracy', MET, bm.PROPOSED,
                                         lower_better=False)
    keys = ['dataset', 'repeat', 'classifier']
    wide = B.pivot_table(index=keys, columns='method', values='accuracy')[MET] \
        .dropna().round(TIE_DP)
    brk = pd.Series(np.apply_along_axis(lambda r: rankdata(-r), 1, wide.values).mean(0),
                    index=MET)
    pd.DataFrame(dict(dataset_avg_rank=avg, block_avg_rank=brk)).round(3) \
        .to_csv(os.path.join(R, 'tableB3_ranks_accuracy.csv'))
    pair.round(5).to_csv(os.path.join(R, 'tableB4_pairwise_accuracy.csv'), index=False)
    NUM['friedman_accuracy'] = {k: float(v) for k, v in fr.items()}
    NUM['ranks_accuracy'] = dict(dataset=avg.round(3).to_dict(), block=brk.round(3).to_dict())
    NUM['pairwise_accuracy'] = pair.round(5).to_dict('records')
    # how far apart the methods land under each protocol (relative spreads,
    # because accuracy and NRMSE have no common unit)
    acc = B.groupby(['dataset', 'method']).accuracy.mean().unstack()[MET]
    nr = A[A.mechanism == 'MCAR'].groupby(['dataset', 'method']).nrmse_pre.mean() \
        .unstack()[MET]
    rows = []
    for d in order:
        if d not in acc.index or d not in nr.index:
            continue
        a, n = acc.loc[d], nr.loc[d]
        rows.append(dict(dataset=d, accuracy_spread=a.max() - a.min(),
                         accuracy_spread_rel_pct=100 * (a.max() - a.min()) / a.mean(),
                         nrmse_spread=n.max() - n.min(),
                         nrmse_spread_rel_pct=100 * (n.max() - n.min()) / n.mean()))
    sp = pd.DataFrame(rows).set_index('dataset')
    sp.loc['MEAN'] = sp.mean()
    sp.round(6).to_csv(os.path.join(R, 'tableB2_protocol_discrimination.csv'))
    NUM['spread'] = sp.round(6).to_dict('index')
    NUM['rank_range_accuracy'] = [float(brk.min()), float(brk.max())]


# ─── Protocol C ──────────────────────────────────────────────────────────────
def protocol_C(C):
    C = C.copy(); C.to_csv(os.path.join(R, 'protocolC_scaling.csv'), index=False)
    meths = [m for m in bm.METHODS + bm.SENSITIVITY_METHODS if m in C.method.unique()]
    t = C.pivot_table(index=['n', 'p'], columns='method', values='seconds')[meths]
    t.round(3).to_csv(os.path.join(R, 'tableC1_runtime.csv'))
    ex = []
    for m in meths:
        s = C[C.method == m]
        X = np.column_stack([np.ones(len(s)), np.log(s.n), np.log(s.p)])
        b = np.linalg.lstsq(X, np.log(s.seconds), rcond=None)[0]
        ex.append(dict(method=m, exp_n=b[1], exp_p=b[2]))
    ex = pd.DataFrame(ex); ex.round(3).to_csv(os.path.join(R, 'tableC2_scaling_exponents.csv'),
                                              index=False)
    NUM['scaling_exponents'] = ex.round(2).set_index('method').to_dict('index')
    big = t.loc[(max(bm.GRID_N), max(bm.GRID_P))]
    NUM['runtime_largest'] = big.round(3).to_dict()
    r = t['CW-CZI (R)']
    NUM['runtime_ratio_vs_R'] = {m: dict(largest=float(big[m] / big['CW-CZI (R)']),
                                         geomean=float(np.exp(np.mean(np.log(t[m] / r)))))
                                 for m in meths}
    NUM['nrmse_synthetic'] = C.pivot_table(index=['n', 'p'], columns='method',
                                           values='nrmse_pre')[meths].round(6) \
        .reset_index().to_dict('records')
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    bp, bn = max(bm.GRID_P), max(bm.GRID_N)
    for ax, (fix, var, lab) in zip(axes, [('p', 'n', f'rows n (p = {bp})'),
                                          ('n', 'p', f'columns p (n = {bn})')]):
        sub = C[C[fix] == (bp if fix == 'p' else bn)].groupby(['method', var]).seconds.mean() \
            .unstack()
        for m in FIG_METHODS:
            if m in sub.index:
                line(ax, sub.columns, sub.loc[m].values, m)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(lab)
        ticks = bm.GRID_N if var == 'n' else bm.GRID_P
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel('seconds per fit (one thread)')
    axes[1].legend(fontsize=5.8, loc='center left', bbox_to_anchor=(1.02, .5))
    save(fig, 'figure5_runtime.png')


# ─── Protocol D ──────────────────────────────────────────────────────────────
def protocol_D(Dc, Ds):
    Dc.to_csv(os.path.join(R, 'protocolD_convergence.csv'), index=False)
    Ds.to_csv(os.path.join(R, 'protocolD_sensitivity.csv'), index=False)
    last = Dc.sort_values('iteration').groupby(['dataset', 'weighting']) \
        .agg(cycles=('iteration', 'max'), last_delta=('delta', 'last')).reset_index()
    last.to_csv(os.path.join(R, 'tableD1_cycles.csv'), index=False)
    NUM['convergence'] = {w: dict(reached=int((g.last_delta < 1e-10).sum()), of=len(g),
                                  cycles=g[g.last_delta < 1e-10].cycles.tolist(),
                                  max_last_delta=float(g.last_delta.max()),
                                  min_last_delta=float(g.last_delta.min()))
                          for w, g in last.groupby('weighting')}
    for mech in bm.MECHANISMS:
        s = Ds[Ds.mechanism == mech]
        if s.empty:
            continue
        d2 = s[s.rule == 'ridge'].pivot_table(index='lam', columns='max_iter', values='nrmse_pre')
        # the ridge rule at its default lambda, as in Figure 7
        d3 = s[(s.rule != 'ridge') | np.isclose(s.lam, 0.10)] \
            .pivot_table(index='rule', columns='max_iter', values='nrmse_pre')
        d2.round(6).to_csv(os.path.join(R, f'tableD2_lambda_x_cycles_{mech}.csv'))
        d3.round(6).to_csv(os.path.join(R, f'tableD3_rule_x_cycles_{mech}.csv'))
        NUM.setdefault('sensitivity', {})[mech] = dict(
            rule_x_cycles=d3.round(6).to_dict('index'),
            lambda_x_cycles=d2.round(6).to_dict('index'),
            best=float(d2.values.min()),
            best_cell=[float(d2.stack().idxmin()[0]), int(d2.stack().idxmin()[1])],
            default=float(d2.loc[0.10, 5]))
        rr = s[(s.rule == 'ridge') & np.isclose(s.lam, 0.10)] \
            .pivot_table(index='dataset', columns='max_iter', values='nrmse_pre')
        g530 = 100 * (rr[5] - rr[30]) / rr[5]
        best_T = rr.idxmin(axis=1)
        NUM['sensitivity'][mech].update(
            ridge_gain_5_to_30_by_dataset=g530.round(3).to_dict(),
            ridge_gain_5_to_30_pooled=float(100 * (rr[5].mean() - rr[30].mean()) / rr[5].mean()),
            ridge_best_cycles_by_dataset={k: int(v) for k, v in best_T.items()},
            marginal_1_vs_30=[float(d3.loc['marginal', 1]), float(d3.loc['marginal', 30])],
            uniform_1_vs_30=[float(d3.loc['uniform', 1]), float(d3.loc['uniform', 30])])
    # Figure 5: convergence
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    # a run stops once its change falls below the tolerance; its last (sub-tolerance)
    # change is carried forward so that the median does not jump when a dataset stops
    it = np.arange(1, int(Dc.iteration.max()) + 1)
    for w, m in [('uniform', 'Mean'), ('marginal', 'CW-CZI (M)'), ('ridge', 'CW-CZI (R)')]:
        W_ = Dc[Dc.weighting == w].pivot_table(index='iteration', columns='dataset', values='delta') \
            .reindex(it).ffill()
        med, lo, hi = W_.median(axis=1), W_.min(axis=1), W_.max(axis=1)
        c = STYLE[m]['color'] if w != 'uniform' else '#7a7974'
        ax.plot(med.index, med.values, color=c, lw=1.6, label=f'{w} rule')
        ax.fill_between(med.index, lo.reindex(med.index).values, hi.reindex(med.index).values,
                        color=c, alpha=.14, lw=0)
    ax.axhline(1e-10, color=MUTED, lw=.6, ls=(0, (2, 2)))
    ax.text(it.max(), 1.6e-10, 'stopping tolerance', ha='right', va='bottom', fontsize=6, color=MUTED)
    ax.set_yscale('log'); ax.set_xlabel('cycle'); ax.set_ylabel(r'normalised $\|\Delta X\|_F$')
    ax.legend(fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 0.13), frameon=False)
    save(fig, 'figure6_convergence.png')
    # Figure 6: cycle budget by rule (one panel per mechanism)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
    for ax, mech in zip(axes, bm.MECHANISMS):
        s = Ds[Ds.mechanism == mech]
        for w, m in [('uniform', 'Mean'), ('marginal', 'CW-CZI (M)'), ('ridge', 'CW-CZI (R)')]:
            g = s[(s.rule == w) & ((s.lam == 0.10) | (w != 'ridge'))].groupby('max_iter') \
                .nrmse_pre.mean()
            c = STYLE[m]['color'] if w != 'uniform' else '#7a7974'
            ax.plot(g.index, g.values, color=c, lw=1.6, marker='o', markersize=3.4,
                    label=f'{w} rule')
        ax.set_xscale('log'); ax.set_xticks(bm.D_CYCLES)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(mech, fontsize=8); ax.set_xlabel('cycle budget')
    axes[0].set_ylabel('NRMSE (Eq. 6)'); axes[-1].legend(fontsize=6.5)
    save(fig, 'figure7_cycle_by_rule.png')


# ─── Protocol E ──────────────────────────────────────────────────────────────
def protocol_E(Er, Es, A):
    if not Er.empty:
        Er = Er.copy()
        if not A.empty:
            mc = A[A.mechanism == 'MCAR'].groupby(['dataset', 'method']).nrmse_pre.mean().unstack()
            Er['R_minus_EM'] = Er.dataset.map(mc['CW-CZI (R)'] - mc['EM'])
            # the imputers able to fit non-linear relations: the forest, and nearest
            # neighbours (Iterative-T is left out because it may select a linear base estimator)
            NL = ['Iterative-RF', 'KNN-T']
            for m in NL:
                if m in mc:
                    Er[f'R_minus_{m}'] = Er.dataset.map(mc['CW-CZI (R)'] - mc[m])
            best_nl = mc[[m for m in NL if m in mc]].min(1)
            Er['R_minus_best_nonlinear_pct'] = Er.dataset.map(
                100 * (mc['CW-CZI (R)'] - best_nl) / mc['CW-CZI (R)'])
            for diag in ['sp_gap', 'eta_gap', 'eta_share']:
                ok = Er['R_minus_best_nonlinear_pct'].notna()
                if ok.sum() > 3:
                    rho, pv = spearmanr(Er.loc[ok, diag], Er.loc[ok, 'R_minus_best_nonlinear_pct'])
                    NUM.setdefault('diagnostic_vs_gap', {})[diag] = dict(rho=float(rho),
                                                                        p=float(pv), n=int(ok.sum()))
        Er.round(6).to_csv(os.path.join(R, 'tableF1_diagnostic_real.csv'), index=False)
        NUM['diagnostic_real'] = Er.round(6).to_dict('records')
    if Es.empty:
        return
    Es.to_csv(os.path.join(R, 'protocolE_synthetic.csv'), index=False)
    t = Es.pivot_table(index=['kind', 'level'], columns='method', values='nrmse_pre')
    dg = Es.groupby(['kind', 'level'])[['sp_gap', 'eta_gap', 'sp_share', 'eta_share']].mean()
    out = t.join(dg)
    if 'nrmse_odd' in Es.columns:
        for part in ['odd', 'even']:
            tp = Es.pivot_table(index=['kind', 'level'], columns='method', values=f'nrmse_{part}')
            out = out.join(tp.add_suffix(f' [{part}]'))
    out.round(6).to_csv(os.path.join(R, 'tableF2_diagnostic_synthetic.csv'))
    NUM['diagnostic_synthetic'] = out.round(6).reset_index().to_dict('records')
    # the best method at each strength, overall and on the transformed columns
    NUM['synthetic_best'] = {f'{k}|{a:g}': t.loc[(k, a)].idxmin() for k, a in t.index}
    if 'nrmse_odd' in Es.columns:
        to = Es[Es.kind == 'non-monotone'].pivot_table(index='level', columns='method', values='nrmse_odd')
        te = Es[Es.kind == 'non-monotone'].pivot_table(index='level', columns='method', values='nrmse_even')
        NUM['synthetic_nonmonotone_parts'] = dict(odd=to.round(6).to_dict('index'),
                                                  even=te.round(6).to_dict('index'))
    # Figure 8: two kinds x (error, diagnostic) — one measure per axis
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4))
    for col, kind in enumerate(['monotone', 'non-monotone']):
        if kind == 'non-monotone' and 'nrmse_odd' in Es.columns:
            s = Es[Es.kind == kind].pivot_table(index='level', columns='method', values='nrmse_odd')
            title = 'non-monotone: transformed columns'
        else:
            s = t.loc[kind]; title = f'{kind} non-linearity'
        ax = axes[0, col]
        for m in [m for m in FIG_METHODS if m in s.columns]:
            line(ax, s.index, s[m].values, m)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel('NRMSE (Eq. 6)' if col == 0 else '')
        ax = axes[1, col]; d = dg.loc[kind]
        ax.plot(d.index, d.sp_gap, color=INK, marker='o', markersize=3.4, lw=1.4,
                label='rank–linear gap (Eq. 7)')
        ax.plot(d.index, d.eta_gap, color=MUTED, marker='s', markersize=3.4, lw=1.4,
                dashes=(5, 2), label='non-linear share (Eq. 8)')
        ax.axhline(0, color=MUTED, lw=.5)
        ax.set_xlabel('strength of non-linearity a')
        ax.set_ylabel('diagnostic (mean over pairs)' if col == 0 else '')
    axes[0, 1].legend(fontsize=5.8, loc='center left', bbox_to_anchor=(1.02, .5))
    axes[1, 1].legend(fontsize=5.8, loc='center left', bbox_to_anchor=(1.02, .5))
    save(fig, 'figure8_nonlinearity.png')


# ─── completeness ────────────────────────────────────────────────────────────
def completeness(A, B):
    lines = []
    want_A = len(bm.MECHANISMS) * len(bm.MASK_RATES) * bm.N_REPEATS
    for name, df, want in [('A', A, want_A), ('B', B, bm.N_REPEATS * 8)]:
        if df.empty:
            lines.append(f'Protocol {name}: no records'); continue
        cnt = df[df.method.isin(bm.METHODS)].groupby(['dataset', 'method']).size().unstack()
        cnt = cnt.reindex(columns=bm.METHODS).fillna(0).astype(int)
        miss = (cnt < want)
        lines.append(f'Protocol {name}: {int((cnt.values >= want).sum())} of {cnt.size} '
                     f'(dataset, method) cells complete ({want} records each)')
        for d, row in miss.iterrows():
            for m in row.index[row.values]:
                lines.append(f'   missing  {d:11s} {m:14s} {cnt.loc[d, m]}/{want}')
    txt = '\n'.join(lines)
    open(os.path.join(R, 'completeness.txt'), 'w', encoding='utf-8').write(txt + '\n')
    print(txt)
    return all('missing' not in l for l in lines)


def main():
    A, B = read('A'), read('B')
    C = read('C'); Dc = read('D', 'convergence'); Ds = read('D', 'sensitivity')
    Er = read('E', 'diagnostic'); Es = read('E', 'synthetic')
    rows, tests = [], []
    for d in bm.DATASETS:
        r, t = bm.describe(bm.load(d)); rows.append(r); tests.append(t)
    t1 = pd.DataFrame(rows).sort_values(['Numeric', 'Dataset'])
    t1.to_csv(os.path.join(R, 'table1_datasets.csv'), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(R, 'table2_missingness_structure.csv'), index=False)
    complete = completeness(A, B)
    order, width = order_datasets(A if not A.empty else pd.DataFrame(dict(dataset=list(bm.DATASETS))))
    NUM['datasets_in_order_of_width'] = order; NUM['width'] = width
    if not A.empty:
        lad = protocol_A(A, order, width)
        figures_A(A[A.method.isin(bm.METHODS)], lad, order, width)
        sens = read('A'); sens = sens[sens.method.isin(bm.SENSITIVITY_METHODS + ['Iterative-RF',
                                                                                 'CW-CZI (R)'])]
        if (sens.method == 'Iterative-RF100').any():
            k = ['dataset', 'mechanism', 'rate', 'repeat']
            w = sens.pivot_table(index=k, columns='method', values='nrmse_pre') \
                .dropna(subset=['Iterative-RF', 'Iterative-RF100', 'CW-CZI (R)'])
            w = w[['CW-CZI (R)', 'Iterative-RF', 'Iterative-RF100']]
            NUM['rf100'] = dict(blocks=len(w), rf30=w['Iterative-RF'].mean(),
                                rf100=w['Iterative-RF100'].mean(), R=w['CW-CZI (R)'].mean(),
                                R_better_than_rf100_pct=100 * float(
                                    (w['CW-CZI (R)'] < w['Iterative-RF100']).mean()),
                                rf100_better_than_rf30_pct=100 * float(
                                    (w['Iterative-RF100'] < w['Iterative-RF']).mean()))
            by = {}
            for m in bm.MECHANISMS:
                if m not in w.index.get_level_values('mechanism'):
                    continue
                wm = w.xs(m, level='mechanism')
                by[m] = dict(blocks=len(wm), rf30=wm['Iterative-RF'].mean(), rf100=wm['Iterative-RF100'].mean(),
                             R=wm['CW-CZI (R)'].mean(),
                             change_pct=100 * (wm['Iterative-RF100'].mean() / wm['Iterative-RF'].mean() - 1),
                             R_beats_rf30=int((wm['CW-CZI (R)'] < wm['Iterative-RF']).sum()),
                             R_beats_rf100=int((wm['CW-CZI (R)'] < wm['Iterative-RF100']).sum()),
                             rf100_beats_rf30=int((wm['Iterative-RF100'] < wm['Iterative-RF']).sum()))
            NUM['rf100']['by_mechanism'] = by
            w.round(5).to_csv(os.path.join(R, 'tableA7_rf100_sensitivity.csv'))
        if (read('A').method == 'CW-CZI (R30)').any():
            NUM['r30'] = sensitivity_r30(read('A'))
    if not B.empty and not A.empty:
        protocol_B(B, A[A.method.isin(bm.METHODS)], order)
    if not C.empty:
        protocol_C(C)
    if not Dc.empty and not Ds.empty:
        protocol_D(Dc, Ds)
    protocol_E(Er, Es, A[A.method.isin(bm.METHODS)] if not A.empty else A)
    NUM['complete'] = bool(complete)
    with open(os.path.join(R, 'numbers.json'), 'w', encoding='utf-8') as f:
        json.dump(NUM, f, indent=1, default=float)
    print('tables, figures and numbers.json written to', R)


if __name__ == '__main__':
    main()
