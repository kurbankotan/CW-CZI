# CW-CZI — Correlation-Weighted Cyclical Z-Score Imputation

CW-CZI is a model-free missing-data imputer for numerical tabular data. It fills
each missing cell from a correlation-weighted combination of the standardised
values observed in the same row, refreshing the column moments and the correlation
matrix once per cycle, and never fits a predictive model. This repository holds the
estimator, the benchmark behind the accompanying article, its raw output, and the
scripts that turn that output into every table and figure of the article.

This is **version 7**, the revised version accepted for publication in the *Düzce
University Journal of Science and Technology* ([how to cite it](#citation)). What changed since the first submission, and
why, is listed under [Changes in the revision](#changes-in-the-revision); the code
and results of the first submission are kept unchanged in [`legacy/`](legacy/).

---

## The estimator

Standardise every column on its observed cells, fill each gap with the column mean,
then repeat until the change falls below a tolerance or the cycle budget runs out:

1. Re-estimate μ, σ and the correlation matrix **R** from the matrix completed so far.
2. Group the rows by missingness pattern. For each distinct pattern (O = observed
   columns, N = missing columns), solve once and write the result to every row that
   carries it.
3. Map the standardised estimate back onto each column's own scale:
   `x̂_ij = μ_j + σ_j · ẑ_ij`.
4. Measure the change `Δ = ‖X⁽ᵗ⁾ − X⁽ᵗ⁻¹⁾‖_F / √m` and go back to step 1.

| Rule | Estimate for the missing cells of a row | Notes |
|---|---|---|
| `uniform` | mean of the row's observed z-scores | ablation floor |
| `marginal` | `ẑ_j = Σ r_jk z_k / Σ \|r_jk\|` | double-counts redundant columns |
| `ridge` | `ẑ_N = R_NO (R_OO + λI)⁻¹ z_O` | **default**; λ = 0.1 (the article compares five cycles and recommends thirty) |
| `auto` | rule, λ and cycle count chosen on held-out cells | `CWCZAuto` |

with `z_O ∈ ℝ^|O|`, `R_OO ∈ ℝ^(|O|×|O|)`, `R_NO ∈ ℝ^(|N|×|O|)` and `ẑ_N ∈ ℝ^|N|`.

```python
from cwczi import CWCZImputer, linearity_diagnostic

print(linearity_diagnostic(X_train))            # screen for non-linear dependence first
imp = CWCZImputer(weighting='ridge', lam=0.1, max_iter=30)   # see "Cycle budget" below
X_train_filled = imp.fit_transform(X_train)     # 2-D float array, NaN for missing
X_test_filled  = imp.transform(X_test)          # stored moments: no leakage
```

`linearity_diagnostic` returns two pairwise statistics computed on the incomplete
matrix: the gap between |Spearman| and |Pearson| correlation (monotone curvature)
and the share of variance the conditional mean explains beyond the best straight
line (any curvature, including U-shapes). CW-CZI reads the data through Pearson
correlations only; when these statistics are well above zero, a model-based imputer
may be worth its cost on the flagged columns. They are a screen, not a predictor: across
the article's eleven datasets they did not predict how much a non-linear imputer gains
(Sections III.H and III.I of the article).

---

## Results

Eleven datasets, three missingness mechanisms, four rates and five repetitions: 660 blocks per method, each dataset counted once in the comparison (Friedman test over datasets, Holm-corrected Wilcoxon tests, effect sizes first).

| Average rank over the eleven datasets (untuned methods) | MCAR | MAR | MNAR |
|---|---|---|---|
| EM | 2.09 | 2.82 | 2.09 |
| **CW-CZI (R)** | **2.09** | **2.27** | **3.27** |
| Iterative-RF | 2.91 | 3.27 | 2.36 |
| Iterative-BR | 4.27 | 3.45 | 3.64 |
| KNN | 4.91 | 4.09 | 5.55 |
| CW-CZI (M) | 5.82 | 6.18 | 6.18 |
| SVD | 6.73 | 6.91 | 5.73 |
| Mean | 8.09 | 7.91 | 7.91 |
| CW-CZI (U) | 8.09 | 8.09 | 8.27 |

CW-CZI (R) against expectation-maximisation, datasets on which CW-CZI (R) is more accurate: MCAR 5/11, MAR 7/11, MNAR 1/11. With 30 cycles instead of 5 (CW-CZI (R30)): MCAR 4/11, MAR 9/11, MNAR 4/11; under MAR the error falls by 3.8% overall and by up to 35% (Libras).

Iterative-RF at 100 trees and 10 rounds instead of 30 and 5 (33 blocks at 30%): mean NRMSE change MCAR -1.4%, MAR +0.9%, MNAR -0.1%; datasets on which CW-CZI (R) is more accurate than the forest, 30 → 100 trees: MCAR 8 → 8, MAR 4 → 5, MNAR 2 → 2.

Seconds per fit at n = 2000, p = 80, 30% MCAR, one thread: CW-CZI (R) 1.79, CW-CZI (R30) 7.32, EM 16.38, Iterative-BR 6.91, Iterative-RF 1005.21, CW-CZI (A) 33.31.

The ladder under MCAR: correlation weighting 19.9% better than the unweighted average, the ridge inverse a further 25.0%; the second step follows the redundancy among the columns more closely than their number.

Every number in the article is in `results/numbers.json`; the tables and figures are the files beside it, and `python check_results.py` re-derives all of them from `results/raw/`.

---

## Changes in the revision

Prompted by the two reviewers' reports, and by a re-examination of the first
submission's design:

| | First submission (v6) | Revision (v7) |
|---|---|---|
| Datasets | 3 (CKD, Heart, Mice) | 11; eight more from PMLB, widths 10–90 columns, four of them with 57 or more |
| Duplicate rows | kept | removed before anything else (240 in Wine, 30 in Libras, 1 in Ionosphere) |
| Masks | one ranking over the whole matrix | drawn column by column: every masked column loses the stated share |
| MAR | the conditioning cell could itself be masked | a quarter of the columns is never masked and conditions the rest |
| Error scale | moments of the cells left after masking | each column's standard deviation before masking (Eq. 6); the old scale is kept as a robustness check |
| Baselines | library defaults | defaults **and** KNN, SVD and chained equations tuned by the same internal validation as CW-CZI (A); the forest imputer also at 100 trees / 10 rounds on a subset |
| Names | MICE, MissForest | Iterative-BR, Iterative-RF (scikit-learn's IterativeImputer, not the original packages) |
| EM baseline | row by row, sample covariance | grouped by missingness pattern like CW-CZI, maximum-likelihood covariance, iteration count recorded |
| Statistics | Friedman and Wilcoxon over 180 nested blocks | per mechanism, over the eleven datasets; effect sizes first |
| Timing | Colab, threads uncontrolled | one machine, one fit at a time, one thread per library |
| Cycle budget | five cycles, tested under MCAR | tested under every mechanism; 30 cycles recommended, because under MAR five leave the bias of the mean fill in place |
| New | — | `linearity_diagnostic`, a screen for non-linear dependence (Protocol E) |

---

## Reproducing the article

```bash
pip install -r requirements-lock.txt            # Python 3.11–3.13
python run_benchmark.py A                       # reconstruction (660 blocks x 13 methods)
python run_benchmark.py B                       # downstream classification
python run_benchmark.py C --workers 1           # wall-clock scaling: run it alone
python run_benchmark.py D                       # CW-CZI convergence and sensitivity
python run_benchmark.py E                       # non-linearity screening
python run_benchmark.py A --methods Iterative-RF100 --rates 0.3 --reps 0   # forest, 100 trees
python run_benchmark.py A --methods "CW-CZI (R30)"                         # ridge rule, 30 cycles
python run_benchmark.py C --methods "CW-CZI (R30)" --workers 1
python analyze_results.py                       # every table, figure and numbers.json
python check_results.py                         # derived files == what the raw records give
pytest tests -q
```

Every block is appended to `results/raw/<protocol>/<dataset>__<method>.csv` as it
finishes; an interrupted run resumes where it stopped when started again with the
same command, and runs on different machines can share the work (`--methods`,
`--datasets`) and merge their `results/raw` folders. Each process is pinned to one
BLAS thread so that blocks running side by side do not compete.

The run reported in the article used two machines: the random-forest imputer of
Protocols A and B (and its 100-tree check) ran mostly on Google Colab, everything
else — Protocol C, the only one whose timings are compared, entirely — on a second
machine. 124 random-forest blocks were run on both machines and gave identical
errors (`results/raw/crosscheck_iterative_rf.json`). Both machines recorded their
environment in `results/raw/environment_*.json`.

---

## Files

| File | What it is |
|---|---|
| `cwczi.py` | the estimator: `CWCZImputer`, `CWCZAuto`, `linearity_diagnostic` |
| `benchmark.py` | datasets, masks, error metrics, baselines and tuned baselines, one function per protocol |
| `run_benchmark.py` | runs a protocol in parallel and resumably, block by block |
| `analyze_results.py` | turns `results/raw` into the article's tables, figures and `numbers.json` |
| `check_results.py` | re-derives every table and number from the raw records and compares |
| `results/` | the run reported in the article: raw records under `raw/`, derived files beside them |
| `datasets/` | the eleven datasets, byte for byte, with checksums — see `datasets/README.md` |
| `tests/` | the estimator's properties, the masks and the metric, the tuned baselines, the loader |
| `legacy/` | the first submission's code and results, unchanged |

---

## Citation

If you use CW-CZI or its benchmark, please cite the article:

> Kotan, K. (in press). Correlation-weighted cyclical Z-score imputation. *Düzce University
> Journal of Science and Technology*.

```bibtex
@article{kotan_cwczi,
  author  = {Kotan, Kurban},
  title   = {Correlation-Weighted Cyclical {Z}-Score Imputation},
  journal = {D{\"u}zce University Journal of Science and Technology},
  year    = {in press},
  note    = {Accepted 2 November 2026}
}
```

The article was accepted on 2 November 2026; its volume, issue, pages and DOI will be
added here when it is published. GitHub's **Cite this repository** button gives the same
reference, read from [`CITATION.cff`](CITATION.cff).

## Author

**Kurban Kotan** — Manisa Celal Bayar University, Faculty of Computer and Information
Sciences, Department of Artificial Intelligence and Machine Learning, Manisa, Türkiye.
ORCID [0000-0002-6660-0565](https://orcid.org/0000-0002-6660-0565) · kurban.kotan@cbu.edu.tr

Released under the MIT License — see `LICENSE`.
