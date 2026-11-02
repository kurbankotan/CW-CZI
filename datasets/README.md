# Datasets

The eleven datasets used in the article, exactly as they enter the pipeline.
Every file is committed, so the results reproduce without a network connection,
and `tests/test_loading.py` checks each one against the SHA-256 below and against
the dimensions Table 1 of the article reports.

| Dataset | File | Original source | DOI | SHA-256 |
|---|---|---|---|---|
| CKD | `kidney_disease.csv` | UCI Chronic Kidney Disease | [10.24432/C5G020](https://doi.org/10.24432/C5G020) | `835c57ec…f5556` |
| Heart | `heart_cleveland.csv` | UCI Heart Disease, Cleveland original | [10.24432/C52P4X](https://doi.org/10.24432/C52P4X) | `0e709332…e239` |
| Mice | `Data_Cortex_Nuclear.csv` | UCI Mice Protein Expression | [10.24432/C50S3Z](https://doi.org/10.24432/C50S3Z) | `1d6722b0…f7ad` |
| Wine | `pmlb/wine_quality_red.tsv.gz` | UCI Wine Quality (red) | [10.24432/C56S3T](https://doi.org/10.24432/C56S3T) | `6402396f…6824` |
| Vehicle | `pmlb/vehicle.tsv.gz` | UCI Statlog (Vehicle Silhouettes) | [10.24432/C5HG6N](https://doi.org/10.24432/C5HG6N) | `dad03273…c4` |
| WDBC | `pmlb/breast_cancer_wisconsin_diagnostic.tsv.gz` | UCI Breast Cancer Wisconsin (Diagnostic) | [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) | `b3845ddf…a291` |
| Ionosphere | `pmlb/ionosphere.tsv.gz` | UCI Ionosphere | [10.24432/C5W01B](https://doi.org/10.24432/C5W01B) | `adb6fc87…0023` |
| Satimage | `pmlb/satimage.tsv.gz` | UCI Statlog (Landsat Satellite) | [10.24432/C55887](https://doi.org/10.24432/C55887) | `1ee8cf96…9f46` |
| Sonar | `pmlb/sonar.tsv.gz` | UCI Connectionist Bench (Sonar, Mines vs. Rocks) | [10.24432/C5T01Q](https://doi.org/10.24432/C5T01Q) | `64378cab…529b` |
| Optdigits | `pmlb/optdigits.tsv.gz` | UCI Optical Recognition of Handwritten Digits | [10.24432/C50P49](https://doi.org/10.24432/C50P49) | `b70b7ac6…33c3` |
| Libras | `pmlb/movement_libras.tsv.gz` | UCI Libras Movement | [10.24432/C5GC82](https://doi.org/10.24432/C5GC82) | `77c9c611…fc85` |

Full checksums are in `tests/test_loading.py`.

## Where the files come from

**CKD and Mice** are the UCI files as downloaded. **Heart** is the 303-row Cleveland
original as `ucimlrepo` returns it (`id=45`), with one change made before caching: the
five-level `num` target is collapsed to 0/1 (`num > 0`), the standard binary
formulation of the Cleveland set. Its `ca` and `thal` columns keep their 4 and 2
genuine missing values.

**The eight files under `pmlb/`** are the PMLB collection's copies of the UCI
datasets (Romano et al., 2022, *Bioinformatics* 38(3), 878–880,
https://doi.org/10.1093/bioinformatics/btab727), downloaded from the PMLB GitHub
repository (`datasets/<name>/<name>.tsv.gz`) and committed byte for byte: the
SHA-256 of each file equals the object id in PMLB's Git LFS pointer, so the copies
can be checked against the source. PMLB stores every dataset as a tab-separated
table whose class column is named `target`; the feature columns are the UCI ones.
The Breast Cancer Wisconsin file is PMLB's `breast_cancer_wisconsin_diagnostic`.

## What the loader does to them

`benchmark.load()` applies the same steps to every file, in this order:

1. Text that is really numbers is coerced to numeric (three CKD columns carry a
   leading tab), and identifier columns (`id`, `MouseID`) are dropped.
2. **Exact duplicate rows are removed**: 240 in Wine, 30 in Libras, one in
   Ionosphere, none elsewhere. A duplicate would let a masked cell be read off an
   identical copy of its own row, and would put a twin of a test row among the
   training rows.
3. The numerical block is every column that is numeric, observed in more than 30%
   of the rows and has more than two distinct values. The rest is the categorical
   block, which is never masked.
4. Rows with no numerical value at all and classes with fewer than ten rows are
   dropped; datasets larger than 600 rows are subsampled to 600 with seed 42, and
   classes left with fewer than ten rows are dropped again (this removes Wine's
   quality grades 3 and 8). A column the subsample leaves with two values or fewer
   moves to the categorical block (this happens in Optdigits).

The resulting dimensions are Table 1 of the article:

| Dataset | Rows before subsampling | n | Numeric | Categorical | Classes | Missing cells | Incomplete rows |
|---|---|---|---|---|---|---|---|
| Heart | 303 | 303 | 10 | 3 | 2 | 0.20% | 2.0% |
| Wine | 1359 | 586 | 11 | 0 | 4 | 0 | 0 |
| CKD | 400 | 400 | 14 | 10 | 2 | 13.89% | 49.2% |
| Vehicle | 846 | 600 | 18 | 0 | 4 | 0 | 0 |
| WDBC | 569 | 569 | 30 | 0 | 2 | 0 | 0 |
| Ionosphere | 350 | 350 | 32 | 2 | 2 | 0 | 0 |
| Satimage | 6435 | 600 | 36 | 0 | 6 | 0 | 0 |
| Optdigits | 5620 | 600 | 57 | 7 | 10 | 0 | 0 |
| Sonar | 208 | 208 | 60 | 0 | 2 | 0 | 0 |
| Mice | 1080 | 600 | 77 | 3 | 8 | 1.65% | 48.8% |
| Libras | 330 | 330 | 90 | 0 | 15 | 0 | 0 |

## Do not substitute the 1025-row `heart.csv`

The 1025-row `heart.csv` that circulates on data-sharing sites is the same study
re-published with three changes that make it unusable here: only 302 of its rows
are unique (an 80/20 split leaves 158 of 159 distinct test rows also present in
the training set), its missing values are recoded as ordinary categories
(`ca = 4`, `thal = 0`), and its categorical codes are shifted.

## Redistribution

All eleven are redistributed under the terms of the UCI repository, which permits
use for research without special permission. Please cite the original dataset
DOIs above, and PMLB for the eight files that come from it.
