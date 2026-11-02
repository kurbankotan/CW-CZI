# The first submission (v6)

These files produced the results of the manuscript as first submitted: three
datasets, cells masked by a single ranking over the whole matrix, errors scaled
by the moments left after masking, and every baseline at its library default.

They are kept unchanged so that every number of the first submission can still
be traced to the code and the run that produced it: `results_v6/` is that run,
and `check_results.py results_v6` still verifies it against its raw files.

The revised article is produced by `benchmark.py`, `run_benchmark.py` and
`analyze_results.py` in the repository root; see the main README for what
changed and why.
