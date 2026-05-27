# TMD Theory Curves

This directory contains compact text files with independently generated TMD-theory curves used for the thesis-level comparison plots.

The main public overlay uses:

```text
epemCrossSection_z0p70.dat
```

The columns are:

```text
qT    OS    SS
```

where `OS` and `SS` denote the opposite-sign and same-sign theory curves before the peak-matching rescaling applied by the plotting macro. In each cut panel, the scale factor is fixed by matching the maximum PYTHIA OS bin; the same factor is then applied to the SS theory curve.

The files are committed so that a reader can run the PYTHIA analysis and reproduce the overlay without regenerating the theory curves.
