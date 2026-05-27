# Tools

The `plotting/` directory contains selected ROOT macros and Python scripts used to inspect qT distributions, OS/SS ratios, momentum-fraction cut dependence, and comparisons to TMD-inspired theory curves.

Some scripts assume local ROOT files or theory-curve text files that are not committed to GitHub. They are included to support regeneration of the thesis plots from externally stored analysis outputs.

The main public plotting entry point is:

```bash
make plots OUT=output.root PLOT_TAG=local
```

For thesis-scale output:

```bash
make plots OUT=output_100M.root PLOT_TAG=100M
```

To regenerate the thesis-style overlay against the included TMD-theory curve:

```bash
make theory-overlay OUT=output.root PLOT_TAG=local
```

The target calls `tools/plotting/figure20_peakmatch_overlay.C`, which follows the thesis convention of matching the maximum PYTHIA OS bin and applying the same scale factor to the SS theory curve.
