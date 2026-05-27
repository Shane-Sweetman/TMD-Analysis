# Results Policy

The thesis figures are included in the thesis and presentation PDFs rather than duplicated here as standalone image files.

This directory keeps small text and CSV summaries from the thesis-level analysis. Large ROOT files and generated samples are excluded from GitHub.

The main omitted files include:

- `output_100M.root`
- per-chunk ROOT files
- merged scan ROOT files
- logs and temporary progress files

To regenerate a local analysis output, run:

```bash
make
make run EVENTS=10000 SEED=12345 OUT=output.root PROGRESS=progress.txt
make theory-overlay OUT=output.root PLOT_TAG=local
```

Thesis-scale numerical results used much larger event samples and should be stored externally if they need to be preserved exactly.
