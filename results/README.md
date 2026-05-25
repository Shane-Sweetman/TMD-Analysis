# Results Policy

The thesis figures are included in the thesis and presentation PDFs rather than duplicated here as standalone image files.

This directory keeps small text and CSV summaries from the thesis-level analysis. Large ROOT files and generated samples are excluded from GitHub.

The main omitted files include:

- `output_100M.root`
- per-chunk ROOT files
- merged scan ROOT files
- logs and temporary progress files

To regenerate a small test output, run:

```bash
EVENTS=10000 SEED=12345 scripts/run_pythia_example.sh
```

Thesis-scale numerical results used much larger event samples and should be stored externally if they need to be preserved exactly.
