# Data Policy

No large generated datasets are committed to this repository.

The repository contains source code, documentation, thesis material, presentations, plotting scripts, compact theory-curve files, and small result summaries. Large ROOT files and generated event samples are excluded.

For reproduction, generate fresh PYTHIA samples with the root `Makefile`, then store the resulting ROOT outputs outside the git tree.

The small text files under `theory/` are committed because they are needed to regenerate the public PYTHIA/TMD overlay plots without rerunning the separate theory calculation.
