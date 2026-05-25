![Wicklow Mountains](doc/assets/wicklow_mountains.jpg)

# TMD-Analysis: Charged-Pion Pair Transverse Momentum in e+e- Collisions

This repository contains the analysis code, documentation, thesis, and presentation material for Shane Sweetman's BSc Theoretical Physics thesis project at the UCD School of Physics:

**A TMD-oriented Analysis of Charged-Pion Pairs in e+e- Collisions**

The project studies charged-pion pairs in opposite jets in simulated e+e- collisions at the Z pole. The main observable is the transverse momentum of a selected pion pair, qT, measured relative to the event thrust axis. Opposite-sign and same-sign pion pairs are compared as a simulation-level probe of charge-dependent fragmentation structure.

This is a simulation and theory-facing first step toward a future archived LEP data analysis. It is not a completed extraction from real collision data.

## Overview

The analysis uses PYTHIA 8 to generate e+e- -> Z/gamma* -> q qbar events at sqrt(s) = 91.2 GeV. Visible final-state particles are clustered with FastJet using anti-kT jets with R = 0.4. Events are selected in a dijet topology, charged pions are identified in opposite jets, and the selected pion pair is classified as opposite-sign or same-sign.

The public repository is intentionally compact: it keeps the final PYTHIA analysis, selected plotting and summary scripts, thesis material, and presentation provenance. Large generated ROOT files, temporary logs, binaries, and exploratory prototypes are not committed.

## Physics Motivation

Transverse-momentum-dependent fragmentation functions describe how partonic transverse motion and hadronization structure appear in final-state hadron distributions. In e+e- annihilation, the absence of hadronic initial-state structure makes the channel a clean environment for studying fragmentation.

This project asks whether a charged-pion-pair observable in opposite jets can show a robust opposite-sign versus same-sign separation in simulation, and whether that separation behaves sensibly as pion momentum-fraction cuts are tightened.

## Main Observable

For a selected charged-pion pair in opposite jets, define

```text
q = p_pi1 + p_pi2
qT = |q - (q . n_thrust) n_thrust|
```

where `n_thrust` is the reconstructed thrust axis. The analysis compares the qT spectra of opposite-sign and same-sign pion pairs. Harder pion momentum-fraction cuts increase the OS/SS separation in the thesis-level PYTHIA study.

See [docs/observable.md](docs/observable.md) for the full definition.

## Repository Structure

```text
TMD-Analysis/
├── README.md
├── doc/assets/                  # README banner image
├── docs/                        # Physics and workflow notes
├── scripts/                     # Bash-first compile/run helpers
├── src/pythia/                  # Main PYTHIA analysis source
├── tools/plotting/              # ROOT/Python plotting and summary scripts
├── results/                     # Small result summaries and regeneration policy
├── data/                        # Data policy; large files are external
├── thesis/                      # Final thesis PDF
└── presentations/               # Final and week-by-week presentation material
```

The presentation source-code map is kept under `presentations/_source-code-map/` so it is available for provenance without dominating the repository front page.

## Dependencies

The original thesis workflow used explicit local compile/run commands. This public version provides a small root-level `Makefile` so the code can be tested with a standard command-line interface. It is a convenience wrapper around the ROOT/PYTHIA/FastJet command, not a CMake project.

Required for the PYTHIA analysis:

- C++17 compiler, tested with `g++`
- ROOT, with `root-config` on `PATH`
- PYTHIA 8
- FastJet

Typical local paths used during the thesis work were:

```bash
export PYTHIA=/Users/shanesweetman/Downloads/pythia/pythia8315
export FJ=/Users/shanesweetman/Downloads/fastjet
```

## Quick Start

From the repository root:

```bash
make test
make open
```

The default run is intentionally small. Thesis-scale runs used much larger samples, up to 100M events, and produced large ROOT files that are not suitable for GitHub.

## Running the PYTHIA Analysis

The main source is [src/pythia/Pythia1.cc](src/pythia/Pythia1.cc). It writes a ROOT output file containing:

- `tPionPairs`, a reduced selected-pair tree
- OS and SS qT histograms for pion momentum-fraction cuts
- ROOT canvases for the cut-dependent OS/SS comparison

Run with local dependency paths:

```bash
PYTHIA=/path/to/pythia8315 \
FJ=/path/to/fastjet \
EVENTS=10000 \
SEED=12345 \
make run
```

Useful targets:

```bash
make help
make print-config
make build
make test
make run EVENTS=100000 SEED=12345 OUT=output.root
make clean
```

For more practical commands, see [docs/running.md](docs/running.md).

## Opening ROOT Output Files

```bash
make open OUT=output.root
```

## Results Summary

The thesis-level PYTHIA study found a clear qualitative trend: tightening pion momentum-fraction cuts increased the separation between opposite-sign and same-sign selected pion-pair qT spectra.

Independent TMD-based theory curves were used as a comparison to the PYTHIA-level observable. The comparison is useful as a physics guide, but this repository should not be read as a full TMD extraction or as a detector-corrected LEP measurement.

The main figures are included in the thesis and presentation PDFs. Small numerical summaries are kept under [results/summaries](results/summaries); large ROOT outputs are excluded.

## Future Archived-Data Direction

A natural next step is to repeat the analysis using archived LEP data from ALEPH or DELPHI. That would require detector-level corrections, pion identification, unfolding, acceptance studies, and systematic uncertainty control. See [docs/archive_data_direction.md](docs/archive_data_direction.md).

## Thesis Reference

The final thesis PDF is included at:

[thesis/Shane_Sweetman_BSc_Thesis_TMD_Analysis.pdf](thesis/Shane_Sweetman_BSc_Thesis_TMD_Analysis.pdf)

## Contact

Shane Sweetman  
UCD School of Physics
