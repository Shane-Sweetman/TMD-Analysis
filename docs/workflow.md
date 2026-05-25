# Analysis Workflow

The main workflow is:

1. Generate e+e- events at the Z pole with PYTHIA 8.
2. Select visible final-state particles.
3. Reconstruct the thrust axis.
4. Reconstruct jets using FastJet anti-kT with `R = 0.4`.
5. Apply a dijet topology selection.
6. Find charged pions in opposite jets.
7. Select the pion pair according to the implemented highest-momentum logic.
8. Compute qT relative to the thrust axis.
9. Fill opposite-sign and same-sign qT histograms.
10. Repeat for pion momentum-fraction cuts.
11. Compare the OS and SS spectra.
12. Compare to independent TMD-based theory curves where applicable.

The final analysis source is [../src/pythia/Pythia1.cc](../src/pythia/Pythia1.cc).

The repository intentionally preserves the final analysis and selected plotting scripts, not every exploratory prototype created during the project.
