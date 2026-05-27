# Running the Analysis

The repository is designed around the root-level `Makefile`. VS Code, Vim, Xcode, or any other editor can be used for editing, but the analysis is built and run from a normal terminal.

The standard workflow is:

```bash
make
make run EVENTS=10000 SEED=12345 OUT=output.root PROGRESS=progress.txt
make open OUT=output.root
```

## Local Configuration

The Makefile first tries to use standard HEP configuration commands:

```bash
root-config
fastjet-config
pythia8-config
```

If those commands are already on `PATH` and point to the intended installations, no additional configuration is needed.

If a command is missing, or if `pythia8-config` points to the wrong PYTHIA installation, copy the example local configuration:

```bash
cp config/local.mk.example local.mk
```

Then edit `local.mk` for your machine:

```make
ROOT_CONFIG = root-config
FASTJET_CONFIG = /path/to/fastjet/bin/fastjet-config
PYTHIA8_CONFIG = /path/to/pythia8315/bin/pythia8-config
```

`local.mk` is ignored by git and stores machine-specific dependency paths outside the tracked repository.

## Compile

```bash
make
```

This builds:

```text
./tmd-pythia
```

Inspect the active dependency paths with:

```bash
make print-config
```

On macOS, if compilation fails with `fatal error: 'iostream' file not found`, repair the Apple Command Line Tools:

```bash
xcode-select --install
```

## Run PYTHIA

Use `make run` for normal work:

```bash
make run EVENTS=10000 SEED=12345 OUT=output.root PROGRESS=progress.txt
```

The Makefile sets the PYTHIA and FastJet runtime library paths only for this command. This avoids conflicts with other local HEP installations.

The executable accepts the same values directly:

```bash
./tmd-pythia 10000 12345 output.root progress.txt
```

For a larger local run:

```bash
make run EVENTS=1000000 SEED=12345 OUT=output_1M.root PROGRESS=progress_1M.txt
```

Thesis-scale runs used up to 100M events:

```bash
make run EVENTS=100000000 SEED=12345 OUT=output_100M.root PROGRESS=progress_100M.txt
```

Large ROOT outputs should be stored externally rather than committed to GitHub.

## Open ROOT Output

```bash
make open OUT=output.root
```

or directly:

```bash
root -l -e 'TFile::Open("output.root"); new TBrowser();'
```

Useful objects to inspect include:

- `tPionPairs`
- `h_qT_highest_OS_pion_cut0`, `h_qT_highest_SS_pion_cut0`
- `h_qT_highest_OS_pion_cut20`, `h_qT_highest_SS_pion_cut20`
- `h_qT_highest_OS_pion_cut40`, `h_qT_highest_SS_pion_cut40`
- `h_qT_highest_OS_pion_cut60`, `h_qT_highest_SS_pion_cut60`
- `c_qT_OSSS_4cuts_pion_counts`
- `c_qT_OSSS_4cuts_pion_norm`

## Regenerate Selected Plots

After producing a ROOT file:

```bash
make plots OUT=output.root PLOT_TAG=local
```

For a thesis-scale file:

```bash
make plots OUT=output_100M.root PLOT_TAG=100M
```

Generated PDFs, PNGs, and ROOT files are ignored by git.

## Regenerate the TMD-Theory Overlay

The compact theory curves used for the overlay are committed under:

```text
data/theory/
```

The overlay macro uses the thesis peak-matching convention: in each pion momentum-fraction cut panel, the TMD OS curve is scaled to the maximum PYTHIA OS bin, and the same scale factor is applied to the TMD SS curve.

For a local output file:

```bash
make theory-overlay OUT=output.root PLOT_TAG=local
make open OUT=figure20_peakmatch_overlay_local.root
```

For a larger reproduction run:

```bash
make theory-overlay OUT=output_1M.root PLOT_TAG=1M_peakmatch
make open OUT=figure20_peakmatch_overlay_1M_peakmatch.root
```

The target calls:

```text
tools/plotting/figure20_peakmatch_overlay.C
```

and writes:

```text
figure20_peakmatch_overlay_<tag>.root
figure20_peakmatch_overlay_<tag>.pdf
figure20_peakmatch_overlay_<tag>.png
figure20_peakmatch_overlay_chi2_<tag>.txt
```

## Build Command Wrapped by the Makefile

The Makefile compiles with the same command structure used during the thesis work:

```bash
g++ -O2 -std=c++17 \
  $(pythia8-config --cxxflags) \
  $(root-config --cflags) \
  $(fastjet-config --cxxflags) \
  src/pythia/Pythia1.cc -o tmd-pythia \
  $(pythia8-config --libs) \
  $(root-config --libs) \
  $(fastjet-config --libs)
```
