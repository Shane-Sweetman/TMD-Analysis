# Running the Analysis

This project used explicit bash commands with local dependency paths. It did not mainly use Makefile or CMake.

Set the local dependency paths:

```bash
export PYTHIA=/Users/shanesweetman/Downloads/pythia/pythia8315
export FJ=/Users/shanesweetman/Downloads/fastjet
```

Run a small PYTHIA test:

```bash
chmod +x scripts/run_pythia_example.sh
EVENTS=10000 SEED=12345 scripts/run_pythia_example.sh
```

Open the output ROOT file:

```bash
chmod +x scripts/open_root_browser.sh
scripts/open_root_browser.sh output.root
```

The script compiles:

```bash
g++ -O2 -std=c++17 \
  -I"$PYTHIA/include" \
  $(root-config --cflags) \
  $("$FJ/bin/fastjet-config" --cxxflags) \
  src/pythia/Pythia1.cc -o TMD \
  -L"$PYTHIA/lib" -lpythia8 \
  $(root-config --libs) \
  $("$FJ/bin/fastjet-config" --libs) \
  -Wl,-rpath,"$PYTHIA/lib" -Wl,-rpath,"$FJ/lib"
```

and runs:

```bash
env DYLD_LIBRARY_PATH="$PYTHIA/lib:$FJ/lib" \
    DYLD_FALLBACK_LIBRARY_PATH="$PYTHIA/lib:$FJ/lib" \
    ./TMD 10000 12345 output.root progress.txt
```

Thesis-scale runs used much larger event counts, up to 100M events. Those outputs are large ROOT files and should be stored externally rather than committed to GitHub.
