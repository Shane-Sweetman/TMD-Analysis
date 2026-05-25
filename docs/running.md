# Running the Analysis

The original thesis workflow used explicit local compile/run commands. This public version provides a small root-level `Makefile` so the analysis can be built and smoke-tested with simple terminal commands.

Set the local dependency paths:

```bash
export PYTHIA=/Users/shanesweetman/Downloads/pythia/pythia8315
export FJ=/Users/shanesweetman/Downloads/fastjet
```

Run a small PYTHIA test:

```bash
make test
```

Open the output ROOT file:

```bash
make open
```

Run a larger sample by overriding variables:

```bash
make run EVENTS=100000 SEED=12345 OUT=output.root
```

Use non-default dependency paths:

```bash
make test PYTHIA=/path/to/pythia8315 FJ=/path/to/fastjet
```

Inspect the active configuration:

```bash
make print-config
```

The Makefile compiles with the same command structure used during the thesis work:

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

and runs the executable as:

```bash
env DYLD_LIBRARY_PATH="$PYTHIA/lib:$FJ/lib" \
    DYLD_FALLBACK_LIBRARY_PATH="$PYTHIA/lib:$FJ/lib" \
    ./TMD 10000 12345 output.root progress.txt
```

Thesis-scale runs used much larger event counts, up to 100M events. Those outputs are large ROOT files and should be stored externally rather than committed to GitHub.
