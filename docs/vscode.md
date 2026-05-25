# VS Code Workflow

This repository includes VS Code tasks that wrap the root `Makefile`.

Open the repository:

```bash
cd /path/to/TMD-Analysis
code .
```

Edit `.vscode/settings.json` if your local dependency paths differ:

```json
{
  "tmdAnalysis.pythiaPath": "/path/to/pythia8315",
  "tmdAnalysis.fastJetPath": "/path/to/fastjet"
}
```

In VS Code, run:

```text
Terminal > Run Task... > TMD: Print Config
Terminal > Run Task... > TMD: Smoke Test (10k events)
Terminal > Run Task... > TMD: Open ROOT Browser
```

For a larger sample, choose:

```text
Terminal > Run Task... > TMD: Run Custom Event Sample
```

and enter the requested event count, seed, output file, and progress file.

The same actions are available from the integrated terminal:

```bash
make print-config
make test
make run EVENTS=100000 SEED=12345 OUT=output.root
make open OUT=output.root
make clean
```

Thesis-scale runs used much larger samples, up to 100M events. Generated ROOT files are intentionally ignored by git.
