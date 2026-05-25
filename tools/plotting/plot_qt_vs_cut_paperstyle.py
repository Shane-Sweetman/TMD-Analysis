#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

DEEP_RED = "#b22222"
DEEP_BLUE = "#1f4fb2"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 18,
    "axes.linewidth": 1.2,
})

def load_table(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            rows.append({
                "cut": float(p[0]),
                "nOS": float(p[1]),
                "meanOS": float(p[2]),
                "semOS": float(p[3]),
                "peakOS": float(p[4]),
                "peakCountOS": float(p[5]),
                "q95OS": float(p[6]),
                "nSS": float(p[7]),
                "meanSS": float(p[8]),
                "semSS": float(p[9]),
                "peakSS": float(p[10]),
                "peakCountSS": float(p[11]),
                "q95SS": float(p[12]),
            })
    if not rows:
        raise ValueError("No rows found in input file.")
    return pd.DataFrame(rows)

def style_axes(ax):
    ax.tick_params(direction="in", top=True, right=True, length=8, width=1.1, labelsize=13)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=4, width=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

def make_mean_plot(df: pd.DataFrame, outfile: Path):
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=240)
    ax.errorbar(df["cut"], df["meanOS"], yerr=df["semOS"], marker="o", lw=2.3, ms=5.8,
                capsize=3.5, color=DEEP_RED, label="OS")
    ax.errorbar(df["cut"], df["meanSS"], yerr=df["semSS"], marker="o", lw=2.3, ms=5.8,
                capsize=3.5, color=DEEP_BLUE, label="SS")
    ax.set_xlabel("Cut on pion momentum fraction (%)")
    ax.set_ylabel(r"$\langle q_T \rangle\ [\mathrm{GeV}]$")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.savefig(outfile, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def make_peak_plot(df: pd.DataFrame, outfile: Path):
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=240)
    # markers only, no connecting lines
    ax.plot(df["cut"], df["peakOS"], marker="o", linestyle="None", ms=6.5,
            color=DEEP_RED, label="OS")
    ax.plot(df["cut"], df["peakSS"], marker="o", linestyle="None", ms=6.5,
            color=DEEP_BLUE, label="SS")
    ax.set_xlabel("Cut on pion momentum fraction (%)")
    ax.set_ylabel(r"$q_T^{\mathrm{peak}}\ [\mathrm{GeV}]$")
    ax.legend(frameon=False, loc="best")
    style_axes(ax)
    fig.savefig(outfile, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def make_q95_plot(df: pd.DataFrame, outfile: Path):
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=240)
    ax.plot(df["cut"], df["q95OS"], marker="o", lw=2.3, ms=5.8, color=DEEP_RED, label="OS")
    ax.plot(df["cut"], df["q95SS"], marker="o", lw=2.3, ms=5.8, color=DEEP_BLUE, label="SS")
    ax.set_xlabel("Cut on pion momentum fraction (%)")
    ax.set_ylabel(r"$q_{T,95\%}\ [\mathrm{GeV}]$")
    ax.legend(frameon=False, loc="best")
    style_axes(ax)
    fig.savefig(outfile, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--prefix", default="qt_vs_cut_paper")
    args = ap.parse_args()

    df = load_table(args.infile)
    base = Path(args.prefix)

    make_mean_plot(df, Path(f"{base}_mean.png"))
    make_peak_plot(df, Path(f"{base}_peak.png"))
    make_q95_plot(df, Path(f"{base}_q95.png"))

    print(f"Saved {base}_mean.png/.pdf")
    print(f"Saved {base}_peak.png/.pdf")
    print(f"Saved {base}_q95.png/.pdf")

if __name__ == "__main__":
    main()
