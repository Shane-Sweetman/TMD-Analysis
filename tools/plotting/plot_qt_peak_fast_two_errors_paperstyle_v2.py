#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

OS_RED = "#EA3323"
SS_BLUE = "#1f4fb2"

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
                "peakOS": float(p[2]),
                "leftOS": float(p[3]),
                "rightOS": float(p[4]),
                "widthErrOS": float(p[5]),
                "nPeakOS": float(p[6]),
                "sqrtNPeakOS": float(p[7]),
                "statErrOS": float(p[8]),
                "nSS": float(p[9]),
                "peakSS": float(p[10]),
                "leftSS": float(p[11]),
                "rightSS": float(p[12]),
                "widthErrSS": float(p[13]),
                "nPeakSS": float(p[14]),
                "sqrtNPeakSS": float(p[15]),
                "statErrSS": float(p[16]),
            })
    if not rows:
        raise ValueError("No rows found.")
    return pd.DataFrame(rows)

def style_axes(ax):
    ax.tick_params(direction="in", top=True, right=True, length=8, width=1.1, labelsize=13)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=4, width=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

def make_plot(df: pd.DataFrame, outfile: Path, err_os: str, err_ss: str):
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=240)

    # Smaller markers so the error bars read more clearly
    ax.errorbar(
        df["cut"], df["peakOS"], yerr=df[err_os],
        fmt="o", ms=5.2, capsize=3.8, elinewidth=1.4, capthick=1.4,
        color=OS_RED, label="OS", linestyle="None", zorder=3
    )
    ax.errorbar(
        df["cut"], df["peakSS"], yerr=df[err_ss],
        fmt="s", ms=4.8, capsize=3.8, elinewidth=1.4, capthick=1.4,
        color=SS_BLUE, label="SS", linestyle="None", zorder=2
    )

    ax.set_xlabel("Cut on pion momentum fraction (%)")
    ax.set_ylabel(r"$q_T^{\mathrm{peak}}\ [\mathrm{GeV}]$")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)

    fig.savefig(outfile, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--prefix", default="qt_peak_fast_two_errors_v2")
    args = ap.parse_args()

    df = load_table(args.infile)
    prefix = Path(args.prefix)

    make_plot(df, Path(f"{prefix}_width.png"), "widthErrOS", "widthErrSS")
    make_plot(df, Path(f"{prefix}_stat.png"), "statErrOS", "statErrSS")

    print(f"Saved {prefix}_width.png/.pdf")
    print(f"Saved {prefix}_stat.png/.pdf")

if __name__ == "__main__":
    main()
