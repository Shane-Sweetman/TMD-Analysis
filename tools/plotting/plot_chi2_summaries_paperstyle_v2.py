#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

RED = "#C62828"
BLUE = "#1f4fb2"
BLACK = "#111111"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 16,
    "axes.linewidth": 1.2,
})


def style_axes(ax):
    ax.tick_params(direction="in", top=True, right=True, length=8, width=1.1, labelsize=13)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=4, width=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def parse_fullgrid(path: str):
    benchmark_score = None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if s.startswith("# benchmark_score"):
                    benchmark_score = float(s.split()[-1])
                continue
            p = s.split()
            if len(p) < 14:
                continue
            rows.append({
                "cut": float(p[0]),
                "z": float(p[1]),
                "scale": float(p[2]),
                "chi2OS": float(p[3]),
                "nOS": float(p[4]),
                "chi2PerOS": float(p[5]),
                "chi2SS": float(p[6]),
                "nSS": float(p[7]),
                "chi2PerSS": float(p[8]),
                "score": float(p[9]),
                "ratio": float(p[10]),
                "osCount": float(p[11]),
                "ssCount": float(p[12]),
                "totalCount": float(p[13]),
            })
    if benchmark_score is None:
        raise ValueError("Could not find benchmark_score in fullgrid file.")
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows parsed from fullgrid file.")
    return benchmark_score, df


def parse_order_chi2_file(path: str):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    m_os = re.search(r"OS\s*:\s*chi2\s*=\s*[-+0-9.eE]+\s*,\s*N\s*=\s*\d+\s*,\s*chi2/N\s*=\s*([-+0-9.eE]+)", text)
    m_ss = re.search(r"SS\s*:\s*chi2\s*=\s*[-+0-9.eE]+\s*,\s*N\s*=\s*\d+\s*,\s*chi2/N\s*=\s*([-+0-9.eE]+)", text)
    if not m_os or not m_ss:
        raise ValueError(f"Could not parse OS/SS chi2/N from {path}")
    osv = float(m_os.group(1))
    ssv = float(m_ss.group(1))
    return osv + ssv, osv, ssv


def best_by_cut(df: pd.DataFrame, max_score=None, always_keep_cut=None):
    grouped = (
        df.sort_values(["cut", "score", "z"])
          .groupby("cut", as_index=False)
          .first()
          .sort_values("cut")
    )
    if max_score is not None:
        mask = grouped["score"] <= max_score
        if always_keep_cut is not None:
            mask = mask | (grouped["cut"] == always_keep_cut)
        grouped = grouped[mask].copy()
    return grouped


def plot_best_chi2_vs_cut(best_df: pd.DataFrame, outfile: Path):
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=240)

    ax.plot(best_df["cut"], best_df["score"], color=RED, lw=2.4, zorder=2)
    ax.scatter(best_df["cut"], best_df["score"], color=RED, edgecolor=BLACK, s=42, zorder=3)

    ax.set_xlabel("Cut on pion momentum fraction (%)")
    ax.set_ylabel(r"Best global $\chi^2$")
    style_axes(ax)

    fig.savefig(outfile, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_best_z_vs_cut(best_df: pd.DataFrame, outfile: Path):
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=240)

    ax.plot(best_df["cut"], best_df["z"], color=BLUE, lw=2.4, zorder=2)
    ax.scatter(best_df["cut"], best_df["z"], color=BLUE, edgecolor=BLACK, s=42, zorder=3)

    ax.set_xlabel("Cut on pion momentum fraction (%)")
    ax.set_ylabel(r"Best-fit $z$")
    style_axes(ax)

    fig.savefig(outfile, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_chi2_vs_order(order_df: pd.DataFrame, outfile: Path, max_score=None):
    if max_score is not None:
        order_df = order_df[order_df["score"] <= max_score].copy()

    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=240)

    x = list(range(len(order_df)))
    ax.plot(x, order_df["score"], color=RED, lw=2.4, zorder=2)
    ax.scatter(x, order_df["score"], color=RED, edgecolor=BLACK, s=48, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(order_df["order"])
    ax.set_xlabel("Perturbative order")
    ax.set_ylabel(r"Global $\chi^2$")
    style_axes(ax)

    fig.savefig(outfile, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fullgrid", help="Full cut-z scan txt file.")
    ap.add_argument("--lo", help="LO chi2 txt file")
    ap.add_argument("--nlo", help="NLO chi2 txt file")
    ap.add_argument("--nnlo", help="NNLO chi2 txt file")
    ap.add_argument("--current-label", default=r"Order 3")
    ap.add_argument("--max-best-score", type=float, default=None,
                    help="Drop cuts whose best global chi2 exceeds this value.")
    ap.add_argument("--keep-cut", type=float, default=60.0,
                    help="Always keep this cut in the best-vs-cut plot even if above max-best-score.")
    ap.add_argument("--max-order-score", type=float, default=None,
                    help="Drop perturbative-order points above this chi2 value.")
    ap.add_argument("--prefix", default="chi2_summary_v2")
    args = ap.parse_args()

    benchmark_score, df = parse_fullgrid(args.fullgrid)
    best_df = best_by_cut(df, max_score=args.max_best_score, always_keep_cut=args.keep_cut)

    prefix = Path(args.prefix)

    plot_best_chi2_vs_cut(best_df, Path(f"{prefix}_best_vs_cut.png"))
    plot_best_z_vs_cut(best_df, Path(f"{prefix}_bestz_vs_cut.png"))

    order_rows = []
    if args.lo:
        s, osv, ssv = parse_order_chi2_file(args.lo)
        order_rows.append({"order": "LO", "score": s, "os": osv, "ss": ssv})
    if args.nlo:
        s, osv, ssv = parse_order_chi2_file(args.nlo)
        order_rows.append({"order": "NLO", "score": s, "os": osv, "ss": ssv})
    if args.nnlo:
        s, osv, ssv = parse_order_chi2_file(args.nnlo)
        order_rows.append({"order": "NNLO", "score": s, "os": osv, "ss": ssv})
    order_rows.append({"order": args.current_label, "score": benchmark_score, "os": None, "ss": None})
    order_df = pd.DataFrame(order_rows)

    plot_chi2_vs_order(order_df, Path(f"{prefix}_vs_order.png"), max_score=args.max_order_score)

    best_df.to_csv(f"{prefix}_best_by_cut.csv", index=False)
    order_df.to_csv(f"{prefix}_order_values.csv", index=False)

    print(f"Benchmark score (current order) = {benchmark_score:.6g}")
    print(f"Saved {prefix}_best_vs_cut.png/.pdf")
    print(f"Saved {prefix}_bestz_vs_cut.png/.pdf")
    print(f"Saved {prefix}_vs_order.png/.pdf")
    print(f"Saved {prefix}_best_by_cut.csv")
    print(f"Saved {prefix}_order_values.csv")


if __name__ == "__main__":
    main()
