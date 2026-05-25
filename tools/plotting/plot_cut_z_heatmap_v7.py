#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


def parse_file(path: str):
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
            if len(p) < 11:
                continue
            rows.append({
                "cut": float(p[0]),
                "z": float(p[1]),
                "ratio": float(p[10]),
            })
    if benchmark_score is None:
        raise ValueError("benchmark_score not found in file header")
    if not rows:
        raise ValueError("No grid rows found")
    return benchmark_score, pd.DataFrame(rows)


def make_stronger_cmap():
    return LinearSegmentedColormap.from_list(
        "strong_blue_white_red",
        [
            (0.00, "#2746bf"),
            (0.18, "#5578da"),
            (0.34, "#88a4ee"),
            (0.47, "#d8dff7"),
            (0.50, "#f2f2f2"),
            (0.53, "#f6dddd"),
            (0.66, "#ef9d9d"),
            (0.82, "#e15c5c"),
            (1.00, "#d42020"),
        ],
    )


def tick_label(v: float) -> str:
    if v >= 1000:
        exp = int(np.floor(np.log10(v)))
        mant = v / (10 ** exp)
        if abs(mant - round(mant)) < 1e-9:
            mant_str = f"{int(round(mant))}"
        else:
            mant_str = f"{mant:.1f}"
        return rf"${mant_str}\times10^{{{exp}}}$"
    if v >= 100:
        return f"{int(round(v))}"
    if v >= 1:
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}"
        return f"{v:.1f}"
    if v >= 0.1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def build_125_ticks(vmin: float, vmax: float):
    if vmin <= 0 or vmax <= 0:
        raise ValueError("vmin and vmax must be positive")

    min_exp = int(np.floor(np.log10(vmin)))
    max_exp = int(np.ceil(np.log10(vmax)))

    vals = []
    for e in range(min_exp - 1, max_exp + 2):
        for m in (1.0, 2.0, 5.0):
            vals.append(m * (10 ** e))

    vals = sorted(v for v in vals if vmin <= v <= vmax)
    vals = sorted(set(vals + [vmin, vmax]))

    cleaned = []
    for v in vals:
        if not cleaned:
            cleaned.append(v)
        else:
            if abs(np.log10(v) - np.log10(cleaned[-1])) > 0.02:
                cleaned.append(v)
    return cleaned


def thin_red_ticks(ticks):
    """Keep all ticks <= 1. Above 1, alternate dropping/keeping starting by dropping the first >1 tick.
    Example: remove 2, keep 5, remove 10, keep 20, ...
    """
    blue_and_one = [t for t in ticks if t <= 1.0]
    red = [t for t in ticks if t > 1.0]

    kept_red = []
    keep = False  # first red tick (>1) is removed
    for t in red:
        if keep:
            kept_red.append(t)
        keep = not keep

    out = blue_and_one + kept_red
    if 1.0 not in out:
        out.append(1.0)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--outfile", default="scan_cut_z_heatmap_v7.png")
    ap.add_argument(
        "--title",
        default=r"$\chi^2_{\mathrm{parameter}} / \chi^2_{\mathrm{base}}$",
    )
    ap.add_argument("--dpi", type=int, default=240)
    args = ap.parse_args()

    benchmark_score, df = parse_file(args.infile)

    xs = sorted(df["z"].unique())
    ys = sorted(df["cut"].unique(), reverse=True)

    pivot = df.pivot(index="cut", columns="z", values="ratio").reindex(index=ys, columns=xs)
    arr = pivot.to_numpy()
    log_ratio = np.log10(arr)

    actual_min = float(np.nanmin(arr))
    actual_max = float(np.nanmax(arr))

    fig, ax = plt.subplots(figsize=(9.0, 6.2), dpi=args.dpi, constrained_layout=True)

    cmap = make_stronger_cmap()
    norm = TwoSlopeNorm(
        vmin=float(np.nanmin(log_ratio)),
        vcenter=0.0,
        vmax=float(np.nanmax(log_ratio)),
    )

    im = ax.imshow(
        log_ratio,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([f".{int(round(x * 100)):02d}" for x in xs], fontsize=11)

    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels([f"{int(y)}" for y in ys], fontsize=12)

    ax.set_xlabel(r"$z$", fontsize=24, labelpad=10)
    ax.set_ylabel("cut (%)", fontsize=24, labelpad=10)
    ax.set_title(args.title, fontsize=28, pad=18)

    ax.tick_params(which="both", length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.045)
    cbar.ax.tick_params(labelsize=11)

    ticks = build_125_ticks(actual_min, actual_max)
    ticks = thin_red_ticks(ticks)

    cbar.set_ticks(np.log10(ticks))
    cbar.set_ticklabels([tick_label(t) for t in ticks])

    out = Path(args.outfile)
    fig.savefig(out, bbox_inches="tight")
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    print(f"Benchmark score = {benchmark_score:.6g}")
    print(f"Actual min ratio = {actual_min:.6g}")
    print(f"Actual max ratio = {actual_max:.6g}")
    print(f"Saved {out}")
    print(f"Saved {pdf_out}")


if __name__ == "__main__":
    main()
