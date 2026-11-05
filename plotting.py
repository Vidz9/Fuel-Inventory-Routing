"""
plotting.py — IRP route level distribution plots.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from results import InstanceResult, MethodResult


def _level_for_route(route_id, route_records, warmstart_label=-1):
    rec = route_records.get(route_id)
    if rec is None:
        return None
    origin = getattr(rec, "origin", None)
    if origin == "dummy":
        return None
    if origin == "warmstart":
        return warmstart_label
    level = getattr(rec, "tree_level", 0)
    return level if level is not None else 0


def _collect_levels(mr, warmstart_label=-1):
    if not mr.selected_route_ids or not mr.route_records:
        return []
    levels = []
    for (v, t), route_id in mr.selected_route_ids.items():
        lv = _level_for_route(route_id, mr.route_records, warmstart_label)
        if lv is not None:
            levels.append(lv)
    return levels


def _level_colors(bin_labels):
    palette = plt.cm.YlOrRd
    max_lv = max((lv for lv in bin_labels if lv >= 1), default=1)
    colors = []
    for lv in bin_labels:
        if lv == -1:
            colors.append("#5cb85c")
        elif lv == 0:
            colors.append("#4878cf")
        else:
            t = lv / max(max_lv, 1)
            colors.append(palette(0.35 + 0.55 * t))
    return colors


def plot_route_level_distributions(
    results,
    results_dir="results",
    demand_coeff=1.0,
    V=2,
    T=5,
    save=True,
    show=False,
):
    """
    Histogram of route assignments by tree level in BP / WBP integer solutions.

    For BP  : levels 0, 1, 2, ...  (all from CG / B&P)
    For WBP : level -1 = heuristic warm-start, then 0, 1, 2, ...

    Bars = average % across instances, error bars = +/-1 std dev.
    """
    inst_dict = results.get(demand_coeff, {}).get((V, T), {})
    if not inst_dict:
        print(f"[plot] No results for demand={demand_coeff}, V={V}, T={T}")
        return {}

    bp_per_inst, wbp_per_inst = [], []
    for inst_id, ir in inst_dict.items():
        if ir.bp is not None:
            lvs = _collect_levels(ir.bp, warmstart_label=-1)
            if lvs:
                bp_per_inst.append(lvs)
        if ir.warmstart_bp is not None:
            lvs = _collect_levels(ir.warmstart_bp, warmstart_label=-1)
            if lvs:
                wbp_per_inst.append(lvs)

    figs = {}
    specs = [
        ("bp",  bp_per_inst,
         "BP — route assignments by tree level"),
        ("wbp", wbp_per_inst,
         "WBP — route assignments by tree level\n(level \u22121 = heuristic warm-start)"),
    ]

    for tag, per_inst, title in specs:
        if not per_inst:
            print(f"[plot] No data for {tag.upper()}, skipping")
            continue

        all_flat   = [lv for inst_lvs in per_inst for lv in inst_lvs]
        counter    = Counter(all_flat)
        bin_labels = sorted(counter.keys())

        pct_matrix = np.zeros((len(per_inst), len(bin_labels)))
        for i, inst_lvs in enumerate(per_inst):
            n = len(inst_lvs)
            if n == 0:
                continue
            c = Counter(inst_lvs)
            for j, lv in enumerate(bin_labels):
                pct_matrix[i, j] = 100.0 * c.get(lv, 0) / n

        mean_pct = pct_matrix.mean(axis=0)
        std_pct  = pct_matrix.std(axis=0)

        fig, ax = plt.subplots(figsize=(max(6, len(bin_labels) * 1.2), 5))
        x      = np.arange(len(bin_labels))
        colors = _level_colors(bin_labels)

        bars = ax.bar(x, mean_pct, color=colors, edgecolor="white",
                      linewidth=0.6, zorder=3)
        ax.errorbar(x, mean_pct, yerr=std_pct, fmt="none",
                    ecolor="#333", elinewidth=1.2, capsize=4, zorder=4)

        for bar, m, s in zip(bars, mean_pct, std_pct):
            if m > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + s + 0.6,
                        f"{m:.1f}%",
                        ha="center", va="bottom", fontsize=8)

        tick_labels = []
        for lv in bin_labels:
            if lv == -1:
                tick_labels.append("Warm-start\n(level \u22121)")
            elif lv == 0:
                tick_labels.append("Root CG\n(level 0)")
            else:
                tick_labels.append(f"Level {lv}")

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel("% of route assignments in integer solution", fontsize=10)
        ax.set_title(
            f"{title}\ndemand={demand_coeff}  V={V}  T={T}  n={len(per_inst)} instances",
            fontsize=11,
        )
        ax.set_ylim(0, min(115, mean_pct.max() + std_pct.max() + 14))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        figs[tag] = fig

        if save:
            Path(results_dir).mkdir(exist_ok=True)
            fname = (Path(results_dir) /
                     f"route_levels_{tag}_{demand_coeff}demand_V{V}_T{T}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"[plot] Saved -> {fname}")

        if show:
            plt.show()

    return figs