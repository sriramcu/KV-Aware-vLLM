"""
kv_cache_visualizer.py
-----------------------
Produces a 6-panel figure from the data collected by kv_cache_monitor.

Usage
-----
    from kv_cache_monitor import to_dataframe, per_request_stats, hottest_blocks, hit_rate_over_time
    from kv_cache_visualizer import visualize

    visualize(save_path="kv_cache_analysis.png")   # saves to disk
    visualize()                                     # shows interactive window
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap


# ── colour palette ──────────────────────────────────────────────────────────
HIT_COLOR  = "#4C9BE8"
MISS_COLOR = "#E8734C"
WARM_CMAP  = LinearSegmentedColormap.from_list("warm", ["#f7f7f7", "#4C9BE8", "#003d6b"])


def visualize(
    save_path: str | None = None,
    top_n_blocks: int = 20,
    time_bucket_s: float = 5.0,
    figsize: tuple = (18, 14),
) -> plt.Figure:
    """
    Build and return a matplotlib Figure.

    Parameters
    ----------
    save_path    : if given, saves the figure here (PNG/PDF/SVG etc.)
    top_n_blocks : how many hottest blocks to show in bar / heatmap panels
    time_bucket_s: bucket width for the temporal hit-rate chart
    """
    from kv_cache_monitor import (
        to_dataframe,
        per_request_stats,
        hottest_blocks,
        hit_rate_over_time,
        global_hit_rate,
    )

    df          = to_dataframe()
    req_stats   = per_request_stats()
    hot_blocks  = hottest_blocks(top_n_blocks)
    time_series = hit_rate_over_time(time_bucket_s)
    ghr         = global_hit_rate()

    if df.empty:
        raise RuntimeError("No data recorded yet. Run your vLLM workload first.")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.suptitle(
        f"KV Cache Analysis  —  global hit rate: {ghr['global_hit_rate']:.1%}  "
        f"({ghr['total_block_hits']} hits / {ghr['total_block_lookups']} lookups)",
        fontsize=14, fontweight="bold",
    )

    gs = fig.add_gridspec(3, 3)

    # ── 1. Hottest blocks — horizontal bar ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    labels   = [r["block_hash"][-28:] for r in hot_blocks]
    hit_vals = [r["hit_count"]        for r in hot_blocks]
    colors   = [HIT_COLOR if r["hit_rate"] > 0.5 else MISS_COLOR for r in hot_blocks]
    bars = ax1.barh(range(len(labels)), hit_vals, color=colors, edgecolor="white", linewidth=0.4)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Hit count")
    ax1.set_title(f"Top {top_n_blocks} hottest blocks")
    for bar, row in zip(bars, hot_blocks):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{row['hit_rate']:.0%}  ({row['unique_request_hits']} reqs)",
                 va="center", fontsize=7, color="#333333")

    # ── 2. Global hit / miss donut ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    wedge_vals  = [ghr["total_block_hits"], ghr["total_block_misses"]]
    wedge_labels = ["Hits", "Misses"]
    wedge_colors = [HIT_COLOR, MISS_COLOR]
    ax2.pie(wedge_vals, labels=wedge_labels, colors=wedge_colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"width": 0.5, "edgecolor": "white"})
    ax2.set_title("Global hit / miss ratio")

    # ── 3. Per-request hit rate — line + fill ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    if req_stats:
        req_df = pd.DataFrame(req_stats)
        x      = range(len(req_df))
        ax3.fill_between(x, req_df["hit_rate"], alpha=0.25, color=HIT_COLOR)
        ax3.plot(x, req_df["hit_rate"], color=HIT_COLOR, linewidth=1.5, marker="o",
                 markersize=3, label="hit rate")
        ax3.axhline(ghr["global_hit_rate"], color="grey", linestyle="--",
                    linewidth=1, label=f"avg {ghr['global_hit_rate']:.1%}")
        ax3.set_ylim(0, 1.05)
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax3.set_xlabel("Request index (arrival order)")
        ax3.set_ylabel("Cache hit rate")
        ax3.set_title("Per-request hit rate (cache warm-up curve)")
        ax3.legend(fontsize=8)

    # ── 4. Hit rate over wall-clock time ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    if time_series:
        ts_df = pd.DataFrame(time_series)
        ax4.bar(ts_df["time_offset_s"], ts_df["hit_rate"],
                width=time_bucket_s * 0.85, color=HIT_COLOR, edgecolor="white")
        ax4.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax4.set_xlabel(f"Time offset (s), bucket = {time_bucket_s}s")
        ax4.set_ylabel("Hit rate")
        ax4.set_title("Hit rate over time")

    # ── 5. Heatmap: block_index × hit/miss counts per request ───────────────
    ax5 = fig.add_subplot(gs[2, :2])
    pivot = (
        df.groupby(["request_id", "block_index"])["hit"]
          .mean()
          .unstack(fill_value=0)
    )
    # Keep at most 40 requests for readability
    pivot = pivot.iloc[:40]
    im = ax5.imshow(pivot.values, aspect="auto", cmap=WARM_CMAP, vmin=0, vmax=1)
    ax5.set_xlabel("Block index (prefix position)")
    ax5.set_ylabel("Request")
    ax5.set_yticks(range(len(pivot.index)))
    ax5.set_yticklabels(pivot.index, fontsize=6)
    ax5.set_title("Hit rate heatmap — requests × block positions")
    plt.colorbar(im, ax=ax5, fraction=0.015, pad=0.01, label="hit rate")

    # ── 6. Block depth histogram — where do misses happen? ──────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    miss_df = df[~df["hit"]]
    if not miss_df.empty:
        ax6.hist(miss_df["block_index"], bins=range(int(miss_df["block_index"].max()) + 2),
                 color=MISS_COLOR, edgecolor="white", linewidth=0.4)
    ax6.set_xlabel("Block index at first miss")
    ax6.set_ylabel("Count")
    ax6.set_title("Distribution of first-miss position")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[kv_cache_visualizer] Saved → {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# CLI entry-point: run after kv_cache_monitor self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time as _time

    # Seed the monitor with the same synthetic data as its own self-test
    import random
    import kvcache_monitor as mon
    from kvcache_monitor import _STORE, BlockEvent, RequestSummary

    t0 = _time.time()
    random.seed(42)
    for req_i in range(30):
        mon.set_current_request_id(f"req-{req_i:03d}")
        n_check    = random.randint(4, 12)
        hit_blocks = min(4, req_i)
        for blk_i in range(n_check):
            bh  = f"hash_{blk_i:04d}" if blk_i < 4 else f"hash_{blk_i:04d}_req{req_i}"
            hit = blk_i < hit_blocks
            _STORE.record_lookup(
                ts=t0 + req_i * 0.15 + blk_i * 0.01,
                request_id=f"req-{req_i:03d}",
                block_hash_repr=bh,
                block_index=blk_i,
                hit=hit,
                group_id=0,
            )
            if not hit:
                break
        _STORE.record_request(RequestSummary(
            request_id=f"req-{req_i:03d}",
            ts_start=t0 + req_i * 0.15,
            ts_end=t0 + req_i * 0.15 + 0.05,
            total_blocks_checked=min(hit_blocks + 1, n_check),
            hit_blocks=hit_blocks,
            miss_blocks=1,
            block_size=16,
            hit_token_count=hit_blocks * 16,
            total_token_count=n_check * 16,
        ))

    out = sys.argv[1] if len(sys.argv) > 1 else "kv_cache_analysis.png"
    visualize(save_path=out)