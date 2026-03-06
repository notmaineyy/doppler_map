"""
plot_static.py
==============
Static Matplotlib plotter for Doppler Visibility Maps.
This is the closest Python equivalent to what you'd produce in MATLAB.

Run directly:
    python plot_static.py

Or import and call:
    from plot_static import plot_visibility_map
    plot_visibility_map(result, dv)

Requires: numpy, matplotlib
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from doppler_visibility import DopplerVisibility, VisibilityResult


# ------------------------------------------------------------------
# Colour scheme — dark radar phosphor aesthetic
# ------------------------------------------------------------------
BG_COLOR      = "#0d1117"
PANEL_COLOR   = "#161b22"
GRID_COLOR    = "#21262d"
GREEN_BRIGHT  = "#39d353"
GREEN_DIM     = "#0e4429"
BLIND_COLOR   = "#30100a"
BLIND_EDGE    = "#ff4444"
TEXT_COLOR    = "#c9d1d9"
ACCENT_COLOR  = "#58a6ff"
COMBINED_COLOR = "#ffd700"


def plot_visibility_map(
    result: VisibilityResult,
    dv: DopplerVisibility,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot a multi-panel Doppler Visibility Map using Matplotlib.

    Layout
    ------
    - One row per PRF showing the binary visibility (green = visible, dark = blind)
    - A final COMBINED row showing jointly-visible velocities
    - Right panel showing coverage % bar chart
    - Title bar with radar parameters

    Parameters
    ----------
    result : VisibilityResult
        Output from DopplerVisibility.compute()
    dv : DopplerVisibility
        The DopplerVisibility instance (for parameter access)
    save_path : str, optional
        If provided, save figure to this path instead of displaying.
    """
    n_prfs = len(result.prfs)
    n_rows = n_prfs + 1  # PRF rows + combined row

    fig = plt.figure(figsize=(16, 2.2 * n_rows + 2), facecolor=BG_COLOR)
    fig.suptitle(
        f"DOPPLER VISIBILITY MAP   |   "
        f"f = {dv.f_radar / 1e9:.1f} GHz   |   "
        f"λ = {result.wavelength * 100:.2f} cm   |   "
        f"Blind zone = {dv.blind_fraction * 100:.0f}% of v_ua",
        fontsize=13,
        color=TEXT_COLOR,
        fontfamily="monospace",
        y=0.98,
    )

    gs = GridSpec(
        n_rows, 2,
        figure=fig,
        width_ratios=[5, 1],
        hspace=0.08,
        wspace=0.04,
        left=0.10, right=0.95,
        top=0.93, bottom=0.07,
    )

    v = result.velocities
    axes_map = []

    # ------------------------------------------------------------------
    # Draw each PRF row
    # ------------------------------------------------------------------
    for i, prf in enumerate(result.prfs):
        ax = fig.add_subplot(gs[i, 0])
        axes_map.append(ax)

        vis = result.visibility_per_prf[i]

        # Fill blind (dark red) and visible (green) regions
        ax.fill_between(v, 0, 1, where=(vis < 0.5), color=BLIND_COLOR, step="mid")
        ax.fill_between(v, 0, 1, where=(vis > 0.5), color=GREEN_DIM, step="mid")

        # Mark blind speed lines
        for bs in result.blind_speeds[i]:
            ax.axvline(bs, color=BLIND_EDGE, linewidth=0.8, alpha=0.7, linestyle="--")

        # Formatting
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(dv.v_range)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        v_ua = result.unambiguous_velocities[i]
        label = (
            f"PRF {i+1}\n"
            f"{prf:.0f} Hz\n"
            f"v_ua={v_ua:.0f} m/s\n"
            f"{result.coverage_per_prf[i]*100:.0f}%"
        )
        ax.set_ylabel(label, fontsize=7.5, color=TEXT_COLOR, fontfamily="monospace",
                      rotation=0, ha="right", va="center", labelpad=55)

        if i < n_prfs - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Radial Velocity (m/s)", color=TEXT_COLOR,
                          fontsize=9, fontfamily="monospace")

    # ------------------------------------------------------------------
    # Combined visibility row
    # ------------------------------------------------------------------
    ax_comb = fig.add_subplot(gs[n_prfs, 0])
    axes_map.append(ax_comb)
    vis_c = result.combined_visibility

    ax_comb.fill_between(v, 0, 1, where=(vis_c < 0.5), color=BLIND_COLOR, step="mid")
    ax_comb.fill_between(v, 0, 1, where=(vis_c > 0.5), color=COMBINED_COLOR, alpha=0.6, step="mid")

    ax_comb.set_facecolor(BG_COLOR)
    ax_comb.set_xlim(dv.v_range)
    ax_comb.set_ylim(0, 1)
    ax_comb.set_yticks([])
    ax_comb.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax_comb.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax_comb.set_ylabel(
        f"COMBINED\n(all PRFs)\n{result.combined_coverage*100:.1f}% visible",
        fontsize=7.5, color=COMBINED_COLOR, fontfamily="monospace",
        rotation=0, ha="right", va="center", labelpad=55
    )
    ax_comb.set_xlabel("Radial Velocity (m/s)", color=TEXT_COLOR,
                       fontsize=9, fontfamily="monospace")

    # ------------------------------------------------------------------
    # Right panel — coverage bar chart
    # ------------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[:, 1])
    ax_bar.set_facecolor(PANEL_COLOR)

    labels = [f"PRF{i+1}\n{prf:.0f}Hz" for i, prf in enumerate(result.prfs)] + ["COMB."]
    coverages = result.coverage_per_prf + [result.combined_coverage]
    colors = [GREEN_DIM] * n_prfs + [COMBINED_COLOR]
    edge_colors = [GREEN_BRIGHT] * n_prfs + [COMBINED_COLOR]

    y_pos = np.arange(len(labels))[::-1]
    bars = ax_bar.barh(y_pos, [c * 100 for c in coverages],
                       color=colors, edgecolor=edge_colors,
                       linewidth=1.0, height=0.6)

    for bar, cov in zip(bars, coverages):
        ax_bar.text(
            min(cov * 100 + 1, 99), bar.get_y() + bar.get_height() / 2,
            f"{cov*100:.0f}%",
            va="center", ha="left", fontsize=7,
            color=TEXT_COLOR, fontfamily="monospace"
        )

    ax_bar.set_facecolor(PANEL_COLOR)
    ax_bar.set_xlim(0, 105)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=7, color=TEXT_COLOR, fontfamily="monospace")
    ax_bar.set_xlabel("Coverage (%)", fontsize=8, color=TEXT_COLOR, fontfamily="monospace")
    ax_bar.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax_bar.axvline(100, color=GRID_COLOR, linewidth=0.8)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    ax_bar.set_title("Coverage", fontsize=8, color=TEXT_COLOR,
                     fontfamily="monospace", pad=6)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    visible_patch = mpatches.Patch(color=GREEN_DIM, edgecolor=GREEN_BRIGHT,
                                   label="Visible (detectable)")
    blind_patch   = mpatches.Patch(color=BLIND_COLOR, edgecolor=BLIND_EDGE,
                                   label="Blind (undetectable)")
    combined_patch = mpatches.Patch(color=COMBINED_COLOR, alpha=0.6,
                                    label="Combined visible")
    fig.legend(
        handles=[visible_patch, blind_patch, combined_patch],
        loc="lower center",
        ncol=3,
        fontsize=8,
        facecolor=PANEL_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        framealpha=0.9,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()

    return fig


# ------------------------------------------------------------------
# Default run
# ------------------------------------------------------------------

if __name__ == "__main__":
    # --- Configure your radar here ---
    dv = DopplerVisibility(
        f_radar=10e9,                          # Hz  (10 GHz = X-band)
        prfs=[1000, 1250, 1500, 1750, 2000, 2500],  # Hz
        v_range=(-600, 600),                   # m/s
        n_points=4000,
        blind_fraction=0.05,                   # 5% blind zone width
    )

    result = dv.compute()
    print(dv.summary(result))
    plot_visibility_map(result, dv, save_path="doppler_visibility_map.png")
