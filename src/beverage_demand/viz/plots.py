"""Shared chart styling. Colours follow the entity (brand), never its rank.

Categorical palette validated for colour-vision deficiency (adjacent-pair CVD dE >= 8); slots with
low contrast against the surface are always direct-labelled.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from beverage_demand.config import figure  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BRAND_COLORS = {
    "Coke": CATEGORICAL[0],
    "Diet Coke": CATEGORICAL[1],
    "Pepsi": CATEGORICAL[2],
    "Diet Pepsi": CATEGORICAL[3],
    "7Up": CATEGORICAL[4],
    "Private Label": CATEGORICAL[5],
}
NEUTRAL = "#9a9992"
ACCENT = CATEGORICAL[0]
CONTRAST = CATEGORICAL[1]
# diverging: red (negative) <-> grey midpoint <-> blue (positive)
DIVERGING = LinearSegmentedColormap.from_list(
    "bev_div", ["#b8322f", "#e98a87", "#f0efec", "#86b6ef", "#1c5cab"]
)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "figure.dpi": 110,
        }
    )


def brand_color(series_or_brand: str) -> str:
    brand = series_or_brand.split(" | ")[0]
    return BRAND_COLORS.get(brand, NEUTRAL)


def save(fig, name: str) -> str:
    out = figure(name)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return str(out)


style()
