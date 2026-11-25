"""Fig. A1 kernel density plots: 5 x 3 panel, Times New Roman, labels below.

Data: Supplementary_data.xlsx, sheet=raw_data, same folder as this script.

Sample (same as BN / descriptive stats):
  - Energy security index (ESI): paper-level rows
  - Other indicators: unique country-year rows

KDE: Gaussian kernel, Scott bandwidth, 1000-point grid (manuscript Sec. 3.1.2).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "Supplementary_data.xlsx"
FIGURE_DIR = SCRIPT_DIR / "kde_figures_simple"

# display name, column in raw_data, sample level — panel order of Fig. A1
INDICATORS = [
    ("Reserve to production", "Reserve to production", "cy"),
    ("Oil price", "Oil price", "cy"),
    ("Electricity price", "Electricity price", "cy"),
    ("WGI", "WGI", "cy"),
    ("Production to consumption", "Production to consumption", "cy"),
    ("Production HHI", "Production HHI", "cy"),
    ("Import to consumption", "Import to consumption", "cy"),
    ("Clean energy production share", "Clean production share", "cy"),
    ("Consumption to GDP", "Consumption to GDP", "cy"),
    ("Consumption HHI", "Consumption HHI", "cy"),
    ("Clean energy consumption share", "Clean consumption share", "cy"),
    ("Electricity consumption per capita", "Electricity consumption per capita", "cy"),
    (r"CO$_2$ emissions per capita", "Carbon emissions per capita", "cy"),
    ("Electricity distribution loss ratio", "Electricity distribution losses per generation", "cy"),
    ("Energy security index", "ESI", "paper"),
]

COLUMN_ALIASES = {
    "Production over consumption": "Production to consumption",
    "Production to consumption (POC)": "Production to consumption",
    "Production HHI (PHHI)": "Production HHI",
    "Reserve to production (RTP)": "Reserve to production",
    "Import over consumption oil+gas": "Import to consumption",
    "Import to consumption oil+gas": "Import to consumption",
    "Import to consumption (IOC)": "Import to consumption",
    "Clean production share (CPS)": "Clean production share",
    "Consumption/GDP": "Consumption to GDP",
    "Consumption to GDP (COG)": "Consumption to GDP",
    "Consumption HHI (CHHI)": "Consumption HHI",
    "Clean consumption share (CCS)": "Clean consumption share",
    "Electricity consumption per capita (ECPC)": "Electricity consumption per capita",
    "Oil price (OP)": "Oil price",
    "Electricity price (EP)": "Electricity price",
    "CO2/population": "Carbon emissions per capita",
    "CO2 per capita (CPC)": "Carbon emissions per capita",
    "CO2 per capita": "Carbon emissions per capita",
    "Electricity distribution losses per generation (EDLPG)": (
        "Electricity distribution losses per generation"
    ),
}


def setup_times() -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Times.ttc",
    ]
    font_path = next((p for p in candidates if Path(p).exists()), None)
    if font_path:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        family = "Times New Roman"
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"
    return family


def normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename = {col: COLUMN_ALIASES[col] for col in out.columns if col in COLUMN_ALIASES}
    return out.rename(columns=rename) if rename else out


def get_values(raw: pd.DataFrame, col: str, level: str) -> np.ndarray:
    if col not in raw.columns:
        raise KeyError(f"Column {col!r} not in raw_data. Available: {list(raw.columns)}")
    if level == "paper":
        s = pd.to_numeric(raw[col], errors="coerce")
    else:
        sub = raw[["Country", "Year", col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        s = sub.groupby(["Country", "Year"], as_index=False)[col].first()[col]
    values = s.to_numpy(dtype=float)
    return values[np.isfinite(values)]


def fit_kde(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < 5:
        raise ValueError("Not enough observations for KDE")
    if np.nanstd(values) < 1e-12:
        grid = np.linspace(float(values.min()) - 1e-6, float(values.max()) + 1e-6, 200)
        density = np.zeros_like(grid)
        density[len(grid) // 2] = 1.0
        return grid, density
    kde = gaussian_kde(values)  # Scott's rule
    pad = 0.05 * (float(values.max()) - float(values.min()) + 1e-12)
    grid = np.linspace(float(values.min()) - pad, float(values.max()) + pad, 1000)
    return grid, kde(grid)


def plot_on_ax(ax, display: str, values: np.ndarray, family: str) -> None:
    grid, density = fit_kde(values)
    ax.plot(grid, density, color="#2c7fb8", linewidth=1.8)
    ax.fill_between(grid, density, color="#2c7fb8", alpha=0.15)
    ax.set_title("")
    ax.set_xlabel(display, fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(family)


def main() -> None:
    family = setup_times()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw = normalize_columns(pd.read_excel(DATA_FILE, sheet_name="raw_data"))
    print(f"Reading {DATA_FILE}")
    print(f"Figures -> {FIGURE_DIR}")

    fig, axes = plt.subplots(5, 3, figsize=(14.5, 17), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for i, (display, col, level) in enumerate(INDICATORS):
        values = get_values(raw, col, level)
        plot_on_ax(axes[i], display, values, family)
        print(f"  {display}: n={len(values)} ({level})")

    for j in range(len(INDICATORS), len(axes)):
        axes[j].axis("off")

    out_png = FIGURE_DIR / "kde_all_3col.png"
    out_pdf = FIGURE_DIR / "kde_all_3col.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"font={family}")
    print(f"saved {out_png.name}")
    print(f"saved {out_pdf.name}")
    print("Done.")


if __name__ == "__main__":
    main()
