from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetworkModel
except ImportError:
    from pgmpy.models import BayesianNetwork as BayesianNetworkModel

from pgmpy.estimators import BayesianEstimator

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "Supplementary_data.xlsx"
OUT_XLSX = SCRIPT_DIR / "Table_ESI_CPT_stability_bootstrap.xlsx"

NODES = [
    "ESI", "RTP", "OP", "EP", "WGI", "POC", "PHHI", "IOC", "CPS",
    "COG", "CHHI", "CCS", "ECPC", "CPC", "EDLPG",
]
EDGES = [
    ("RTP", "CPS"),
    ("RTP", "EDLPG"),
    ("OP", "IOC"),
    ("EP", "POC"),
    ("EP", "CPC"),
    ("WGI", "CHHI"),
    ("WGI", "CPC"),
    ("POC", "CPS"),
    ("POC", "IOC"),
    ("PHHI", "COG"),
    ("PHHI", "ESI"),
    ("IOC", "EDLPG"),
    ("IOC", "COG"),
    ("CPS", "PHHI"),
    ("CPS", "CCS"),
    ("COG", "ECPC"),
    ("CHHI", "PHHI"),
    ("CCS", "ESI"),
    ("CPC", "ECPC"),
    ("EDLPG", "CCS"),
    ("CPC", "ESI"),
]
ESI_PARENTS = ["CPC", "CCS", "PHHI"]
ESI_STATES = ["High", "Medium", "Low"]
PARENT_ROWS = [
    ("High", "High", "High"),
    ("High", "High", "Low"),
    ("High", "Low", "High"),
    ("High", "Low", "Low"),
    ("Low", "High", "Low"),
    ("Low", "Low", "High"),
    ("Low", "High", "High"),
    ("Low", "Low", "Low"),
]

EQUIV_SAMPLE = 5
N_BOOT = 200
RANDOM_SEED = 42


def load_data() -> pd.DataFrame:
    raw = pd.read_excel(DATA_FILE, sheet_name="discretized_data")
    raw.columns = [str(c).strip() for c in raw.columns]
    data = raw[NODES].astype(str)
    for col in data.columns:
        data[col] = pd.Categorical(data[col], categories=sorted(data[col].unique()))
    return data


def fit_model(data: pd.DataFrame):
    model = BayesianNetworkModel(EDGES)
    for node in NODES:
        if node not in model.nodes():
            model.add_node(node)
    estimator = BayesianEstimator(model, data)
    cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=EQUIV_SAMPLE)
    model.add_cpds(*cpds)
    model.check_model()
    return model


def esi_cpt(model) -> dict[tuple[str, str, str], dict[str, float]]:
    cpd = model.get_cpds("ESI")
    parents = list(cpd.variables[1:])
    state_names = cpd.state_names
    values = np.asarray(cpd.values)
    out = {}
    parent_states = [state_names[p] for p in parents]
    for idx in np.ndindex(*values.shape[1:]):
        cfg = {parents[j]: str(parent_states[j][idx[j]]) for j in range(len(parents))}
        key = (cfg["CPC"], cfg["CCS"], cfg["PHHI"])
        out[key] = {
            str(state_names["ESI"][i]): float(values[(i, *idx)])
            for i in range(len(state_names["ESI"]))
        }
    return out


def parent_counts(data: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    grouped = data.groupby(ESI_PARENTS, observed=False).size().to_dict()
    return {(str(a), str(b), str(c)): int(n) for (a, b, c), n in grouped.items()}


def bootstrap_cpt(data: pd.DataFrame) -> list[dict[tuple[str, str, str], dict[str, float]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(data)
    cats = {col: list(data[col].cat.categories) for col in data.columns}
    draws = []
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot = data.iloc[idx].reset_index(drop=True)
        for col, levels in cats.items():
            boot[col] = pd.Categorical(boot[col], categories=levels)
        draws.append(esi_cpt(fit_model(boot)))
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{N_BOOT}")
    return draws


def build_table(
    full: dict[tuple[str, str, str], dict[str, float]],
    boots: list[dict[tuple[str, str, str], dict[str, float]]],
    counts: dict[tuple[str, str, str], int],
) -> pd.DataFrame:
    rows = []
    for key in PARENT_ROWS:
        n_ik = counts.get(key, 0)
        pf = full.get(key, {s: np.nan for s in ESI_STATES})
        acc = {s: [] for s in ESI_STATES}
        for draw in boots:
            probs = draw.get(key)
            if probs is None:
                continue
            for state in ESI_STATES:
                acc[state].append(probs.get(state, np.nan))
        rows.append(
            {
                "CPC": key[0],
                "CCS": key[1],
                "PHHI": key[2],
                "Full_High": pf.get("High", np.nan),
                "Full_Medium": pf.get("Medium", np.nan),
                "Full_Low": pf.get("Low", np.nan),
                "Boot_High": float(np.nanmean(acc["High"])) if acc["High"] else np.nan,
                "Boot_Medium": float(np.nanmean(acc["Medium"])) if acc["Medium"] else np.nan,
                "Boot_Low": float(np.nanmean(acc["Low"])) if acc["Low"] else np.nan,
                "N": n_ik,
            }
        )
    return pd.DataFrame(rows)


def write_excel(df: pd.DataFrame, path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Table"
    thin = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    header_fill = PatternFill("solid", fgColor="D9E2F3")

    ws.merge_cells("A1:C1")
    ws["A1"] = "Parent configuration"
    ws.merge_cells("D1:F1")
    ws["D1"] = "Full sample"
    ws.merge_cells("G1:I1")
    ws["G1"] = "Bootstrap mean"

    headers = ["CPC", "CCS", "PHHI", "High", "Medium", "Low", "High", "Medium", "Low"]
    for col, header in enumerate(headers, 1):
        ws.cell(2, col, header)
    for col in range(1, 10):
        for row in range(1, 3):
            cell = ws.cell(row, col)
            cell.alignment = center
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.border = thin

    for i, r in df.iterrows():
        rr = i + 3
        vals = [
            r.CPC, r.CCS, r.PHHI,
            r.Full_High, r.Full_Medium, r.Full_Low,
            r.Boot_High, r.Boot_Medium, r.Boot_Low,
        ]
        for col, v in enumerate(vals, 1):
            cell = ws.cell(rr, col, v)
            cell.alignment = center
            cell.border = thin
            if col >= 4 and isinstance(v, (float, int, np.floating)) and pd.notna(v):
                cell.number_format = "0.000"

    for i, w in enumerate([10, 10, 10, 10, 10, 10, 10, 10, 10], 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws2 = wb.create_sheet("Flat")
    flat = df.rename(
        columns={
            "Full_High": "Full sample High",
            "Full_Medium": "Full sample Medium",
            "Full_Low": "Full sample Low",
            "Boot_High": "Bootstrap mean High",
            "Boot_Medium": "Bootstrap mean Medium",
            "Boot_Low": "Bootstrap mean Low",
        }
    )
    for row in dataframe_to_rows(flat.round(3), index=False, header=True):
        ws2.append(row)
    wb.save(path)


def main() -> None:
    print(f"Reading {DATA_FILE}  sheet=discretized_data")
    data = load_data()
    print(f"n={len(data)}")

    print("Full-sample CPT...")
    full = esi_cpt(fit_model(data))
    counts = parent_counts(data)

    print(f"Bootstrap B={N_BOOT}, seed={RANDOM_SEED}...")
    boots = bootstrap_cpt(data)
    table = build_table(full, boots, counts)
    write_excel(table, OUT_XLSX)

    show = table.copy()
    for col in [
        "Full_High", "Full_Medium", "Full_Low",
        "Boot_High", "Boot_Medium", "Boot_Low",
    ]:
        show[col] = show[col].round(3)
    print(show.to_string(index=False))
    print(f"saved {OUT_XLSX}")


if __name__ == "__main__":
    main()
