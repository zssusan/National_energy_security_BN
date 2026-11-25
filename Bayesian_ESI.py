from __future__ import annotations

import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetworkModel
except ImportError:
    from pgmpy.models import BayesianNetwork as BayesianNetworkModel

try:
    from pgmpy.structure_score import BIC as BicScorer
except ImportError:
    try:
        from pgmpy.estimators import BicScore as BicScorer
    except ImportError:
        from pgmpy.estimators import BIC as BicScorer

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "Supplementary_data.xlsx"
OUTPUT_DIR = SCRIPT_DIR
OIL_COL, OIL_SHORT = "Oil price", "OilP"
RTP_COL = "Reserve to production"
IOC_COL = "Import to consumption"
POC_COL = "Production to consumption"
COG_COL = "Consumption to GDP"
CPC_COL = "Carbon emissions per capita"
N_RESTARTS = 50
N_OPERATIONS = 40
INIT_DENSITY = 0.3
RANDOM_SEED = 42

UPSTREAM = [RTP_COL, OIL_COL, "Electricity price", "WGI"]
MIDSTREAM = [
    POC_COL,
    "Production HHI",
    IOC_COL,
    "Clean production share",
    COG_COL,
    "Consumption HHI",
    "Clean consumption share",
    "Electricity consumption per capita",
    CPC_COL,
    "Electricity distribution losses per generation",
]
DOWNSTREAM = ["ESI"]
TARGET_COLUMNS = DOWNSTREAM + UPSTREAM + MIDSTREAM
STAGE_A_NODES = UPSTREAM + MIDSTREAM

NEGATIVE_INDICATORS = [
    "Production HHI",
    IOC_COL,
    COG_COL,
    "Consumption HHI",
    "Electricity consumption per capita",
    OIL_COL,
    "Electricity price",
    CPC_COL,
    "Electricity distribution losses per generation",
]

INDICATOR_DIRECTION = {
    POC_COL: "+",
    "Production HHI": "-",
    RTP_COL: "+",
    IOC_COL: "-",
    "Clean production share": "+",
    COG_COL: "-",
    "Consumption HHI": "-",
    "Clean consumption share": "+",
    "Electricity consumption per capita": "-",
    OIL_COL: "-",
    "Electricity price": "-",
    CPC_COL: "-",
    "Electricity distribution losses per generation": "-",
    "WGI": "+",
    "ESI": "+",
}

N_BINS = {
    "ESI": 3,
    POC_COL: 2,
    "Production HHI": 2,
    RTP_COL: 2,
    IOC_COL: 2,
    "Clean production share": 2,
    COG_COL: 2,
    "Consumption HHI": 2,
    "Clean consumption share": 2,
    "Electricity consumption per capita": 3,
    OIL_COL: 3,
    "Electricity price": 3,
    CPC_COL: 2,
    "Electricity distribution losses per generation": 2,
    "WGI": 2,
}

BIN_LABELS = {
    2: {0: "Low", 1: "High"},
    3: {0: "Low", 1: "Medium", 2: "High"},
}

SHORT_LABELS = {
    "ESI": "ESI",
    RTP_COL: "RTP",
    IOC_COL: "IOC",
    OIL_COL: OIL_SHORT,
    "Electricity price": "EP",
    "WGI": "WGI",
    POC_COL: "POC",
    "Production HHI": "PHHI",
    "Clean production share": "CPS",
    COG_COL: "COG",
    "Consumption HHI": "CHHI",
    "Clean consumption share": "CCS",
    "Electricity consumption per capita": "ECPC",
    CPC_COL: "CPC",
    "Electricity distribution losses per generation": "EDLPG",
}


def get_level(var: str) -> int:
    if var in UPSTREAM:
        return 2
    if var in MIDSTREAM:
        return 1
    if var in DOWNSTREAM:
        return 0
    return 1


def get_level_name(var: str) -> str:
    return {2: "Upstream", 1: "Midstream", 0: "Downstream"}[get_level(var)]


def is_allowed_edge(u: str, v: str) -> bool:
    if u == v:
        return False
    lu, lv = get_level(u), get_level(v)
    if lu > lv:
        return True
    if lu == lv == 1:
        return True
    return False


def kmeans_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1, 1)
    x = x[np.isfinite(x[:, 0])]
    if len(x) < n_bins:
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        return np.quantile(values, qs)

    km = KMeans(
        n_clusters=n_bins,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    km.fit(x)
    centers = np.sort(km.cluster_centers_.ravel())
    return (centers[:-1] + centers[1:]) / 2.0


def assign_by_thresholds(values: np.ndarray, thresholds: np.ndarray, labels: dict[int, str]) -> list[str]:
    thresholds = np.sort(thresholds)
    out = []
    for v in values:
        idx = int(np.searchsorted(thresholds, v, side="right"))
        out.append(labels[idx])
    return out


def apply_indicator_direction(scaled: pd.DataFrame) -> pd.DataFrame:
    out = scaled.copy()
    for col in NEGATIVE_INDICATORS:
        if col in out.columns:
            out[col] = 1.0 - out[col]
    return out


def build_direction_table() -> pd.DataFrame:
    rows = []
    for col in TARGET_COLUMNS:
        rows.append(
            {
                "indicator": col,
                "short": SHORT_LABELS.get(col, col),
                "level": get_level_name(col),
                "direction": INDICATOR_DIRECTION.get(col, ""),
                "reversed_for_learning": col in NEGATIVE_INDICATORS,
            }
        )
    return pd.DataFrame(rows)


def load_raw_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}\n"
            "Place Supplementary_data.xlsx in the same folder as this script."
        )
    raw = pd.read_excel(DATA_FILE, sheet_name="raw_data")
    raw.columns = [str(c).strip() for c in raw.columns]
    if "Paper" not in raw.columns:
        raw["Paper"] = pd.NA
    missing = [c for c in ["ID", "Year", "Country", *TARGET_COLUMNS] if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns in raw_data: {missing}")
    return raw


def prepare_discretized_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = raw[TARGET_COLUMNS].copy()
    if work.isnull().any().any():
        work = work.fillna(work.median(numeric_only=True))

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(work),
        columns=TARGET_COLUMNS,
        index=work.index,
    )
    scaled = apply_indicator_direction(scaled)

    country_year = (
        pd.concat([raw[["Country", "Year"]], scaled], axis=1)
        .groupby(["Country", "Year"], as_index=False)[TARGET_COLUMNS]
        .first()
    )

    threshold_rows = []
    labeled = pd.DataFrame(index=raw.index)
    for col in TARGET_COLUMNS:
        n_bins = N_BINS[col]
        if col == "ESI":
            fit_values = scaled[col].to_numpy(dtype=float)
            sample_rule = "paper_rows"
        else:
            fit_values = country_year[col].to_numpy(dtype=float)
            sample_rule = "country_year"

        thresholds = kmeans_bin_edges(fit_values, n_bins)
        labels = BIN_LABELS[n_bins]
        labeled[col] = assign_by_thresholds(scaled[col].to_numpy(dtype=float), thresholds, labels)

        row = {
            "indicator": col,
            "level": get_level_name(col),
            "direction": INDICATOR_DIRECTION.get(col, ""),
            "reversed": col in NEGATIVE_INDICATORS,
            "n_bins": n_bins,
            "sample_rule_for_fit": sample_rule,
            "fit_n": len(fit_values),
        }
        for i, thr in enumerate(thresholds, start=1):
            row[f"threshold_{i}"] = float(thr)
        vc = pd.Series(labeled[col]).value_counts()
        for lab in labels.values():
            row[f"count_{lab.replace(' ', '_')}"] = int(vc.get(lab, 0))
        threshold_rows.append(row)

    numeric = labeled.copy()
    for col in TARGET_COLUMNS:
        numeric[col] = pd.Categorical(
            numeric[col],
            categories=list(BIN_LABELS[N_BINS[col]].values()),
            ordered=True,
        ).codes

    meta = raw[["ID", "Year", "Country", "Paper"]].copy()
    labeled_out = pd.concat([meta, labeled], axis=1)
    threshold_table = pd.DataFrame(threshold_rows)

    cy_keys = raw[["Country", "Year"]].copy()
    cy_keys["__row__"] = raw.index
    first_idx = cy_keys.groupby(["Country", "Year"], as_index=False)["__row__"].first()["__row__"]
    numeric_cy = numeric.loc[first_idx].reset_index(drop=True)

    return labeled_out, numeric, threshold_table, numeric_cy


class HierarchicalHillClimbing:
    def __init__(
        self,
        nodes: list[str],
        n_restarts: int = N_RESTARTS,
        n_operations: int = N_OPERATIONS,
        midstream_max_parents: int | None = None,
        init_density: float = INIT_DENSITY,
        fixed_edges: list[tuple[str, str]] | None = None,
        searchable_children: set[str] | None = None,
        stage_name: str = "HC",
    ):
        self.nodes = list(nodes)
        self.n_restarts = n_restarts
        self.n_operations = n_operations
        self.midstream_max_parents = midstream_max_parents
        self.init_density = init_density
        self.fixed_edges = list(fixed_edges or [])
        self.searchable_children = searchable_children
        self.stage_name = stage_name
        self.best_edges: list[tuple[str, str]] = []
        self.best_bic = float("-inf")
        self.search_history: list[dict] = []
        self.bic_scorer = None
        self._bic_error_logged = False

    def _build_model(self, edges):
        model = BayesianNetworkModel(edges)
        for node in self.nodes:
            if node not in model.nodes():
                model.add_node(node)
        return model

    def _in_degree(self, edges, node: str) -> int:
        return sum(1 for _, v in edges if v == node)

    def _parent_limit_ok(self, edges, child: str) -> bool:
        if self.midstream_max_parents is None:
            return True
        if child in MIDSTREAM:
            return self._in_degree(edges, child) <= self.midstream_max_parents
        return True

    def has_cycle_after_operation(self, current_edges, operation) -> bool:
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(current_edges)
        op_type, edge = operation
        if op_type == "add":
            g.add_edge(*edge)
        elif op_type == "remove":
            if g.has_edge(*edge):
                g.remove_edge(*edge)
        elif op_type == "reverse":
            u, v = edge
            if g.has_edge(u, v):
                g.remove_edge(u, v)
            g.add_edge(v, u)
        return not nx.is_directed_acyclic_graph(g)

    def generate_hierarchical_dag(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        for u, v in self.fixed_edges:
            if u in self.nodes and v in self.nodes:
                g.add_edge(u, v)

        allowed = [
            (u, v)
            for u in self.nodes
            for v in self.nodes
            if is_allowed_edge(u, v)
            and (u, v) not in self.fixed_edges
            and (self.searchable_children is None or v in self.searchable_children)
        ]
        random.shuffle(allowed)
        for u, v in allowed:
            if random.random() >= self.init_density:
                continue
            if (
                self.midstream_max_parents is not None
                and v in MIDSTREAM
                and g.in_degree(v) >= self.midstream_max_parents
            ):
                continue
            g.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(g):
                g.remove_edge(u, v)
        return g

    def get_possible_operations(self, current_edges):
        edge_set = set(current_edges)
        fixed_set = set(self.fixed_edges)
        operations = []

        for u in self.nodes:
            for v in self.nodes:
                if (u, v) in edge_set:
                    continue
                if not is_allowed_edge(u, v):
                    continue
                if self.searchable_children is not None and v not in self.searchable_children:
                    continue
                trial = self.apply_operation(current_edges, ("add", (u, v)))
                if not self._parent_limit_ok(trial, v):
                    continue
                op = ("add", (u, v))
                if not self.has_cycle_after_operation(current_edges, op):
                    operations.append(op)

        for edge in current_edges:
            if edge in fixed_set:
                continue
            u, v = edge
            if self.searchable_children is not None and v not in self.searchable_children:
                continue
            operations.append(("remove", edge))

        for edge in current_edges:
            if edge in fixed_set:
                continue
            u, v = edge
            if self.searchable_children is not None and u not in self.searchable_children:
                continue
            if not is_allowed_edge(v, u):
                continue
            trial = self.apply_operation(current_edges, ("reverse", edge))
            if not self._parent_limit_ok(trial, u):
                continue
            op = ("reverse", edge)
            if not self.has_cycle_after_operation(current_edges, op):
                operations.append(op)

        return operations

    def apply_operation(self, edges, operation):
        op_type, edge = operation
        new_edges = list(edges)
        if op_type == "add":
            if edge not in new_edges:
                new_edges.append(edge)
        elif op_type == "remove":
            if edge in new_edges:
                new_edges.remove(edge)
        elif op_type == "reverse":
            u, v = edge
            if (u, v) in new_edges:
                new_edges.remove((u, v))
                new_edges.append((v, u))
        return new_edges

    def calculate_bic(self, edges) -> float:
        try:
            model = self._build_model(edges)
            return float(self.bic_scorer.score(model))
        except Exception as exc:
            if not self._bic_error_logged:
                print(f"BIC calculation failed: {exc}")
                self._bic_error_logged = True
            return float("-inf")

    def search(self, data_numeric: pd.DataFrame, show_progress: bool = True):
        self.bic_scorer = BicScorer(data_numeric[self.nodes].copy())

        best_edges: list[tuple[str, str]] = []
        best_bic = float("-inf")

        print("=" * 72)
        print(f"{self.stage_name}: hierarchical hill climbing")
        print(f"restarts={self.n_restarts}, operations/restart={self.n_operations}")
        print(
            "constraints: upstream->mid/down, mid->mid/down; "
            f"midstream max_parents={self.midstream_max_parents}; "
            f"learning_rows={len(data_numeric)}; fixed_edges={len(self.fixed_edges)}"
        )
        print("=" * 72)

        for restart in range(self.n_restarts):
            if show_progress and restart % 20 == 0:
                print(f"Progress: {restart}/{self.n_restarts}")

            dag = self.generate_hierarchical_dag()
            current_edges = list(dag.edges())
            current_bic = self.calculate_bic(current_edges)

            for _ in range(self.n_operations):
                ops = self.get_possible_operations(current_edges)
                if not ops:
                    break
                sampled = random.sample(ops, min(100, len(ops)))
                best_op = None
                best_op_bic = current_bic
                for op in sampled:
                    new_edges = self.apply_operation(current_edges, op)
                    new_bic = self.calculate_bic(new_edges)
                    if new_bic > best_op_bic:
                        best_op = op
                        best_op_bic = new_bic
                if best_op is not None and best_op_bic > current_bic:
                    current_edges = self.apply_operation(current_edges, best_op)
                    current_bic = best_op_bic

            self.search_history.append(
                {
                    "stage": self.stage_name,
                    "restart": restart,
                    "final_bic": current_bic,
                    "n_edges": len(current_edges),
                }
            )
            if current_bic > best_bic:
                best_bic = current_bic
                best_edges = current_edges
                if show_progress:
                    print(f"  Restart {restart}: better BIC={current_bic:.2f}, edges={len(best_edges)}")

        self.best_edges = best_edges
        self.best_bic = best_bic
        print(f"\n{self.stage_name} done. Best BIC={best_bic:.2f}, edges={len(best_edges)}")
        return self.best_edges


def analyze_edges(edges) -> pd.DataFrame:
    rows = []
    for u, v in edges:
        rows.append(
            {
                "From": u,
                "To": v,
                "From_level": get_level_name(u),
                "To_level": get_level_name(v),
                "Connection_type": f"{get_level_name(u)}→{get_level_name(v)}",
            }
        )
    return pd.DataFrame(rows)


def node_degree_table(edges) -> pd.DataFrame:
    g = nx.DiGraph()
    g.add_nodes_from(TARGET_COLUMNS)
    g.add_edges_from(edges)
    rows = []
    for node in TARGET_COLUMNS:
        rows.append(
            {
                "Node": node,
                "Short": SHORT_LABELS[node],
                "Level": get_level_name(node),
                "In_Degree": g.in_degree(node),
                "Out_Degree": g.out_degree(node),
                "Parents": ", ".join(sorted(g.predecessors(node))),
                "Children": ", ".join(sorted(g.successors(node))),
            }
        )
    return pd.DataFrame(rows)


def plot_network(edges, output_path: Path, title: str):
    g = nx.DiGraph()
    g.add_nodes_from(TARGET_COLUMNS)
    g.add_edges_from(edges)

    pos = {}
    for i, node in enumerate(UPSTREAM):
        pos[node] = (0.0, -i * 1.4)
    for i, node in enumerate(MIDSTREAM):
        pos[node] = (3.0, -i * 1.05)
    for i, node in enumerate(DOWNSTREAM):
        pos[node] = (6.0, -4.5 - i)

    colors = []
    for node in g.nodes():
        if node in UPSTREAM:
            colors.append("#4ECDC4")
        elif node in MIDSTREAM:
            colors.append("#FFA500")
        else:
            colors.append("#FF6B6B")

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=2200, edgecolors="black", linewidths=1.2)
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="#555555",
        arrows=True,
        arrowsize=18,
        arrowstyle="->",
        width=1.4,
        alpha=0.8,
        connectionstyle="arc3,rad=0.04",
    )
    labels = {n: SHORT_LABELS[n] for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=9, font_weight="bold")

    legend = [
        Patch(facecolor="#4ECDC4", label="Upstream"),
        Patch(facecolor="#FFA500", label="Midstream"),
        Patch(facecolor="#FF6B6B", label="Downstream (ESI)"),
    ]
    plt.legend(handles=legend, loc="upper right")
    plt.title(title, fontsize=14, pad=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def score_structure(edges, data_numeric: pd.DataFrame, columns: list[str]) -> float:
    scorer = BicScorer(data_numeric[columns])
    model = BayesianNetworkModel(edges)
    for node in columns:
        if node not in model.nodes():
            model.add_node(node)
    return float(scorer.score(model))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    raw = load_raw_data()
    tag = f"OilP_noGDPpc_R{N_RESTARTS}_O{N_OPERATIONS}"
    print(f"DATA: {DATA_FILE}  sheet=raw_data  shape={raw.shape}")
    print(f"FINAL: {tag}  (no GDPpc, OilP, IOC, seed={RANDOM_SEED})")

    labeled, numeric, thresholds, numeric_cy = prepare_discretized_data(raw)
    direction_df = build_direction_table()

    stage_a = HierarchicalHillClimbing(
        nodes=STAGE_A_NODES,
        n_restarts=N_RESTARTS,
        n_operations=N_OPERATIONS,
        midstream_max_parents=None,
        stage_name=f"Stage A {tag}",
    )
    edges_a = stage_a.search(numeric_cy[STAGE_A_NODES], show_progress=True)

    stage_b = HierarchicalHillClimbing(
        nodes=TARGET_COLUMNS,
        n_restarts=N_RESTARTS,
        n_operations=N_OPERATIONS,
        midstream_max_parents=None,
        fixed_edges=edges_a,
        searchable_children={"ESI"},
        stage_name=f"Stage B {tag}",
    )
    edges = stage_b.search(numeric[TARGET_COLUMNS], show_progress=True)
    edge_set = set(edges)
    for e in edges_a:
        if e not in edge_set:
            edges.append(e)
            edge_set.add(e)

    g = nx.DiGraph()
    g.add_nodes_from(TARGET_COLUMNS)
    g.add_edges_from(edges)
    direct = sorted(g.predecessors("ESI"))
    indirect = sorted(set(nx.ancestors(g, "ESI")) - set(direct))
    iso = [SHORT_LABELS[n] for n in TARGET_COLUMNS if g.in_degree(n) + g.out_degree(n) == 0]
    oil_edges = [(u, v) for u, v in edges if OIL_COL in (u, v)]
    has_oil_ioc = (OIL_COL, IOC_COL) in set(edges)
    ioc_esi = nx.has_path(g, IOC_COL, "ESI")
    oil_esi = nx.has_path(g, OIL_COL, "ESI")
    bic = score_structure(edges, numeric, TARGET_COLUMNS)

    summary = {
        "GDPpc": "无",
        "oil_short": OIL_SHORT,
        "n_restarts": N_RESTARTS,
        "n_operations": N_OPERATIONS,
        "n_nodes": len(TARGET_COLUMNS),
        "final_bic_on_papers": bic,
        "stage_a_bic": stage_a.best_bic,
        "stage_b_bic": stage_b.best_bic,
        "n_edges_total": len(edges),
        "孤立点": ", ".join(iso) if iso else "无",
        "Oil边": "; ".join(f"{SHORT_LABELS[u]}→{SHORT_LABELS[v]}" for u, v in oil_edges) or "无",
        "Oil→IoC": "是" if has_oil_ioc else "否",
        "IoC→…→ESI": "是" if ioc_esi else "否",
        "Oil→…→ESI": "是" if oil_esi else "否",
        "ESI直接": ", ".join(SHORT_LABELS[p] for p in direct),
        "间接": ", ".join(SHORT_LABELS[p] for p in indirect) if indirect else "(none)",
        "合计": len(direct) + len(indirect),
        "Oil→ESI路径": (
            " → ".join(SHORT_LABELS.get(x, x) for x in nx.shortest_path(g, OIL_COL, "ESI"))
            if oil_esi
            else ""
        ),
    }

    out_xlsx = OUTPUT_DIR / f"{tag}.xlsx"
    out_png = OUTPUT_DIR / f"{tag}.png"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        labeled.to_excel(writer, sheet_name="Discretized Data", index=False)
        thresholds.to_excel(writer, sheet_name="Discretization Thresholds", index=False)
        direction_df.to_excel(writer, sheet_name="Indicator Direction", index=False)
        analyze_edges(edges_a).to_excel(writer, sheet_name="Stage A Structure", index=False)
        analyze_edges(edges).to_excel(writer, sheet_name="Network Structure", index=False)
        node_degree_table(edges).to_excel(writer, sheet_name="Node Relations", index=False)
        pd.DataFrame(
            [{"relation": "direct_to_ESI", "short": SHORT_LABELS[p], "indicator": p} for p in direct]
            + [
                {"relation": "indirect_to_ESI", "short": SHORT_LABELS[p], "indicator": p}
                for p in indirect
            ]
        ).to_excel(writer, sheet_name="ESI Connections", index=False)
        pd.DataFrame(stage_a.search_history + stage_b.search_history).to_excel(
            writer, sheet_name="Search History", index=False
        )
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)

    plot_network(edges, out_png, f"OilP+IOC noGDPpc | R{N_RESTARTS} O{N_OPERATIONS}")
    print(f"{tag}: BIC={bic:.2f}, Oil->IoC={has_oil_ioc}, isolated={summary['孤立点']}")
    print(f"  Oil边: {summary['Oil边']}")
    print(f"  path: {summary['Oil→ESI路径']}")
    print(f"  ESI直接: {summary['ESI直接']}")
    print(f"  间接: {summary['间接']}")
    print(f"  saved {out_xlsx}")
    print(f"  saved {out_png}")


if __name__ == "__main__":
    main()
