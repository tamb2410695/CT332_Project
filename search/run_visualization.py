import argparse
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


root = Path(__file__).resolve().parents[1]
defaultInput = root / "results" / "benchmark.csv"
defaultOutput = root / "results" / "benchmark-figures"

sns.set_theme(style="whitegrid", font="DejaVu Sans")

requiredColumns = {
    "problem",
    "layout",
    "algorithm",
    "heuristic",
    "solved",
    "pathLength",
    "pathCost",
    "expandedNodes",
    "generatedNodes",
    "maximumFrontierSize",
    "runtime_ms",
}

policies = {
    ("dfs", "none"): ("DFS", "#64748b"),
    ("bfs", "none"): ("BFS", "#0891b2"),
    ("ucs", "none"): ("UCS", "#475569"),
    ("gbfs", "manhattan"): ("GBFS Manhattan", "#7c3aed"),
    ("gbfs", "euclidean"): ("GBFS Euclidean", "#c026d3"),
    ("astar", "null"): ("A* null", "#d97706"),
    ("astar", "manhattan"): ("A* Manhattan", "#16a34a"),
    ("astar", "euclidean"): ("A* Euclidean", "#059669"),
}

positionOrder = [
    "DFS",
    "BFS",
    "UCS",
    "GBFS Manhattan",
    "GBFS Euclidean",
    "A* null",
    "A* Manhattan",
    "A* Euclidean",
]
policyHatches = {
    "DFS": "///",
    "BFS": "\\\\\\",
    "UCS": "xxx",
    "GBFS Manhattan": "...",
    "GBFS Euclidean": "+++",
    "A* null": "ooo",
    "A* Manhattan": "***",
    "A* Euclidean": "---",
}
policyMarkers = {
    "DFS": "o",
    "BFS": "s",
    "UCS": "^",
    "GBFS Manhattan": "D",
    "GBFS Euclidean": "v",
    "A* null": "P",
    "A* Manhattan": "X",
    "A* Euclidean": "<",
}
policyColors = {label: color for label, color in policies.values()}
monochromeFill = "#f1f5f9"

layoutOrder = ["tinyMaze", "mediumMaze", "bigMaze", "customMaze"]
layoutMethodOrder = [
    "BFS",
    "A* null",
    "GBFS Manhattan",
    "GBFS Euclidean",
    "A* Manhattan",
    "A* Euclidean",
]
metrics = [
    ("pathCost", "Path cost"),
    ("expandedNodes", "Expanded nodes"),
    ("generatedNodes", "Generated frontier nodes"),
    ("maximumFrontierSize", "Maximum frontier size (entries; memory proxy)"),
    ("runtime_ms", "Median runtime (ms; machine-dependent)"),
]

def loadBenchmarks(csvPath: Path) -> pd.DataFrame:
    data = pd.read_csv(csvPath, keep_default_na=False)
    columnAliases = {
        "pathLength": "pathLength",
        "pathCost": "pathCost",
        "expandedNodes": "expandedNodes",
        "generatedNodes": "generatedNodes",
        "maximumFrontierSize": "maximumFrontierSize",
        "runIndex": "runIndex",
        "runsPerCase": "runsPerCase",
    }
    data = data.rename(columns=columnAliases)
    missing = requiredColumns - set(data.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Benchmark CSV is missing required columns: {names}")

    solved = data["solved"].astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
    data = data.loc[solved].copy()
    if data.empty:
        raise ValueError(f"No solved benchmark rows found in {csvPath}")

    hasRunIndex = "runIndex" in data.columns
    hasRunsPerCase = "runsPerCase" in data.columns
    if hasRunIndex != hasRunsPerCase:
        raise ValueError("Run-level CSV must include both runIndex and runsPerCase")
    hasRunLevelData = hasRunIndex and hasRunsPerCase

    numericColumns = [
        "pathLength", "pathCost", "expandedNodes", "generatedNodes",
        "maximumFrontierSize", "runtime_ms",
    ]
    if hasRunLevelData:
        numericColumns.extend(("runIndex", "runsPerCase"))
    for column in numericColumns:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not data[column].map(math.isfinite).all() or (data[column] < 0).any():
            raise ValueError(f"Invalid metric values in column {column!r}")
    integerColumns = [
        "pathLength", "pathCost", "expandedNodes", "generatedNodes",
        "maximumFrontierSize",
    ]
    if hasRunLevelData:
        integerColumns.extend(("runIndex", "runsPerCase"))
    if any((data[column] % 1 != 0).any() for column in integerColumns):
        raise ValueError("Node counts, path length and run indices must be integers")
    positionRows = data["problem"] == "PositionSearchProblem"
    if (data.loc[positionRows, "pathLength"] != data.loc[positionRows, "pathCost"]).any():
        raise ValueError(
            "Position benchmark expects unit step costs: pathCost must equal pathLength"
        )

    caseColumns = ["problem", "layout", "algorithm", "heuristic"]
    if hasRunLevelData:
        replicateKey = caseColumns + ["runIndex"]
        if data.duplicated(replicateKey).any():
            raise ValueError("Benchmark CSV contains duplicate runIndex values for a case")
        for case, group in data.groupby(caseColumns, sort=False):
            runCounts = group["runsPerCase"].unique()
            indices = sorted(group["runIndex"].astype(int).tolist())
            expectedCount = int(runCounts[0]) if len(runCounts) == 1 else -1
            if expectedCount < 1 or len(runCounts) != 1 or indices != list(range(1, expectedCount + 1)):
                raise ValueError(
                    f"Incomplete or inconsistent runs for {case}: "
                    f"found run indices {indices} with runsPerCase={runCounts.tolist()}"
                )
    elif data.duplicated(caseColumns).any():
        raise ValueError("Summary benchmark CSV contains duplicate rows for a case")

    data["heuristic"] = data["heuristic"].astype(str).str.strip()
    missingAStarHeuristic = (
        data["algorithm"].eq("astar")
        & data["heuristic"].str.lower().isin(("", "nan", "none"))
    )
    data.loc[missingAStarHeuristic, "heuristic"] = "null"

    data["policy"] = [
        policies.get(
            (algorithm, heuristic),
            (f"{algorithm} {heuristic}", "#334155"),
        )[0]
        for algorithm, heuristic in zip(data["algorithm"], data["heuristic"])
    ]
    data["color"] = [
        policies.get((algorithm, heuristic), ("", "#334155"))[1]
        for algorithm, heuristic in zip(data["algorithm"], data["heuristic"])
    ]
    extraPolicies = [
        policy for policy in data["policy"].unique() if policy not in policyColors
    ]
    extraColors = sns.color_palette("colorblind", n_colors=max(1, len(extraPolicies)))
    for policy, color in zip(extraPolicies, extraColors):
        policyColors[policy] = matplotlib.colors.to_hex(color)
        policyHatches.setdefault(policy, ("///", "xxx", "...", "ooo", "+++", "\\\\\\")[(len(policyHatches)) % 6])
        policyMarkers.setdefault(policy, ("D", "v", "P", "X", "<", ">")[(len(policyMarkers)) % 6])
    data.attrs["hasRunLevelData"] = hasRunLevelData
    return data


def summarizeBenchmarks(data: pd.DataFrame) -> pd.DataFrame:
    groupingColumns = ["problem", "layout", "algorithm", "heuristic", "policy", "color"]
    metricColumns = [
        "pathCost", "expandedNodes", "generatedNodes",
        "maximumFrontierSize", "runtime_ms",
    ]
    summary = data.groupby(groupingColumns, as_index=False, sort=False)[metricColumns].median()
    summary["solved"] = True
    return summary


def requireRows(data: pd.DataFrame, problemName: str, layoutName: str, expectedRows: int) -> pd.DataFrame:
    selected = data.loc[(data["problem"] == problemName) & (data["layout"] == layoutName)].copy()
    if len(selected) != expectedRows:
        raise ValueError(
            f"Expected {expectedRows} solved rows for {problemName}/{layoutName}; found {len(selected)}"
        )
    if selected["policy"].duplicated().any():
        raise ValueError(
            f"Duplicate algorithm/heuristic rows for {problemName}/{layoutName}"
        )
    return selected


def metricLabel(metric: str) -> str:
    return {
        "pathCost": "Path cost",
        "expandedNodes": "Number of expanded nodes",
        "generatedNodes": "Generated frontier nodes",
        "maximumFrontierSize": "Maximum frontier size (entries)",
        "runtime_ms": "Median runtime (ms)",
    }[metric]


def policyPalette(policyNames: list[str], visualStyle: str) -> dict[str, str]:
    if visualStyle == "pattern":
        return {policy: monochromeFill for policy in policyNames}
    return {policy: policyColors.get(policy, "#334155") for policy in policyNames}


def policyHatch(policyName: str, visualStyle: str) -> str:
    return policyHatches.get(policyName, "///") if visualStyle in {"pattern", "both"} else ""


def policyMarker(policyName: str, visualStyle: str) -> str:
    return policyMarkers.get(policyName, "o") if visualStyle in {"pattern", "both"} else "o"


def policyLegendHandles(
    policyNames: list[str], palette: dict[str, str], visualStyle: str
):
    return [
        Patch(
            facecolor=palette[policy], edgecolor="#1f2937",
            hatch=policyHatch(policy, visualStyle), label=policy,
        )
        for policy in policyNames
    ]


def styleAxes(axis, metric: str, *, orientation: str = "horizontal") -> None:
    if orientation == "horizontal":
        axis.set_xlabel(metricLabel(metric))
        axis.set_ylabel("")
        axis.grid(axis="x", color="#dbe3ed", linewidth=0.8)
    else:
        axis.set_xlabel("Maze layout")
        axis.set_ylabel(metricLabel(metric))
        axis.grid(axis="y", color="#dbe3ed", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def annotateHorizontalBars(axis, metric: str) -> None:
    labels = [
        f"{bar.get_width():.3f}" if metric == "runtime_ms" else f"{bar.get_width():g}"
        for bar in axis.patches
    ]
    axis.bar_label(axis.containers[0], labels=labels, padding=4, fontsize=9)
    for bar in axis.patches:
        if bar.get_width() == 0:
            axis.annotate(
                "0",
                (0, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
    axis.margins(x=0.16)


def makePolicyFigure(data: pd.DataFrame, *, problemName: str, layoutName: str,
                       policyOrder: list[str], chartTitle: str, fileStem: str,
                       note: str, outputDir: Path, visualStyle: str) -> None:
    selectedData = requireRows(data, problemName, layoutName, len(policyOrder))
    policyRank = {policy: rank for rank, policy in enumerate(policyOrder)}
    selectedData["policyRank"] = selectedData["policy"].map(policyRank)
    selectedData = selectedData.sort_values("policyRank")

    for metric, heading in metrics:
        figure, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
        sns.barplot(
            data=selectedData,
            x=metric,
            y="policy",
            order=policyOrder,
            color="#64748b",
            errorbar=None,
            ax=axis,
        )
        for bar, color, policy in zip(
            axis.patches, selectedData["color"], selectedData["policy"]
        ):
            bar.set_facecolor(
                monochromeFill if visualStyle == "pattern" else color
            )
            bar.set_edgecolor("#1f2937")
            bar.set_linewidth(0.65)
            bar.set_hatch(policyHatch(policy, visualStyle))
        styleAxes(axis, metric)
        annotateHorizontalBars(axis, metric)
        figure.suptitle(f"{chartTitle} — {heading}", fontsize=16, fontweight="bold")
        metricNote = {
            "generatedNodes": (
                "Counts frontier insertions; duplicate suppression/relaxation rules "
                "differ by algorithm."
            ),
            "maximumFrontierSize": (
                "Counts frontier entries, not bytes of RAM."
            ),
        }.get(metric, note)
        figure.text(0.02, -0.015, metricNote, fontsize=9, color="#475569")
        saveFigure(figure, outputDir / f"{fileStem}-{metric}")


def makeMazeFigures(data: pd.DataFrame, outputDir: Path, visualStyle: str):
    """Draw every solved problem/layout pair present in the input CSV."""
    chartSpecs = []
    preferredRank = {policy: rank for rank, policy in enumerate(positionOrder)}
    mazeGroups = data.groupby(["problem", "layout"], sort=True)

    for (problemName, layoutName), mazeData in mazeGroups:
        policyNames = sorted(
            mazeData["policy"].unique().tolist(),
            key=lambda policy: (preferredRank.get(policy, len(preferredRank)), policy),
        )
        selectedData = mazeData.copy()
        selectedData["policy"] = pd.Categorical(
            selectedData["policy"], policyNames, ordered=True
        )
        selectedData = selectedData.sort_values("policy")
        safeProblem = re.sub(r"[^A-Za-z0-9]+", "-", str(problemName)).strip("-").lower()
        safeLayout = re.sub(r"[^A-Za-z0-9]+", "-", str(layoutName)).strip("-").lower()
        fileStem = f"maze-{safeProblem}-{safeLayout}"

        for metric, heading in metrics:
            figure, axis = plt.subplots(figsize=(9, max(4.2, 0.55 * len(policyNames))),
                                        constrained_layout=True)
            sns.barplot(
                data=selectedData,
                x=metric,
                y="policy",
                order=policyNames,
                color="#64748b",
                errorbar=None,
                ax=axis,
            )
            for bar, policy in zip(axis.patches, policyNames):
                color = policyColors.get(policy, "#334155")
                bar.set_facecolor(monochromeFill if visualStyle == "pattern" else color)
                bar.set_edgecolor("#1f2937")
                bar.set_linewidth(0.65)
                bar.set_hatch(policyHatch(policy, visualStyle))
            styleAxes(axis, metric)
            annotateHorizontalBars(axis, metric)
            figure.suptitle(
                f"{problemName} — {layoutName}: {heading}",
                fontsize=15,
                fontweight="bold",
            )
            figure.text(
                0.02,
                -0.015,
                "Bars show recorded values; run-level inputs are summarized by median.",
                fontsize=9,
                color="#475569",
            )
            saveFigure(figure, outputDir / f"{fileStem}-{metric}")
            chartSpecs.append((
                f"{fileStem}-{metric}",
                f"{problemName} — {layoutName}: {heading}",
                "One bar per algorithm/heuristic configuration in the source CSV.",
            ))
    return chartSpecs


def makeLayoutFigures(data: pd.DataFrame, outputDir: Path, visualStyle: str) -> None:
    selectedData = data.loc[
        (data["problem"] == "PositionSearchProblem")
        & data["layout"].isin(layoutOrder)
        & data["policy"].isin(layoutMethodOrder)
    ].copy()
    expectedRows = len(layoutOrder) * len(layoutMethodOrder)
    if len(selectedData) != expectedRows or selectedData.duplicated(["layout", "policy"]).any():
        raise ValueError(
            "Layout chart requires one BFS, A* null, GBFS Manhattan/Euclidean and "
            "A* Manhattan/Euclidean row for each selected layout"
        )

    selectedData["layout"] = pd.Categorical(selectedData["layout"], layoutOrder, ordered=True)
    selectedData["policy"] = pd.Categorical(
        selectedData["policy"], layoutMethodOrder, ordered=True
    )
    selectedData = selectedData.sort_values(["layout", "policy"])
    palette = policyPalette(layoutMethodOrder, visualStyle)

    for metric, heading in metrics:
        figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
        sns.barplot(
            data=selectedData,
            x="layout",
            y=metric,
            hue="policy",
            order=layoutOrder,
            hue_order=layoutMethodOrder,
            palette=palette,
            errorbar=None,
            ax=axis,
        )
        for policy, container in zip(layoutMethodOrder, axis.containers):
            for bar in container:
                bar.set_edgecolor("#1f2937")
                bar.set_linewidth(0.55)
                bar.set_hatch(policyHatch(policy, visualStyle))
        styleAxes(axis, metric, orientation="vertical")
        axis.tick_params(axis="x", rotation=0)
        axis.legend(
            handles=policyLegendHandles(layoutMethodOrder, palette, visualStyle),
            title=f"Algorithm ({visualStyle})", frameon=False, ncol=3,
            loc="upper center", bbox_to_anchor=(0.5, 1.12),
        )
        annotateGroupedBars(axis, metric)
        figure.suptitle(f"Position search across layouts — {heading}", fontsize=16,
                        fontweight="bold")
        metricNote = {
            "runtime_ms": "Runtime is the recorded median and depends on the machine.",
            "generatedNodes": (
                "Counts frontier insertions; duplicate suppression/relaxation rules "
                "differ by algorithm."
            ),
            "maximumFrontierSize": (
                "Counts frontier entries, not bytes of RAM."
            ),
        }.get(metric, "Values are medians across independent runs.")
        figure.text(
            0.02,
            -0.015,
            metricNote,
            fontsize=9,
            color="#475569",
        )
        saveFigure(figure, outputDir / f"position-layouts-{metric}")


def makeTradeoffFigure(data: pd.DataFrame, outputDir: Path, visualStyle: str) -> None:
    selectedData = requireRows(data, "PositionSearchProblem", "mediumMaze", len(positionOrder))

    points = []
    for (pathCost, expandedNodes), group in selectedData.groupby(
        ["pathCost", "expandedNodes"], sort=True
    ):
        policyNames = sorted(group["policy"].tolist(), key=positionOrder.index)
        points.append({
            "pathCost": pathCost,
            "expandedNodes": expandedNodes,
            "label": " / ".join(policyNames),
            "color": (
                monochromeFill if visualStyle == "pattern"
                else group.iloc[0]["color"]
            ),
            "policies": policyNames,
        })

    frontier = []
    for point in points:
        isDominated = any(
            other["pathCost"] <= point["pathCost"]
            and other["expandedNodes"] <= point["expandedNodes"]
            and (other["pathCost"] < point["pathCost"]
                 or other["expandedNodes"] < point["expandedNodes"])
            for other in points
        )
        if not isDominated:
            frontier.append(point)
    frontier.sort(key=lambda point: point["pathCost"])

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for point in points:
        markerPolicy = point["policies"][0]
        axis.scatter(
            point["pathCost"], point["expandedNodes"],
            s=105, color=point["color"], marker=policyMarker(markerPolicy, visualStyle),
            edgecolor="#1f2937", linewidth=0.8,
            zorder=3,
        )
        offsets = {
            "DFS": (8, 8),
            "BFS / UCS / A* null": (8, 7),
            "GBFS Manhattan": (8, 8),
            "GBFS Euclidean": (8, 8),
            "A* Manhattan": (8, -17),
            "A* Euclidean": (8, -2),
        }
        axis.annotate(
            point["label"],
            (point["pathCost"], point["expandedNodes"]),
            xytext=offsets.get(point["label"], (8, 8)),
            textcoords="offset points", fontsize=9,
        )

    if len(frontier) > 1:
        axis.plot(
            [point["pathCost"] for point in frontier],
            [point["expandedNodes"] for point in frontier],
            linestyle="--", color="#334155", linewidth=1.3, zorder=2,
            label="Pareto frontier",
        )
        axis.legend(frameon=False, loc="upper right")

    axis.set_xlabel("Path cost (steps; lower is better)")
    axis.set_ylabel("Expanded nodes (lower is better)")
    axis.set_title(
        "mediumMaze: solution quality vs search effort",
        fontsize=16, fontweight="bold",
    )
    axis.grid(color="#dbe3ed", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.text(
        0.02, -0.015,
        "Each point is one policy; overlapping BFS, UCS and A* null results are grouped. "
        "Lower-left is preferable; the dashed line marks nondominated measured choices.",
        fontsize=9, color="#475569",
    )
    saveFigure(figure, outputDir / "position-medium-tradeoff")


def makeRuntimeDistributionFigure(
    data: pd.DataFrame, outputDir: Path, visualStyle: str
) -> None:
    if "runIndex" not in data.columns:
        raise ValueError(
            "Runtime distribution needs run-level CSV data with runIndex and runsPerCase"
        )
    selectedData = data.loc[
        (data["problem"] == "PositionSearchProblem")
        & (data["layout"] == "mediumMaze")
    ].copy()
    expectedPolicies = len(positionOrder)
    if selectedData["policy"].nunique() != expectedPolicies:
        raise ValueError(
            f"Runtime distribution chart requires {expectedPolicies} mediumMaze policies"
        )
    observations = selectedData.groupby("policy").size()
    if observations.nunique() != 1 or observations.iloc[0] < 2:
        raise ValueError("Runtime distribution chart requires at least two raw runs per policy")
    selectedData["policy"] = pd.Categorical(
        selectedData["policy"], positionOrder, ordered=True
    )
    selectedData = selectedData.sort_values("policy")
    palette = policyPalette(positionOrder, visualStyle)

    figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    sns.boxplot(
        data=selectedData, x="runtime_ms", y="policy", order=positionOrder,
        hue="policy", hue_order=positionOrder, palette=palette,
        legend=False, showfliers=False, width=0.62, ax=axis,
    )
    for patch, policy in zip(axis.patches, positionOrder):
        patch.set_edgecolor("#1f2937")
        patch.set_linewidth(0.65)
        patch.set_hatch(policyHatch(policy, visualStyle))

    verticalJitter = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    for policyIndex, policy in enumerate(positionOrder):
        policyRuns = selectedData.loc[
            selectedData["policy"] == policy
        ].sort_values("runIndex")
        offsets = [verticalJitter[i % len(verticalJitter)]
                   for i in range(len(policyRuns))]
        axis.scatter(
            policyRuns["runtime_ms"],
            [policyIndex + offset for offset in offsets],
            marker=policyMarker(policy, visualStyle), s=38,
            color=("#111827" if visualStyle == "pattern" else policyColors[policy]),
            edgecolor="white", linewidth=0.35, alpha=0.9, zorder=3,
        )
    axis.set_xlabel("Runtime per run (ms; machine-dependent)")
    axis.set_ylabel("")
    axis.set_title(
        "mediumMaze: runtime variation across independent runs",
        fontsize=16, fontweight="bold",
    )
    axis.grid(axis="x", color="#dbe3ed", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.text(
        0.02, -0.015,
        f"Each dot is one of {int(observations.iloc[0])} measured runs; boxes show the "
        "interquartile range and median. Runtime varies with machine load.",
        fontsize=9, color="#475569",
    )
    saveFigure(figure, outputDir / "position-medium-runtime-distribution")


def annotateGroupedBars(axis, metric: str) -> None:
    for container in axis.containers:
        labels = [
            f"{bar.get_height():.2f}" if metric == "runtime_ms" else f"{bar.get_height():g}"
            for bar in container
        ]
        axis.bar_label(container, labels=labels, padding=2, fontsize=8)
        for bar in container:
            if bar.get_height() == 0:
                axis.annotate(
                    "0",
                    (bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    axis.margins(y=0.18)


def saveFigure(figure, pathWithoutExtension: Path) -> None:
    figure.savefig(
        pathWithoutExtension.with_suffix(".png"), dpi=180, bbox_inches="tight"
    )
    figure.savefig(pathWithoutExtension.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def writeDashboard(
    outputDir: Path, inputPath: Path, chartSpecs: list[tuple[str, str, str]]
) -> None:
    cards = []
    for fileStem, chartTitle, chartDescription in chartSpecs:
        pngName = f"{fileStem}.png"
        svgName = f"{fileStem}.svg"
        cards.append(
            f'<section><h2>{chartTitle}</h2><p>{chartDescription}</p>'
            f'<a href="{svgName}"><img src="{pngName}" alt="{chartTitle}"></a>'
            f'<p><a href="{svgName}">Open vector SVG</a></p></section>'
        )
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pacman Search benchmark charts</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#172033;background:#f8fafc}
.source{color:#526174} section{background:#fff;border:1px solid #dbe3ed;border-radius:12px;padding:20px;margin:24px 0}
img{display:block;width:100%;height:auto} h2{margin:0 0 6px} p{line-height:1.5}
</style></head><body>
<h1>Pacman Search benchmark charts</h1>
<p class="source">Source CSV: INPUT_PATH. Charts use solved rows; searches are not rerun.</p>
CARDS</body></html>
""".replace("INPUT_PATH", str(inputPath)).replace("CARDS", "\n".join(cards))
    (outputDir / "index.html").write_text(page, encoding="utf-8")


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=defaultInput, help="benchmark CSV path")
    parser.add_argument("--output-dir", dest="outputDir", type=Path, default=defaultOutput,
                        help="directory for generated figures and dashboard")
    parser.add_argument(
        "--charts",
        nargs="+",
        choices=("all", "mazes", "medium", "layouts", "tradeoff", "runtime-distribution"),
        default=["all"],
        metavar="CHART",
        help=(
            "chart groups to generate: mazes (every problem/layout in the CSV), "
            "medium (five mediumMaze metric bars), layouts "
            "(five layout metric comparisons), tradeoff, runtime-distribution; "
            "choose one or more, or all (default; selects charts supported by the input)"
        ),
    )
    parser.add_argument(
        "--style",
        choices=("pattern", "color", "both"),
        default="pattern",
        help=(
            "visual encoding: monochrome patterns, color only, or color plus patterns "
            "(default: pattern)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    rawData = loadBenchmarks(args.input)
    hasRunLevelData = rawData.attrs.get("hasRunLevelData", False)
    data = summarizeBenchmarks(rawData)
    sns.set_theme(
        style="whitegrid", context="notebook", palette="colorblind", font="DejaVu Sans"
    )
    args.outputDir.mkdir(parents=True, exist_ok=True)

    selectedCharts = set(args.charts)
    if "all" in selectedCharts:
        mediumData = data.loc[
            (data["problem"] == "PositionSearchProblem")
            & (data["layout"] == "mediumMaze")
        ]
        positionLayouts = set(
            data.loc[data["problem"] == "PositionSearchProblem", "layout"]
        )
        hasStandardPositionCases = (
            hasRunLevelData
            and set(mediumData["policy"]) == set(positionOrder)
            and set(layoutOrder).issubset(positionLayouts)
        )
        if hasStandardPositionCases:
            selectedCharts = {"medium", "layouts", "tradeoff"}
            mediumRunCounts = rawData.loc[
                (rawData["problem"] == "PositionSearchProblem")
                & (rawData["layout"] == "mediumMaze")
            ].groupby("policy").size()
            if len(mediumRunCounts) == len(positionOrder) and mediumRunCounts.min() >= 2:
                selectedCharts.add("runtime-distribution")
        else:
            selectedCharts = {"mazes"}

    mazeChartSpecs = []
    if "mazes" in selectedCharts:
        mazeChartSpecs = makeMazeFigures(data, args.outputDir, args.style)

    if "medium" in selectedCharts:
        makePolicyFigure(
            data,
            problemName="PositionSearchProblem",
            layoutName="mediumMaze",
            policyOrder=positionOrder,
            chartTitle="mediumMaze: path quality, search work and runtime",
            fileStem="position-medium",
            note="Unit-cost navigation. Runtime is the recorded median and depends on the machine.",
            outputDir=args.outputDir,
            visualStyle=args.style,
        )
    if "layouts" in selectedCharts:
        makeLayoutFigures(data, args.outputDir, args.style)
    if "tradeoff" in selectedCharts:
        makeTradeoffFigure(data, args.outputDir, args.style)
    if "runtime-distribution" in selectedCharts:
        makeRuntimeDistributionFigure(rawData, args.outputDir, args.style)

    metricDescriptions = dict(metrics)
    chartSpecs = mazeChartSpecs.copy()
    comparisonSpecs = []
    if "medium" in selectedCharts:
        comparisonSpecs.append((
            "position-medium",
            "mediumMaze",
            "Eight policies compare uninformed search, Euclidean/Manhattan GBFS and A*.",
        ))
    if "layouts" in selectedCharts:
        comparisonSpecs.append((
            "position-layouts",
            "Position search across layouts",
            "BFS and A* null baselines plus Manhattan/Euclidean GBFS and A* over four layouts.",
        ))
    for fileStem, comparison, chartDescription in comparisonSpecs:
        for metric, metricTitle in metricDescriptions.items():
            chartSpecs.append(
                (
                    f"{fileStem}-{metric}",
                    f"{comparison}: {metricTitle}",
                    chartDescription,
                )
            )
    if "tradeoff" in selectedCharts:
        chartSpecs.append((
            "position-medium-tradeoff",
            "mediumMaze: path cost vs expanded nodes",
            "A scatter plot compares solution quality and search effort; the dashed line "
            "shows the Pareto frontier among measured policies.",
        ))
    if "runtime-distribution" in selectedCharts:
        chartSpecs.append((
            "position-medium-runtime-distribution",
            "mediumMaze: runtime distribution across runs",
            "Each dot is one measured run; box plots summarize the median and interquartile range.",
        ))
    writeDashboard(args.outputDir, args.input, chartSpecs)
    print(
        f"Read {len(rawData)} solved benchmark rows across "
        f"{len(data)} benchmark cases from {args.input}"
    )
    print(f"Selected chart groups: {', '.join(sorted(selectedCharts))}")
    print(f"Visual style: {args.style}")
    print(f"Wrote {len(chartSpecs)} separate PNG/SVG figures and index.html to {args.outputDir}")

if __name__ == "__main__":
    main()
