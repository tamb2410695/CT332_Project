
import argparse
import csv
import statistics
import sys
import time
import math
from dataclasses import asdict, dataclass
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))


import layout
import pacman
import search
import searchAgents

@dataclass(frozen=True)
class BenchmarkRow:
    problem: str
    layout: str
    algorithm: str
    heuristic: str
    solved: bool
    pathLength: int
    pathCost: float
    expandedNodes: int
    generatedNodes: int
    maximumFrontierSize: int
    runtime_ms: float
    runIndex: int
    runsPerCase: int


positionCases = (
    ("dfs", "none", search.depthFirstSearch, None),
    ("bfs", "none", search.breadthFirstSearch, None),
    ("ucs", "none", search.uniformCostSearch, None),
    ("gbfs", "manhattan", search.greedyBestFirstSearch, searchAgents.manhattanHeuristic),
    ("gbfs", "euclidean", search.greedyBestFirstSearch, searchAgents.euclideanHeuristic),
    ("astar", "null", search.aStarSearch, search.nullHeuristic),
    ("astar", "euclidean", search.aStarSearch, searchAgents.euclideanHeuristic),
    ("astar", "manhattan", search.aStarSearch, searchAgents.manhattanHeuristic),
)

def loadGameState(layoutName: str):
    selectedLayout = layout.getLayout(layoutName)
    if selectedLayout is None:
        raise ValueError(f"Unknown layout: {layoutName}")
    state = pacman.GameState()
    state.initialize(selectedLayout, 0)
    return state


def validateResult(problem, actions, metrics):
    state = problem.getStartState()
    pathCost = 0
    for action in actions:
        matches = [(successor, stepCost)
                   for successor, candidate, stepCost in problem.getSuccessors(state)
                   if candidate == action]
        if len(matches) != 1:
            raise ValueError('Benchmark returned an illegal or ambiguous action: %r' % action)
        state, stepCost = matches[0]
        pathCost += stepCost
    if bool(problem.isGoalState(state)) != metrics.solved:
        raise ValueError('Benchmark solved flag disagrees with the returned path')
    if not metrics.solved and actions:
        raise ValueError('Failed search must return an empty action list')
    if not math.isclose(problem.getCostOfActions(actions), pathCost,
                        rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('Benchmark path cost disagrees with successor costs')


def executeCaseRuns(problemFactory, algorithm, heuristic, runs):
    """Return one validated benchmark observation for each independent run."""
    if runs < 1:
        raise ValueError('runs must be at least 1')
    observations = []
    for runIndex in range(1, runs + 1):
        problem = problemFactory()
        started = time.perf_counter_ns()
        if heuristic is None:
            actions = algorithm(problem)
        else:
            actions = algorithm(problem, heuristic)
        elapsedMs = (time.perf_counter_ns() - started) / 1_000_000
        metrics = search.getLastSearchMetrics()
        # Use a separate instance: getSuccessors used for replay increments _expanded.
        validateResult(problemFactory(), actions, metrics)
        observations.append({
            "solved": metrics.solved,
            "pathLength": len(actions),
            "pathCost": problem.getCostOfActions(actions),
            "expandedNodes": getattr(problem, "_expanded", metrics.expandedNodes),
            "generatedNodes": metrics.generatedNodes,
            "maximumFrontierSize": metrics.maximumFrontierSize,
            "runtime_ms": round(elapsedMs, 6),
            "runIndex": runIndex,
            "runsPerCase": runs,
        })
    return observations

def executeCase(problemFactory, algorithm, heuristic, runs):
    observations = executeCaseRuns(problemFactory, algorithm, heuristic, runs)
    summary = observations[-1].copy()
    summary["runtime_ms"] = round(
        statistics.median(row["runtime_ms"] for row in observations), 6
    )
    return summary


def positionRows(algorithmNames, layoutNames, runs):
    for layoutName in layoutNames:
        for algorithmName, heuristicName, algorithm, heuristic in positionCases:
            if algorithmName not in algorithmNames:
                continue
            def factory(name=layoutName):
                return searchAgents.PositionSearchProblem(
                    loadGameState(name), warn=False, visualize=False
                )

            observations = executeCaseRuns(factory, algorithm, heuristic, runs)
            for values in observations:
                yield BenchmarkRow(
                    "PositionSearchProblem",
                    layoutName,
                    algorithmName,
                    heuristicName,
                    **values,
                )

def parseArgs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r", "--runs", type=int, default=5,
        help="number of measurements per case; CSV stores their median",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=projectRoot / "results" / "benchmark.csv",
        help="path to the output CSV file (default: results/benchmark.csv)",
    )
    parser.add_argument(
        "-s", "--search-algorithms", dest="searchAlgorithms",
        nargs="+",
        choices=["dfs", "bfs", "ucs", "gbfs", "astar"],
        default=["dfs", "bfs", "ucs", "astar", "gbfs"],
        help="search algorithms to benchmark (default: all)",
    )
    parser.add_argument(
        "-l", "--position-layouts", dest="positionLayouts",
        nargs="+",
        choices=["tinyMaze", "mediumMaze", "bigMaze", "customMaze", "reportExampleMaze"],
        default=["tinyMaze", "mediumMaze", "bigMaze", "customMaze", "reportExampleMaze"],
        help="maze layouts to benchmark (default: all)",
    )
    return parser.parse_args()


def main():
    args = parseArgs()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    rows = list(positionRows(args.searchAlgorithms, args.positionLayouts, args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as outputFile:
        writer = csv.DictWriter(
            outputFile, fieldnames=list(asdict(rows[0]).keys())
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    print(f"Wrote {len(rows)} measured rows to {args.output}")

if __name__ == "__main__":
    main()
