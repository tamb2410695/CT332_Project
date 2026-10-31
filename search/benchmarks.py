import argparse
import csv
import statistics
import sys
import time
import math
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    pathCost: int
    expandedNodes: int
    generatedNodes: int
    maximumFrontierSize: int
    runtime_ms: float
    runs: int


POSITION_CASES = (
    ("dfs", "none", search.depthFirstSearch, None),
    ("bfs", "none", search.breadthFirstSearch, None),
    ("ucs", "none", search.uniformCostSearch, None),
    ("gbfs", "manhattan", search.greedyBestFirstSearch, searchAgents.manhattanHeuristic),
    ("astar", "null", search.aStarSearch, search.nullHeuristic),
    ("astar", "euclidean", search.aStarSearch, search.nullHeuristic),
    (
        "astar",
        "manhattan",
        search.aStarSearch,
        searchAgents.manhattanHeuristic,
    ),
)


def load_game_state(layout_name: str):
    selected_layout = layout.getLayout(layout_name)
    if selected_layout is None:
        raise ValueError(f"Unknown layout: {layout_name}")
    state = pacman.GameState()
    state.initialize(selected_layout, 0)
    return state


def validate_result(problem, actions, metrics):
    state = problem.getStartState()
    cost = 0
    for action in actions:
        matches = [(successor, step_cost)
                   for successor, candidate, step_cost in problem.getSuccessors(state)
                   if candidate == action]
        if len(matches) != 1:
            raise ValueError('Benchmark returned an illegal or ambiguous action: %r' % action)
        state, step_cost = matches[0]
        cost += step_cost
    if bool(problem.isGoalState(state)) != metrics.solved:
        raise ValueError('Benchmark solved flag disagrees with the returned path')
    if not metrics.solved and actions:
        raise ValueError('Failed search must return an empty action list')
    if not math.isclose(problem.getCostOfActions(actions), cost,
                        rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('Benchmark path cost disagrees with successor costs')


def execute_case(problem_factory, algorithm, heuristic, runs):
    if runs < 1:
        raise ValueError('runs must be at least 1')
    timings = []
    final = None
    for _ in range(runs):
        problem = problem_factory()
        started = time.perf_counter_ns()
        print(algorithm, heuristic)
        if heuristic is None:
            actions = algorithm(problem)
        else:
            actions = algorithm(problem, heuristic)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        timings.append(elapsed_ms)
        metrics = search.getLastSearchMetrics()
        validate_result(problem_factory(), actions, metrics)
        final = (problem, actions, metrics)

    problem, actions, metrics = final
    return {
        "solved": metrics.solved,
        "pathLength": len(actions),
        "pathCost": problem.getCostOfActions(actions),
        "expandedNodes": getattr(problem, "_expanded", metrics.expandedNodes),
        "generatedNodes": metrics.generatedNodes,
        "maximumFrontierSize": metrics.maximumFrontierSize,
        "runtime_ms": round(statistics.median(timings), 6),
    }

def position_rows(algorithm_names, layout_names, runs):
    for layout_name in layout_names:
        for algorithm_name, heuristic_name, algorithm, heuristic in POSITION_CASES:
            if algorithm_name not in algorithm_names:
                continue
            def factory(name=layout_name):
                return searchAgents.PositionSearchProblem(
                    load_game_state(name), warn=False, visualize=False
                )

            values = execute_case(factory, algorithm, heuristic, runs)
            yield BenchmarkRow(
                "PositionSearchProblem",
                layout_name,
                algorithm_name,
                heuristic_name,
                **values,
                runs=runs,
            )

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r", "--runs", type=int, default=5,
        help="number of measurements per case; CSV stores their median",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "benchmark.csv",
        help="path to the output CSV file (default: results/benchmark.csv)",
    )
    parser.add_argument(
        "-s", "--search-algorithms",
        nargs="+",
        choices=["dfs", "bfs", "ucs", "gbfs", "astar"],
        default=["dfs", "bfs", "ucs", "astar", "gbfs"],
        help="search algorithms to benchmark (default: all)",
    )
    parser.add_argument(
        "-l", "--position-layouts",
        nargs="+",
        choices=["tinyMaze", "mediumMaze", "bigMaze", "customMaze"],
        default=["tinyMaze", "mediumMaze", "bigMaze", "customMaze"],
        help="maze layouts to benchmark (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    rows = list(position_rows(args.search_algorithms, args.position_layouts, args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(asdict(rows[0]).keys())
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    print(f"Wrote {len(rows)} measured rows to {args.output}")

if __name__ == "__main__":
    main()