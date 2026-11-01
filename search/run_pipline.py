import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
benchmarkScript = projectRoot / "search" / "run_benchmarks.py"
visualizationScript = projectRoot / "search" / "run_visualization.py"
defaultOutput = projectRoot / "results" / "benchmark.csv"
mplConfigDir = Path(tempfile.gettempdir()) / "pacman-benchmark-mpl"


def resolveProjectPath(path: Path) -> Path:
    return path if path.is_absolute() else projectRoot / path


def childEnvironment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(mplConfigDir))
    return env


def visualizationDependenciesAvailable() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import matplotlib, pandas, seaborn"],
        cwd=projectRoot,
        env=childEnvironment(),
        check=False,
    )
    if result.returncode:
        print(
            "Visualization dependencies are missing or could not be imported. "
            "Install them with: python3 -m pip install "
            "-r benchmark/requirements-visualization.txt",
            file=sys.stderr,
        )
        return False
    return True


def runStep(label: str, command: list[str]) -> int:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=projectRoot / "search" if label == "Benchmark" else projectRoot,
        env=childEnvironment(), check=False
    )
    if result.returncode:
        print(f"{label} failed with exit code {result.returncode}.", file=sys.stderr)
    return result.returncode


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="repeat each case this many times; CSV stores one row per run (default: 7)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaultOutput,
        help="benchmark CSV path (relative paths use the repository root)",
    )
    parser.add_argument(
        "--figures-dir", dest="figuresDir",
        type=Path,
        help="chart output directory (default: sibling <csv-name>-figures directory)",
    )
    parser.add_argument(
        "--charts",
        nargs="+",
        choices=("all", "mazes", "medium", "layouts", "tradeoff", "runtime-distribution"),
        default=["all"],
        metavar="CHART",
        help="chart groups to generate (default: all)",
    )
    parser.add_argument(
        "--style",
        choices=("pattern", "color", "both"),
        default="pattern",
        help="chart encoding: patterns, colors, or both (default: pattern)",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    args.output = resolveProjectPath(args.output)
    args.figuresDir = resolveProjectPath(
        args.figuresDir
        if args.figuresDir is not None
        else args.output.parent / f"{args.output.stem}-figures"
    )
    return args


def main() -> int:
    args = parseArgs()
    if not visualizationDependenciesAvailable():
        return 2

    benchmarkCommand = [
        sys.executable,
        str(benchmarkScript),
        "--runs",
        str(args.runs),
        "--output",
        str(args.output),
    ]
    status = runStep("Benchmark", benchmarkCommand)
    if status:
        return status

    visualizationCommand = [
        sys.executable,
        str(visualizationScript),
        "--input",
        str(args.output),
        "--output-dir",
        str(args.figuresDir),
        "--charts",
        *args.charts,
        "--style",
        args.style,
    ]
    status = runStep("Visualization", visualizationCommand)
    if status:
        return status

    print(f"\nPipeline complete. CSV: {args.output}")
    print(f"Dashboard: {args.figuresDir / 'index.html'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
