"""Run one registered bound solver (unknown target, at most K swaps) on a problem file."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import load, save
from .solvers import get_solver, list_solvers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", choices=list_solvers("bound"))
    parser.add_argument("problem", type=Path, help="Problem JSON file.")
    parser.add_argument("-k", "--moves", type=int, default=3, help="Swap budget K.")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-t", "--timeout", type=float, default=None, help="Seconds.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("solution"),
        help="Output stem (writes <stem>.json and <stem>.txt).",
    )
    args = parser.parse_args()

    solver_cls = get_solver("bound", args.solver)
    problem = load(args.problem).problem
    solution = solver_cls(num_moves=args.moves).solve(
        problem, seed=args.seed, timeout=args.timeout
    )
    save(args.output, solution)
    print(
        f"{args.solver}: status={solution.status.value} "
        f"score {solution.initial_score} -> {solution.final_score} "
        f"({len(solution.moves)} swaps)"
    )


if __name__ == "__main__":
    main()
