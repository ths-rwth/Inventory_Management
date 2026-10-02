"""Pipeline — compose a Generator + UnboundSolver into a bound-shaped solver."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..io import load, save
from ..models import Problem, Solution, Status
from ..scoring import directions_for, score
from . import get_solver
from ._solution_builder import wrap_end_state
from .generators import Generator
from .unbound import UnboundSolver


class Pipeline:
    """Duck-types as a bound solver: solve(problem, seed, timeout) -> Solution.

    Composes a Generator with an UnboundSolver. Splits `timeout` 50/50
    between stages (Policy A; spec §9). When unbound declares
    `compatible_generators`, rejects mismatched pairings at construction.
    """

    def __init__(
        self,
        generator: Generator,
        unbound: UnboundSolver,
        name: str | None = None,
    ) -> None:
        if unbound.compatible_generators is not None:
            if generator.name not in unbound.compatible_generators:
                raise ValueError(
                    f"Pipeline pairing rejected: unbound solver "
                    f"{unbound.name!r} declares "
                    f"compatible_generators={unbound.compatible_generators!r}, "
                    f"got generator {generator.name!r}. Either pair with a "
                    f"compatible generator or remove the restriction."
                )
        self.generator = generator
        self.unbound = unbound
        self.name = name or f"{generator.name}+{unbound.name}"

    def solve(
        self,
        problem: Problem,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> Solution:
        # Policy A: split 50/50. Treat non-positive timeout as "no limit"
        # so a benchmark passing timeout=0 doesn't starve both stages.
        if timeout is not None and timeout > 0:
            gen_timeout: float | None = timeout / 2.0
            unb_timeout: float | None = timeout / 2.0
        else:
            gen_timeout = None
            unb_timeout = None

        # Inline generator wrap to capture gen status. We can't go through
        # `self.generator.solve(...)` because that helper drops the status.
        end_state, gen_status = wrap_end_state(
            problem, self.generator.generate_end_state, seed, gen_timeout
        )

        # Build initial grid from problem.items
        initial_grid = tuple(
            tuple(
                next((i.type for i in problem.items if i.pos.row == r and i.pos.col == c), 0)
                for c in range(problem.num_cols)
            )
            for r in range(problem.num_rows)
        )

        if gen_status in (Status.SOLVED, Status.OPTIMAL) and end_state.grid == initial_grid:
            initial_sc = score(problem.items, directions=directions_for(problem.neighbors))
            return Solution(
                problem=problem,
                moves=(),
                initial_items=problem.items,
                final_items=problem.items,
                initial_score=initial_sc,
                final_score=initial_sc,
                initial_grid=initial_grid,
                final_grid=initial_grid,
                status=Status.ALREADY_OPTIMAL,
                effective_moves=(),
            )

        solution = self.unbound.solve(
            problem, end_state, seed=seed, timeout=unb_timeout
        )

        # Status-fold: if generator was non-SOLVED, the pipeline outcome
        # inherits that status (a TIMEOUT-generator + SOLVED-unbound is
        # still a TIMEOUT pipeline because the EndState is degenerate).
        # Unbound CRASHED / UNAVAILABLE are set later by benchmark.py only.
        if gen_status not in (Status.SOLVED, Status.OPTIMAL):
            solution = replace(solution, status=gen_status)
        return solution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose a generator + unbound solver and run on a problem.",
    )
    parser.add_argument("problem", type=Path, help="Path to a problem/solution JSON.")
    parser.add_argument(
        "--generator",
        required=True,
        help="Registered generator name (see invsorter.solvers.list_solvers('generator')).",
    )
    parser.add_argument(
        "--unbound",
        required=True,
        help="Registered unbound solver name.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output stem (writes <stem>.json and <stem>.txt).",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    GenCls = get_solver("generator", args.generator)
    UnbCls = get_solver("unbound", args.unbound)
    pipeline = Pipeline(GenCls(), UnbCls())

    problem = load(args.problem).problem
    solution = pipeline.solve(problem, seed=args.seed)
    save(args.output, solution)


if __name__ == "__main__":
    main()
