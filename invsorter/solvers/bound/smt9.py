import argparse
import random
import sys
import time
from pathlib import Path

from z3 import (
    Optimize,
    Bool,
    Int,
    Or,
    Not,
    And,
    Implies,
    If,
    Sum,
    sat,
    unsat,
    IntSort,
    Function,
)

from ...grid import apply_move, valid_targets
from ...io import load, save, to_ascii
from ...models import Move, Problem, Pos, Item, GenerateResult, Status, Swap, Size
from . import BoundSolver
from .._engine_utils import apply_timeout_z3, apply_seed_z3
from .._utils import (
    test_similarity,
    add_placeholders,
    von_neumann_neighbours,
    moore_neighbours,
)


class SMT9Solver(BoundSolver):
    """Explicit Encoding of the item types in each step using uninterpreted functions."""

    name = "smt9"

    def __init__(self, num_moves: int = 10) -> None:
        self.num_moves = num_moves

    def generate_moves(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        items = problem.items
        if problem.neighbors == 8:
            neighbour_fn = moore_neighbours
        else:
            neighbour_fn = von_neumann_neighbours
        moves: list[Move] = []

        _t0 = time.perf_counter()
        o = Optimize()
        apply_timeout_z3(o, timeout)
        apply_seed_z3(o, rng.randint(0, 2**31 - 1))

        extendedItems = add_placeholders(problem.num_rows, problem.num_cols, items)

        ######## defining variables and symbols ########

        t = {}
        s_1_x = {}
        s_1_y = {}
        s_2_x = {}
        s_2_y = {}

        I = IntSort()

        for k in range(self.num_moves + 1):
            t[k] = Function(f"t_k:{k}", I, I, I)

        for k in range(1, self.num_moves + 1):
            s_1_x[k] = Int(f"s_1_x_k:{k}")
            s_1_y[k] = Int(f"s_1_y_k:{k}")
            s_2_x[k] = Int(f"s_2_x_k:{k}")
            s_2_y[k] = Int(f"s_2_y_k:{k}")

        ######## defining the formula ########

        # initial positions
        for i in extendedItems:
            o.add(t[0](i.pos.col, i.pos.row) == i.type)

        # swaps inside grid bounds
        for k in range(1, self.num_moves + 1):
            o.add(0 <= s_1_x[k])
            o.add(s_1_x[k] < problem.num_cols)
            o.add(0 <= s_1_y[k])
            o.add(s_1_y[k] < problem.num_rows)
            o.add(0 <= s_2_x[k])
            o.add(s_2_x[k] < problem.num_cols)
            o.add(0 <= s_2_y[k])
            o.add(s_2_y[k] < problem.num_rows)

        # update types according to swaps
        for k in range(1, self.num_moves + 1):
            o.add(t[k](s_1_x[k], s_1_y[k]) == t[k - 1](s_2_x[k], s_2_y[k]))
            o.add(t[k](s_2_x[k], s_2_y[k]) == t[k - 1](s_1_x[k], s_1_y[k]))

        # type consistency
        for k in range(1, self.num_moves + 1):
            for x in range(problem.num_cols):
                for y in range(problem.num_rows):
                    o.add(
                        Implies(
                            And(
                                Or(x != s_1_x[k], y != s_1_y[k]),
                                Or(x != s_2_x[k], y != s_2_y[k]),
                            ),
                            t[k](x, y) == t[k - 1](x, y),
                        )
                    )

        # symmetry breaking
        for k in range(1, self.num_moves + 1):
            o.add(
                Or(
                    s_1_y[k] < s_2_y[k],
                    And(
                        s_1_y[k] == s_2_y[k],
                        s_1_x[k] <= s_2_x[k],
                    ),
                )
            )


        ######### defining the objective ########

        evaluation = []
        K = self.num_moves

        for x in range(problem.num_cols):
            for y in range(problem.num_rows):
                for y2, x2 in neighbour_fn(problem.num_rows, problem.num_cols, y, x, 0):
                    for t1 in list(set([item.type for item in items])):
                        for t2 in list(set([item.type for item in items])):
                            evaluation.append(
                                If(And(t[K](x, y) == t1, t[K](x2, y2) == t2), 1, 0)
                                * test_similarity(t1, t2)
                            )

        o.maximize(Sum(evaluation))

        _t1 = time.perf_counter()
        _model_build_ms = (_t1 - _t0) * 1000.0
        z3_status = o.check()
        _t2 = time.perf_counter()
        _engine_solve_ms = (_t2 - _t1) * 1000.0
        if z3_status == unsat:
            self._timing = {
                "model_build_ms": _model_build_ms,
                "engine_solve_ms": _engine_solve_ms,
                "readback_ms": 0.0,
            }
            return GenerateResult((), Status.UNSAT)
        if z3_status != sat:
            print(f"[smt9] no plan: status={z3_status}", file=sys.stderr)
            self._timing = {
                "model_build_ms": _model_build_ms,
                "engine_solve_ms": _engine_solve_ms,
                "readback_ms": 0.0,
            }
            return GenerateResult((), Status.TIMEOUT)
        model = o.model()

        # reconstructing the moves
        _t3 = time.perf_counter()
        for k in range(1, self.num_moves + 1):
            if (
                model.eval(t[k](s_1_x[k], s_1_y[k])).as_long()
                != model.eval(t[k](s_2_x[k], s_2_y[k])).as_long()
            ):
                moves.append(
                    Swap(
                        Pos(
                            row=model.eval(s_1_y[k]).as_long(),
                            col=model.eval(s_1_x[k]).as_long(),
                        ),
                        Pos(
                            row=model.eval(s_2_y[k]).as_long(),
                            col=model.eval(s_2_x[k]).as_long(),
                        ),
                    )
                )

        result_moves = tuple(moves)
        readback_ms = (time.perf_counter() - _t3) * 1000.0
        self._timing = {
            "model_build_ms": _model_build_ms,
            "engine_solve_ms": _engine_solve_ms,
            "readback_ms": readback_ms,
        }
        if not result_moves:
            return GenerateResult(result_moves, Status.ALREADY_OPTIMAL)
        return GenerateResult(result_moves, Status.SOLVED)


def main() -> None:
    parser = argparse.ArgumentParser(description="SMT Solver 9")
    parser.add_argument("input", type=str, help="Problem JSON file")
    parser.add_argument("-k", "--moves", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-o", "--output", type=str, default="solution")
    args = parser.parse_args()

    problem = load(Path(args.input))
    solver = SMT9Solver(num_moves=args.moves)
    result = solver.solve(problem.problem, seed=args.seed)
    save(Path(args.output), result)
    print(
        f"Score: {result.initial_score} -> {result.final_score} "
        f"({len(result.moves)} moves)"
    )


if __name__ == "__main__":
    main()
