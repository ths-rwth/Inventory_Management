import random
import time
import argparse
import sys

from z3 import Optimize, Bool, And, Or, Not, Implies, If, Sum, sat, unsat, is_true
from typing import Callable, override
from pathlib import Path

from ...io import load, save
from ...models import Problem, Item, Pos, Swap, GenerateResult, Status
from . import BoundSolver
from .._engine_utils import apply_timeout_z3, apply_seed_z3
from .._utils import (
    von_neumann_neighbours,
    moore_neighbours,
    add_args_bit_blasting,
    test_similarity,
)


def solve_sat_type(
    num_rows: int,
    num_cols: int,
    items: list[Item],
    unfolding_depth,
    similarity: Callable[[int, int], int],
    neighbours: Callable[[int, int, int, int, int], list[tuple[int, int]]],
    types: list[int],
    *,
    timeout: float | None = None,
    seed: int | None = None,
    no_diff_noop: bool = False,
    upper_left_noop: bool = True,
):
    _timing = {}
    _t0 = time.perf_counter()
    o = Optimize()
    apply_timeout_z3(o, timeout)
    apply_seed_z3(o, seed)

    # --------------------------- create variables ---------------------------
    x = {}
    z = {}
    delta = {}
    for row1 in range(1, num_rows + 1):
        for col1 in range(1, num_cols + 1):
            for t in types:
                for k in range(unfolding_depth + 1):
                    x[row1, col1, t, k] = Bool(f"x_{row1}_{col1}_{t}_{k}")
                    # = true iff at time t item is at position (row, col)

            for k in range(unfolding_depth):
                for row2 in range(1, num_rows + 1):
                    for col2 in range(1, num_cols + 1):
                        if (row1, col1) <= (row2, col2):
                            if upper_left_noop:
                                if (row1, col1) != (row2, col2) or (row1, col1) == (
                                    1,
                                    1,
                                ):
                                    z[row1, col1, row2, col2, k] = Bool(
                                        f"z_{row1}_{col1}_{row2}_{col2}_{k}"
                                    )
                                    # = true iff at the time step from t to t+1, we swap the items
                                    # at positions (row, col) and (row2, col2)
                                else:
                                    z[row1, col1, row2, col2, k] = False
                            else:
                                z[row1, col1, row2, col2, k] = Bool(
                                    f"z_{row1}_{col1}_{row2}_{col2}_{k}"
                                )

    for k in range(unfolding_depth):
        delta[k] = Bool(f"delta_{k}")

    # --------------------------- build formula ---------------------------

    # at k = 0, all items are at their initial positions
    phi1 = And([x[item.pos.row + 1, item.pos.col + 1, item.type, 0] for item in items])

    # at k = 0, all positions where there is no item are empty
    phi2 = And(
        [
            Not(x[row, col, t, 0])
            for t in types
            for (row, col) in [
                (r + 1, c + 1)
                for r in range(num_rows)
                for c in range(num_cols)
                if (r, c) not in [(item.pos.row, item.pos.col) for item in items]
            ]
        ]
    )

    # at every point in time every position contains at most one type
    phi3 = And(
        [
            Or([Not(x[row, col, t1, k]), Not(x[row, col, t2, k])])
            for row in range(1, num_rows + 1)
            for col in range(1, num_cols + 1)
            for k in range(unfolding_depth + 1)
            for t1 in types
            for t2 in [t for t in types if t > t1]
        ]
    )

    # in every time step at least one swap is performed
    phi4 = And(
        [
            Or(
                [
                    z[row1, col1, row2, col2, k]
                    for row1 in range(1, num_rows + 1)
                    for col1 in range(1, num_cols + 1)
                    for (row2, col2) in [
                        (row, col)
                        for row in range(1, num_rows + 1)
                        for col in range(1, num_cols + 1)
                        if (row1, col1) <= (row, col) and (row1, col1) != (row, col)
                    ]
                ]
            )
            for k in range(unfolding_depth)
        ]
    )

    # in every time step at most one swap is performed
    phi5 = And(
        [
            Or(
                Not(z[row1_1, col1_1, row1_2, col1_2, k]),
                Not(z[row2_1, col2_1, row2_2, col2_2, k]),
            )
            for k in range(unfolding_depth)
            for row1_1 in range(1, num_rows + 1)
            for col1_1 in range(1, num_cols + 1)
            for (row1_2, col1_2) in [
                (row, col)
                for row in range(1, num_rows + 1)
                for col in range(1, num_cols + 1)
                if (row1_1, col1_1) <= (row, col)
            ]
            for (row2_1, col2_1, row2_2, col2_2) in [
                (row1, col1, row2, col2)
                for row1 in range(1, num_rows + 1)
                for col1 in range(1, num_cols + 1)
                for row2 in range(1, num_rows + 1)
                for col2 in range(1, num_cols + 1)
                if (
                    (
                        row1 != row1_1
                        or col1 != col1_1
                        or row2 != row1_2
                        or col2 != col1_2
                    )
                    and ((row1, col1) <= (row2, col2))
                    and ((row1_1, col1_1, row1_2, col1_2) < (row1, col1, row2, col2))
                )
            ]
        ]
    )

    # if z[row1, col1, row2, col2, k] is true,
    # then positions (row1, col1) and (row2, col2) are swapped at step k to k +1
    phi6 = And(
        [
            Implies(
                z[row1, col1, row2, col2, k],
                And(
                    [
                        x[row1, col1, t, k] == x[row2, col2, t, k + 1],
                        x[row2, col2, t, k] == x[row1, col1, t, k + 1],
                    ]
                ),
            )
            for row1 in range(1, num_rows + 1)
            for col1 in range(1, num_cols + 1)
            for k in range(unfolding_depth)
            for (row2, col2) in [
                (r, c)
                for r in range(1, num_rows + 1)
                for c in range(1, num_cols + 1)
                if (row1, col1) <= (r, c)
            ]
            for t in types
        ]
    )

    # if a position (r, s) is not involved in a swap at time step k to k + 1,
    # then it stays the same
    phi7 = And(
        [
            Implies(
                Not(
                    Or(
                        [
                            z[row1, col1, row2, col2, k]
                            for (row2, col2) in [
                                (r, c)
                                for r in range(1, num_rows + 1)
                                for c in range(1, num_cols + 1)
                                if (row1, col1) <= (r, c)
                            ]
                        ]
                        + [
                            z[row2, col2, row1, col1, k]
                            for (row2, col2) in [
                                (r, c)
                                for r in range(1, num_rows + 1)
                                for c in range(1, num_cols + 1)
                                if (r, c) < (row1, col1)
                            ]
                        ]
                    )
                ),
                x[row1, col1, t, k] == x[row1, col1, t, k + 1],
            )
            for k in range(unfolding_depth)
            for t in types
            for row1 in range(1, num_rows + 1)
            for col1 in range(1, num_cols + 1)
        ]
    )

    # ---------- no-op symmetry breaking constraints ----------
    # the only no-op swaps are swaps of positions with them selves
    phi8 = And(
        [
            Implies(
                z[row1, col1, row2, col2, k],
                And(
                    [
                        Or(Not(x[row1, col1, t, k]), Not(x[row2, col2, t, k]))
                        for t in types
                    ]
                ),
            )
            for k in range(unfolding_depth)
            for row1 in range(1, num_rows + 1)
            for col1 in range(1, num_cols + 1)
            for (row2, col2) in [
                (r, c)
                for r in range(1, num_rows + 1)
                for c in range(1, num_cols + 1)
                if (row1, col1) < (r, c)
            ]
        ]
    )

    constraints = [phi1, phi2, phi3, phi4, phi5, phi6, phi7]

    if no_diff_noop:
        constraints.append(phi8)

    o.add(And(constraints))

    # we have a maximization objective and sum over all positions pos1 and all its neighbouring positions pos2
    maximize = o.maximize(
        Sum(
            [
                If(
                    And(
                        x[row1, col1, t1, unfolding_depth],
                        x[row2, col2, t2, unfolding_depth],
                    ),
                    1,
                    0,
                )
                * similarity(t1, t2)
                for row1 in range(1, num_rows + 1)
                for col1 in range(1, num_cols + 1)
                for t1 in types
                for (row2, col2) in neighbours(num_rows, num_cols, row1, col1, 1)
                for t2 in types
            ]
        )
    )

    _t1 = time.perf_counter()
    _timing["model_build_ms"] = (_t1 - _t0) * 1000.0
    status = o.check()
    _t2 = time.perf_counter()
    _timing["engine_solve_ms"] = (_t2 - _t1) * 1000.0
    if status != sat:
        print(f"[sat_type] no plan: status={status}", file=sys.stderr)
        return None, status, _timing
    return (x, z, o.model()), status, _timing


class SatBitBlastingType(BoundSolver):
    """
    A Sat Solver Model that uses the 'bit-blasting' approach with type-indexed
    Boolean variables: for every time step, every type, and every position there is
    one Boolean ``x[row, col, type, k]`` that is true iff a cell of that type
    occupies that position at step k. This is in contrast to the item-indexed ``sat``
    encoding, which has a variable per item id; folding the variables over types
    instead of item ids exploits the symmetry between same-type items at the cost
    of losing per-item identity in the readback. Realized via the Z3 engine.
    """

    name = "sat_type"

    def __init__(
        self,
        num_moves: int = 10,
        neighbours: Callable[
            [int, int, int, int, int], list[tuple[int, int]]
        ] = von_neumann_neighbours,
        *,
        no_diff_noop: bool = False,
        upper_left_noop: bool = True,
    ) -> None:
        self.num_moves = num_moves
        self.neighbours = neighbours
        self.no_diff_noop = no_diff_noop
        self.upper_left_noop = upper_left_noop

    @override
    # TODO(fairness)[critical]: reconstruct_moves IndexError on 1-cell-diff
    def generate_moves(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:

        items = list(problem.items)
        types = list(set([item.type for item in items]))

        # Select neighbour function based on problem.neighbors
        if problem.neighbors == 8:
            neighbour_fn = moore_neighbours
        else:
            neighbour_fn = von_neumann_neighbours

        seed = rng.randint(0, 2**31 - 1)
        result, z3_status, _timing = solve_sat_type(
            problem.num_rows,
            problem.num_cols,
            items,
            self.num_moves,
            test_similarity,
            neighbour_fn,
            types,
            timeout=timeout,
            seed=seed,
            no_diff_noop=self.no_diff_noop,
            upper_left_noop=self.upper_left_noop,
        )
        # Z3 status: sat → SOLVED|ALREADY_OPTIMAL, unsat → UNSAT, unknown → TIMEOUT
        if z3_status == unsat:
            self._timing = _timing
            return GenerateResult((), Status.UNSAT)
        if z3_status != sat:
            self._timing = _timing
            return GenerateResult((), Status.TIMEOUT)
        x, z, model = result
        _t3 = time.perf_counter()
        swaps_list: list[Swap] = []
        for k in range(self.num_moves):
            found = None
            for row1 in range(1, problem.num_rows + 1):
                for col1 in range(1, problem.num_cols + 1):
                    for row2 in range(1, problem.num_rows + 1):
                        for col2 in range(1, problem.num_cols + 1):
                            if (row1, col1) <= (row2, col2):
                                if self.upper_left_noop:
                                    if (row1, col1) != (row2, col2) or (row1, col1) == (
                                        1,
                                        1,
                                    ):
                                        if is_true(
                                            model.eval(
                                                z[row1, col1, row2, col2, k],
                                                model_completion=True,
                                            )
                                        ):
                                            found = (row1, col1, row2, col2)
                                            break
                                else:
                                    if is_true(
                                        model.eval(
                                            z[row1, col1, row2, col2, k],
                                            model_completion=True,
                                        )
                                    ):
                                        found = (row1, col1, row2, col2)
                                        break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found is not None and not (
                found[0] == found[2] and found[1] == found[3]
            ):
                swaps_list.append(
                    Swap(
                        pos1=Pos(row=found[0] - 1, col=found[1] - 1),
                        pos2=Pos(row=found[2] - 1, col=found[3] - 1),
                    )
                )
        moves = tuple(swaps_list)
        readback_ms = (time.perf_counter() - _t3) * 1000.0
        self._timing = {
            "model_build_ms": _timing.get("model_build_ms", 0.0),
            "engine_solve_ms": _timing.get("engine_solve_ms", 0.0),
            "readback_ms": readback_ms,
        }
        if not moves:
            return GenerateResult(moves, Status.ALREADY_OPTIMAL)
        return GenerateResult(moves, Status.SOLVED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bit-Blasting Sat Solver")
    args = add_args_bit_blasting(parser).parse_args()

    problem = load(Path(args.input))
    solver = SatBitBlastingType(num_moves=args.moves)
    result = solver.solve(problem.problem)
    save(Path(args.output), result)
    print(
        f"Score: {result.initial_score} -> {result.final_score} "
        f"({len(result.moves)} moves)"
    )


if __name__ == "__main__":
    main()
