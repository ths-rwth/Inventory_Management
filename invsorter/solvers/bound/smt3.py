import random
import time
import argparse
import sys

from z3 import Optimize, Int, And, Or, Not, Implies, If, Sum, sat, unsat
from typing import Callable, override
from pathlib import Path

from ...io import load, save
from ...models import Problem, Item, Swap, Pos, GenerateResult, Status
from . import BoundSolver
from .._engine_utils import apply_timeout_z3, apply_seed_z3
from .._utils import (
    von_neumann_neighbours,
    moore_neighbours,
    inv,
    test_similarity,
    get_neighbours,
)


def solve_smt3(
    num_rows: int,
    num_cols: int,
    items: list[Item],
    unfolding_depth: int,
    neighbours: Callable[[int, int, int, int, int], list[tuple[int, int]]],
    *,
    timeout: float | None = None,
    seed: int | None = None,
    swap_pos_lex_order: bool = True,
    no_diff_noop: bool = True,
    upper_left_noop: bool = True,
):

    for item in items:
        if item.id <= 0:
            raise Exception("Item IDs need to be positive for this model")

    _timing = {}
    _t0 = time.perf_counter()
    o = Optimize()
    apply_timeout_z3(o, timeout)
    apply_seed_z3(o, seed)

    # --------------------------- create variables ---------------------------
    t_start = {}
    t_end = {}
    p = {}
    t = {}

    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            t_end[row, col] = Int(f"t_end_{row}_{col}")
    for item in items:
        t_start[item.pos.row + 1, item.pos.col + 1] = item.type
    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            if (row, col) not in [
                (item.pos.row + 1, item.pos.col + 1) for item in items
            ]:
                t_start[row, col] = 0

    for k in range(unfolding_depth):
        for h in [1, 2]:
            p[h, "row", k] = Int(f"p_{h}_row_{k}")
            p[h, "col", k] = Int(f"p_{h}_col_{k}")

            t[h, k] = Int(f"t_{h}_{k}")

    # --------------------------- build formula ---------------------------
    # (check out the pdf in the docs if you actually want to read this)

    # all swap positions are within the grid bounds
    phi3 = And(
        [
            And(
                [
                    p[h, "row", k] >= 1,
                    p[h, "col", k] >= 1,
                    p[h, "row", k] <= num_rows,
                    p[h, "col", k] <= num_cols,
                ]
            )
            for k in range(unfolding_depth)
            for h in [1, 2]
        ]
    )

    # for all swaps (k to k + 1) we have that if position h of the swap was not part of any swap
    # (k' to k' + 1) that happened before, then the type of the item at position h of the swap
    # (k to k + 1) is the same as the type of the item that was there at the beginning
    phi4 = And(
        [
            Implies(
                And(
                    [
                        And(
                            Or(
                                p[h, "row", k1] != p[1, "row", k2],
                                p[h, "col", k1] != p[1, "col", k2],
                            ),
                            Or(
                                p[h, "row", k1] != p[2, "row", k2],
                                p[h, "col", k1] != p[2, "col", k2],
                            ),
                        )
                        for k2 in range(k1)
                    ]
                ),
                Or(
                    [
                        And(
                            [
                                p[h, "row", k1] == row,
                                p[h, "col", k1] == col,
                                t[h, k1] == t_start[row, col],
                            ]
                        )
                        for row in range(1, num_rows + 1)
                        for col in range(1, num_cols + 1)
                    ]
                ),
            )
            for k1 in range(unfolding_depth)
            for h in [1, 2]
        ]
    )

    # for all pairs of swaps (k to k + 1), (k' to k' + 1) we have that if position h of swap
    # (k to k + 1) is equal to position h' of swap (k' to k' + 1) and this position is not part
    # of any swap in-between those two swaps, then the type of the item at position h′ in swap
    # (k' to k' + 1) is equal to the type of the item at position inv(h) in swap (k to k + 1).
    phi5 = And(
        [
            Implies(
                And(
                    [
                        p[h1, "row", k1] == p[h2, "row", k2],
                        p[h1, "col", k1] == p[h2, "col", k2],
                        And(
                            [
                                And(
                                    Or(
                                        p[1, "row", k3] != p[h1, "row", k1],
                                        p[1, "col", k3] != p[h1, "col", k1],
                                    ),
                                    Or(
                                        p[2, "row", k3] != p[h1, "row", k1],
                                        p[2, "col", k3] != p[h1, "col", k1],
                                    ),
                                )
                                for k3 in range(k1 + 1, k2)
                            ]
                        ),
                    ]
                ),
                t[h2, k2] == t[inv(h1), k1],
            )
            for k1 in range(unfolding_depth)
            for k2 in range(k1 + 1, unfolding_depth)
            for h1 in [1, 2]
            for h2 in [1, 2]
        ]
    )

    # if a position is not part of any swap, then there is the same item type there at the end as in the beginning
    phi6 = And(
        [
            Implies(
                And(
                    [
                        And(
                            Or(p[1, "row", k] != row, p[1, "col", k] != col),
                            Or(p[2, "row", k] != row, p[2, "col", k] != col),
                        )
                        for k in range(unfolding_depth)
                    ]
                ),
                t_end[row, col] == t_start[row, col],
            )
            for row in range(1, num_rows + 1)
            for col in range(1, num_cols + 1)
        ]
    )

    phi7 = And(
        [
            Implies(
                And(
                    [
                        p[h, "row", k1] == row,
                        p[h, "col", k1] == col,
                        And(
                            [
                                And(
                                    Or(
                                        p[1, "row", k2] != p[h, "row", k1],
                                        p[1, "col", k2] != p[h, "col", k1],
                                    ),
                                    Or(
                                        p[2, "row", k2] != p[h, "row", k1],
                                        p[2, "col", k2] != p[h, "col", k1],
                                    ),
                                )
                                for k2 in range(k1 + 1, unfolding_depth)
                            ]
                        ),
                    ]
                ),
                t_end[row, col] == t[inv(h), k1],
            )
            for row in range(1, num_rows + 1)
            for col in range(1, num_cols + 1)
            for h in [1, 2]
            for k1 in range(unfolding_depth)
        ]
    )

    constraints = [phi3, phi4, phi5, phi6, phi7]

    # --------------- Additional constraints for symmetry breaking ---------------
    if swap_pos_lex_order:
        # We demand that the first swap position is lexicographically smaller than the second one to reduce symmetry
        phi9 = And(
            [
                Or(
                    p[1, "row", k] < p[2, "row", k],
                    And(
                        p[1, "row", k] == p[2, "row", k],
                        p[1, "col", k] <= p[2, "col", k],
                    ),
                )
                for k in range(unfolding_depth)
            ]
        )
        constraints.append(phi9)

    if no_diff_noop:
        # If a swap is not a swap of a position with itself, then the item types that it swaps are different,
        # i.e., every no-op swap is a swap of a position with itself
        phi11 = And(
            [
                Implies(
                    Or(
                        p[1, "row", k] != p[2, "row", k],
                        p[1, "col", k] != p[2, "col", k],
                    ),
                    t[1, k] != t[2, k],
                )
                for k in range(unfolding_depth)
            ]
        )
        constraints.append(phi11)

    if upper_left_noop:
        # if a swap is a no-op swap, i.e., it swaps a position with itself, then the position that is swapped
        # must be the one in the upper left corner.
        phi12 = And(
            [
                Not(
                    And(
                        [
                            p[1, "row", k] == row,
                            p[2, "row", k] == row,
                            p[1, "col", k] == col,
                            p[2, "col", k] == col,
                        ]
                    )
                )
                for k in range(unfolding_depth)
                for (row, col) in [
                    (r, s)
                    for r in range(1, num_rows + 1)
                    for s in range(1, num_cols + 1)
                    if (r, s) != (1, 1)
                ]
            ]
        )

        constraints.append(phi12)

    o.add(And(constraints))

    # we have a maximization objective and sum over all positions pos1 and all its neighbouring positions pos2.
    # if an item of type t1 is at pos1 and an item of type2 is at then neighbouring position pos2,
    # then we accumulate their similarity.
    # maximize = o.maximize(Sum([similarity(t_end[row1, col1], t_end[row2, col2])
    # for row1 in range(1, num_rows + 1)
    # for col1 in range(1, num_cols + 1)
    # for (row2, col2) in neighbouring(num_rows, num_cols, row1, col1)]))

    maximize = o.maximize(
        Sum(
            [
                If(And(t_end[row1, col1] == t1, t_end[row2, col2] == t2), 1, 0)
                * test_similarity(t1, t2)
                for row1 in range(1, num_rows + 1)
                for col1 in range(1, num_cols + 1)
                for (row2, col2) in neighbours(num_rows, num_cols, row1, col1, 1)
                for t1 in list(set([item.type for item in items]))
                for t2 in list(set([item.type for item in items]))
            ]
        )
    )

    _t1 = time.perf_counter()
    _timing["model_build_ms"] = (_t1 - _t0) * 1000.0
    z3_status = o.check()
    _t2 = time.perf_counter()
    _timing["engine_solve_ms"] = (_t2 - _t1) * 1000.0
    if z3_status != sat:
        print(f"[smt3] no plan: status={z3_status}", file=sys.stderr)
        return None, z3_status, _timing
    return (p, o.model()), z3_status, _timing


class SMT3(BoundSolver):
    name = "smt3"

    def __init__(
        self,
        num_moves: int = 10,
        neighbours: Callable[
            [int, int, int, int, int], list[tuple[int, int]]
        ] = von_neumann_neighbours,
        *,
        swap_pos_lex_order: bool = True,
        no_diff_noop: bool = True,
        upper_left_noop: bool = True,
    ) -> None:
        self.num_moves = num_moves
        self.neighbours = neighbours
        self.swap_pos_lex_order = swap_pos_lex_order
        self.no_diff_noop = no_diff_noop
        self.upper_left_noop = upper_left_noop

    @override
    # TODO(fairness)[medium]: no constraint p[1]!=p[2] — degenerate self-swap plans sat-feasible
    # TODO(fairness)[high]: old CLI flag removed; neighbors now read from problem.neighbors
    def generate_moves(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:

        # this model uses ID 0 as the representation for no item, so it is important that no item has ID 0.
        items = [
            Item(id=item.id + 1, pos=item.pos, size=item.size, type=item.type)
            for item in problem.items
        ]

        # Select neighbor function from problem config
        if problem.neighbors == 8:
            neighbour_fn = moore_neighbours
        else:
            neighbour_fn = von_neumann_neighbours
        seed = rng.randint(0, 2**31 - 1)
        result, z3_status, _timing = solve_smt3(
            problem.num_rows,
            problem.num_cols,
            items,
            self.num_moves,
            neighbours=neighbour_fn,
            timeout=timeout,
            seed=seed,
            swap_pos_lex_order=self.swap_pos_lex_order,
            no_diff_noop=self.no_diff_noop,
            upper_left_noop=self.upper_left_noop,
        )
        if z3_status == unsat:
            self._timing = _timing
            return GenerateResult((), Status.UNSAT)
        if z3_status != sat:
            self._timing = _timing
            return GenerateResult((), Status.TIMEOUT)
        p, model = result

        _t3 = time.perf_counter()
        swaps = []
        for k in range(self.num_moves):
            p1_row = int(model[p[1, "row", k]].__str__()) - 1
            p1_col = int(model[p[1, "col", k]].__str__()) - 1
            p2_row = int(model[p[2, "row", k]].__str__()) - 1
            p2_col = int(model[p[2, "col", k]].__str__()) - 1
            if p1_row == p2_row and p1_col == p2_col:
                continue  # skip identity self-swap
            swaps.append(
                Swap(pos1=Pos(row=p1_row, col=p1_col), pos2=Pos(row=p2_row, col=p2_col))
            )
        moves = tuple(swaps)
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
    parser = argparse.ArgumentParser(description="Partial Assignment SMT Solver")
    parser.add_argument("input", type=str, help="Problem JSON file")
    parser.add_argument("-k", "--moves", type=int, default=10)
    parser.add_argument("-o", "--output", type=str, default="solution")
    args = parser.parse_args()

    problem = load(Path(args.input))
    solver = SMT3(num_moves=args.moves)
    result = solver.solve(problem.problem)
    save(Path(args.output), result)
    print(
        f"Score: {result.initial_score} -> {result.final_score} "
        f"({len(result.moves)} moves)"
    )


if __name__ == "__main__":
    main()
