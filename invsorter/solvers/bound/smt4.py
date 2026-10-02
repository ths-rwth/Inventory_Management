import random
import time
import argparse
import sys

from z3 import Optimize, Int, And, Or, Implies, Sum, Not, sat, unsat
from typing import Callable, override
from pathlib import Path

from ...io import load, save
from ...models import Problem, Item, Swap, Pos, GenerateResult, Status
from . import BoundSolver
from .._engine_utils import apply_timeout_z3, apply_seed_z3
from .._utils import (
    von_neumann_neighbours,
    moore_neighbours,
    get_neighbours,
    inv,
    first_l_positions,
    test_similarity,
)


def solve_smt4(
    num_rows: int,
    num_cols: int,
    items: list[Item],
    unfolding_depth: int,
    swaps_per_step: int,
    similarity: Callable[[Int, Int], int],
    neighbours: Callable[[int, int, int, int, int], list[tuple[int, int]]],
    *,
    timeout: float | None = None,
    seed: int | None = None,
):

    max_type = 0
    for item in items:
        if item.id <= 0:
            raise Exception("Item IDs need to be positive for this model")
        if item.type > max_type:
            max_type = item.type

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
            # t_start[row, col] = Int(f"t_start_{row}_{col}")
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
        for l1 in range(1, swaps_per_step + 1):
            for h in [1, 2]:
                p[h, "row", k, l1] = Int(f"p_{h}_row_{k}_{l1}")
                p[h, "col", k, l1] = Int(f"p_{h}_col_{k}_{l1}")

                t[h, k, l1] = Int(f"t_{h}_{k}_{l1}")

    # --------------------------- build formula ---------------------------
    # (check out the pdf in the docs if you actually want to read this)

    # for every item the type that is initially at the position of the item is the type of the item
    # phi1 = And([t_start[item.pos.row + 1, item.pos.col + 1] == item.type for item in items])

    # all the other positions not occupied by an item in the beginning are assigned the type 0
    # phi2 = And([t_start[row, col] == 0
    #             for row in range(1, num_rows + 1)
    #             for col in range(1, num_cols + 1)
    #             if (row, col) not in [(item.pos.row + 1, item.pos.col + 1)
    #                                   for item in items]])

    # all swap positions are within the grid bounds
    phi3 = And(
        [
            And(
                [
                    p[h, "row", k, l1] >= 1,
                    p[h, "col", k, l1] >= 1,
                    p[h, "row", k, l1] <= num_rows,
                    p[h, "col", k, l1] <= num_cols,
                ]
            )
            for k in range(unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
            for h in [1, 2]
        ]
    )

    # all swaps that happen in one time step are independent of each other
    phi4 = And(
        [
            Or(
                p[h1, "row", k, l1] != p[h2, "row", k, l2],
                p[h1, "col", k, l1] != p[h2, "col", k, l2],
            )
            for k in range(unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
            for l2 in range(l1 + 1, swaps_per_step + 1)
            for h1 in [1, 2]
            for h2 in [1, 2]
        ]
    )

    # for all time steps (k to k + 1) and all swaps l in these time steps we have that
    # if position h of swap l in time step (k to k + 1) was not part of any swap l' in a time
    # step (k' to k' + 1) that happened before, then the type of the item at swap position h
    # of the swap l in time step (k to k + 1) is the same as the type of the item that was
    # there at the beginning
    phi5 = And(
        [
            Implies(
                And(
                    [
                        Or(
                            p[h1, "row", k1, l1] != p[h2, "row", k2, l2],
                            p[h1, "col", k1, l1] != p[h2, "col", k2, l2],
                        )
                        for k2 in range(k1)
                        for l2 in range(1, swaps_per_step + 1)
                        for h2 in [1, 2]
                    ]
                ),
                Or(
                    [
                        And(
                            [
                                p[h1, "row", k1, l1] == row,
                                p[h1, "col", k1, l1] == col,
                                t[h1, k1, l1] == t_start[row, col],
                            ]
                        )
                        for row in range(1, num_rows + 1)
                        for col in range(1, num_cols + 1)
                    ]
                ),
            )
            for k1 in range(unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
            for h1 in [1, 2]
        ]
    )

    # for all pairs of swaps (k to k + 1), (k' to k' + 1) we have that if position h of swap
    # (k to k + 1) is equal to position h' of swap (k' to k' + 1) and this position is not part
    # of any swap in-between those two swaps, then the type of the item at position h′ in swap
    # (k' to k' + 1) is equal to the type of the item at position inv(h) in swap (k to k + 1).
    phi6 = And(
        [
            Implies(
                And(
                    [
                        p[h1, "row", k1, l1] == p[h2, "row", k2, l2],
                        p[h1, "col", k1, l1] == p[h2, "col", k2, l2],
                        And(
                            [
                                Or(
                                    p[h3, "row", k3, l3] != p[h1, "row", k1, l1],
                                    p[h3, "col", k3, l3] != p[h1, "col", k1, l1],
                                )
                                for k3 in range(k1 + 1, k2)
                                for l3 in range(1, swaps_per_step + 1)
                                for h3 in [1, 2]
                            ]
                        ),
                    ]
                ),
                t[h2, k2, l2] == t[inv(h1), k1, l1],
            )
            for k1 in range(unfolding_depth)
            for k2 in range(k1 + 1, unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
            for l2 in range(1, swaps_per_step + 1)
            for h1 in [1, 2]
            for h2 in [1, 2]
        ]
    )

    # if a position is not part of any swap, then there is the same item type there at the end as in the beginning
    phi7 = And(
        [
            Implies(
                And(
                    [
                        Or(p[h, "row", k, l1] != row, p[h, "col", k, l1] != col)
                        for k in range(unfolding_depth)
                        for l1 in range(1, swaps_per_step + 1)
                        for h in [1, 2]
                    ]
                ),
                t_end[row, col] == t_start[row, col],
            )
            for row in range(1, num_rows + 1)
            for col in range(1, num_cols + 1)
        ]
    )

    # for every position (r, s) and every time step (k to k + 1) and every swap l in that time step
    # we have that if swap l of step (k to k + 1) is the last swap in which (r, s) is involved
    # and (r, s) is swap position h of swap l in step (k to k + 1),
    # then the type of the item at position (r, s) at the end is equal to the type of the item
    # at swap position inv(h) of swap l in time step (k to k + 1)
    phi8 = And(
        [
            Implies(
                And(
                    [
                        p[h1, "row", k1, l1] == row,
                        p[h1, "col", k1, l1] == col,
                        And(
                            [
                                Or(
                                    p[h2, "row", k2, l2] != p[h1, "row", k1, l1],
                                    p[h2, "col", k2, l2] != p[h1, "col", k1, l1],
                                )
                                for k2 in range(k1 + 1, unfolding_depth)
                                for l2 in range(1, swaps_per_step + 1)
                                for h2 in [1, 2]
                            ]
                        ),
                    ]
                ),
                t_end[row, col] == t[inv(h1), k1, l1],
            )
            for row in range(1, num_rows + 1)
            for col in range(1, num_cols + 1)
            for h1 in [1, 2]
            for k1 in range(unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
        ]
    )

    # --------------- Additional constraints for symmetry breaking ---------------

    # We demand that the first swap position is lexicographically smaller than the second one to reduce symmetry
    phi9 = And(
        [
            Or(
                p[1, "row", k, l1] < p[2, "row", k, l1],
                And(
                    p[1, "row", k, l1] == p[2, "row", k, l1],
                    p[1, "col", k, l1] <= p[2, "col", k, l1],
                ),
            )
            for k in range(unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
        ]
    )

    # We demand that the swaps that we do in one time step are ordered lexicographically to reduce symmetry
    phi10 = And(
        [
            Or(
                p[1, "row", k, l1] < p[1, "row", k, l2],
                And(
                    p[1, "row", k, l1] == p[1, "row", k, l2],
                    p[1, "col", k, l1] < p[1, "col", k, l2],
                ),
            )
            for k in range(unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
            for l2 in range(l1 + 1, swaps_per_step + 1)
        ]
    )

    # If a swap is not a swap of a position with itself, then the item types that it swaps are different,
    # i.e. every no-op swap is a swap of a position with itself
    phi11 = And(
        [
            Implies(
                Or(
                    p[1, "row", k, l1] != p[2, "row", k, l1],
                    p[1, "col", k, l1] != p[2, "col", k, l1],
                ),
                t[1, k, l1] != t[2, k, l1],
            )
            for k in range(unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
        ]
    )

    # if a swap is a no-op swap, i.e., it swaps a position with itself, then the position that is swapped
    # must be one of the first l positions (where positions are enumerated row by row from left to right)
    phi12 = And(
        [
            Not(
                And(
                    [
                        p[1, "row", k, l1] == row,
                        p[2, "row", k, l1] == row,
                        p[1, "col", k, l1] == col,
                        p[2, "col", k, l1] == col,
                    ]
                )
            )
            for k in range(unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
            for (row, col) in [
                (r, s)
                for r in range(1, num_rows + 1)
                for s in range(1, num_cols + 1)
                if (r, s) not in first_l_positions(num_rows, num_cols, swaps_per_step)
            ]
        ]
    )

    # restrict type domains
    phi13 = And(
        [
            And(t[h, k, l1] >= 0, t[h, k, l1] <= max_type)
            for k in range(unfolding_depth)
            for l1 in range(1, swaps_per_step + 1)
            for h in [1, 2]
        ]
    )
    phi14 = And(
        [
            And(t_end[row, col] >= 0, t_end[row, col] <= max_type)
            for row in range(1, num_rows + 1)
            for col in range(1, num_cols + 1)
        ]
    )

    constraints = [phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10, phi11, phi12]

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
                And(t_end[row1, col1] == t1, t_end[row2, col2] == t2)
                * similarity(t1, t2)
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
        print(f"[smt4] no plan: status={z3_status}", file=sys.stderr)
        return None, z3_status, _timing
    return (p, o.model()), z3_status, _timing


class SMT4(BoundSolver):
    name = "smt4"

    def __init__(
        self,
        num_moves: int = 10,
        swaps_per_step: int = 3,
        neighbours: Callable[
            [int, int, int, int, int], list[tuple[int, int]]
        ] = von_neumann_neighbours,
    ) -> None:
        self.num_moves = num_moves
        self.neighbours = neighbours
        self.swaps_per_step = swaps_per_step

    @override
    # TODO(fairness)[high]: phi9+phi12 symmetry-breaking may eliminate all optima on small/dense grids
    # TODO(fairness)[medium]: maximize uses BoolRef * int (non-idiomatic, possibly version-dependent)
    # TODO(fairness)[medium]: unknown silently degrades to identity solution
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
        result, z3_status, _timing = solve_smt4(
            problem.num_rows,
            problem.num_cols,
            items,
            self.num_moves,
            self.swaps_per_step,
            test_similarity,
            neighbour_fn,
            timeout=timeout,
            seed=seed,
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
            for l1 in range(1, self.swaps_per_step + 1):
                swaps.append(
                    Swap(
                        pos1=Pos(
                            row=int(model[p[1, "row", k, l1]].__str__()) - 1,
                            col=int(model[p[1, "col", k, l1]].__str__()) - 1,
                        ),
                        pos2=Pos(
                            row=int(model[p[2, "row", k, l1]].__str__()) - 1,
                            col=int(model[p[2, "col", k, l1]].__str__()) - 1,
                        ),
                    )
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
    parser.add_argument("-l", "--swaps_per_step", type=int, default=3)
    parser.add_argument("-o", "--output", type=str, default="solution")
    args = parser.parse_args()

    problem = load(Path(args.input))
    solver = SMT4(num_moves=args.moves, swaps_per_step=args.swaps_per_step)
    result = solver.solve(problem.problem)
    save(Path(args.output), result)
    print(
        f"Score: {result.initial_score} -> {result.final_score} "
        f"({len(result.moves)} moves)"
    )


if __name__ == "__main__":
    main()
