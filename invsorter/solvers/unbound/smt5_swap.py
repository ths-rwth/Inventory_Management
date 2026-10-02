"""SMT5SwapRouter — Z3 SMT swap-sequence search to a target EndState.

Lifted from invsorter/solvers/smt5_unbounded_k.py:solve_smt5.
"""

from __future__ import annotations

import random
import sys
import time
from typing import Callable, ClassVar

from z3 import And, Bool, Int, Not, Optimize, Or, Implies, Sum, sat, unsat

from . import UnboundSolver
from ...models import EndState, GenerateResult, Item, Pos, Problem, Status, Swap
from .._engine_utils import apply_seed_z3, apply_timeout_z3
from .._utils import get_max_type, inv, von_neumann_neighbours


class SMT5SwapRouter(UnboundSolver):
    name = "smt5_swap"
    description = "Z3 SMT swap-sequence search; unfolding_depth = #items."

    def __init__(
        self,
        neighbours: Callable[
            [int, int, int, int, int], list[tuple[int, int]]
        ] = von_neumann_neighbours,
        reorder: bool = True,
    ) -> None:
        self.neighbours = neighbours
        self.reorder = reorder

    # TODO(fairness)[medium]: stdout debug spam skews wall-clock
    def generate_moves(
        self,
        problem: Problem,
        target: EndState,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        type_counts = {}
        for item in problem.items:
            if item.type not in type_counts:
                type_counts[item.type] = 1
            else:
                type_counts[item.type] += 1

        # IDs are 0-based; no rewrite needed.
        items = [
            Item(id=item.id, pos=item.pos, size=item.size, type=item.type)
            for item in problem.items
        ]
        types = list(set([item.type for item in items]))

        max_type = get_max_type(items)
        empty = max_type + 1

        final_positions = {}
        for row in range(1, problem.num_rows + 1):
            for col in range(1, problem.num_cols + 1):
                t = target.grid[row - 1][col - 1]
                final_positions[row, col] = empty if t == 0 else t

        reorder = self.reorder

        if reorder:
            num_empty = problem.num_rows * problem.num_cols - sum(
                [type_counts[t] for t in types]
            )
            types.append(empty)
            type_counts[empty] = num_empty
            types = sorted(types, key=type_counts.get)

            temp_types = []
            temp_counts = {}
            temp_items = []
            temp_f_pos = {}

            i = 1
            for t in types:
                temp_types.append(i)
                temp_counts[i] = type_counts[t]
                i += 1
            for item in items:
                temp_items.append(
                    Item(
                        id=item.id,
                        type=temp_types[types.index(item.type)],
                        pos=item.pos,
                        size=item.size,
                    )
                )
            for row in range(1, problem.num_rows + 1):
                for col in range(1, problem.num_cols + 1):
                    temp_f_pos[row, col] = temp_types[
                        types.index(final_positions[row, col])
                    ]

            empty = temp_types[types.index(empty)]
            items = temp_items
            removed = temp_types.pop()
            del temp_counts[removed]
            types = temp_types
            type_counts = temp_counts
            final_positions = temp_f_pos

        unfolding_depth = len(types)

        # --------------------------- create variables ---------------------------
        t_start = {}
        p = {}
        t = {}
        delta = {}

        # generate constants (dicts) for t_start and t_end, as they are given now
        for item in items:
            t_start[item.pos.row + 1, item.pos.col + 1] = item.type
        for row in range(1, problem.num_rows + 1):
            for col in range(1, problem.num_cols + 1):
                if (row, col) not in [
                    (item.pos.row + 1, item.pos.col + 1) for item in items
                ]:
                    t_start[row, col] = empty
        t_end = final_positions

        for k in range(unfolding_depth):
            for l1 in range(1, type_counts[k + 1] + 1):
                # we can fix the type of the item at the first swap position
                t[1, k, l1] = k + 1

        for k in range(unfolding_depth):
            for l1 in range(1, type_counts[k + 1] + 1):
                t[2, k, l1] = Int(f"t_{2}_{k}_{l1}")
                for h in [1, 2]:
                    p[h, "row", k, l1] = Int(f"p_{h}_row_{k}_{l1}")
                    p[h, "col", k, l1] = Int(f"p_{h}_col_{k}_{l1}")

                delta[k, l1] = Bool(f"delta_{k}_{l1}")

        # --------------------------- build formula ---------------------------

        # all swap positions are within the grid bounds
        phi5 = And(
            [
                And(
                    [
                        p[h, "row", k, l1] >= 1,
                        p[h, "col", k, l1] >= 1,
                        p[h, "row", k, l1] <= problem.num_rows,
                        p[h, "col", k, l1] <= problem.num_cols,
                    ]
                )
                for k in range(unfolding_depth)
                for l1 in range(1, type_counts[k + 1] + 1)
                for h in [1, 2]
            ]
        )

        # all swaps that happen in one time step are independent of each other
        phi6 = And(
            [
                Or(
                    p[h1, "row", k, l1] != p[h2, "row", k, l2],
                    p[h1, "col", k, l1] != p[h2, "col", k, l2],
                )
                for k in range(unfolding_depth)
                for l1 in range(1, type_counts[k + 1] + 1)
                for l2 in range(l1 + 1, type_counts[k + 1] + 1)
                for h1 in [1, 2]
                for h2 in [1, 2]
            ]
        )

        # for all time steps (k to k + 1) and all swaps l in these time steps we have that
        # if position h of swap l in time step (k to k + 1) was not part of any swap l' in a time
        # step (k' to k' + 1) that happened before, then the type of the item at swap position h
        # of the swap l in time step (k to k + 1) is the same as the type of the item that was
        # there at the beginning
        phi7 = And(
            [
                Implies(
                    And(
                        [
                            Or(
                                p[h1, "row", k1, l1] != p[h2, "row", k2, l2],
                                p[h1, "col", k1, l1] != p[h2, "col", k2, l2],
                            )
                            for k2 in range(k1)
                            for l2 in range(1, type_counts[k2 + 1] + 1)
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
                            for row in range(1, problem.num_rows + 1)
                            for col in range(1, problem.num_cols + 1)
                        ]
                    ),
                )
                for k1 in range(unfolding_depth)
                for l1 in range(1, type_counts[k1 + 1] + 1)
                for h1 in [1, 2]
            ]
        )

        # for all pairs of swaps (k to k + 1), (k' to k' + 1) we have that if position h of swap
        # (k to k + 1) is equal to position h' of swap (k' to k' + 1) and this position is not part
        # of any swap in-between those two swaps, then the type of the item at position h′ in swap
        # (k' to k' + 1) is equal to the type of the item at position inv(h) in swap (k to k + 1).
        phi8 = And(
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
                                    for l3 in range(1, type_counts[k3 + 1] + 1)
                                    for h3 in [1, 2]
                                ]
                            ),
                        ]
                    ),
                    t[h2, k2, l2] == t[inv(h1), k1, l1],
                )
                for k1 in range(unfolding_depth)
                for k2 in range(k1 + 1, unfolding_depth)
                for l1 in range(1, type_counts[k1 + 1] + 1)
                for l2 in range(1, type_counts[k2 + 1] + 1)
                for h1 in [1, 2]
                for h2 in [1, 2]
            ]
        )

        # for every position (r, s) and every time step (k to k + 1) and every swap l in that time step
        # we have that if swap l of step (k to k + 1) is the last swap in which (r, s) is involved
        # and (r, s) is swap position h of swap l in step (k to k + 1),
        # then the type of the item at position (r, s) at the end is equal to the type of the item
        # at swap position inv(h) of swap l in time step (k to k + 1)
        phi10 = And(
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
                                    for l2 in range(1, type_counts[k2 + 1] + 1)
                                    for h2 in [1, 2]
                                ]
                            ),
                        ]
                    ),
                    t_end[row, col] == t[inv(h1), k1, l1],
                )
                for row in range(1, problem.num_rows + 1)
                for col in range(1, problem.num_cols + 1)
                for h1 in [1, 2]
                for k1 in range(unfolding_depth)
                for l1 in range(1, type_counts[k1 + 1] + 1)
            ]
        )

        # We define the meaning of the delta variables as representing no-op swaps:
        phi11 = And(
            [
                delta[k, l1]
                == And(
                    p[1, "row", k, l1] == p[2, "row", k, l1],
                    p[1, "col", k, l1] == p[2, "col", k, l1],
                )
                for k in range(unfolding_depth)
                for l1 in range(1, type_counts[k + 1] + 1)
            ]
        )

        # The only no-op swaps are swaps of a position with itself:
        phi12 = And(
            [
                Implies(Not(delta[k, l1]), t[1, k, l1] != t[2, k, l1])
                for k in range(unfolding_depth)
                for l1 in range(1, type_counts[k + 1] + 1)
            ]
        )

        # Swaps in time step k only swap positions containing items of type k + 1 with positions
        # containing items of a type >= k + 1 (or empty positions)
        phi13 = And(
            [
                t[2, k, l1] >= k + 1
                for k in range(unfolding_depth)
                for l1 in range(1, type_counts[k + 1] + 1)
            ]
        )

        # We demand that swaps that are done during a single time step are ordered lexicographically
        phi14 = And(
            [
                Or(
                    p[1, "row", k, l1] < p[1, "row", k, l2],
                    And(
                        p[1, "row", k, l1] == p[1, "row", k, l2],
                        p[1, "col", k, l1] < p[1, "col", k, l2],
                    ),
                )
                for k in range(unfolding_depth)
                for l1 in range(1, type_counts[k + 1] + 1)
                for l2 in range(l1 + 1, type_counts[k + 1] + 1)
            ]
        )

        # Symmetry breaking: for two swaps of items of the same type in time step k,
        # the second position of the first swap is lexicographically smaller than the
        # second position of the second swap. Not currently included in active constraints.
        phi15 = And(
            [
                Implies(
                    t[2, k, l1] == t[2, k, l2],
                    Or(
                        p[2, "row", k, l1] < p[2, "row", k, l2],
                        And(
                            p[2, "row", k, l1] == p[2, "row", k, l2],
                            p[2, "col", k, l1] < p[2, "col", k, l2],
                        ),
                    ),
                )
                for k in range(unfolding_depth)
                for l1 in range(1, type_counts[k + 1] + 1)
                for l2 in range(l1 + 1, type_counts[k + 1] + 1)
            ]
        )

        o = Optimize()
        constraints = [phi5, phi6, phi7, phi8, phi10, phi11, phi12, phi13, phi14]

        o.add(And(constraints))

        maximize = o.maximize(
            Sum(
                [
                    delta[k, l1]
                    for k in range(unfolding_depth)
                    for l1 in range(1, type_counts[k + 1] + 1)
                ]
            )
        )

        apply_timeout_z3(o, timeout)
        seed_val = rng.randint(0, 2**31 - 1)
        apply_seed_z3(o, seed_val)

        start_time = time.time()
        status = o.check()
        if status == sat:
            model = o.model()

            swaps = []
            for k in range(len(types)):
                for l1 in range(1, type_counts[k + 1] + 1):
                    if not bool(model[delta[k, l1]]):
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
            return GenerateResult(tuple(swaps), Status.SOLVED)
        elif status == unsat:
            return GenerateResult((), Status.UNSAT)
        else:
            return GenerateResult((), Status.TIMEOUT)
