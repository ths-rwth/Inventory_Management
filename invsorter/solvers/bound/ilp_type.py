import argparse
import random
import time
from typing import Callable, override
from gurobipy import GRB, Model, quicksum
from pathlib import Path

from ...io import load, save
from ...models import Problem, Item, Pos, Swap, GenerateResult, Status
from . import BoundSolver
from .._utils import (
    moore_neighbours,
    von_neumann_neighbours,
    get_neighbours,
    test_similarity,
)
from .._engine_utils import apply_timeout_gurobi, apply_seed_gurobi, apply_quiet_gurobi


def solve_ilp2(
    num_rows: int,
    num_cols: int,
    items: list[Item],
    unfolding_depth,
    similarity: Callable[[int, int], int],
    neighbours: Callable[[int, int, int, int, int], list[tuple[int, int]]],
    types: list[int],
    type_counts: dict[int, int],
    timeout: float | None = None,
    seed: int | None = None,
    *,
    no_diff_noop: bool = False,
    upper_left_noop: bool = True,
    cardinality_constraints: bool = True,
) -> tuple[Model, dict[str, float]]:

    _timing = {}
    _t0 = time.perf_counter()
    x = {}
    y = {}
    z = {}
    delta = {}

    model = Model("inventory_sorting")
    apply_quiet_gurobi(model)

    # ----- Variables -----
    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            for t1 in types:
                for row2, col2 in neighbours(num_rows, num_cols, row, col, 1):
                    for t2 in types:
                        y[row, col, row2, col2, t1, t2] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"y_{row}_{col}_{row2}_{col2}_{t1}_{t2}",
                        )
                        # iff x[row, col, item, k] = x[row2, col2, item2] = 1 so iff at time unfolding_depth item is at
                        # (col, row) and item2 is at (col2, row2) (linearization of the logic "And")
                for k in range(unfolding_depth + 1):
                    x[row, col, t1, k] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_{row}_{col}_{t1}_{k}"
                    )

    # z is type-INDEPENDENT (keyed only by positions and time) — must be
    # created once per (row, col, row2, col2, k). Previously this lived inside
    # the type loop, producing len(types) duplicate Gurobi variables under the
    # same name; downstream readback via model.getVarByName returned an
    # unconstrained zero-valued duplicate instead of the constrained one.
    for k in range(unfolding_depth):
        for row in range(1, num_rows + 1):
            for col in range(1, num_cols + 1):
                for row2 in range(1, num_rows + 1):
                    for col2 in range(1, num_cols + 1):
                        if (row, col) <= (row2, col2):
                            if upper_left_noop:
                                if (row, col) != (row2, col2) or (row, col) == (1, 1):
                                    z[row, col, row2, col2, k] = model.addVar(
                                        vtype=GRB.BINARY,
                                        name=f"z_{row}_{col}_{row2}_{col2}_{k}",
                                    )
                                else:
                                    z[row, col, row2, col2, k] = 0
                            else:
                                z[row, col, row2, col2, k] = model.addVar(
                                    vtype=GRB.BINARY,
                                    name=f"z_{row}_{col}_{row2}_{col2}_{k}",
                                )

    # ----- Objective -----
    # we want to maximize the similarity between an item and its surrounding items
    model.setObjective(
        quicksum(
            (y[row, col, row2, col2, t1, t2] * similarity(t1, t2))
            for row in range(1, num_rows + 1)
            for col in range(1, num_cols + 1)
            for t1 in types
            for (row2, col2) in neighbours(num_rows, num_cols, row, col, 1)
            for t2 in types
        ),
        GRB.MAXIMIZE,
    )

    # ----- Constraints -----
    # at k=0 the given items are at their given positions and all other positions are occupied by the respective
    # placeholder items
    model.addConstrs(
        (
            x[(item.pos.row + 1), (item.pos.col + 1), item.type, 0] == 1
            for item in items
        ),
        name="start_positions",
    )
    model.addConstrs(
        x[row, col, t, 0] == 0
        for t in types
        for (row, col) in [
            (r + 1, c + 1)
            for r in range(num_rows)
            for c in range(num_cols)
            if (r, c) not in [(item.pos.row, item.pos.col) for item in items]
        ]
    )

    model.addConstrs(
        quicksum(x[row, col, t, k] for t in types) <= 1
        for k in range(unfolding_depth + 1)
        for row in range(1, num_rows + 1)
        for col in range(1, num_cols + 1)
    )

    # encoding the logical "And" for the y-Variables
    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            for row2, col2 in neighbours(num_rows, num_cols, row, col, 1):
                for t1 in types:
                    for t2 in types:
                        model.addConstr(
                            (
                                y[row, col, row2, col2, t1, t2]
                                <= x[row, col, t1, unfolding_depth]
                            ),
                            name=f"And1_{row}_{col}_{row2}_{col2}_{t1}_{t2}",
                        )
                        model.addConstr(
                            (
                                y[row, col, row2, col2, t1, t2]
                                <= x[row2, col2, t2, unfolding_depth]
                            ),
                            name=f"And2_{row}_{col}_{row2}_{col2}_{t1}_{t2}",
                        )
                        model.addConstr(
                            (
                                y[row, col, row2, col2, t1, t2]
                                >= x[row, col, t1, unfolding_depth]
                                + x[row2, col2, t2, unfolding_depth]
                                - 1
                            ),
                            name=f"And3_{row}_{col}_{row2}_{col2}_{t1}_{t2}",
                        )

    # in every time step (k to k+1) we do exactly one swap operation
    model.addConstrs(
        (
            quicksum(
                z[row, col, row2, col2, k]
                for row in range(1, num_rows + 1)
                for col in range(1, num_cols + 1)
                for (row2, col2) in [
                    (r, c)
                    for r in range(1, num_rows + 1)
                    for c in range(1, num_cols + 1)
                    if (row, col) <= (r, c)
                ]
            )
            == 1
            for k in range(unfolding_depth)
        ),
        name="Exactly one swap",
    )

    # encoding of the swapping decision
    for k in range(unfolding_depth):
        for row in range(1, num_rows + 1):
            for col in range(1, num_cols + 1):
                for row2 in range(1, num_rows + 1):
                    for col2 in range(1, num_cols + 1):
                        if (row, col) <= (row2, col2):
                            for t in types:
                                # z[row, col, row2, col2, k] implies x[row, col, item, k] = x[row2, col2, item, k + 1]
                                model.addConstr(
                                    x[row, col, t, k]
                                    >= x[row2, col2, t, k + 1]
                                    + z[row, col, row2, col2, k]
                                    - 1
                                )
                                model.addConstr(
                                    x[row, col, t, k]
                                    <= x[row2, col2, t, k + 1]
                                    - z[row, col, row2, col2, k]
                                    + 1
                                )

                                # z[row, col, row2, col2, k] implies x[row2, col2, item, k] = x[row, col, item, k + 1]
                                model.addConstr(
                                    x[row2, col2, t, k]
                                    >= x[row, col, t, k + 1]
                                    + z[row, col, row2, col2, k]
                                    - 1
                                )
                                model.addConstr(
                                    x[row2, col2, t, k]
                                    <= x[row, col, t, k + 1]
                                    - z[row, col, row2, col2, k]
                                    + 1
                                )

    # if a position is not part of a swap operation in time step (k to k+1) then its occupancy does not change
    for k in range(unfolding_depth):
        for t in types:
            for row in range(1, num_rows + 1):
                for col in range(1, num_cols + 1):
                    model.addConstr(
                        x[row, col, t, k]
                        >= x[row, col, t, k + 1]
                        - quicksum(
                            z[row, col, row2, col2, k]
                            for (row2, col2) in [
                                (r, c)
                                for r in range(1, num_rows + 1)
                                for c in range(1, num_cols + 1)
                                if (row, col) <= (r, c)
                            ]
                        )
                        - quicksum(
                            z[row2, col2, row, col, k]
                            for (row2, col2) in [
                                (r, c)
                                for r in range(1, num_rows + 1)
                                for c in range(1, num_cols + 1)
                                if (r, c) < (row, col)
                            ]
                        )
                    )

                    model.addConstr(
                        x[row, col, t, k]
                        <= x[row, col, t, k + 1]
                        + quicksum(
                            z[row, col, row2, col2, k]
                            for (row2, col2) in [
                                (r, c)
                                for r in range(1, num_rows + 1)
                                for c in range(1, num_cols + 1)
                                if (row, col) <= (r, c)
                            ]
                        )
                        + quicksum(
                            z[row2, col2, row, col, k]
                            for (row2, col2) in [
                                (r, c)
                                for r in range(1, num_rows + 1)
                                for c in range(1, num_cols + 1)
                                if (r, c) < (row, col)
                            ]
                        )
                    )

    # cardinality constraints
    if cardinality_constraints:
        model.addConstrs(
            quicksum(
                x[row, col, t, k]
                for row in range(1, num_rows + 1)
                for col in range(1, num_cols + 1)
            )
            == type_counts[t]
            for k in range(1, unfolding_depth + 1)
            for t in types
        )

    # ---------- no-op symmetry breaking constraints ----------

    # if the type of a position stays the same in a time step, then that position was not swapped with a position
    # that is not itself
    # In the tests I did this did not improve performance however it might be the case for larger instances,
    # so I left it in here with a flag

    enforce_noop_with_itself = False

    if enforce_noop_with_itself:
        model.addConstrs(
            quicksum(
                z[row2, col2, row1, col1, k]
                for (row2, col2) in [
                    (r, c)
                    for r in range(1, num_rows + 1)
                    for c in range(1, num_cols + 1)
                    if (r, c) < (row1, col1)
                ]
            )
            + quicksum(
                z[row1, col1, row2, col2, k]
                for (row2, col2) in [
                    (r, c)
                    for r in range(1, num_rows + 1)
                    for c in range(1, num_cols + 1)
                    if (row1, col1) <= (r, c)
                ]
            )
            <= 2 - x[row1, col1, t, k] - x[row1, col1, t, k + 1]
            for k in range(unfolding_depth)
            for t in types
            for row1 in range(1, num_rows + 1)
            for col1 in range(1, num_cols + 1)
        )

    if no_diff_noop:
        # but I think we can encode that in a smarter way:
        # we say that for two different positions (row1, col1) and (row2, col2) and for every type t
        # the sum x[row1, col1, t, k] + x[row2, col2, t, k] + z[row1, col1, row2, col2, k] <= 2
        # meaning that we never swap different positions that contain the same item
        model.addConstrs(
            x[row1, col1, t, k] + x[row2, col2, t, k] + z[row1, col1, row2, col2, k]
            <= 2
            for k in range(unfolding_depth)
            for row1 in range(1, num_rows + 1)
            for col1 in range(1, num_cols + 1)
            for (row2, col2) in [
                (r, c)
                for r in range(1, num_rows + 1)
                for c in range(1, num_cols)
                if (row1, col1) < (r, c)
            ]
            for t in types
        )

        # enforcing that no-ops need to happen at (1, 1)
        # Generally one should completely eliminate these variables for sparsity
        # most likely gurobi does that anyway
        # model.addConstrs(z[row, col, row, col, k] == 0
        #                  for row in range(1, num_rows + 1)
        #                  for col in range(1, num_cols + 1)
        #                  for k in range(unfolding_depth)
        #                  if (row, col) != (1, 1))
    _t1 = time.perf_counter()
    _timing["model_build_ms"] = (_t1 - _t0) * 1000.0
    model.update()
    apply_timeout_gurobi(model, timeout)
    apply_seed_gurobi(model, seed)
    model.optimize()
    _t2 = time.perf_counter()
    _timing["engine_solve_ms"] = (_t2 - _t1) * 1000.0

    return model, _timing


class ILPType(BoundSolver):
    """
    An integer (or to be more precise: binary) linear programming solver indexed by
    item TYPE rather than item id. Variables ``x[row, col, type, k]`` track which
    type occupies each cell at each step, with cardinality constraints enforcing
    that the multiset of types is preserved across swaps. This trades per-item
    identity (lost in the readback) for a smaller variable count when many items
    share the same type, exploiting the natural symmetry between
    type-indistinguishable items. Like ``ilp``, it implements the bit-blasting
    pattern as a binary ILP with Gurobi.
    """

    name = "ilp_type"

    def __init__(
        self,
        num_moves: int = 10,
        neighbours: Callable[
            [int, int, int, int, int], list[tuple[int, int]]
        ] = von_neumann_neighbours,
        *,
        no_diff_noop: bool = False,
        upper_left_noop: bool = True,
        cardinality_constraints: bool = True,
    ) -> None:
        self.num_moves = num_moves
        self.neighbours = neighbours
        self.no_diff_noop = no_diff_noop
        self.upper_left_noop = upper_left_noop
        self.cardinality_constraints = cardinality_constraints

    @override
    def generate_moves(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        items = list(problem.items)

        types = list(set([item.type for item in items]))
        type_counts = {}
        for item in items:
            if item.type not in type_counts:
                type_counts[item.type] = 1
            else:
                type_counts[item.type] += 1

        # Select neighbor function from problem config
        if problem.neighbors == 8:
            neighbour_fn = moore_neighbours
        else:
            neighbour_fn = von_neumann_neighbours
        seed_val = rng.randint(0, 2**31 - 1)
        model, _timing = solve_ilp2(
            problem.num_rows,
            problem.num_cols,
            items,
            self.num_moves,
            test_similarity,
            neighbour_fn,
            types,
            type_counts,
            timeout=timeout,
            seed=seed_val,
            no_diff_noop=self.no_diff_noop,
            upper_left_noop=self.upper_left_noop,
            cardinality_constraints=self.cardinality_constraints,
        )

        return self._readback(model, problem, _timing)

    def _readback(
        self, model, problem: Problem, _timing: dict[str, float]
    ) -> GenerateResult:
        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            _t3 = time.perf_counter()
            # decode z variables directly: exactly one z entry ≈ 1 per time step
            if self.upper_left_noop:
                z = {
                    (row, col, row2, col2, k): model.getVarByName(
                        f"z_{row}_{col}_{row2}_{col2}_{k}"
                    )
                    for row in range(1, problem.num_rows + 1)
                    for col in range(1, problem.num_cols + 1)
                    for row2 in range(1, problem.num_rows + 1)
                    for col2 in range(1, problem.num_cols + 1)
                    for k in range(self.num_moves)
                    if (row, col) <= (row2, col2)
                    if ((row, col) != (row2, col2) or (row, col) == (1, 1))
                }
            else:
                z = {
                    (row, col, row2, col2, k): model.getVarByName(
                        f"z_{row}_{col}_{row2}_{col2}_{k}"
                    )
                    for row in range(1, problem.num_rows + 1)
                    for col in range(1, problem.num_cols + 1)
                    for row2 in range(1, problem.num_rows + 1)
                    for col2 in range(1, problem.num_cols + 1)
                    for k in range(self.num_moves)
                    if (row, col) <= (row2, col2)
                }

            moves = []
            for k in range(self.num_moves):
                for (row, col, row2, col2, kk), var in z.items():
                    if kk == k and round(var.x) == 1:
                        moves.append(
                            Swap(
                                pos1=Pos(row=row - 1, col=col - 1),
                                pos2=Pos(row=row2 - 1, col=col2 - 1),
                            )
                        )
                        break
            moves = tuple(moves)
            readback_ms = (time.perf_counter() - _t3) * 1000.0
            self._timing = {
                "model_build_ms": _timing.get("model_build_ms", 0.0),
                "engine_solve_ms": _timing.get("engine_solve_ms", 0.0),
                "readback_ms": readback_ms,
            }
            if model.Status == GRB.OPTIMAL:
                return GenerateResult(moves, Status.OPTIMAL)
            else:
                return GenerateResult(moves, Status.TIMEOUT)
        else:
            self._timing = _timing
            if model.Status == GRB.INFEASIBLE:
                return GenerateResult((), Status.UNSAT)
            elif model.Status == GRB.TIME_LIMIT:
                return GenerateResult((), Status.TIMEOUT)
            else:
                return GenerateResult((), Status.TIMEOUT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integer Linear Programming Solver using Types"
    )
    parser.add_argument("input", type=str, help="Problem JSON file")
    parser.add_argument("-k", "--moves", type=int, default=10)
    parser.add_argument("-o", "--output", type=str, default="solution")
    args = parser.parse_args()

    problem = load(Path(args.input))
    solver = ILPType(num_moves=args.moves)
    result = solver.solve(problem.problem)
    save(Path(args.output), result)
    print(
        f"Score: {result.initial_score} -> {result.final_score} "
        f"({len(result.moves)} moves)"
    )


if __name__ == "__main__":
    main()
