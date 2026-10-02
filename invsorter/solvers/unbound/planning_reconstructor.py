"""PlanningReconstructor — planning swap reconstructor lifted from twolayer._reconstruct_planning."""

from __future__ import annotations

import random

try:
    import unified_planning as up
    from unified_planning.shortcuts import (
        BoolType, Equals, Fluent, InstantaneousAction,
        MinimizeSequentialPlanLength, Not, Object,
        OneshotPlanner, OptimalityGuarantee,
        Problem as UPProblem, UserType,
    )
    _BACKEND_AVAILABLE = True
    _BACKEND_ERROR = None
except ImportError as e:
    _BACKEND_AVAILABLE = False
    _BACKEND_ERROR = str(e)

from . import UnboundSolver
from ...models import EndState, GenerateResult, Item, Pos, Problem, Status, Swap, TypeGrid


def _build_type_grid(
    items: tuple[Item, ...],
    num_rows: int,
    num_cols: int,
) -> TypeGrid:
    grid = [[0] * num_cols for _ in range(num_rows)]
    for it in items:
        grid[it.pos.row][it.pos.col] = it.type
    return tuple(tuple(row) for row in grid)


class PlanningReconstructor(UnboundSolver):
    name = "planning_reconstructor"
    description = "Planning-based swap reconstruction toward a target type-grid."

    def __init__(self) -> None:
        if not _BACKEND_AVAILABLE:
            raise ImportError(
                f"planning_reconstructor requires unified-planning and up-fast-downward. "
                f"Install with: pip install unified-planning up-fast-downward. "
                f"Original error: {_BACKEND_ERROR}"
            )

    # TODO(fairness)[medium]: mismatched cells include empty cells but swap action has no empty-cell semantics
    def generate_moves(
        self,
        problem: Problem,
        target: EndState,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        init_grid = _build_type_grid(problem.items, problem.num_rows, problem.num_cols)
        goal_grid = target.grid

        H = len(init_grid)
        W = len(init_grid[0])

        up.shortcuts.get_environment().credits_stream = None

        mismatched: list[Pos] = []
        types_involved: set[int] = set()
        for r in range(H):
            for c in range(W):
                if init_grid[r][c] != goal_grid[r][c]:
                    mismatched.append(Pos(r, c))
                    types_involved.add(init_grid[r][c])
                    types_involved.add(goal_grid[r][c])

        if not mismatched:
            return GenerateResult((), Status.SOLVED)

        problem = UPProblem('swap_reconstruction')
        CellType = UserType('Cell')
        IType = UserType('IType')

        has = Fluent('has', BoolType(), cell=CellType, itype=IType)
        problem.add_fluent(has, default_initial_value=False)

        cell_objs: dict[Pos, Object] = {}
        for p in mismatched:
            cell_objs[p] = Object(f'c_{p.row}_{p.col}', CellType)
        problem.add_objects(list(cell_objs.values()))

        type_objs: dict[int, Object] = {}
        for t in sorted(types_involved):
            type_objs[t] = Object(f't{t}', IType)
        problem.add_objects(list(type_objs.values()))

        swap_act = InstantaneousAction(
            'swap', c1=CellType, c2=CellType, t1=IType, t2=IType,
        )
        c1_p = swap_act.parameter('c1')
        c2_p = swap_act.parameter('c2')
        t1_p = swap_act.parameter('t1')
        t2_p = swap_act.parameter('t2')
        swap_act.add_precondition(has(c1_p, t1_p))
        swap_act.add_precondition(has(c2_p, t2_p))
        swap_act.add_precondition(Not(Equals(t1_p, t2_p)))
        swap_act.add_precondition(Not(Equals(c1_p, c2_p)))
        swap_act.add_effect(has(c1_p, t1_p), False)
        swap_act.add_effect(has(c1_p, t2_p), True)
        swap_act.add_effect(has(c2_p, t2_p), False)
        swap_act.add_effect(has(c2_p, t1_p), True)
        problem.add_action(swap_act)

        for p in mismatched:
            problem.set_initial_value(
                has(cell_objs[p], type_objs[init_grid[p.row][p.col]]), True,
            )
            problem.add_goal(
                has(cell_objs[p], type_objs[goal_grid[p.row][p.col]]),
            )

        problem.add_quality_metric(MinimizeSequentialPlanLength())

        with OneshotPlanner(
            name='fast-downward-opt',
            optimality_guarantee=OptimalityGuarantee.SOLVED_OPTIMALLY,
        ) as planner:
            result = planner.solve(problem, timeout=timeout)

        if result.plan is None:
            return GenerateResult((), Status.TIMEOUT)

        swaps: list[Swap] = []
        for action in result.plan.actions:
            params = action.actual_parameters
            r1, c1 = str(params[0]).split('_')[1:]
            r2, c2 = str(params[1]).split('_')[1:]
            swaps.append(Swap(Pos(int(r1), int(c1)), Pos(int(r2), int(c2))))
        return GenerateResult(tuple(swaps), Status.SOLVED)
