"""Shared helpers for wrapping solver outputs into Solution / EndState
contracts. Used by all three ABCs (BoundSolver, UnboundSolver, Generator)
and by Pipeline.

`wrap_moves_into_solution` introspects the solver's `generate_moves`
signature: if the solver declares a `timeout` parameter, the wrap threads
the wall-clock budget; otherwise it calls the legacy two-arg form. This is
pure backward-compat — no solver needs to be touched in the foundations plan.

`wrap_end_state` is the analogous helper for Generator. It returns a
`(EndState, Status)` tuple so Pipeline can fold the generator status into
the final Solution status.
"""

from __future__ import annotations

import random
from time import perf_counter
from typing import Callable

from ..grid import apply_moves
from ..models import (
    EndState,
    GenerateEndStateResult,
    GenerateResult,
    Item,
    Move,
    Problem,
    Solution,
    Status,
    Swap,
    TypeGrid,
)
from ..scoring import score, directions_for
from ._engine_utils import _accepts_timeout, filter_effective_moves


def make_rng(seed: int | None, problem: Problem) -> random.Random:
    """Mirror the old Solver.solve seed-fallback."""
    return random.Random(seed if seed is not None else problem.seed + 1000)


def _items_to_type_grid_tuple(
    items: tuple[Item, ...], num_rows: int, num_cols: int
) -> TypeGrid:
    """Build a (H, W) tuple-of-tuples TypeGrid from items. type 0 = empty."""
    grid = [[0] * num_cols for _ in range(num_rows)]
    for it in items:
        # Multi-cell items occupy a w×h rectangle anchored at it.pos.
        for dr in range(it.size.h):
            for dc in range(it.size.w):
                r = it.pos.row + dr
                c = it.pos.col + dc
                if 0 <= r < num_rows and 0 <= c < num_cols:
                    grid[r][c] = it.type
    return tuple(tuple(row) for row in grid)


def _infer_status_heuristic(
    moves: tuple[Move | Swap, ...],
    elapsed_ms: float,
    timeout: float | None,
) -> Status:
    """Infer Status when solver returned a bare tuple (no GenerateResult).

    Rule:
    - non-empty moves → SOLVED
    - empty moves AND timeout was set AND elapsed >= 0.95 * timeout → TIMEOUT
    - empty moves otherwise → ALREADY_OPTIMAL

    UNSAT can only be solver-declared via GenerateResult; it is never inferred.
    """
    if moves != ():
        return Status.SOLVED
    if timeout is not None and elapsed_ms >= timeout * 1000.0 * 0.95:
        return Status.TIMEOUT
    return Status.ALREADY_OPTIMAL


def wrap_moves_into_solution(
    problem: Problem,
    moves_fn: Callable[..., tuple[Move | Swap, ...] | GenerateResult],
    seed: int | None = None,
    timeout: float | None = None,
) -> Solution:
    """Run a moves-producing callable, wrap into a Solution.

    Signature-introspects `moves_fn`: if it accepts `timeout`, threads the
    budget through; else calls the legacy `(problem, rng)` form. Accepts bare
    tuple OR `GenerateResult` return. Computes `effective_moves` via the
    no-op-swap filter. Computes `initial_grid` / `final_grid` as TypeGrids.

    The `moves_fn` shape is `(problem, rng)` for BoundSolver and
    `(problem, target, rng)` for UnboundSolver — UnboundSolver wraps the
    method in a `lambda p, r: self.generate_moves(p, target, r)` before
    passing in (existing pattern in `unbound/__init__.py`).
    """
    rng = make_rng(seed, problem)
    initial_items = problem.items
    initial_grid = _items_to_type_grid_tuple(
        initial_items, problem.num_rows, problem.num_cols
    )
    initial_score = score(
        initial_items,
        directions=directions_for(problem.neighbors),
    )

    start = perf_counter()
    if _accepts_timeout(moves_fn):
        result = moves_fn(problem, rng, timeout=timeout)
    else:
        result = moves_fn(problem, rng)
    elapsed_ms = (perf_counter() - start) * 1000.0

    if isinstance(result, GenerateResult):
        moves: tuple[Move | Swap, ...] = result.moves
        status = result.status
    else:
        moves = result
        status = _infer_status_heuristic(moves, elapsed_ms, timeout)

    effective_moves = filter_effective_moves(moves)
    final_items = apply_moves(initial_items, moves)
    final_grid = _items_to_type_grid_tuple(
        final_items, problem.num_rows, problem.num_cols
    )
    final_score = score(
        final_items,
        directions=directions_for(problem.neighbors),
    )

    return Solution(
        problem=problem,
        moves=moves,
        initial_items=initial_items,
        final_items=final_items,
        initial_score=initial_score,
        final_score=final_score,
        initial_grid=initial_grid,
        final_grid=final_grid,
        status=status,
        effective_moves=effective_moves,
    )


def wrap_end_state(
    problem: Problem,
    end_state_fn: Callable[..., EndState | GenerateEndStateResult],
    seed: int | None = None,
    timeout: float | None = None,
) -> tuple[EndState, Status]:
    """Run a generator's `generate_end_state`, returning `(EndState, Status)`.

    Signature-introspects for `timeout` like `wrap_moves_into_solution`.
    Accepts bare `EndState` OR `GenerateEndStateResult` return. Returns the
    pair so callers (Pipeline) can fold status downstream.

    For bare `EndState` returns: SOLVED if returned, TIMEOUT inferred when
    `timeout` was set and 95% consumed. UNSAT/ALREADY_OPTIMAL never
    inferred for generators.
    """
    rng = make_rng(seed, problem)
    start = perf_counter()
    if _accepts_timeout(end_state_fn):
        result = end_state_fn(problem, rng, timeout=timeout)
    else:
        result = end_state_fn(problem, rng)
    elapsed_ms = (perf_counter() - start) * 1000.0

    if isinstance(result, GenerateEndStateResult):
        return result.end_state, result.status
    # Bare EndState
    if timeout is not None and elapsed_ms >= timeout * 1000.0 * 0.95:
        return result, Status.TIMEOUT
    return result, Status.SOLVED
