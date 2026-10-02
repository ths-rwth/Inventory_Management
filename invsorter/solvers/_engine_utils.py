"""Backend-neutral helpers for threading wall-clock timeout and seed into the
optimization engines used by invsorter solvers (Z3, Gurobi, CP-SAT).

All helpers are free-standing functions; no inheritance constraints.
Each helper is a no-op when its primary argument is None — solvers can call
unconditionally without checking budget presence themselves.

Solvers opt in by calling the appropriate helper inside their
`generate_moves` body.

Module import-safety: this module deliberately does NOT import z3, gurobipy
or ortools at module top. Each helper accepts a pre-constructed backend
object (`o`, `model`, `solver`) and operates on it. This keeps the module
loadable even when none of the solver backends are installed.
"""

from __future__ import annotations

import inspect as _inspect
from typing import Any as _Any, Callable as _Callable

from ..models import Move, Swap


# --- Z3 (Optimize / Solver) ---

def apply_timeout_z3(o: _Any, timeout_s: float | None) -> None:
    """Set Z3 wall-clock timeout. Z3 expects milliseconds.

    Pass the result of `Optimize()` or `Solver()` as `o`. No-op if
    `timeout_s is None`.
    """
    if timeout_s is None:
        return
    o.set("timeout", int(timeout_s * 1000))


def apply_seed_z3(o: _Any, seed: int | None) -> None:
    """Best-effort Z3 random_seed wiring; silently no-op on Optimize.

    No-op if `seed is None`.

    Z3 `Solver` instances accept the bare `random_seed` parameter and this
    helper wires it. Z3 `Optimize` instances do NOT expose ANY settable
    random_seed parameter — verified empirically: `Optimize.help()` lists
    zero seed-related parameters, every candidate name (`random_seed`,
    `smt.random_seed`, `sat.random_seed`, `maxres.*.random_seed`, etc.) is
    rejected with `unknown parameter`, and module-level
    `z3.set_param("smt.random_seed", N)` does not propagate determinism
    into Optimize either. For Optimize-based solvers (sat, sat2, sat_type,
    most SMT-family solvers), this helper silently no-ops on the
    unknown-parameter exception.

    Optimize-based solvers are inherently nondeterministic at the
    move-sequence level. The Z3 maintainer's claim that Optimize is
    deterministic for fixed input (issue #1821) does NOT hold for all
    encodings — verified empirically that some Optimize encodings (notably
    uninterpreted-function-based ones like bound/uf, and the smt2 encoding)
    produce different final_scores across same-seed runs on non-trivial
    problems. Smaller / simpler problems are typically stable. Callers that
    need final_score or move-sequence determinism on Optimize must either
    use a plain Solver (which DOES accept random_seed) and encode
    optimization manually, or accept some run-to-run variance and report
    distributional metrics rather than point estimates.
    """
    if seed is None:
        return
    try:
        o.set("random_seed", seed)
    except Exception:
        # Z3 Optimize does not accept random_seed in any form. Silently no-op
        # so callers can use a uniform helper across Solver and Optimize.
        pass


# --- Gurobi (Model) ---

def apply_timeout_gurobi(model: _Any, timeout_s: float | None) -> None:
    """Set Gurobi `Params.TimeLimit` (seconds). No-op if `timeout_s is None`."""
    if timeout_s is None:
        return
    model.Params.TimeLimit = float(timeout_s)


def apply_seed_gurobi(model: _Any, seed: int | None) -> None:
    """Set Gurobi `Params.Seed`. Modulo-clamped to Gurobi's 0..2_000_000_000
    range. No-op if `seed is None`."""
    if seed is None:
        return
    model.Params.Seed = int(seed) % 2_000_000_000

def apply_quiet_gurobi(model: _Any) -> None:
    """Suppress all Gurobi console output (license banner, solver log, etc.)."""
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0

# --- CP-SAT (CpSolver) ---

def apply_timeout_cpsat(solver: _Any, timeout_s: float | None) -> None:
    """Set CP-SAT `parameters.max_time_in_seconds`. No-op if `timeout_s is None`."""
    if timeout_s is None:
        return
    solver.parameters.max_time_in_seconds = float(timeout_s)


def apply_seed_cpsat(solver: _Any, seed: int | None) -> None:
    """Set CP-SAT `parameters.random_seed`. No-op if `seed is None`."""
    if seed is None:
        return
    solver.parameters.random_seed = int(seed)


# --- Signature introspection ---

_SIGNATURE_CACHE: dict[_Callable[..., _Any], bool] = {}


def _accepts_timeout(fn: _Callable[..., _Any]) -> bool:
    """Return True iff `fn`'s signature has a parameter LITERALLY named
    `timeout`. Result is memoized per callable identity.

    Used by `_solution_builder.wrap_moves_into_solution` /
    `wrap_end_state` (to decide whether to thread `timeout` into the
    generate_* call) AND by every ABC's `solve` method (to decide whether
    the inner lambda should forward `timeout` to the concrete method).
    Single source of truth — both layers read the same cache.

    Bound-method identity: bound-method `__hash__` / `__eq__` are based on
    `(__self__, __func__)`, so distinct attribute accesses returning
    structurally-equivalent bound methods hit the same cache entry. The
    cache stores one entry per (instance, method) pair across loops.
    `inspect.signature` cost is one-shot per pair and negligible vs.
    solver runtime.

    Limitation — `**kwargs` blind spot: a solver written as
    `def generate_moves(self, problem, rng, **kwargs):` has a VAR_KEYWORD
    parameter named `kwargs`, NOT `timeout` — this function returns False
    and the wrap takes the no-forward path. Solvers that want to opt into
    `timeout` threading MUST declare `timeout` explicitly in the signature.
    """
    cached = _SIGNATURE_CACHE.get(fn)
    if cached is not None:
        return cached
    try:
        sig = _inspect.signature(fn)
        result = "timeout" in sig.parameters
    except (TypeError, ValueError):
        result = False
    _SIGNATURE_CACHE[fn] = result
    return result


# --- Move filtering ---

def is_noop_swap(move: Move | Swap) -> bool:
    """A swap whose two positions coincide is a no-op. Encoding-forced padding
    (smt1, cp, etc.) commonly emits Swap(Pos(0,0), Pos(0,0)) for unused k slots.
    """
    return isinstance(move, Swap) and move.pos1 == move.pos2


def filter_effective_moves(
    moves: tuple[Move | Swap, ...],
) -> tuple[Move | Swap, ...]:
    """Drop no-op swaps; preserve everything else (including all Move values
    and any Swap with distinct endpoints)."""
    return tuple(m for m in moves if not is_noop_swap(m))
