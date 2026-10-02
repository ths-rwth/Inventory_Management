from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple


class Pos(NamedTuple):
    row: int
    col: int


class Size(NamedTuple):
    w: int
    h: int


class Item(NamedTuple):
    id: int
    type: int
    pos: Pos
    size: Size


class Move(NamedTuple):
    item_id: int
    target: Pos


class Swap(NamedTuple):
    pos1: Pos
    pos2: Pos


@dataclass(frozen=True, slots=True)
class Problem:
    num_cols: int
    num_rows: int
    items: tuple[Item, ...]
    neighbors: int = 4          # 4 = von Neumann, 8 = Moore
    seed: int = 0

    def __post_init__(self):
        if self.neighbors not in (4, 8):
            raise ValueError(
                f"neighbors must be 4 or 8, got {self.neighbors}"
            )


TypeGrid = tuple[tuple[int, ...], ...]


class Status(StrEnum):
    """Outcome of a solver invocation. Set by the solver (via GenerateResult /
    GenerateEndStateResult) or inferred by the wrap helpers when the solver
    returns a bare tuple.

    SOLVED          — solver returned a non-empty plan.
    ALREADY_OPTIMAL — solver returned () and budget was not consumed.
    TIMEOUT         — wall-clock budget exhausted (declared or inferred).
    UNSAT           — solver-declared infeasibility / unreachable target.
    CRASHED         — set by benchmark.py wrapper on caught Exception.
    UNAVAILABLE     — set by benchmark.py wrapper on ImportError (missing dep).
    """

    SOLVED = "solved"
    OPTIMAL = "optimal"
    ALREADY_OPTIMAL = "already_optimal"
    TIMEOUT = "timeout"
    UNSAT = "unsat"
    CRASHED = "crashed"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    BUDGET_EXCEEDED = "budget_exceeded"


class GenerateResult(NamedTuple):
    """Optional return shape from `BoundSolver.generate_moves` /
    `UnboundSolver.generate_moves`. When a solver returns this, the wrap
    helper trusts the embedded status verbatim. Bare-tuple returns continue
    to work — status is inferred heuristically by `_solution_builder`."""

    moves: tuple[Move | Swap, ...]
    status: Status


class GenerateEndStateResult(NamedTuple):
    """Optional return shape from `Generator.generate_end_state`.
    Same trust contract as GenerateResult."""

    end_state: "EndState"
    status: Status


@dataclass(frozen=True, slots=True)
class Solution:
    problem: Problem
    moves: tuple[Move, ...] | tuple[Swap, ...]   # raw, as solver returned
    initial_items: tuple[Item, ...]               # NEW (required)
    final_items: tuple[Item, ...]
    initial_score: int
    final_score: int
    initial_grid: TypeGrid = ()                   # NEW; default empty
    final_grid: TypeGrid = ()                     # NEW
    status: Status = Status.SOLVED                # NEW
    effective_moves: tuple[Move, ...] | tuple[Swap, ...] = ()  # NEW; filtered no-ops


@dataclass(frozen=True, slots=True)
class EndState:
    """A target inventory configuration for an UnboundSolver to reconstruct.

    grid: TypeGrid
        Row-major (H, W) tuple of tuples of int type IDs.
        type 0 = empty cell. This convention is uniform across all unbound
        consumers; solvers may remap internally but the EndState shape itself
        is canonical.

    neighbors: int
        Adjacency graph used to produce this EndState. 4 = von Neumann,
        8 = Moore. Default 4. Propagated to Problem when the EndState is
        consumed downstream.
    """

    grid: TypeGrid
    neighbors: int = 4
