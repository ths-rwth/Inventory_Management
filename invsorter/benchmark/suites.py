from __future__ import annotations

import random
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from itertools import product as _product

from invsorter.generate import generate
from invsorter.models import EndState, Problem, Status
from invsorter.solvers.generators.genetic_endstate import GeneticEndStateGenerator


class BoundedSuite(Enum):
    """Names of solvers in the bounded benchmark suite."""

    CP = "cp"
    GENETIC = "genetic"
    GREEDY = "greedy"
    ILP_TYPE = "ilp_type"
    ILP_TYPE_NOCARD = "ilp_type_nocard"
    SAT_TYPE = "sat_type"
    SMT3 = "smt3"
    SMT3_MAX = "smt3_max"
    SMT9 = "smt9"

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(s.value for s in cls)


class SMT4Suite(Enum):
    """SMT4 is tested separately with swaps_per_step variation."""

    SMT4 = "smt4"

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(s.value for s in cls)


class RuntimeCompareSuite(Enum):
    """Names of solvers in the runtime-compare benchmark suite."""

    GREEDY = "greedy"
    GENETIC = "genetic"

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(s.value for s in cls)


DEFAULT_DIMENSIONS: list[tuple[int, int, int, int]] = [
    (3, 3, 4, 2),
    (3, 3, 8, 2),
    (4, 4, 8, 2),
    (4, 4, 10, 3),
    (4, 4, 12, 3),
    (4, 4, 16, 4),
    (6, 6, 20, 4),
    (6, 6, 30, 5),
    (10, 10, 30, 5),
    (10, 10, 50, 8),
]
DEFAULT_KS = (1, 2, 3)
DEFAULT_NEIGHBORS = 4
INSTANCES_PER_CELL = 30
SMT4_SWAPS_PER_STEP_VALUES = (1, 2, 3, 4)

SYMMETRY_SWEEP_DIMENSIONS: list[tuple[int, int, int, int]] = [
    (3, 3, 4, 2),
    (3, 3, 8, 2),
    (4, 4, 8, 2),
    (4, 4, 12, 3),
    (5, 5, 20, 4),
    (6, 6, 30, 5),
    (8, 8, 20, 4),
    (8, 8, 48, 8),
]
SYMMETRY_SWEEP_KS: tuple[int, ...] = (1, 3)
SYMMETRY_SWEEP_INSTANCES_PER_CELL: int = 5
SYMMETRY_SWEEP_SOLVERS: tuple[str, ...] = ("ilp_type", "sat_type", "smt3")
SYMMETRY_SWEEP_ILP_ONLY_MIN_SIZE: int = 5
RUNTIME_COMPARE_DIMENSIONS: list[tuple[int, int, int, int]] = [
    (10, 10, 80, 10),
    (25, 25, 500, 10),
    (50, 50, 2000, 10),
    (100, 100, 8000, 10),
]
RUNTIME_COMPARE_KS: tuple[int, ...] = (10,)
RUNTIME_COMPARE_INSTANCES_PER_CELL: int = 2


_SYMMETRY_FLAG_NAMES_BY_SOLVER: dict[str, tuple[str, ...]] = {
    "ilp_type": ("no_diff_noop", "upper_left_noop", "cardinality_constraints"),
    "sat_type": ("no_diff_noop", "upper_left_noop"),
    "smt3": ("swap_pos_lex_order", "no_diff_noop", "upper_left_noop"),
}


def expected_symmetry_flags(solver: str) -> tuple[str, ...]:
    """Canonical flag names (in serialisation order) for *solver*.

    Raises ``KeyError`` if *solver* is not a symmetry-sweep solver. Used by
    the runner to verify a ``SymmetrySweepCell.flag_combo`` is well-formed
    before constructing the solver.
    """
    return _SYMMETRY_FLAG_NAMES_BY_SOLVER[solver]


def _enumerate_flag_combos(
    flag_names: tuple[str, ...],
) -> tuple[tuple[tuple[str, bool], ...], ...]:
    return tuple(
        tuple(zip(flag_names, values, strict=True))
        for values in _product((False, True), repeat=len(flag_names))
    )


_SYMMETRY_FLAG_COMBOS_BY_SOLVER: dict[str, tuple[tuple[tuple[str, bool], ...], ...]] = {
    solver: _enumerate_flag_combos(flags)
    for solver, flags in _SYMMETRY_FLAG_NAMES_BY_SOLVER.items()
}


def _serialize_flag_combo(combo: tuple[tuple[str, bool], ...]) -> str:
    return ",".join(f"{name}={value}" for name, value in combo)


# Per-cell wall-clock budgets. ``DEFAULT_TIMEOUT_S`` applies to bounded /
# smt4 / unbounded suites; ``RUNTIME_COMPARE_TIMEOUT_S`` is a generous
# budget so larger runtime-compare cells finish (CLI --timeout overrides).
DEFAULT_TIMEOUT_S: float = 900.0
RUNTIME_COMPARE_TIMEOUT_S: float = (
    14400.0  # 4h; generous so cells finish. CLI --timeout overrides.
)


def _stable_seed(cell_key: str, master_seed: int = 0) -> int:
    """Deterministic seed from cell_key + master_seed using CRC32 hash."""
    h = zlib.crc32((cell_key + str(master_seed)).encode("utf-8"))
    return h & 0x7FFFFFFF  # positive 31-bit


def _default_instances_for_dimension(num_rows: int, num_cols: int) -> int:
    """Return default instances for a grid dimension.

    Small grids (<=4x4) get more instances since they're faster to solve.
    Large grids (>=6x6) get fewer instances due to longer solve times.
    """
    max_dim = max(num_rows, num_cols)
    if max_dim <= 4:
        return 30
    return 10


@dataclass(frozen=True, slots=True)
class BoundedCell:
    """A single benchmark cell: one solver × one problem dimension × one k."""

    solver: str
    k: int
    num_rows: int
    num_cols: int
    num_items: int
    num_types: int
    neighbors: int
    seed: int
    instance_index: int

    @property
    def cell_key(self) -> str:
        return (
            f"bounded:{self.solver}:k{self.k}:"
            f"{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}"
        )

    @property
    def instance_key(self) -> str:
        """Identifier for the underlying problem instance (no solver, no k).

        Used as the seed source so every (solver, k) configuration in the
        suite operates on the SAME instance. The full ``cell_key`` (including
        solver and k) is still unique per row in the output JSONL.
        """
        return (
            f"bounded:"
            f"{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}"
        )

    def to_problem(self) -> Problem:
        sol = generate(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            num_items=self.num_items,
            num_types=self.num_types,
            seed=self.seed,
            neighbors=self.neighbors,
        )
        return sol.problem


@dataclass(frozen=True, slots=True)
class RuntimeCompareCell:
    """A single benchmark cell: one solver × one problem dimension × one k."""

    solver: str
    k: int
    num_rows: int
    num_cols: int
    num_items: int
    num_types: int
    neighbors: int
    seed: int
    instance_index: int

    @property
    def cell_key(self) -> str:
        return (
            f"runtime_compare:{self.solver}:k{self.k}:"
            f"{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}"
        )

    @property
    def instance_key(self) -> str:
        """Identifier for the underlying problem instance (no solver, no k).

        Used as the seed source so every (solver, k) combination operates on
        the SAME generated problem.
        """
        return (
            f"runtime_compare:"
            f"{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}"
        )

    def to_problem(self) -> Problem:
        sol = generate(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            num_items=self.num_items,
            num_types=self.num_types,
            seed=self.seed,
            neighbors=self.neighbors,
        )
        return sol.problem


@dataclass(frozen=True, slots=True)
class SMT4Cell:
    """A single benchmark cell for SMT4 with swaps_per_step variation."""

    solver: str = "smt4"
    swaps_per_step: int = 1
    num_rows: int = 4
    num_cols: int = 4
    num_items: int = 8
    num_types: int = 2
    neighbors: int = 4
    seed: int = 0
    instance_index: int = 0

    @property
    def cell_key(self) -> str:
        return (
            f"smt4:sp{self.swaps_per_step}:"
            f"{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}"
        )

    @property
    def instance_key(self) -> str:
        """Identifier for the underlying problem instance (no solver, no sps).

        Used as the seed source so every ``swaps_per_step`` configuration
        operates on the SAME instance. The full ``cell_key`` (including
        sps) is still unique per row in the output JSONL.
        """
        return (
            f"smt4:"
            f"{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}"
        )

    def to_problem(self) -> Problem:
        sol = generate(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            num_items=self.num_items,
            num_types=self.num_types,
            seed=self.seed,
            neighbors=self.neighbors,
        )
        return sol.problem


def iter_bounded_cells(
    dimensions: list[tuple[int, int, int, int]] | None = None,
    k_values: tuple[int, ...] | None = None,
    neighbors: int = DEFAULT_NEIGHBORS,
    instances_per_cell: int | None = None,
    master_seed: int = 0,
) -> Iterator[BoundedCell]:
    """
    Enumerate all valid bounded benchmark cells.

    Skips combinations where item_count > grid capacity or type_count > item_count.
    """
    if dimensions is None:
        dimensions = DEFAULT_DIMENSIONS
    if k_values is None:
        k_values = DEFAULT_KS

    for num_rows, num_cols, num_items, num_types in dimensions:
        capacity = num_rows * num_cols
        if num_items > capacity:
            continue
        if num_types > num_items:
            continue

        n_instances = (
            instances_per_cell
            if instances_per_cell is not None
            else _default_instances_for_dimension(num_rows, num_cols)
        )
        for instance_index in range(n_instances):
            # Per-instance seed key - independent of solver and k so every
            # (solver, k) combination operates on the SAME generated problem.
            instance_key = (
                f"bounded:"
                f"{num_rows}x{num_cols}:"
                f"items{num_items}:types{num_types}:"
                f"n{neighbors}:i{instance_index}"
            )
            seed = _stable_seed(instance_key, master_seed)
            for solver in BoundedSuite.names():
                for k in k_values:
                    yield BoundedCell(
                        solver=solver,
                        k=k,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        num_items=num_items,
                        num_types=num_types,
                        neighbors=neighbors,
                        seed=seed,
                        instance_index=instance_index,
                    )


def iter_runtime_compare_cells(
    dimensions: list[tuple[int, int, int, int]] | None = None,
    k_values: tuple[int, ...] | None = None,
    neighbors: int = DEFAULT_NEIGHBORS,
    instances_per_cell: int | None = None,
    master_seed: int = 0,
) -> Iterator[RuntimeCompareCell]:
    """Enumerate all valid runtime-compare benchmark cells (greedy + genetic only).

    Skips combinations where item_count > grid capacity or type_count > item_count.
    """
    if dimensions is None:
        dimensions = RUNTIME_COMPARE_DIMENSIONS
    if k_values is None:
        k_values = RUNTIME_COMPARE_KS

    for num_rows, num_cols, num_items, num_types in dimensions:
        capacity = num_rows * num_cols
        if num_items > capacity:
            continue
        if num_types > num_items:
            continue

        n_instances = (
            instances_per_cell
            if instances_per_cell is not None
            else RUNTIME_COMPARE_INSTANCES_PER_CELL
        )
        for instance_index in range(n_instances):
            # Per-instance seed key - independent of solver and k so every
            # (solver, k) combination operates on the SAME generated problem.
            instance_key = (
                f"runtime_compare:"
                f"{num_rows}x{num_cols}:"
                f"items{num_items}:types{num_types}:"
                f"n{neighbors}:i{instance_index}"
            )
            seed = _stable_seed(instance_key, master_seed)
            for solver in RuntimeCompareSuite.names():
                for k in k_values:
                    yield RuntimeCompareCell(
                        solver=solver,
                        k=k,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        num_items=num_items,
                        num_types=num_types,
                        neighbors=neighbors,
                        seed=seed,
                        instance_index=instance_index,
                    )


def iter_smt4_cells(
    dimensions: list[tuple[int, int, int, int]] | None = None,
    swaps_per_step_values: tuple[int, ...] | None = None,
    neighbors: int = DEFAULT_NEIGHBORS,
    instances_per_cell: int | None = None,
    master_seed: int = 0,
) -> Iterator[SMT4Cell]:
    """
    Enumerate all valid SMT4 benchmark cells.
    """
    if dimensions is None:
        dimensions = DEFAULT_DIMENSIONS
    if swaps_per_step_values is None:
        swaps_per_step_values = SMT4_SWAPS_PER_STEP_VALUES

    for num_rows, num_cols, num_items, num_types in dimensions:
        capacity = num_rows * num_cols
        if num_items > capacity:
            continue
        if num_types > num_items:
            continue

        n_instances = (
            instances_per_cell
            if instances_per_cell is not None
            else _default_instances_for_dimension(num_rows, num_cols)
        )
        for instance_index in range(n_instances):
            # Per-instance seed key - independent of swaps_per_step so every
            # sps configuration operates on the SAME generated problem.
            instance_key = (
                f"smt4:"
                f"{num_rows}x{num_cols}:"
                f"items{num_items}:types{num_types}:"
                f"n{neighbors}:i{instance_index}"
            )
            seed = _stable_seed(instance_key, master_seed)
            for sps in swaps_per_step_values:
                yield SMT4Cell(
                    solver="smt4",
                    swaps_per_step=sps,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    num_items=num_items,
                    num_types=num_types,
                    neighbors=neighbors,
                    seed=seed,
                    instance_index=instance_index,
                )


@dataclass(frozen=True, slots=True)
class SymmetrySweepCell:
    solver: str
    k: int
    num_rows: int
    num_cols: int
    num_items: int
    num_types: int
    neighbors: int
    seed: int
    instance_index: int
    flag_combo: str

    @property
    def cell_key(self) -> str:
        return (
            f"symmetry_sweep:{self.solver}:k{self.k}:"
            f"{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}:"
            f"flags={self.flag_combo}"
        )

    @property
    def instance_key(self) -> str:
        return (
            f"symmetry_sweep:"
            f"{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}"
        )

    def to_problem(self) -> Problem:
        sol = generate(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            num_items=self.num_items,
            num_types=self.num_types,
            seed=self.seed,
            neighbors=self.neighbors,
        )
        return sol.problem


def iter_symmetry_sweep_cells(
    dimensions: list[tuple[int, int, int, int]] | None = None,
    k_values: tuple[int, ...] | None = None,
    neighbors: int = DEFAULT_NEIGHBORS,
    instances_per_cell: int = SYMMETRY_SWEEP_INSTANCES_PER_CELL,
    master_seed: int = 0,
) -> Iterator[SymmetrySweepCell]:
    if dimensions is None:
        dimensions = SYMMETRY_SWEEP_DIMENSIONS
    if k_values is None:
        k_values = SYMMETRY_SWEEP_KS

    for num_rows, num_cols, num_items, num_types in dimensions:
        capacity = num_rows * num_cols
        if num_items > capacity:
            continue
        if num_types > num_items:
            continue

        for instance_index in range(instances_per_cell):
            instance_key = (
                f"symmetry_sweep:"
                f"{num_rows}x{num_cols}:"
                f"items{num_items}:types{num_types}:"
                f"n{neighbors}:i{instance_index}"
            )
            seed = _stable_seed(instance_key, master_seed)
            if num_rows >= SYMMETRY_SWEEP_ILP_ONLY_MIN_SIZE:
                solvers = ("ilp_type",)
            else:
                solvers = SYMMETRY_SWEEP_SOLVERS
            for solver in solvers:
                for combo in _SYMMETRY_FLAG_COMBOS_BY_SOLVER[solver]:
                    for k in k_values:
                        yield SymmetrySweepCell(
                            solver=solver,
                            k=k,
                            num_rows=num_rows,
                            num_cols=num_cols,
                            num_items=num_items,
                            num_types=num_types,
                            neighbors=neighbors,
                            seed=seed,
                            instance_index=instance_index,
                            flag_combo=_serialize_flag_combo(combo),
                        )


def dry_run(
    dimensions: list[tuple[int, int, int, int]] | None = None,
    k_values: tuple[int, ...] | None = None,
    neighbors: int = DEFAULT_NEIGHBORS,
    instances_per_cell: int = INSTANCES_PER_CELL,
) -> dict[str, int]:
    """
    Return cell counts without yielding problems.
    """
    if dimensions is None:
        dimensions = DEFAULT_DIMENSIONS
    if k_values is None:
        k_values = DEFAULT_KS

    valid_dims = [(r, c, i, t) for r, c, i, t in dimensions if i <= r * c and t <= i]
    num_bounded_solvers = len(BoundedSuite.names())
    num_k = len(k_values)
    num_instances = instances_per_cell
    bounded_cells = num_bounded_solvers * num_k * len(valid_dims) * num_instances
    num_smt4_solvers = len(SMT4Suite.names())
    num_sps = len(SMT4_SWAPS_PER_STEP_VALUES)
    smt4_cells = num_smt4_solvers * num_sps * len(valid_dims) * num_instances

    return {
        "bounded_solvers": num_bounded_solvers,
        "k_values": num_k,
        "dimensions": len(dimensions),
        "valid_dimensions": len(valid_dims),
        "bounded_cells": bounded_cells,
        "smt4_solvers": num_smt4_solvers,
        "smt4_swaps": num_sps,
        "smt4_cells": smt4_cells,
        "total_cells": bounded_cells + smt4_cells,
    }


# ---------------------------------------------------------------------------
# Unbounded benchmark suite
# ---------------------------------------------------------------------------


class UnboundedSuite(Enum):
    """Names of solvers in the unbounded benchmark suite."""

    GREEDY_RECONSTRUCTOR = "greedy_reconstructor"
    PLANNING_RECONSTRUCTOR = "planning_reconstructor"
    SMT5_SWAP = "smt5_swap"
    SMT6 = "smt6"

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(s.value for s in cls)


@dataclass(frozen=True, slots=True)
class UnboundedInstance:
    """One benchmark instance: dimension + index + pregenerated target."""

    num_rows: int
    num_cols: int
    num_items: int
    num_types: int
    neighbors: int
    seed: int
    instance_index: int
    target: EndState | None
    pregeneration_status: Status

    @property
    def instance_key(self) -> str:
        return (
            f"unbound:{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}"
        )

    def to_problem(self) -> Problem:
        sol = generate(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            num_items=self.num_items,
            num_types=self.num_types,
            seed=self.seed,
            neighbors=self.neighbors,
        )
        return sol.problem


@dataclass(frozen=True, slots=True)
class UnboundedCell:
    """One benchmark cell: one solver × one UnboundedInstance."""

    solver: str
    num_rows: int
    num_cols: int
    num_items: int
    num_types: int
    neighbors: int
    seed: int
    instance_index: int
    target: EndState | None
    pregeneration_status: Status | None = None

    @property
    def cell_key(self) -> str:
        return (
            f"unbound:{self.solver}:"
            f"{self.num_rows}x{self.num_cols}:"
            f"items{self.num_items}:types{self.num_types}:"
            f"n{self.neighbors}:i{self.instance_index}"
        )

    def to_problem(self) -> Problem:
        sol = generate(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            num_items=self.num_items,
            num_types=self.num_types,
            seed=self.seed,
            neighbors=self.neighbors,
        )
        return sol.problem


def pregenerate_instance(
    num_rows: int,
    num_cols: int,
    num_items: int,
    num_types: int,
    instance_index: int,
    neighbors: int = DEFAULT_NEIGHBORS,
    master_seed: int = 0,
    timeout: float | None = None,
) -> UnboundedInstance:
    """Generate one UnboundedInstance, running genetic_endstate to obtain a target.

    Falls back to target=None / Status.TIMEOUT on any exception (ImportError,
    GurobiError, RuntimeError, etc.).
    """
    instance_key = (
        f"unbound:{num_rows}x{num_cols}:"
        f"items{num_items}:types{num_types}:"
        f"n{neighbors}:i{instance_index}"
    )
    seed = _stable_seed(instance_key, master_seed)
    sol = generate(
        num_rows=num_rows,
        num_cols=num_cols,
        num_items=num_items,
        num_types=num_types,
        seed=seed,
        neighbors=neighbors,
    )
    problem = sol.problem
    pregen_seed = _stable_seed(instance_key + ":pregen", master_seed)
    target: EndState | None
    pregen_status: Status
    try:
        result = GeneticEndStateGenerator(neighbors=neighbors).generate_end_state(
            problem, random.Random(pregen_seed), timeout
        )
        target = result.end_state
        pregen_status = result.status
    except Exception:
        target = None
        pregen_status = Status.TIMEOUT
    return UnboundedInstance(
        num_rows=num_rows,
        num_cols=num_cols,
        num_items=num_items,
        num_types=num_types,
        neighbors=neighbors,
        seed=seed,
        instance_index=instance_index,
        target=target,
        pregeneration_status=pregen_status,
    )


def iter_unbounded_instances(
    dimensions: Sequence[tuple[int, int, int, int]] = DEFAULT_DIMENSIONS,
    instances_per_cell: int | None = None,
    neighbors: int = DEFAULT_NEIGHBORS,
    master_seed: int = 0,
    timeout: float | None = None,
) -> Iterator[UnboundedInstance]:
    """Enumerate all valid unbounded benchmark instances.

    Skips combinations where item_count > grid capacity or type_count > item_count.
    Runs the genetic_endstate generator pregeneration for each instance.
    """
    for num_rows, num_cols, num_items, num_types in dimensions:
        if num_items > num_rows * num_cols:
            continue
        if num_types > num_items:
            continue
        n_instances = (
            instances_per_cell
            if instances_per_cell is not None
            else _default_instances_for_dimension(num_rows, num_cols)
        )
        for instance_index in range(n_instances):
            yield pregenerate_instance(
                num_rows=num_rows,
                num_cols=num_cols,
                num_items=num_items,
                num_types=num_types,
                instance_index=instance_index,
                neighbors=neighbors,
                master_seed=master_seed,
                timeout=timeout,
            )


def iter_unbounded_cells(
    instance: UnboundedInstance,
) -> Iterator[UnboundedCell]:
    """Yield one UnboundedCell per solver for the given instance."""
    for solver in UnboundedSuite.names():
        yield UnboundedCell(
            solver=solver,
            num_rows=instance.num_rows,
            num_cols=instance.num_cols,
            num_items=instance.num_items,
            num_types=instance.num_types,
            neighbors=instance.neighbors,
            seed=instance.seed,
            instance_index=instance.instance_index,
            target=instance.target,
            pregeneration_status=instance.pregeneration_status,
        )
