from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from os import PathLike
from pathlib import Path
from typing import Any

from .suites import (
    BoundedCell,
    RuntimeCompareCell,
    SMT4Cell,
    UnboundedCell,
    UnboundedInstance,
    SymmetrySweepCell,
)

Pathish = str | PathLike[str]


@dataclass(frozen=True, slots=True)
class BenchmarkResultRow:
    suite: str
    solver: str
    result_key: str
    num_rows: int
    num_cols: int
    num_items: int
    num_types: int
    instance_index: int
    seed: int
    neighbors: int
    status: str
    elapsed_ms: float | None = None
    timeout_s: float | None = None
    moves_generated: int | None = None
    raw_moves: int | None = None
    effective_moves: int | None = None
    initial_score: int | None = None
    final_score: int | None = None
    improvement: int | None = None
    k: int | None = None
    swaps_per_step: int | None = None
    flag_combo: str | None = None
    note: str | None = None
    traceback: str | None = None
    pregeneration_status: str | None = None
    run_index: int = 0
    schema_version: int = 2
    imported_from_legacy: bool = False
    run_started_at: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-compatible object with every schema field present."""
        return {name: getattr(self, name) for name in _RESULT_ROW_FIELD_NAMES}


_RESULT_ROW_FIELD_NAMES = tuple(field.name for field in fields(BenchmarkResultRow))


def result_key_for_cell(
    cell: BoundedCell
    | RuntimeCompareCell
    | SMT4Cell
    | UnboundedCell
    | SymmetrySweepCell,
) -> str:
    if isinstance(
        cell,
        (
            BoundedCell,
            RuntimeCompareCell,
            SMT4Cell,
            UnboundedCell,
            SymmetrySweepCell,
        ),
    ):
        return cell.cell_key
    raise TypeError(f"unsupported benchmark cell type: {type(cell).__name__}")


def result_key_for_unbounded_pregeneration(instance: UnboundedInstance) -> str:
    if not isinstance(instance, UnboundedInstance):
        raise TypeError(
            "unsupported unbounded pregeneration instance type: "
            f"{type(instance).__name__}"
        )
    return f"pregenerate:{instance.instance_key}"


def row_for_cell(
    cell: BoundedCell
    | RuntimeCompareCell
    | SMT4Cell
    | UnboundedCell
    | SymmetrySweepCell,
    status: str,
    elapsed_ms: float | None = None,
    timeout_s: float | None = None,
    moves_generated: int | None = None,
    raw_moves: int | None = None,
    effective_moves: int | None = None,
    initial_score: int | None = None,
    final_score: int | None = None,
    improvement: int | None = None,
    note: str | None = None,
    traceback: str | None = None,
    run_index: int = 0,
    run_started_at: str | None = None,
) -> BenchmarkResultRow:
    if isinstance(cell, BoundedCell):
        return BenchmarkResultRow(
            suite="bounded",
            solver=cell.solver,
            result_key=result_key_for_cell(cell),
            num_rows=cell.num_rows,
            num_cols=cell.num_cols,
            num_items=cell.num_items,
            num_types=cell.num_types,
            instance_index=cell.instance_index,
            seed=cell.seed,
            neighbors=cell.neighbors,
            status=str(status),
            elapsed_ms=elapsed_ms,
            timeout_s=timeout_s,
            moves_generated=moves_generated,
            raw_moves=raw_moves,
            effective_moves=effective_moves,
            initial_score=initial_score,
            final_score=final_score,
            improvement=improvement,
            k=cell.k,
            swaps_per_step=None,
            note=note,
            traceback=traceback,
            pregeneration_status=None,
            run_index=run_index,
            run_started_at=run_started_at,
        )
    if isinstance(cell, RuntimeCompareCell):
        return BenchmarkResultRow(
            suite="runtime_compare",
            solver=cell.solver,
            result_key=result_key_for_cell(cell),
            num_rows=cell.num_rows,
            num_cols=cell.num_cols,
            num_items=cell.num_items,
            num_types=cell.num_types,
            instance_index=cell.instance_index,
            seed=cell.seed,
            neighbors=cell.neighbors,
            status=str(status),
            elapsed_ms=elapsed_ms,
            timeout_s=timeout_s,
            moves_generated=moves_generated,
            raw_moves=raw_moves,
            effective_moves=effective_moves,
            initial_score=initial_score,
            final_score=final_score,
            improvement=improvement,
            k=cell.k,
            swaps_per_step=None,
            note=note,
            traceback=traceback,
            pregeneration_status=None,
            run_index=run_index,
            run_started_at=run_started_at,
        )
    if isinstance(cell, SMT4Cell):
        return BenchmarkResultRow(
            suite="smt4",
            solver=cell.solver,
            result_key=result_key_for_cell(cell),
            num_rows=cell.num_rows,
            num_cols=cell.num_cols,
            num_items=cell.num_items,
            num_types=cell.num_types,
            instance_index=cell.instance_index,
            seed=cell.seed,
            neighbors=cell.neighbors,
            status=str(status),
            elapsed_ms=elapsed_ms,
            timeout_s=timeout_s,
            moves_generated=moves_generated,
            raw_moves=raw_moves,
            effective_moves=effective_moves,
            initial_score=initial_score,
            final_score=final_score,
            improvement=improvement,
            k=None,
            swaps_per_step=cell.swaps_per_step,
            note=note,
            traceback=traceback,
            pregeneration_status=None,
            run_index=run_index,
            run_started_at=run_started_at,
        )
    if isinstance(cell, UnboundedCell):
        return BenchmarkResultRow(
            suite="unbounded",
            solver=cell.solver,
            result_key=result_key_for_cell(cell),
            num_rows=cell.num_rows,
            num_cols=cell.num_cols,
            num_items=cell.num_items,
            num_types=cell.num_types,
            instance_index=cell.instance_index,
            seed=cell.seed,
            neighbors=cell.neighbors,
            status=str(status),
            elapsed_ms=elapsed_ms,
            timeout_s=timeout_s,
            moves_generated=moves_generated,
            raw_moves=raw_moves,
            effective_moves=effective_moves,
            initial_score=initial_score,
            final_score=final_score,
            improvement=improvement,
            k=None,
            swaps_per_step=None,
            note=note,
            traceback=traceback,
            pregeneration_status=_optional_str(
                getattr(cell, "pregeneration_status", None)
            ),
            run_index=run_index,
            run_started_at=run_started_at,
        )
    if isinstance(cell, SymmetrySweepCell):
        return BenchmarkResultRow(
            suite="symmetry_sweep",
            solver=cell.solver,
            result_key=result_key_for_cell(cell),
            num_rows=cell.num_rows,
            num_cols=cell.num_cols,
            num_items=cell.num_items,
            num_types=cell.num_types,
            instance_index=cell.instance_index,
            seed=cell.seed,
            neighbors=cell.neighbors,
            status=str(status),
            elapsed_ms=elapsed_ms,
            timeout_s=timeout_s,
            moves_generated=moves_generated,
            raw_moves=raw_moves,
            effective_moves=effective_moves,
            initial_score=initial_score,
            final_score=final_score,
            improvement=improvement,
            k=cell.k,
            swaps_per_step=None,
            flag_combo=cell.flag_combo,
            note=note,
            traceback=traceback,
            pregeneration_status=None,
            run_index=run_index,
            run_started_at=run_started_at,
        )
    raise TypeError(f"unsupported benchmark cell type: {type(cell).__name__}")


def row_for_unbounded_pregeneration(
    instance: UnboundedInstance,
    *,
    status: str,
    elapsed_ms: float | None = None,
    timeout_s: float | None = None,
    note: str | None = None,
    traceback: str | None = None,
    run_index: int = 0,
    run_started_at: str | None = None,
) -> BenchmarkResultRow:
    if not isinstance(instance, UnboundedInstance):
        raise TypeError(
            "unsupported unbounded pregeneration instance type: "
            f"{type(instance).__name__}"
        )
    return BenchmarkResultRow(
        suite="unbounded_pregeneration",
        solver="genetic_endstate",
        result_key=result_key_for_unbounded_pregeneration(instance),
        num_rows=instance.num_rows,
        num_cols=instance.num_cols,
        num_items=instance.num_items,
        num_types=instance.num_types,
        instance_index=instance.instance_index,
        seed=instance.seed,
        neighbors=instance.neighbors,
        status=str(status),
        elapsed_ms=elapsed_ms,
        timeout_s=timeout_s,
        moves_generated=None,
        raw_moves=None,
        effective_moves=None,
        initial_score=None,
        final_score=None,
        improvement=None,
        k=None,
        swaps_per_step=None,
        note=note,
        traceback=traceback,
        pregeneration_status=_optional_str(instance.pregeneration_status),
        run_index=run_index,
        run_started_at=run_started_at,
    )


def _row_json_object(
    row: BenchmarkResultRow | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(row, BenchmarkResultRow):
        return row.to_json_dict()
    if isinstance(row, Mapping):
        result_key = row.get("result_key")
        if not isinstance(result_key, str):
            raise TypeError("row mapping must contain a string result_key")
        return dict(row)
    raise TypeError(f"unsupported result row type: {type(row).__name__}")


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "BenchmarkResultRow",
    "result_key_for_cell",
    "result_key_for_unbounded_pregeneration",
    "row_for_cell",
    "row_for_unbounded_pregeneration",
]
