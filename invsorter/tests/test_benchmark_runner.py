"""Unit tests for invsorter.benchmark.runner."""
from __future__ import annotations

import json
from pathlib import Path

from invsorter.benchmark.results import result_key_for_cell, row_for_cell
from invsorter.benchmark.runner import run_cells
from invsorter.benchmark.store import BenchmarkStore
from invsorter.benchmark.suites import BoundedCell

_TINY_CELL = BoundedCell(
    solver="greedy",
    k=3,
    num_rows=3,
    num_cols=3,
    num_items=4,
    num_types=2,
    neighbors=4,
    seed=42,
    instance_index=0,
)

_BAD_SOLVER_CELL = BoundedCell(
    solver="__no_such_solver__",
    k=3,
    num_rows=3,
    num_cols=3,
    num_items=4,
    num_types=2,
    neighbors=4,
    seed=42,
    instance_index=0,
)


def _write_completed_row(path: Path, cell: BoundedCell, *, run_index: int = 0) -> None:
    row = row_for_cell(cell, status="ok", timeout_s=30.0)
    BenchmarkStore(path).append(row, run_index=run_index)


def test_run_cells_empty_input(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"
    rows = run_cells([], store=BenchmarkStore(out), resume=False, timeout_s=30.0)
    assert rows == []
    assert not out.exists() or out.stat().st_size == 0


def test_resume_skips_completed_cells_for_replicas_one(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"
    _write_completed_row(out, _TINY_CELL)

    store = BenchmarkStore(out)
    assert store.count_runs(result_key_for_cell(_TINY_CELL)) == 1

    rows = run_cells([_TINY_CELL], store=store, resume=True, timeout_s=30.0, workers=1, replicas=1)

    assert rows == []
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1


def test_resume_backfills_partial_replication(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"
    _write_completed_row(out, _TINY_CELL, run_index=0)

    store = BenchmarkStore(out)
    rows = run_cells([_TINY_CELL], store=store, resume=True, timeout_s=60.0, workers=1, replicas=3)

    assert len(rows) == 2
    index = store.load_index()
    stored = index[result_key_for_cell(_TINY_CELL)]
    assert len(stored) == 3
    assert sorted(r["run_index"] for r in stored) == [0, 1, 2]


def test_run_cells_success_produces_row_and_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"
    store = BenchmarkStore(out)
    rows = run_cells([_TINY_CELL], store=store, resume=False, timeout_s=60.0, workers=1, replicas=1)

    assert len(rows) == 1
    row = rows[0]
    assert row.suite == "bounded"
    assert row.solver == "greedy"
    assert row.result_key == result_key_for_cell(_TINY_CELL)
    assert row.num_rows == 3
    assert row.num_cols == 3
    assert row.num_items == 4
    assert row.num_types == 2
    assert row.k == 3
    assert row.status not in ("timeout", "crash")
    assert isinstance(row.elapsed_ms, float)
    assert row.elapsed_ms >= 0.0
    assert isinstance(row.initial_score, int)
    assert isinstance(row.final_score, int)
    assert row.timeout_s == 60.0
    assert row.run_index == 0
    assert row.schema_version == 2

    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["result_key"] == row.result_key
    assert obj["status"] == row.status
    assert obj["run_index"] == 0
    assert obj["schema_version"] == 2
    assert obj["timeout_s"] == 60.0


def test_run_cells_bad_solver_produces_crash_row_and_counts_complete(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"
    store = BenchmarkStore(out)
    rows = run_cells([_BAD_SOLVER_CELL], store=store, resume=False, timeout_s=60.0, workers=1, replicas=1)

    assert len(rows) == 1
    row = rows[0]
    assert row.status == "crash"
    assert row.traceback is not None
    assert "__no_such_solver__" in row.traceback
    assert row.timeout_s == 60.0

    second = run_cells([_BAD_SOLVER_CELL], store=store, resume=True, timeout_s=60.0, workers=1, replicas=1)
    assert second == []
    assert store.count_runs(result_key_for_cell(_BAD_SOLVER_CELL)) == 1


def test_run_cells_multiple_workers_and_replicas_assign_unique_pairs(tmp_path: Path) -> None:
    cells = [
        BoundedCell(
            solver="greedy",
            k=3,
            num_rows=3,
            num_cols=3,
            num_items=4,
            num_types=2,
            neighbors=4,
            seed=42,
            instance_index=i,
        )
        for i in range(3)
    ]
    out = tmp_path / "runs.jsonl"
    store = BenchmarkStore(out)
    rows = run_cells(cells, store=store, resume=False, timeout_s=60.0, workers=3, replicas=3)

    assert len(rows) == 9
    index = store.load_index()
    assert len(index) == 3
    for cell in cells:
        grp = index[result_key_for_cell(cell)]
        assert len(grp) == 3
        assert sorted(r["run_index"] for r in grp) == [0, 1, 2]

    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 9


def test_resume_false_reruns_targeted_cells_but_preserves_unaffected_rows(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"
    store = BenchmarkStore(out)
    other = BoundedCell(
        solver="greedy",
        k=3,
        num_rows=3,
        num_cols=3,
        num_items=4,
        num_types=2,
        neighbors=4,
        seed=42,
        instance_index=99,
    )
    store.append(row_for_cell(_TINY_CELL, status="solved", timeout_s=30.0), run_index=0)
    store.append(row_for_cell(other, status="solved", timeout_s=30.0), run_index=0)

    rows = run_cells([_TINY_CELL], store=store, resume=False, timeout_s=60.0, workers=1, replicas=1)

    assert len(rows) == 1
    index = store.load_index()
    assert len(index[result_key_for_cell(_TINY_CELL)]) == 1
    assert len(index[result_key_for_cell(other)]) == 1
