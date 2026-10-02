from __future__ import annotations

import json
from pathlib import Path

import pytest

from invsorter.benchmark import BenchmarkStore, BenchmarkStoreError, row_for_cell
from invsorter.benchmark.suites import BoundedCell
import invsorter.benchmark.store as store_mod


def _cell(instance_index: int) -> BoundedCell:
    return BoundedCell(
        solver="greedy",
        k=3,
        num_rows=3,
        num_cols=3,
        num_items=4,
        num_types=2,
        neighbors=4,
        seed=42,
        instance_index=instance_index,
    )


def test_append_and_load_index_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    store = BenchmarkStore(path)
    row0 = row_for_cell(_cell(0), status="solved", timeout_s=5.0, elapsed_ms=10.0)
    row1 = row_for_cell(_cell(0), status="timeout", timeout_s=5.0, elapsed_ms=5_000.0)

    store.append(row0, run_index=0)
    store.append(row1, run_index=1)

    index = store.load_index()
    assert set(index) == {row0.result_key}
    assert [r["run_index"] for r in index[row0.result_key]] == [0, 1]
    assert [r["status"] for r in index[row0.result_key]] == ["solved", "timeout"]


def test_count_runs_and_next_run_index(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    store = BenchmarkStore(path)
    row = row_for_cell(_cell(1), status="solved", timeout_s=5.0)

    assert store.count_runs(row.result_key) == 0
    assert store.next_run_index(row.result_key) == 0

    store.append(row, run_index=0)
    store.append(row, run_index=2)

    assert store.count_runs(row.result_key) == 2
    assert store.next_run_index(row.result_key) == 3


def test_compact_removes_matching_rows_and_preserves_others(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    store = BenchmarkStore(path)
    keep = row_for_cell(_cell(0), status="solved", timeout_s=5.0)
    drop = row_for_cell(_cell(1), status="crash", timeout_s=5.0)

    store.append(keep, run_index=0)
    store.append(drop, run_index=0)

    dropped = store.compact(lambda row: row.get("status") != "crash")
    assert dropped == 1

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["result_key"] == keep.result_key


def test_compact_preserves_original_file_if_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "runs.jsonl"
    store = BenchmarkStore(path)
    row = row_for_cell(_cell(0), status="solved", timeout_s=5.0)
    store.append(row, run_index=0)
    before = path.read_text(encoding="utf-8")

    def boom(_src: str, _dst: str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(store_mod.os, "replace", boom)

    with pytest.raises(OSError):
        store.compact(lambda _row: True)

    assert path.read_text(encoding="utf-8") == before


def test_append_rejects_live_lock_from_different_pid(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    store = BenchmarkStore(path)
    lock = path.with_suffix(".lock")
    lock.write_text("999999", encoding="utf-8")

    with pytest.raises(BenchmarkStoreError):
        store.append(row_for_cell(_cell(0), status="solved", timeout_s=5.0), run_index=0)


def test_append_overwrites_imported_from_legacy_with_kwarg(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    store = BenchmarkStore(path)
    row = row_for_cell(_cell(0), status="solved", timeout_s=5.0).to_json_dict()

    row["imported_from_legacy"] = True
    store.append(row, run_index=0, imported_from_legacy=False)
    row["imported_from_legacy"] = False
    store.append(row, run_index=1, imported_from_legacy=True)

    rows = store.load_index()[row["result_key"]]
    assert [r["imported_from_legacy"] for r in rows] == [False, True]

