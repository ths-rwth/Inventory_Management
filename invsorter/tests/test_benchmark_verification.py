"""End-to-end verification tests for the benchmark system."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import invsorter.benchmark.runner as runner_mod
from invsorter.benchmark.runner import run_cells
from invsorter.benchmark.store import BenchmarkStore
from invsorter.benchmark.suites import (
    BoundedCell,
    SymmetrySweepCell,
    iter_bounded_cells,
    iter_smt4_cells,
    iter_symmetry_sweep_cells,
    expected_symmetry_flags,
    _SYMMETRY_FLAG_COMBOS_BY_SOLVER,
)

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

_REQUIRED_KEYS = frozenset(
    {
        "suite",
        "solver",
        "result_key",
        "num_rows",
        "num_cols",
        "num_items",
        "num_types",
        "instance_index",
        "seed",
        "neighbors",
        "status",
        "elapsed_ms",
        "timeout_s",
        "run_index",
        "schema_version",
    }
)


def test_cell_enumeration_is_deterministic_and_skips_invalid_dimensions():
    dims_with_invalid = [
        (3, 3, 4, 2),
        (3, 3, 10, 2),
        (2, 2, 5, 3),
    ]

    run1 = list(iter_bounded_cells(dims_with_invalid, instances_per_cell=1, master_seed=7))
    run2 = list(iter_bounded_cells(dims_with_invalid, instances_per_cell=1, master_seed=7))
    assert run1 == run2
    assert all(c.num_rows == 3 and c.num_cols == 3 and c.num_items == 4 for c in run1)
    assert len(run1) > 0

    n_valid_dims = sum(1 for r, c, i, t in dims_with_invalid if i <= r * c and t <= i)
    assert n_valid_dims < len(dims_with_invalid)

    smt4_run1 = list(iter_smt4_cells(dims_with_invalid, instances_per_cell=1, master_seed=7))
    smt4_run2 = list(iter_smt4_cells(dims_with_invalid, instances_per_cell=1, master_seed=7))
    assert smt4_run1 == smt4_run2

    from invsorter.benchmark.suites import iter_unbounded_instances, iter_unbounded_cells

    tiny_dim = [(3, 3, 4, 2)]
    ub_inst1 = list(iter_unbounded_instances(tiny_dim, instances_per_cell=1, master_seed=7))
    ub_inst2 = list(iter_unbounded_instances(tiny_dim, instances_per_cell=1, master_seed=7))
    assert len(ub_inst1) == len(ub_inst2) == 1
    assert ub_inst1[0].seed == ub_inst2[0].seed

    ub_cells1 = list(iter_unbounded_cells(ub_inst1[0]))
    ub_cells2 = list(iter_unbounded_cells(ub_inst2[0]))
    assert ub_cells1 == ub_cells2


def test_run_cells_end_to_end_row_has_required_schema_fields(tmp_path: Path) -> None:
    output = tmp_path / "schema_test.jsonl"
    store = BenchmarkStore(output)

    rows = run_cells(
        [_TINY_CELL],
        store=store,
        timeout_s=60,
        workers=1,
        resume=False,
        replicas=1,
    )
    assert len(rows) == 1

    raw = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) >= 1
    row = json.loads(raw[-1])

    missing = _REQUIRED_KEYS - set(row.keys())
    assert not missing
    assert row["status"] == "solved"
    assert row["solver"] == "greedy"
    assert row["run_index"] == 0
    assert row["schema_version"] == 2


    output = tmp_path / "timeout_test.jsonl"
    store = BenchmarkStore(output)

    cell = BoundedCell(
        solver="greedy",
        k=100,
        num_rows=20,
        num_cols=20,
        num_items=400,
        num_types=10,
        neighbors=4,
        seed=99,
        instance_index=0,
    )

    run_cells([cell], store=store, timeout_s=0.3, workers=1, resume=False, replicas=1)

    raw = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1
    row = json.loads(raw[0])
    assert row["status"] == "timeout"
    assert row["timeout_s"] == 0.3
    assert row["elapsed_ms"] == 300.0


def test_cli_help_and_tiny_invocation(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "invsorter.benchmark", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--db" in result.stdout

    db_path = tmp_path / "runs.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "invsorter.benchmark",
            "run",
            "--suites",
            "bounded",
            "--db",
            str(db_path),
            "--instances",
            "1",
            "--timeout",
            "0.5",
            "--workers",
            "1",
            "--replicas",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr
    assert db_path.exists()
    assert db_path.read_text(encoding="utf-8").strip()



def test_symmetry_sweep_enumeration():
    cells1 = list(iter_symmetry_sweep_cells())
    cells2 = list(iter_symmetry_sweep_cells())

    assert len(cells1) == 1120
    assert cells1 == cells2

    keys = [c.cell_key for c in cells1]
    assert len(keys) == len(set(keys))

    from collections import defaultdict
    by_instance = defaultdict(list)
    for c in cells1:
        by_instance[c.instance_key].append(c)
    for inst_key, group in by_instance.items():
        assert all(c.instance_key == inst_key for c in group)


def test_symmetry_sweep_row_has_flag_combo(tmp_path: Path) -> None:
    output = tmp_path / "symmetry_test.jsonl"
    store = BenchmarkStore(output)

    cell = SymmetrySweepCell(
        solver="ilp_type",
        k=3,
        num_rows=4,
        num_cols=4,
        num_items=8,
        num_types=2,
        neighbors=4,
        seed=42,
        instance_index=0,
        flag_combo="no_diff_noop=False,upper_left_noop=True,cardinality_constraints=True",
    )

    rows = run_cells([cell], store=store, timeout_s=30, workers=1, resume=False, replicas=1)
    assert len(rows) == 1

    raw = output.read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(raw[-1])
    assert row["suite"] == "symmetry_sweep"
    assert row["solver"] == "ilp_type"
    assert row["flag_combo"] == (
        "no_diff_noop=False,upper_left_noop=True,cardinality_constraints=True"
    )
    assert row["k"] == 3


def test_symmetry_sweep_flag_override_in_child(tmp_path: Path) -> None:
    output = tmp_path / "symmetry_override_test.jsonl"
    store = BenchmarkStore(output)

    cell = SymmetrySweepCell(
        solver="ilp_type",
        k=3,
        num_rows=4,
        num_cols=4,
        num_items=8,
        num_types=2,
        neighbors=4,
        seed=42,
        instance_index=0,
        flag_combo="no_diff_noop=True,upper_left_noop=False,cardinality_constraints=False",
    )

    rows = run_cells([cell], store=store, timeout_s=30, workers=1, resume=False, replicas=1)
    assert len(rows) == 1
    tb = rows[0].traceback or ""
    if rows[0].status == "crash" and "size-limited license" in tb:
        pytest.skip("Gurobi size-limited license cannot run this model")
    assert rows[0].status != "crash" or rows[0].traceback is None


def test_symmetry_sweep_resume_skips_completed_and_partial_replication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "_MIN_FREE_BYTES", 0)
    output = tmp_path / "symmetry_resume_test.jsonl"
    store = BenchmarkStore(output)

    cell = SymmetrySweepCell(
        solver="sat_type",
        k=3,
        num_rows=4,
        num_cols=4,
        num_items=8,
        num_types=2,
        neighbors=4,
        seed=42,
        instance_index=0,
        flag_combo="no_diff_noop=False,upper_left_noop=True,cardinality_constraints=True",
    )

    store.append(
        {
            "suite": "symmetry_sweep",
            "solver": "sat_type",
            "result_key": cell.cell_key,
            "num_rows": cell.num_rows,
            "num_cols": cell.num_cols,
            "num_items": cell.num_items,
            "num_types": cell.num_types,
            "instance_index": cell.instance_index,
            "seed": cell.seed,
            "neighbors": cell.neighbors,
            "status": "solved",
            "elapsed_ms": 1.0,
            "timeout_s": 30.0,
            "k": cell.k,
            "flag_combo": cell.flag_combo,
            "schema_version": 2,
            "run_index": 0,
            "imported_from_legacy": False,
            "run_started_at": None,
        },
        run_index=0,
    )

    rows = run_cells([cell], store=store, timeout_s=30, workers=1, resume=True, replicas=1)
    assert len(rows) == 0

    rows = run_cells([cell], store=store, timeout_s=30, workers=1, resume=True, replicas=3)
    assert len(rows) == 2
    raw = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 3


def test_dimension_aware_defaults():
    cells = list(iter_bounded_cells(instances_per_cell=None))

    by_grid = {}
    for c in cells:
        key = (c.num_rows, c.num_cols)
        by_grid.setdefault(key, set()).add(c.instance_index)

    assert len(by_grid[(3, 3)]) == 30
    assert len(by_grid[(4, 4)]) == 30
    assert len(by_grid[(6, 6)]) == 10
    assert len(by_grid[(10, 10)]) == 10
    assert len(cells) == 5940


def test_instances_override_works():
    cells = list(iter_bounded_cells(instances_per_cell=5))

    by_grid = {}
    for c in cells:
        key = (c.num_rows, c.num_cols)
        by_grid.setdefault(key, set()).add(c.instance_index)

    assert len(by_grid[(3, 3)]) == 5
    assert len(by_grid[(4, 4)]) == 5
    assert len(by_grid[(6, 6)]) == 5
    assert len(by_grid[(10, 10)]) == 5
    assert len(cells) == 1350

def test_symmetry_sweep_flag_combo_missing_flag_crashes(tmp_path: Path) -> None:
    output = tmp_path / "symmetry_missing_test.jsonl"
    store = BenchmarkStore(output)

    cell = SymmetrySweepCell(
        solver="ilp_type",
        k=1,
        num_rows=3,
        num_cols=3,
        num_items=4,
        num_types=2,
        neighbors=4,
        seed=42,
        instance_index=0,
        flag_combo="no_diff_noop=False,upper_left_noop=True",
    )

    rows = run_cells([cell], store=store, timeout_s=30, workers=1, resume=False, replicas=1)
    assert len(rows) == 1
    assert rows[0].status == "crash"
    assert "missing=['cardinality_constraints']" in (rows[0].traceback or "")


def test_symmetry_sweep_flag_combo_unknown_flag_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "_MIN_FREE_BYTES", 0)
    output = tmp_path / "symmetry_unknown_test.jsonl"
    store = BenchmarkStore(output)

    cell = SymmetrySweepCell(
        solver="sat_type",
        k=1,
        num_rows=3,
        num_cols=3,
        num_items=4,
        num_types=2,
        neighbors=4,
        seed=42,
        instance_index=0,
        flag_combo="no_diff_noop=False,upper_left_noop=True,bogus=True",
    )

    rows = run_cells([cell], store=store, timeout_s=30, workers=1, resume=False, replicas=1)
    assert len(rows) == 1
    assert rows[0].status == "crash"
    assert "unknown=['bogus']" in (rows[0].traceback or "")


def test_expected_symmetry_flags_matches_combo_enumeration():
    for solver in ("ilp_type", "sat_type", "smt3"):
        names = set(expected_symmetry_flags(solver))
        for combo in _SYMMETRY_FLAG_COMBOS_BY_SOLVER[solver]:
            assert set(name for name, _ in combo) == names

