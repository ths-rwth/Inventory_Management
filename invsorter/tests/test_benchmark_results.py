import json

from invsorter.benchmark import (
    BenchmarkStore,
    BoundedCell,
    SMT4Cell,
    UnboundedCell,
    UnboundedInstance,
    result_key_for_cell,
    result_key_for_unbounded_pregeneration,
    row_for_cell,
    row_for_unbounded_pregeneration,
)
from invsorter.models import Status


def test_result_keys_use_existing_cell_keys_and_distinguish_parameters():
    bounded = BoundedCell(
        solver="greedy",
        k=2,
        num_rows=3,
        num_cols=4,
        num_items=5,
        num_types=2,
        neighbors=8,
        seed=123,
        instance_index=7,
    )
    smt4 = SMT4Cell(
        solver="smt4",
        swaps_per_step=3,
        num_rows=4,
        num_cols=4,
        num_items=8,
        num_types=2,
        neighbors=4,
        seed=456,
        instance_index=9,
    )
    unbounded = UnboundedCell(
        solver="smt6",
        num_rows=6,
        num_cols=6,
        num_items=20,
        num_types=4,
        neighbors=4,
        seed=789,
        instance_index=11,
        target=None,
        pregeneration_status=Status.SOLVED,
    )

    bounded_key = result_key_for_cell(bounded)
    smt4_key = result_key_for_cell(smt4)
    unbounded_key = result_key_for_cell(unbounded)

    assert bounded_key == bounded.cell_key
    assert ":greedy:" in bounded_key
    assert ":k2:" in bounded_key
    assert bounded_key.endswith(":n8:i7")

    assert smt4_key == smt4.cell_key
    assert smt4_key.startswith("smt4:sp3:")
    assert smt4_key.endswith(":n4:i9")

    assert unbounded_key == unbounded.cell_key
    assert ":smt6:" in unbounded_key
    assert unbounded_key.endswith(":n4:i11")
    unbounded_row = row_for_cell(unbounded, status="solved")
    assert unbounded_row.suite == "unbounded"
    assert unbounded_row.solver == "smt6"
    assert unbounded_row.k is None
    assert unbounded_row.swaps_per_step is None
    assert unbounded_row.pregeneration_status == "solved"


def test_unbounded_pregeneration_row_has_distinct_key_and_status():
    instance = UnboundedInstance(
        num_rows=4,
        num_cols=4,
        num_items=8,
        num_types=2,
        neighbors=4,
        seed=1234,
        instance_index=5,
        target=None,
        pregeneration_status=Status.TIMEOUT,
    )

    row = row_for_unbounded_pregeneration(
        instance,
        status=Status.TIMEOUT,
        timeout_s=30.0,
        elapsed_ms=30_000.0,
        note="solver timed out",
    )

    assert result_key_for_unbounded_pregeneration(instance) == (
        f"pregenerate:{instance.instance_key}"
    )
    assert row.suite == "unbounded_pregeneration"
    assert row.solver == "genetic_endstate"
    assert row.result_key == f"pregenerate:{instance.instance_key}"
    assert row.pregeneration_status == "timeout"
    assert row.timeout_s == 30.0
    assert row.elapsed_ms == 30_000.0


def test_store_append_round_trip_and_count_runs(tmp_path):
    path = tmp_path / "runs.jsonl"
    store = BenchmarkStore(path)
    first = row_for_cell(
        BoundedCell(
            solver="cp",
            k=1,
            num_rows=3,
            num_cols=3,
            num_items=4,
            num_types=2,
            neighbors=4,
            seed=1,
            instance_index=0,
        ),
        status="solved",
        elapsed_ms=12.5,
        timeout_s=5.0,
        moves_generated=2,
    )
    second = row_for_cell(
        SMT4Cell(
            solver="smt4",
            swaps_per_step=2,
            num_rows=3,
            num_cols=3,
            num_items=4,
            num_types=2,
            neighbors=4,
            seed=2,
            instance_index=1,
        ),
        status="timeout",
        timeout_s=5.0,
        elapsed_ms=5_000.0,
    )

    store.append(first, run_index=0)
    assert store.count_runs(first.result_key) == 1

    store.append(second, run_index=0)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert {json.loads(line)["result_key"] for line in lines} == {
        first.result_key,
        second.result_key,
    }
    assert store.count_runs(first.result_key) == 1
    assert store.count_runs(second.result_key) == 1
    assert store.next_run_index(first.result_key) == 1


def test_store_load_index_ignores_malformed_and_non_object_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "\n".join(
            (
                '{"result_key":"ok","run_index":0}',
                "",
                "not-json",
                '["not", "an", "object"]',
                '{"result_key":3}',
                '{"missing_key":"ignored"}',
                '{"result_key":"later","run_index":1}',
                '{"result_key":"partial"',
            )
        ),
        encoding="utf-8",
    )

    store = BenchmarkStore(path)
    index = store.load_index()
    assert set(index) == {"ok", "later"}
    assert BenchmarkStore(tmp_path / "missing.jsonl").load_index() == {}


def test_row_dict_contains_stable_schema_fields():
    row = row_for_cell(
        BoundedCell(
            solver="genetic",
            k=3,
            num_rows=4,
            num_cols=5,
            num_items=6,
            num_types=3,
            neighbors=8,
            seed=42,
            instance_index=4,
        ),
        status="solved",
        elapsed_ms=7.25,
        timeout_s=9.0,
        moves_generated=10,
        raw_moves=12,
        effective_moves=8,
        initial_score=30,
        final_score=11,
        improvement=19,
    )

    obj = row.to_json_dict()

    expected_fields = {
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
        "moves_generated",
        "raw_moves",
        "effective_moves",
        "initial_score",
        "final_score",
        "improvement",
        "k",
        "swaps_per_step",
        "flag_combo",
        "note",
        "traceback",
        "pregeneration_status",
        "run_index",
        "schema_version",
        "imported_from_legacy",
        "run_started_at",
    }
    assert expected_fields <= obj.keys()
    assert obj["suite"] == "bounded"
    assert obj["solver"] == "genetic"
    assert obj["num_rows"] == 4
    assert obj["num_cols"] == 5
    assert obj["num_items"] == 6
    assert obj["num_types"] == 3
    assert obj["instance_index"] == 4
    assert obj["seed"] == 42
    assert obj["neighbors"] == 8
    assert obj["status"] == "solved"
    assert obj["elapsed_ms"] == 7.25
    assert obj["timeout_s"] == 9.0
    assert obj["moves_generated"] == 10
    assert obj["raw_moves"] == 12
    assert obj["effective_moves"] == 8
    assert obj["initial_score"] == 30
    assert obj["final_score"] == 11
    assert obj["improvement"] == 19
    assert obj["k"] == 3
    assert obj["swaps_per_step"] is None
    assert obj["flag_combo"] is None
    assert obj["note"] is None
    assert obj["traceback"] is None
    assert obj["pregeneration_status"] is None
    assert obj["run_index"] == 0
    assert obj["schema_version"] == 2
    assert obj["imported_from_legacy"] is False
    assert obj["run_started_at"] is None
