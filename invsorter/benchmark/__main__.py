"""Entry point for ``python -m invsorter.benchmark``.

The benchmark CLI operates on a single append-only raw-run database:

    results/<run-dir>/runs.jsonl

Subcommands:
    run       produce raw runs into runs.jsonl
    status    show completion counts per (suite, grid, solver)
    verify    consistency checks on the DB
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Sequence

from invsorter.benchmark import (
    iter_bounded_cells,
    iter_runtime_compare_cells,
    iter_smt4_cells,
    iter_symmetry_sweep_cells,
    iter_unbounded_cells,
    iter_unbounded_instances,
    run_cells,
)
from invsorter.benchmark.store import BenchmarkStore, resolve_db_path
from invsorter.benchmark.suites import (
    DEFAULT_DIMENSIONS,
    DEFAULT_TIMEOUT_S,
    RUNTIME_COMPARE_TIMEOUT_S,
    BoundedCell,
    RuntimeCompareCell,
    SMT4Cell,
    UnboundedCell,
    SymmetrySweepCell,
)

_AnyCell = (
    BoundedCell
    | RuntimeCompareCell
    | SMT4Cell
    | UnboundedCell
    | SymmetrySweepCell
)
_SUITE_ORDER: tuple[str, ...] = (
    "bounded",
    "runtime_compare",
    "smt4",
    "unbounded",
    "symmetry_sweep",
)
_OPT_IN_SUITES: frozenset[str] = frozenset({"runtime_compare"})


def _load_runs(db_path: Path) -> list[dict[str, Any]]:
    """All raw runs as plain dicts; one per JSONL line. Skips malformed."""
    result: list[dict[str, Any]] = []
    try:
        with db_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    result.append(obj)
    except FileNotFoundError:
        return result
    return result


def _parse_dim(spec: str) -> tuple[int, int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--dims expects RxCxIxT (e.g. 3x3x4x2); got {spec!r}"
        )
    try:
        rows, cols, items, types = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--dims components must be integers; got {spec!r}"
        ) from exc
    if min(rows, cols, items, types) < 1:
        raise argparse.ArgumentTypeError(
            f"--dims components must be positive; got {spec!r}"
        )
    return (rows, cols, items, types)


def _grid_key(cell: _AnyCell) -> tuple[int, int]:
    return (cell.num_rows, cell.num_cols)


def _group_cells_by_grid(
    cells: Sequence[_AnyCell],
) -> list[tuple[tuple[int, int], list[_AnyCell]]]:
    grid_cells: dict[tuple[int, int], list[_AnyCell]] = {}
    for cell in cells:
        grid_cells.setdefault(_grid_key(cell), []).append(cell)
    return sorted(grid_cells.items())


def _find_fully_timed_out_solvers(
    rows: list[dict[str, Any]],
    grid_size: tuple[int, int],
    policy: str,
) -> set[str]:
    """Return solvers eliminated at *grid_size* under the chosen policy.

    ``rows`` are raw runs; they are grouped per solver and per ``result_key``
    (cell) so the policy sees the per-cell replica structure.
    """
    solver_cells: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        if (row.get("num_rows"), row.get("num_cols")) != grid_size:
            continue
        cells = solver_cells.setdefault(str(row.get("solver")), {})
        cells.setdefault(str(row.get("result_key")), []).append(
            str(row.get("status"))
        )

    eliminated: set[str] = set()
    for solver, cells in solver_cells.items():
        if not cells:
            continue
        if policy == "strict":
            if all(
                status == "timeout"
                for statuses in cells.values()
                for status in statuses
            ):
                eliminated.add(solver)
        elif policy == "any-timeout":
            if all("timeout" in statuses for statuses in cells.values()):
                eliminated.add(solver)
        else:
            raise ValueError(f"Unknown elimination policy: {policy!r}")
    return eliminated


def _is_solver_eliminated(
    solver: str,
    grid_size: tuple[int, int],
    eliminated: list[tuple[str, tuple[int, int]]],
) -> bool:
    rows, cols = grid_size
    for elim_solver, (er, ec) in eliminated:
        if elim_solver == solver and rows >= er and cols >= ec:
            return True
    return False


def _add_common_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--suites",
        nargs="+",
        choices=(
            "bounded",
            "runtime_compare",
            "smt4",
            "unbounded",
            "symmetry_sweep",
            "all",
        ),
        default=["all"],
        metavar="SUITE",
        help="Suites to run: bounded, smt4, unbounded, symmetry_sweep, runtime_compare, all (default: all except runtime_compare).",
    )
    p.add_argument(
        "--db", required=True, metavar="PATH", help="DB path (runs.jsonl) or directory."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="Master random seed (default: 0).",
    )
    p.add_argument(
        "--instances",
        type=int,
        default=None,
        metavar="N",
        help="Number of instances per cell.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Per-cell wall-clock timeout in seconds. When omitted, each suite uses its own default: "
            f"{DEFAULT_TIMEOUT_S:.0f}s for bounded/smt4/unbounded, "
            f"{RUNTIME_COMPARE_TIMEOUT_S:.0f}s for runtime_compare."
        ),
    )
    p.add_argument(
        "--workers", type=int, default=1, metavar="N", help="Parallel worker threads."
    )
    p.add_argument("--neighbors", type=int, default=4, choices=(4, 8), metavar="{4,8}")
    p.add_argument(
        "--dims", nargs="+", type=_parse_dim, default=None, metavar="RxCxIxT"
    )
    p.add_argument(
        "--replicas",
        type=int,
        default=3,
        metavar="N",
        help="Desired runs per cell (default: 3).",
    )
    p.add_argument(
        "--elimination-policy",
        choices=("strict", "any-timeout"),
        default="strict",
        help="Solver elimination policy for larger grids.",
    )
    resume_group = p.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true", default=True
    )
    resume_group.add_argument("--force", dest="resume", action="store_false")


def _build_run_parser(p: argparse.ArgumentParser) -> None:
    _add_common_run_flags(p)
    p.add_argument(
        "--solver",
        action="append",
        default=[],
        metavar="SOLVER",
        help="Only run cells for the named solver (can be given multiple times).",
    )


def _build_status_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", required=True, metavar="PATH")
    p.add_argument("--replicas", type=int, default=3, metavar="N")
    p.add_argument("--suite", action="append", default=[])
    p.add_argument("--solver", action="append", default=[])


def _build_verify_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", required=True, metavar="PATH")
    p.add_argument("--strict", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m invsorter.benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _build_run_parser(sub.add_parser("run"))
    _build_status_parser(sub.add_parser("status"))
    _build_verify_parser(sub.add_parser("verify"))
    return parser


def _resolve_suites(suites_raw: list[str]) -> list[str]:
    if "all" in suites_raw:
        return [s for s in _SUITE_ORDER if s not in _OPT_IN_SUITES]
    seen = set(suites_raw)
    return [s for s in _SUITE_ORDER if s in seen]


def _collect_cells(
    suite: str,
    *,
    dims: list[tuple[int, int, int, int]] | None,
    neighbors: int,
    instances: int | None,
    seed: int,
    timeout: float,
) -> list[_AnyCell]:
    if suite == "bounded":
        return list(
            iter_bounded_cells(
                dimensions=dims,
                neighbors=neighbors,
                instances_per_cell=instances,
                master_seed=seed,
            )
        )
    if suite == "smt4":
        return list(
            iter_smt4_cells(
                dimensions=dims,
                neighbors=neighbors,
                instances_per_cell=instances,
                master_seed=seed,
            )
        )
    if suite == "unbounded":
        instance_gen = iter_unbounded_instances(
            dimensions=dims if dims is not None else DEFAULT_DIMENSIONS,
            instances_per_cell=instances,
            neighbors=neighbors,
            master_seed=seed,
            timeout=timeout,
        )
        return list(
            chain.from_iterable(iter_unbounded_cells(inst) for inst in instance_gen)
        )
    if suite == "symmetry_sweep":
        kwargs: dict[str, Any] = {
            "dimensions": dims,
            "neighbors": neighbors,
            "master_seed": seed,
        }
        if instances is not None:
            kwargs["instances_per_cell"] = instances
        return list(iter_symmetry_sweep_cells(**kwargs))
    if suite == "runtime_compare":
        return list(
            iter_runtime_compare_cells(
                dimensions=dims,
                neighbors=neighbors,
                instances_per_cell=instances,
                master_seed=seed,
            )
        )
    raise AssertionError(f"unexpected suite: {suite!r}")


def _resolve_timeout(suite: str, timeout_override: float | None) -> float:
    if timeout_override is not None:
        return timeout_override
    if suite == "runtime_compare":
        return RUNTIME_COMPARE_TIMEOUT_S
    return DEFAULT_TIMEOUT_S


def _run_suite_with_pruning(
    cells: list[_AnyCell],
    store: BenchmarkStore,
    timeout: float,
    workers: int,
    replicas: int,
    resume: bool,
    suite_label: str,
    *,
    elimination_policy: str = "strict",
) -> None:
    groups = _group_cells_by_grid(cells)
    eliminated: list[tuple[str, tuple[int, int]]] = []
    total = len(cells)
    completed = 0

    for grid_size, group in groups:
        rows_val, cols_val = grid_size
        active = [
            c
            for c in group
            if not _is_solver_eliminated(c.solver, grid_size, eliminated)
        ]
        skipped = len(group) - len(active)
        completed += skipped
        if skipped:
            print(
                f"[{suite_label}] {rows_val}x{cols_val}: skipped {skipped} cells (eliminated solvers)",
                flush=True,
            )

        if not active:
            continue

        print(
            f"[{suite_label}] {rows_val}x{cols_val}: running {len(active)} cells ({completed + len(active)}/{total})",
            flush=True,
        )

        status_counts: dict[str, int] = {}
        group_done = [0]

        def _on_row(row: object) -> None:
            status = getattr(row, "status", "?")
            status_counts[status] = status_counts.get(status, 0) + 1
            group_done[0] += 1
            n = group_done[0]
            if n <= 20 or n % 10 == 0 or n == len(active) * replicas:
                parts = "  ".join(f"{s}={c}" for s, c in sorted(status_counts.items()))
                print(
                    f"  [{suite_label}] {rows_val}x{cols_val}: {n}/{len(active) * replicas}  {parts}",
                    flush=True,
                )

        run_cells(
            active,
            store=store,
            timeout_s=timeout,
            workers=workers,
            replicas=replicas,
            resume=resume,
            on_row=_on_row,
        )
        completed += len(active)

        # Re-read current DB rows for this grid and apply the elimination
        # policy.
        raw = [
            row
            for runs in store.load_index().values()
            for row in runs
        ]
        timed_out = _find_fully_timed_out_solvers(
            raw, grid_size, elimination_policy
        )
        for solver in sorted(timed_out):
            eliminated.append((solver, grid_size))
            print(
                f"[{suite_label}] {rows_val}x{cols_val}: eliminating {solver} from all larger grids",
                flush=True,
            )


def _handle_run(args: argparse.Namespace) -> None:
    suites = _resolve_suites(args.suites)
    db_path = resolve_db_path(args.db)
    store = BenchmarkStore(db_path)

    workers = max(1, args.workers)
    seed = args.seed
    instances = args.instances
    timeout_override = args.timeout
    neighbors = args.neighbors
    resume = args.resume
    dims = args.dims
    replicas = max(1, args.replicas)

    for suite in suites:
        timeout = _resolve_timeout(suite, timeout_override)
        print(f"[{suite}] collecting cells ...", flush=True)
        cells = _collect_cells(
            suite,
            dims=dims,
            neighbors=neighbors,
            instances=instances,
            seed=seed,
            timeout=timeout,
        )
        if args.solver:
            cells = [c for c in cells if getattr(c, "solver", None) in args.solver]
        print(
            f"[{suite}] running {len(cells)} cells -> {db_path} ...",
            flush=True,
        )
        _run_suite_with_pruning(
            cells,
            store,
            timeout,
            workers,
            replicas,
            resume,
            suite,
            elimination_policy=args.elimination_policy,
        )
        print(f"[{suite}] done.", flush=True)


def _replica_target(rows: list[dict[str, Any]], fallback: int) -> int:
    targets = [
        target
        for row in rows
        for target in [row.get("legacy_complete_count")]
        if isinstance(target, int) and target > 0
    ]
    if targets:
        return max(targets)
    return fallback


def _handle_status(args: argparse.Namespace) -> None:
    db_path = resolve_db_path(args.db)
    raw = _load_runs(db_path)
    replicas = max(1, args.replicas)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw:
        if args.suite and row.get("suite") not in args.suite:
            continue
        if args.solver and row.get("solver") not in args.solver:
            continue
        key = (
            str(row.get("suite")),
            f"{row.get('num_rows')}x{row.get('num_cols')}x{row.get('num_items')}x{row.get('num_types')}",
            str(row.get("solver")),
        )
        grouped.setdefault(key, []).append(row)

    total_runs = len(raw)
    total_cells = 0
    total_complete = 0
    total_partial = 0
    total_missing = 0

    # Further group each bucket by result_key to compute completion.
    for (suite, grid, solver), rows in sorted(grouped.items()):
        by_key: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_key.setdefault(str(r.get("result_key")), []).append(r)
        cells = len(by_key)
        complete = 0
        partial = 0
        missing = 0
        for grp in by_key.values():
            n = len(grp)
            target = _replica_target(grp, replicas)
            if n >= target:
                complete += 1
            elif n > 0:
                partial += 1
            else:
                missing += 1
        total_cells += cells
        total_complete += complete
        total_partial += partial
        total_missing += missing
        print(
            f"suite={suite} grid={grid} solver={solver} "
            f"cells={cells} complete={complete} partial={partial} missing={missing}"
        )

    print(
        f"overall: runs={total_runs} cells={total_cells} "
        f"complete={total_complete} partial={total_partial} missing={total_missing}"
    )


def _verify_rows(raw: list[dict[str, Any]], strict: bool) -> tuple[bool, list[str]]:
    errors: list[str] = []
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in raw:
        rk = row.get("result_key")
        if isinstance(rk, str):
            by_key.setdefault(rk, []).append(row)

    strict_identity_fields = (
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
        "k",
        "swaps_per_step",
        "flag_combo",
        "pregeneration_status",
    )

    for rk, grp in sorted(by_key.items()):
        for field in strict_identity_fields:
            values = {r.get(field) for r in grp if r.get(field) is not None}
            if len(values) > 1:
                errors.append(f"{rk}: field {field!r} disagrees: {sorted(values)}")
        indices = [r.get("run_index") for r in grp]
        if any(not isinstance(i, int) or i < 0 for i in indices):
            errors.append(f"{rk}: invalid run_index values: {indices}")
        if len(indices) != len(set(indices)):
            errors.append(f"{rk}: duplicate run_index values: {indices}")
        if strict:
            expected = set(range(len(indices)))
            if set(indices) != expected:
                errors.append(
                    f"{rk}: non-contiguous run_index values: {sorted(indices)}"
                )
        if any(r.get("schema_version") != 2 for r in grp):
            errors.append(f"{rk}: schema_version != 2")
        timeout_vals = {
            r.get("timeout_s") for r in grp if r.get("timeout_s") is not None
        }
        if len(timeout_vals) > 1:
            print(
                f"warning: timeout_s disagrees within result_key={rk!r}: {sorted(timeout_vals)}",
                file=sys.stderr,
            )
    return (not errors), errors


def _handle_verify(args: argparse.Namespace) -> None:
    db_path = resolve_db_path(args.db)
    raw = _load_runs(db_path)
    ok, errors = _verify_rows(raw, strict=args.strict)
    if not ok:
        for err in errors:
            print(f"verify FAIL: {err}", file=sys.stderr)
        raise SystemExit(1)
    print("verify OK")


_HANDLERS = {
    "run": _handle_run,
    "status": _handle_status,
    "verify": _handle_verify,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])
    _HANDLERS[args.cmd](args)


if __name__ == "__main__":
    main()
