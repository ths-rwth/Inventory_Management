"""Subprocess-isolated benchmark runner with configurable parallelism.

Each cell runs in a fresh child process so that solver crashes, memory leaks,
and native-extension timeouts do not corrupt the parent.  A thread pool on the
main process manages the subprocesses and serialises JSONL writes.

Windows note: the default ``multiprocessing`` start method is ``spawn``, which
re-imports the worker module in the child.  All worker functions and cell types
must therefore be defined at module scope (no closures, no lambdas), which is
satisfied here.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import queue as _queue_module
import time
import traceback as _traceback_module
from collections.abc import Callable, Sequence
from dataclasses import replace
import os as _os_module
import sys as _sys_module
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from os import PathLike
from pathlib import Path
from typing import Any

from invsorter.benchmark.results import (
    BenchmarkResultRow,
    result_key_for_cell,
    row_for_cell,
)
from invsorter.benchmark.store import BenchmarkStore
from invsorter.benchmark.suites import (
    BoundedCell,
    RuntimeCompareCell,
    SMT4Cell,
    UnboundedCell,
    SymmetrySweepCell,
)

Pathish = str | PathLike[str]
_AnyCell = (
    BoundedCell
    | RuntimeCompareCell
    | SMT4Cell
    | UnboundedCell
    | SymmetrySweepCell
)


def _parse_flag_combo(flag_combo: str) -> dict[str, bool]:
    """Parse 'a=True,b=False' -> {'a': True, 'b': False}.

    Raises ``ValueError`` on:
      - empty input
      - a part missing '='
      - a value that is not exactly 'True' or 'False'
      - a duplicate flag name
    """
    if not flag_combo:
        raise ValueError("flag_combo is empty")
    out: dict[str, bool] = {}
    for part in flag_combo.split(","):
        if "=" not in part:
            raise ValueError(f"flag_combo part {part!r} missing '='")
        name, value = part.split("=", 1)
        if value not in ("True", "False"):
            raise ValueError(
                f"flag_combo value for {name!r} must be 'True' or 'False'; got {value!r}"
            )
        if name in out:
            raise ValueError(f"flag_combo has duplicate flag {name!r}")
        out[name] = value == "True"
    return out


# How long (s) to wait for the result queue after the child has already exited.
# The child's feeder thread should flush before process exit, so 5 s is ample.
_QUEUE_DRAIN_TIMEOUT: float = 5.0

_MIN_FREE_BYTES = 10 * 1024**3  # 10 GiB
_COOLDOWN_S = 0.3


def _replica_target(existing: list[dict[str, Any]], requested: int) -> int:
    targets = [
        target
        for row in existing
        for target in [row.get("legacy_complete_count")]
        if isinstance(target, int) and target > 0
    ]
    if targets:
        return max(targets)
    return requested


def _available_memory_bytes() -> int | float:
    """Return available physical memory in bytes.

    Falls back to ``float('inf')`` if the platform-specific call fails.
    """
    if _sys_module.platform == "win32":
        try:

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = _MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return int(mem.ullAvailPhys)
        except Exception:
            pass
        return float("inf")

    # Linux /proc/meminfo
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass

    return float("inf")


# ---------------------------------------------------------------------------
# Subprocess worker - MUST stay at module scope for Windows spawn pickling.
# ---------------------------------------------------------------------------


def _cell_worker(
    cell: _AnyCell,
    timeout_s: float,
    queue: multiprocessing.Queue,  # type: ignore[type-arg]
) -> None:
    """Run one benchmark cell and put the result dict into *queue*.

    Always puts exactly one item so the parent can always ``get()`` without
    blocking indefinitely.  Catches all exceptions and converts them to a
    ``status="crash"`` row.
    ``timeout_s`` is the wall-clock budget the parent will enforce by killing
    the process if it overruns.  Child-side solver timeouts are handled by
    the parent's process kill; cell types ignore ``timeout_s`` otherwise.
    """
    # Suppress all solver stdout/stderr (Gurobi license banners, solver
    # logs, etc.).  The worker communicates exclusively through *queue* -
    # nothing useful goes to the inherited file descriptors.
    _sys_module.stdout = open(_os_module.devnull, "w")
    _sys_module.stderr = open(_os_module.devnull, "w")
    try:
        # Local imports: they happen inside the child process only, keeping
        # the import graph for the parent slim and avoiding pickling issues.
        from invsorter.solvers import get_solver  # noqa: PLC0415

        worker_t0 = time.perf_counter()
        problem = cell.to_problem()
        gen_elapsed_s = time.perf_counter() - worker_t0

        t0 = time.perf_counter()
        if isinstance(cell, UnboundedCell):
            solver_cls = get_solver("unbound", cell.solver)
            solution = solver_cls().solve(problem, cell.target, seed=cell.seed)
        elif isinstance(cell, SMT4Cell):
            solver_cls = get_solver("bound", cell.solver)
            solution = solver_cls(swaps_per_step=cell.swaps_per_step).solve(
                problem, seed=cell.seed
            )
        elif isinstance(cell, SymmetrySweepCell):
            from invsorter.benchmark.suites import expected_symmetry_flags  # noqa: PLC0415

            solver_cls = get_solver("bound", cell.solver)
            flags = _parse_flag_combo(cell.flag_combo)
            expected = set(expected_symmetry_flags(cell.solver))
            actual = set(flags.keys())
            if actual != expected:
                missing = sorted(expected - actual)
                unknown = sorted(actual - expected)
                raise ValueError(
                    f"SymmetrySweepCell flag_combo for {cell.solver!r} "
                    f"has missing={missing!r} unknown={unknown!r}; "
                    f"expected exactly {sorted(expected)!r}, "
                    f"got {cell.flag_combo!r}"
                )
            solution = solver_cls(num_moves=cell.k, **flags).solve(
                problem,
                seed=cell.seed,
            )
        else:  # BoundedCell
            solver_cls = get_solver("bound", cell.solver)
            solution = solver_cls(num_moves=cell.k).solve(problem, seed=cell.seed)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        row = row_for_cell(
            cell,
            status=str(solution.status),
            elapsed_ms=elapsed_ms,
            timeout_s=timeout_s,
            moves_generated=len(solution.moves),
            raw_moves=len(solution.moves),
            effective_moves=len(solution.effective_moves),
            initial_score=solution.initial_score,
            final_score=solution.final_score,
            improvement=solution.final_score - solution.initial_score,
        )
        queue.put(row.to_json_dict())

    except Exception:  # noqa: BLE001
        tb = _traceback_module.format_exc()
        try:
            crash_row = row_for_cell(
                cell, status="crash", traceback=tb, timeout_s=timeout_s
            )
            queue.put(crash_row.to_json_dict())
        except Exception:  # noqa: BLE001
            # row_for_cell itself failed (should not happen for valid cells).
            # Build a complete dict manually so the parent can reconstruct.
            queue.put(
                {
                    "suite": "unknown",
                    "solver": getattr(cell, "solver", "unknown"),
                    "result_key": cell.cell_key,
                    "num_rows": cell.num_rows,
                    "num_cols": cell.num_cols,
                    "num_items": cell.num_items,
                    "num_types": cell.num_types,
                    "instance_index": cell.instance_index,
                    "seed": cell.seed,
                    "neighbors": cell.neighbors,
                    "status": "crash",
                    "elapsed_ms": None,
                    "timeout_s": timeout_s,
                    "moves_generated": None,
                    "raw_moves": None,
                    "effective_moves": None,
                    "initial_score": None,
                    "final_score": None,
                    "improvement": None,
                    "k": None,
                    "swaps_per_step": None,
                    "flag_combo": None,
                    "note": None,
                    "traceback": tb,
                    "pregeneration_status": None,
                }
            )


# ---------------------------------------------------------------------------
# Thread-level helper - one thread per worker slot, one subprocess per call.
# ---------------------------------------------------------------------------


def _run_one_cell(cell: _AnyCell, timeout_s: float) -> BenchmarkResultRow:
    """Spawn a child process to run *cell*; return a ``BenchmarkResultRow``.

    * Timeout: if the process is still alive after *timeout_s* seconds it is
      killed (SIGKILL / TerminateProcess) and a ``status="timeout"`` row is
      returned with ``elapsed_ms = timeout_s * 1000``.
    * Crash: if the process exited but left the queue empty (e.g. OOM kill),
      a ``status="crash"`` row is returned.
    * Success / solver-reported crash: the worker put a result dict; it is
      used to construct the ``BenchmarkResultRow``.
    """
    mp_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_cell_worker, args=(cell, timeout_s, mp_queue)
    )
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return row_for_cell(
            cell,
            status="timeout",
            timeout_s=timeout_s,
            elapsed_ms=timeout_s * 1000.0,
        )

    # Process has exited; retrieve the result the worker put in the queue.
    # Use a short timeout to let the queue's internal feeder thread flush.
    try:
        row_dict: dict[str, Any] = mp_queue.get(timeout=_QUEUE_DRAIN_TIMEOUT)
    except _queue_module.Empty:
        note = f"subprocess exited with code {proc.exitcode!r} and no result"
        return row_for_cell(cell, status="crash", note=note, timeout_s=timeout_s)

    return BenchmarkResultRow(**row_dict)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_cells(
    cells: Sequence[_AnyCell],
    output_path: Pathish | None = None,
    *,
    store: BenchmarkStore | None = None,
    timeout_s: float = 900.0,
    workers: int = 1,
    replicas: int = 3,
    resume: bool = True,
    on_row: Callable[[BenchmarkResultRow], None] | None = None,
) -> list[BenchmarkResultRow]:
    """Run a list of benchmark cells, writing results to a JSONL store.

    Args:
        cells:        Ordered sequence of cells to evaluate.
        output_path:  (Deprecated) Destination JSONL file path.
        store:        ``BenchmarkStore`` instance for the new DB API.
        timeout_s:    Per-cell wall-clock timeout in seconds.  Default 900 s
                      (15 minutes).
        workers:      Number of parallel worker threads.  Each thread runs
                      exactly one subprocess at a time.  Default 1.
        replicas:     Desired runs per cell.  Default 3.
        resume:       If True, cells with ``count_runs >= replicas`` are skipped.
        on_row:       Optional callback invoked after each row is written.
                      Called on the main thread, safe for printing.

    Returns:
        Rows in completion order (may differ from *cells* order when
        ``workers > 1``).
    """
    # Backward compat: accept output_path if store is not given.
    if store is None:
        if output_path is None:
            raise ValueError("Either store or output_path must be provided")
        store = BenchmarkStore(output_path)

    # Build work list: (cell, run_index) pairs.
    idx = store.load_index()
    work_items: list[tuple[_AnyCell, int]] = []

    if not resume:
        # Compact away all rows for cells in this invocation, then re-run.
        cell_keys = {result_key_for_cell(c) for c in cells}
        if cell_keys and idx:
            store.compact(lambda row: row.get("result_key", "") not in cell_keys)
            # Re-read index after compaction.
            idx = store.load_index()

    for c in cells:
        rk = result_key_for_cell(c)
        existing = idx.get(rk, [])
        current_count = len(existing)
        target_count = _replica_target(existing, replicas)
        if resume and current_count >= target_count:
            continue  # cell already complete
        needed = target_count - current_count
        next_idx = 0
        if existing:
            next_idx = max(r.get("run_index", 0) for r in existing) + 1
        for i in range(needed):
            work_items.append((c, next_idx + i))

    rows: list[BenchmarkResultRow] = []
    if not work_items:
        return rows

    with ThreadPoolExecutor(max_workers=workers) as executor:
        active_futures: dict[Any, tuple[_AnyCell, int]] = {}
        pending_idx = 0

        # Seed initial batch (no cooldown to avoid unnecessary delay).
        while len(active_futures) < workers and pending_idx < len(work_items):
            if _available_memory_bytes() < _MIN_FREE_BYTES:
                break
            cell, run_index = work_items[pending_idx]
            pending_idx += 1
            future = executor.submit(_run_one_cell, cell, timeout_s)
            active_futures[future] = (cell, run_index)

        while active_futures:
            done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
            for future in done:
                cell, run_index = active_futures.pop(future)
                row = future.result()
                store.append(row.to_json_dict(), run_index=run_index)
                row = replace(row, run_index=run_index)
                rows.append(row)
                if on_row is not None:
                    on_row(row)

            # Dispatch loop: replenish up to workers limit.
            while len(active_futures) < workers and pending_idx < len(work_items):
                if _available_memory_bytes() < _MIN_FREE_BYTES:
                    break
                # Cooldown removed: it added ~0.3 s of dead time per cell.
                # Re-enable only if dispatch loop becomes CPU-bound.
                # time.sleep(_COOLDOWN_S)
                cell, run_index = work_items[pending_idx]
                pending_idx += 1
                future = executor.submit(_run_one_cell, cell, timeout_s)
                active_futures[future] = (cell, run_index)

    return rows


if __name__ == "__main__":
    pass
