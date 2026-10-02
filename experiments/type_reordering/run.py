"""Type-reordering micro-benchmark on the two example instances of the paper.

Runs ``smt5_swap`` (OMTI-Known) and ``smt6`` (OMTE-Known) on the instances in
``experiments/type_reordering/instances/`` with ``reorder`` toggled on and
off, multiple replicas per condition.

Each replica runs in its own subprocess so a stuck Z3 instance cannot
poison the rest. Targets were pregenerated once per example via
``GeneticEndStateGenerator`` and are shipped in
``experiments/type_reordering/targets/``; the cached files are reused, so
reruns use exactly the targets of the paper.

Output: one JSON line per (example, solver, reorder, replica) cell at
``results/type_reordering.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXAMPLES: tuple[tuple[str, str], ...] = (
    ("order_example", "experiments/type_reordering/instances/order_example.json"),
    ("order_example_empty", "experiments/type_reordering/instances/order_example_empty.json"),
)

SOLVERS: tuple[str, ...] = ("smt5_swap", "smt6")

DEFAULT_REPLICAS: int = 5
DEFAULT_TIMEOUT_S: float = 600.0
PREGEN_TIMEOUT_S: float = 120.0
PREGEN_MASTER_SEED: int = 20260525


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _solver_worker(
    example_path: str,
    target_grid: list[list[int]],
    solver_name: str,
    reorder: bool,
    seed: int,
    timeout_s: float,
    queue: "multiprocessing.Queue",
) -> None:
    # Silence solver stdout/stderr — smt5_swap / smt6 print debug info.
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        from invsorter.io import load
        from invsorter.models import EndState
        from invsorter.solvers import get_solver

        solution = load(Path(example_path))
        problem = solution.problem
        target = EndState(grid=tuple(tuple(row) for row in target_grid))

        solver_cls = get_solver("unbound", solver_name)
        instance = solver_cls(reorder=reorder)

        t0 = time.perf_counter()
        sol = instance.solve(problem, target, seed=seed, timeout=timeout_s)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        queue.put(
            {
                "status": str(sol.status),
                "elapsed_ms": elapsed_ms,
                "moves_generated": len(sol.moves),
                "effective_moves": len(sol.effective_moves),
                "initial_score": sol.initial_score,
                "final_score": sol.final_score,
            }
        )
    except Exception:
        queue.put(
            {
                "status": "crash",
                "elapsed_ms": None,
                "traceback": traceback.format_exc(),
            }
        )


def _run_isolated(
    example_path: str,
    target_grid: list[list[int]],
    solver_name: str,
    reorder: bool,
    seed: int,
    timeout_s: float,
) -> dict:
    ctx = multiprocessing.get_context("spawn")
    queue: "multiprocessing.Queue" = ctx.Queue()
    proc = ctx.Process(
        target=_solver_worker,
        args=(example_path, target_grid, solver_name, reorder, seed, timeout_s, queue),
    )
    proc.start()
    # Wall-clock budget = solver timeout + slack for import + queue drain.
    proc.join(timeout_s + 60.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "status": "timeout",
            "elapsed_ms": None,
            "traceback": "parent-killed (wall clock exceeded)",
        }
    try:
        return queue.get(timeout=5.0)
    except Exception:
        return {
            "status": "crash",
            "elapsed_ms": None,
            "traceback": "no result from worker queue",
        }


# ---------------------------------------------------------------------------
# Target pregeneration
# ---------------------------------------------------------------------------


def _pregen_worker(
    example_path: str,
    seed: int,
    timeout_s: float,
    queue: "multiprocessing.Queue",
) -> None:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        from invsorter.io import load
        from invsorter.solvers.generators.genetic_endstate import (
            GeneticEndStateGenerator,
        )

        problem = load(Path(example_path)).problem
        result = GeneticEndStateGenerator(
            neighbors=problem.neighbors
        ).generate_end_state(problem, random.Random(seed), timeout_s)
        queue.put(
            {
                "status": str(result.status),
                "grid": [list(row) for row in result.end_state.grid],
            }
        )
    except Exception:
        queue.put({"status": "crash", "traceback": traceback.format_exc()})


def _pregenerate_target(example_key: str, example_path: str, cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    ctx = multiprocessing.get_context("spawn")
    queue: "multiprocessing.Queue" = ctx.Queue()
    # Hash the key into the seed for stability across renamings.
    seed = (PREGEN_MASTER_SEED + sum(ord(c) for c in example_key)) & 0x7FFFFFFF
    proc = ctx.Process(
        target=_pregen_worker,
        args=(example_path, seed, PREGEN_TIMEOUT_S, queue),
    )
    proc.start()
    proc.join(PREGEN_TIMEOUT_S + 30.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        raise RuntimeError(f"pregeneration timed out for {example_key}")
    payload = queue.get(timeout=5.0)
    if "grid" not in payload:
        raise RuntimeError(f"pregeneration failed for {example_key}: {payload}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    example: str
    example_path: str
    solver: str
    reorder: bool
    replica: int
    seed: int
    status: str
    elapsed_ms: float | None
    moves_generated: int | None = None
    effective_moves: int | None = None
    initial_score: int | None = None
    final_score: int | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    traceback: str | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "type_reordering.jsonl",
    )
    parser.add_argument(
        "--target-cache-dir",
        type=Path,
        default=EXPERIMENT_DIR / "targets",
    )
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Pregenerate targets (one per example).
    targets: dict[str, list[list[int]]] = {}
    for example_key, example_rel in EXAMPLES:
        example_path = str(REPO_ROOT / example_rel)
        cache_path = args.target_cache_dir / f"{example_key}.json"
        print(f"[pregen] {example_key} ...", flush=True)
        payload = _pregenerate_target(example_key, example_path, cache_path)
        print(f"[pregen] {example_key}: status={payload['status']}", flush=True)
        targets[example_key] = payload["grid"]

    # 2. Run all (example, solver, reorder, replica) cells.
    rng = random.Random(args.seed)
    with args.out.open("w", encoding="utf-8") as fh:
        total = len(EXAMPLES) * len(SOLVERS) * 2 * args.replicas
        done = 0
        for example_key, example_rel in EXAMPLES:
            example_path = str(REPO_ROOT / example_rel)
            target_grid = targets[example_key]
            for solver_name in SOLVERS:
                for reorder in (True, False):
                    for replica in range(args.replicas):
                        seed = rng.randint(0, 2**31 - 1)
                        done += 1
                        label = (
                            f"[{done:>3}/{total}] {example_key} "
                            f"{solver_name} reorder={reorder} r{replica}"
                        )
                        print(label + " ...", flush=True)
                        t0 = time.perf_counter()
                        result = _run_isolated(
                            example_path=example_path,
                            target_grid=target_grid,
                            solver_name=solver_name,
                            reorder=reorder,
                            seed=seed,
                            timeout_s=args.timeout,
                        )
                        wall = time.perf_counter() - t0
                        row = Row(
                            example=example_key,
                            example_path=example_rel,
                            solver=solver_name,
                            reorder=reorder,
                            replica=replica,
                            seed=seed,
                            status=str(result.get("status", "crash")),
                            elapsed_ms=result.get("elapsed_ms"),
                            moves_generated=result.get("moves_generated"),
                            effective_moves=result.get("effective_moves"),
                            initial_score=result.get("initial_score"),
                            final_score=result.get("final_score"),
                            timeout_s=args.timeout,
                            traceback=result.get("traceback"),
                        )
                        fh.write(json.dumps(asdict(row)) + "\n")
                        fh.flush()
                        ms_str = (
                            f"{row.elapsed_ms:>9.1f}ms"
                            if row.elapsed_ms is not None
                            else "        n/a"
                        )
                        print(
                            f"{label}: {row.status:<10s} {ms_str} (wall {wall:.1f}s)",
                            flush=True,
                        )

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
