"""Append-only JSONL store for benchmark raw runs.

The store is a single ``runs.jsonl`` file where each line is a JSON object
representing one raw run.  Normal operation is pure append; the only rewrite
path is ``compact()`` (targeted deletion), which uses a temp file +
``os.replace`` for crash-safety.

Windows atomicity: the temp file is created as a sibling of the target
(same parent directory) so that ``os.replace`` is atomic on Windows too.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path
from typing import Any

from .results import BenchmarkResultRow, _row_json_object

Pathish = str | PathLike[str]


class BenchmarkStoreError(Exception):
    """Raised on store-level integrity or locking violations."""


def resolve_db_path(raw: Pathish) -> Path:
    """Resolve a user-supplied ``--db`` argument to a concrete file path.

    If *raw* points to an existing directory (or has no suffix), treat it as
    ``raw/runs.jsonl``.  Otherwise use *raw* verbatim.
    """
    p = Path(raw)
    if p.is_dir() or (not p.suffix and not p.exists()):
        return p / "runs.jsonl"
    return p


class BenchmarkStore:
    """Append-only JSONL store for raw benchmark runs.

    Construction is cheap; the index is rebuilt on each call to
    ``load_index()`` so callers can re-read after external writes.
    """

    def __init__(self, path: Pathish) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lock helpers
    # ------------------------------------------------------------------

    @property
    def _lock_path(self) -> Path:
        return self._path.with_suffix(".lock")

    def _acquire_lock(self) -> None:
        self._check_stale_lock()
        lock = self._lock_path
        lock.write_text(str(os.getpid()), encoding="utf-8")

    def _release_lock(self) -> None:
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    def _check_stale_lock(self) -> None:
        """Raise ``BenchmarkStoreError`` if a non-stale lock exists."""
        lock = self._lock_path
        if not lock.exists():
            return
        try:
            pid_str = lock.read_text(encoding="utf-8").strip()
            pid = int(pid_str)
            if pid == os.getpid():
                return
            mtime = lock.stat().st_mtime
            age = time.time() - mtime
            if age > 300:  # 5 minutes → stale
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
                return
            raise BenchmarkStoreError(
                f"DB locked by pid {pid} (held for {age:.0f}s). "
                f"If stale, remove {lock} manually or wait for the 5-min timeout."
            )
        except (ValueError, FileNotFoundError):
            pass

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def append(
        self,
        row: BenchmarkResultRow | dict[str, Any],
        *,
        run_index: int,
        imported_from_legacy: bool = False,
        run_started_at: str | None = None,
    ) -> None:
        """Append a single raw-run row to the JSONL file.

        Extra metadata (``run_index``, ``schema_version``, etc.) is injected
        here so callers never have to worry about it.
        """
        self._check_stale_lock()
        obj = _row_json_object(row)
        obj["run_index"] = run_index
        obj["schema_version"] = 2
        obj["imported_from_legacy"] = imported_from_legacy
        if run_started_at is not None:
            obj["run_started_at"] = run_started_at
        self._write_line(obj)

    def append_batch(
        self, rows: Iterable[tuple[BenchmarkResultRow | dict[str, Any], int]]
    ) -> None:
        """Append multiple ``(row, run_index)`` pairs efficiently."""
        self._check_stale_lock()
        with self._path.open("a", encoding="utf-8", newline="\n") as fh:
            for row, run_index in rows:
                obj = _row_json_object(row)
                obj["run_index"] = run_index
                obj["schema_version"] = 2
                fh.write(json.dumps(obj, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")
            fh.flush()

    def _write_line(self, obj: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False, allow_nan=False,
                                separators=(",", ":")) + "\n")
            fh.flush()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def load_index(self) -> dict[str, list[dict[str, Any]]]:
        """Group raw runs by ``result_key``; lists sorted by ``run_index``.

        Malformed lines are silently skipped.
        """
        by_key: dict[str, list[dict[str, Any]]] = {}
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    rk = obj.get("result_key")
                    if isinstance(rk, str):
                        by_key.setdefault(rk, []).append(obj)
        except FileNotFoundError:
            return {}
        # Sort each list by run_index.
        for rows in by_key.values():
            rows.sort(key=lambda r: r.get("run_index", 0))
        return by_key

    def count_runs(self, result_key: str) -> int:
        """Number of raw runs stored for *result_key*."""
        idx = self.load_index()
        return len(idx.get(result_key, []))

    def next_run_index(self, result_key: str) -> int:
        """Next ``run_index`` to use for *result_key*.

        Returns ``max(existing_indices) + 1``, or 0 if no rows exist.
        """
        idx = self.load_index()
        rows = idx.get(result_key, [])
        if not rows:
            return 0
        return max(r.get("run_index", 0) for r in rows) + 1

    # ------------------------------------------------------------------
    # Mutation (rewrite-in-place)
    # ------------------------------------------------------------------

    def compact(self, predicate: Callable[[dict[str, Any]], bool]) -> int:
        """Atomically rewrite the file, keeping rows where *predicate* returns ``True``.

        Returns the number of rows **dropped**.

        Uses a temp file (sibling of target, NOT ``tempfile.gettempdir()``)
        plus ``os.replace`` for crash-safety.  Acquires the advisory lock
        for the duration.
        """
        self._acquire_lock()
        try:
            return self._compact_inner(predicate)
        except BaseException:
            self._release_lock()
            raise

    def _compact_inner(self, predicate: Callable[[dict[str, Any]], bool]) -> int:
        tmp = self._path.with_suffix(".jsonl.tmp")
        assert tmp.parent == self._path.parent, (
            "temp file must be sibling of target for atomic os.replace"
        )
        kept = 0
        dropped = 0
        try:
            with self._path.open("r", encoding="utf-8") as src, \
                 tmp.open("w", encoding="utf-8", newline="\n") as dst:
                for line in src:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if predicate(obj):
                        dst.write(json.dumps(obj, ensure_ascii=False, allow_nan=False,
                                             separators=(",", ":")) + "\n")
                        kept += 1
                    else:
                        dropped += 1
                dst.flush()
            os.replace(str(tmp), str(self._path))
        except BaseException:
            # Clean up temp file on any error; original is untouched.
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise
        self._release_lock()
        return dropped

