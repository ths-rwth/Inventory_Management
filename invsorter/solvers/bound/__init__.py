"""BoundSolver ABC."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from ...models import GenerateResult, Move, Problem, Solution, Swap
from .._engine_utils import _accepts_timeout
from .._solution_builder import wrap_moves_into_solution


class BoundSolver(ABC):
    """Bound k-cap solvers. Produce a Solution from (problem, seed) within
    a hard k-cap (`num_moves`).

    Canonical objective contract: subclasses SHOULD optimize
        `scoring.score(items, directions=directions_for(problem.neighbors))`
        over DIRECTIONS matching problem.neighbors (4-connected by default).
    Subclasses optimizing a different objective MUST flag this with a
    comment line IMMEDIATELY ABOVE the class
    declaration (NOT inside this docstring).
    """

    name: ClassVar[str] = "unnamed"
    description: ClassVar[str] = "no description"

    # Optional hooks (default None — opt-in per solver, deferred wiring).
    # A solver may set these as ClassVars or instance attrs to declare
    # warm-start callables / tuning-config dicts. Currently NO solver sets
    # them; per the foundations plan these are placeholders the user will
    # gate per-solver later.
    warm_start: ClassVar[Callable[..., tuple[Any, ...]] | None] = None
    tuning_config: ClassVar[dict[str, Any] | None] = None

    @abstractmethod
    def generate_moves(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> tuple[Move, ...] | tuple[Swap, ...] | GenerateResult: ...

    def solve(
        self,
        problem: Problem,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> Solution:
        # Auto-detect whether the concrete generate_moves accepts `timeout`.
        # If yes, forward as kwarg; if no, discard. Per-solver wiring becomes
        # a single-file edit in the concrete solver — no ABC change needed.
        if _accepts_timeout(self.generate_moves):
            def moves_fn(
                p: Problem,
                r: random.Random,
                timeout: float | None = None,
            ) -> tuple[Move, ...] | tuple[Swap, ...] | GenerateResult:
                return self.generate_moves(p, r, timeout=timeout)
        else:
            def moves_fn(
                p: Problem,
                r: random.Random,
                timeout: float | None = None,
            ) -> tuple[Move, ...] | tuple[Swap, ...] | GenerateResult:
                return self.generate_moves(p, r)
        return wrap_moves_into_solution(problem, moves_fn, seed, timeout)
