"""Generator ABC."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from ...models import EndState, GenerateEndStateResult, Problem
from .._engine_utils import _accepts_timeout
from .._solution_builder import wrap_end_state


class Generator(ABC):
    """Generators produce target `EndState`s for unbound solvers to reconstruct.

    Canonical objective contract: subclasses SHOULD produce an `EndState`
    that is reachable from `problem.items` via swaps AND maximizes
    `scoring.score` over the EndState grid. Subclasses optimizing a
    different objective MUST flag with a comment
    line IMMEDIATELY ABOVE the class declaration (NOT inside this docstring).

    `generate_end_state` may return a bare `EndState` (legacy) or a
    `GenerateEndStateResult(end_state, status)` (preferred — wraps know
    the solver's declared status without heuristics).
    """

    name: ClassVar[str] = "unnamed"
    description: ClassVar[str] = "no description"

    warm_start: ClassVar[Callable[..., "EndState"] | None] = None
    tuning_config: ClassVar[dict[str, Any] | None] = None

    @abstractmethod
    def generate_end_state(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> EndState | GenerateEndStateResult: ...

    def solve(
        self,
        problem: Problem,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> EndState:
        # Auto-detect whether the concrete generate_end_state accepts
        # `timeout`. Same pattern as Bound/UnboundSolver.solve.
        if _accepts_timeout(self.generate_end_state):
            def end_state_fn(
                p: Problem,
                r: random.Random,
                timeout: float | None = None,
            ) -> EndState | GenerateEndStateResult:
                return self.generate_end_state(p, r, timeout=timeout)
        else:
            def end_state_fn(
                p: Problem,
                r: random.Random,
                timeout: float | None = None,
            ) -> EndState | GenerateEndStateResult:
                return self.generate_end_state(p, r)
        end_state, _status = wrap_end_state(
            problem, end_state_fn, seed, timeout
        )
        return end_state
