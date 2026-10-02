"""UnboundSolver ABC."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from ...models import (
    EndState,
    GenerateResult,
    Move,
    Problem,
    Solution,
    Swap,
)
from .._engine_utils import _accepts_timeout
from .._solution_builder import wrap_moves_into_solution


class UnboundSolver(ABC):
    """Unbound solvers. Reconstruct a target `EndState` from a `Problem`,
    emitting a sequence of swaps/moves.

    Canonical objective contract: subclasses SHOULD reach `target.grid`
    type-by-type AND minimize `len(effective_moves)` as a tiebreaker.
    Subclasses optimizing a different objective MUST flag with a
    comment line IMMEDIATELY ABOVE the class
    declaration (NOT inside this docstring).

    Pairing contract: when `compatible_generators` is not None, a Pipeline
    composing this UnboundSolver with a generator whose `name` is not in the
    set raises `ValueError` at Pipeline construction time. Default `None`
    means any pairing is accepted.
    """

    name: ClassVar[str] = "unnamed"
    description: ClassVar[str] = "no description"
    compatible_generators: ClassVar[tuple[str, ...] | None] = None

    warm_start: ClassVar[Callable[..., tuple[Any, ...]] | None] = None
    tuning_config: ClassVar[dict[str, Any] | None] = None

    @abstractmethod
    def generate_moves(
        self,
        problem: Problem,
        target: EndState,
        rng: random.Random,
        timeout: float | None = None,
    ) -> tuple[Move, ...] | tuple[Swap, ...] | GenerateResult: ...

    def solve(
        self,
        problem: Problem,
        target: EndState,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> Solution:
        # Auto-detect whether the concrete generate_moves accepts `timeout`.
        # The wrap layer always sees a 3-arg lambda (so it threads timeout
        # for elapsed-ms heuristics); the lambda body decides whether to
        # forward to the concrete method based on its signature.
        if _accepts_timeout(self.generate_moves):
            def moves_fn(
                p: Problem,
                r: random.Random,
                timeout: float | None = None,
            ) -> tuple[Move, ...] | tuple[Swap, ...] | GenerateResult:
                return self.generate_moves(p, target, r, timeout=timeout)
        else:
            def moves_fn(
                p: Problem,
                r: random.Random,
                timeout: float | None = None,
            ) -> tuple[Move, ...] | tuple[Swap, ...] | GenerateResult:
                return self.generate_moves(p, target, r)
        return wrap_moves_into_solution(problem, moves_fn, seed, timeout)
