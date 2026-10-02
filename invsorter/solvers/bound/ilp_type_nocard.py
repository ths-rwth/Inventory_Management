"""ILP Type solver without cardinality constraints."""
from __future__ import annotations

import random
from typing import Callable, override

from ...models import Problem, GenerateResult
from . import BoundSolver
from .._utils import moore_neighbours, von_neumann_neighbours, test_similarity
from .ilp_type import solve_ilp2, ILPType


class ILPTypeNoCard(BoundSolver):
    """
    Same as ILPType, but cardinality constraints are disabled.
    This may change the search space and solver behaviour.
    """

    name = "ilp_type_nocard"

    def __init__(
        self,
        num_moves: int = 10,
        neighbours: Callable[
            [int, int, int, int, int], list[tuple[int, int]]
        ] = von_neumann_neighbours,
    ) -> None:
        self.num_moves = num_moves
        self.neighbours = neighbours

    @override
    def generate_moves(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        items = list(problem.items)
        types = list({item.type for item in items})
        type_counts: dict[int, int] = {}
        for item in items:
            type_counts[item.type] = type_counts.get(item.type, 0) + 1

        if problem.neighbors == 8:
            neighbour_fn = moore_neighbours
        else:
            neighbour_fn = von_neumann_neighbours

        seed_val = rng.randint(0, 2**31 - 1)
        model, _timing = solve_ilp2(
            problem.num_rows, problem.num_cols, items, self.num_moves,
            test_similarity, neighbour_fn, types, type_counts,
            timeout=timeout, seed=seed_val,
            no_diff_noop=False,
            upper_left_noop=True,
            cardinality_constraints=False,
        )

        proxy = ILPType(
            num_moves=self.num_moves,
            neighbours=self.neighbours,
            upper_left_noop=True,    # must match the call to solve_ilp2 below
            no_diff_noop=False,
            cardinality_constraints=False,
        )
        return proxy._readback(model, problem, _timing)
