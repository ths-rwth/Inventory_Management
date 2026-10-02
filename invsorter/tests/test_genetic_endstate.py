"""Unit tests for GeneticEndStateGenerator (spec §8)."""

from __future__ import annotations

import random
import time
from collections import Counter

import pytest

from invsorter.generate import generate
from invsorter.models import EndState, Item, Pos, Problem, Size, Status
from invsorter.scoring import directions_for, score
from invsorter.solvers.generators.genetic_endstate import (
    GeneticEndStateGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _items_from(end_state: EndState) -> tuple[Item, ...]:
    """Materialise `tuple[Item, ...]` from a decoded grid.

    Item IDs are sequential; `scoring.score` only reads `item.type` and
    `item.pos` so IDs are irrelevant for fitness comparisons.
    """
    items: list[Item] = []
    next_id = 0
    for r, row in enumerate(end_state.grid):
        for c, t in enumerate(row):
            if t != 0:
                items.append(Item(next_id, t, Pos(r, c), Size(1, 1)))
                next_id += 1
    return tuple(items)


# ---------------------------------------------------------------------------
# 8.1 Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_seed_same_grid():
    sol = generate(
        num_rows=10, num_cols=10, num_items=50, num_types=8,
        seed=42, neighbors=8,
    )
    problem = sol.problem

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)

    r1 = gen.generate_end_state(problem, random.Random(123), timeout=3.0)
    r2 = gen.generate_end_state(problem, random.Random(123), timeout=3.0)

    assert r1.status == Status.SOLVED
    assert r2.status == Status.SOLVED
    assert r1.end_state.grid == r2.end_state.grid


# ---------------------------------------------------------------------------
# 8.2 Time budget
# ---------------------------------------------------------------------------


def test_time_budget_upper_bound():
    sol = generate(
        num_rows=10, num_cols=10, num_items=50, num_types=8,
        seed=42, neighbors=8,
    )
    problem = sol.problem

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)

    t0 = time.monotonic()
    result = gen.generate_end_state(problem, random.Random(7), timeout=900.0)
    elapsed = time.monotonic() - t0

    assert result.status == Status.SOLVED
    assert elapsed <= 3.5, f"elapsed {elapsed:.3f}s exceeds 3.5s budget"


# ---------------------------------------------------------------------------
# 8.3 Beats identity score
# ---------------------------------------------------------------------------


def test_beats_identity_on_nontrivial_instance():
    sol = generate(
        num_rows=10, num_cols=10, num_items=50, num_types=8,
        seed=7, neighbors=4,
    )
    problem = sol.problem
    dirs = directions_for(problem.neighbors)
    identity_score = score(problem.items, directions=dirs)

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)
    result = gen.generate_end_state(problem, random.Random(2026), timeout=3.0)

    assert result.status == Status.SOLVED
    ga_items = _items_from(result.end_state)
    ga_score = score(ga_items, directions=dirs)
    assert ga_score > identity_score, (
        f"GA score {ga_score} did not beat identity {identity_score}"
    )


# ---------------------------------------------------------------------------
# 8.4 Multiset preserved
# ---------------------------------------------------------------------------


def test_multiset_preserved():
    sol = generate(
        num_rows=10, num_cols=10, num_items=50, num_types=8,
        seed=11, neighbors=4,
    )
    problem = sol.problem

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)
    result = gen.generate_end_state(problem, random.Random(31), timeout=3.0)

    assert result.status == Status.SOLVED

    grid_counter = Counter(
        t for row in result.end_state.grid for t in row
    )
    expected = Counter(item.type for item in problem.items)
    empties = problem.num_rows * problem.num_cols - len(problem.items)
    expected[0] += empties
    assert grid_counter == expected


# ---------------------------------------------------------------------------
# 8.5 Edge: N=0
# ---------------------------------------------------------------------------


def test_edge_case_zero_items():
    problem = Problem(num_rows=4, num_cols=4, items=(), neighbors=4)

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)

    t0 = time.monotonic()
    result = gen.generate_end_state(problem, random.Random(0), timeout=3.0)
    elapsed = time.monotonic() - t0

    assert result.status == Status.SOLVED
    assert result.end_state.grid == ((0,) * 4,) * 4
    assert elapsed <= 0.1, f"N=0 short-circuit took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# 8.6 Edge: N=1
# ---------------------------------------------------------------------------


def test_edge_case_one_item():
    problem = Problem(
        num_rows=3,
        num_cols=3,
        items=(Item(0, 5, Pos(1, 2), Size(1, 1)),),
        neighbors=4,
    )

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)

    t0 = time.monotonic()
    result = gen.generate_end_state(problem, random.Random(0), timeout=3.0)
    elapsed = time.monotonic() - t0

    assert result.status == Status.SOLVED
    grid = result.end_state.grid
    assert grid[1][2] == 5
    for r in range(3):
        for c in range(3):
            if (r, c) != (1, 2):
                assert grid[r][c] == 0, f"cell ({r},{c}) is {grid[r][c]}, expected 0"
    assert elapsed <= 0.1, f"N=1 short-circuit took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# 8.7 Edge: all items share a single type
# ---------------------------------------------------------------------------


def test_edge_case_all_same_type():
    sol = generate(
        num_rows=4, num_cols=4, num_items=8, num_types=1,
        seed=3, neighbors=4,
    )
    problem = sol.problem
    # Sanity: all items collapsed to type 1.
    assert len({item.type for item in problem.items}) == 1

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)

    # Build the expected identity grid from item positions.
    expected = [[0] * 4 for _ in range(4)]
    for item in problem.items:
        expected[item.pos.row][item.pos.col] = item.type
    expected_grid = tuple(tuple(row) for row in expected)

    t0 = time.monotonic()
    result = gen.generate_end_state(problem, random.Random(0), timeout=3.0)
    elapsed = time.monotonic() - t0

    assert result.status == Status.SOLVED
    assert result.end_state.grid == expected_grid
    assert elapsed <= 0.1, f"all-same-type short-circuit took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# 8.8 Multi-cell assertion
# ---------------------------------------------------------------------------


def test_multi_cell_assertion():
    problem = Problem(
        num_rows=3,
        num_cols=3,
        items=(Item(0, 1, Pos(0, 0), Size(2, 1)),),
        neighbors=4,
    )

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)

    with pytest.raises(AssertionError) as exc_info:
        gen.generate_end_state(problem, random.Random(0), timeout=3.0)

    assert "1x1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 8.9 Typical small call returns SOLVED
# ---------------------------------------------------------------------------


def test_status_solved_on_typical_call():
    sol = generate(
        num_rows=3, num_cols=3, num_items=4, num_types=2,
        seed=1, neighbors=4,
    )
    problem = sol.problem

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)
    result = gen.generate_end_state(problem, random.Random(0), timeout=0.5)

    assert result.status == Status.SOLVED


# ---------------------------------------------------------------------------
# 8.10 Timeout=0 returns best of seed population
# ---------------------------------------------------------------------------


def test_timeout_zero_returns_seed_population_best():
    sol = generate(
        num_rows=10, num_cols=10, num_items=50, num_types=8,
        seed=11, neighbors=4,
    )
    problem = sol.problem

    gen = GeneticEndStateGenerator(neighbors=problem.neighbors)

    t0 = time.monotonic()
    result = gen.generate_end_state(problem, random.Random(0), timeout=0.0)
    elapsed = time.monotonic() - t0

    assert result.status == Status.SOLVED
    assert elapsed <= 1.0, f"timeout=0 setup took {elapsed:.3f}s"

    grid_counter = Counter(
        t for row in result.end_state.grid for t in row
    )
    expected = Counter(item.type for item in problem.items)
    empties = problem.num_rows * problem.num_cols - len(problem.items)
    expected[0] += empties
    assert grid_counter == expected
