"""GeneticEndStateGenerator: time-budgeted GA over type-multiset permutations."""

from __future__ import annotations

import random
import time
from typing import ClassVar

from . import Generator
from ...models import (
    EndState,
    GenerateEndStateResult,
    Problem,
    Size,
    Status,
)
from ...scoring import directions_for
from .._utils import test_similarity


# ---------------------------------------------------------------------------
# Module-level GA primitives (unit-testable from tests/test_genetic_endstate.py).
# ---------------------------------------------------------------------------


def _full_fitness(
    pi: list[int],
    slots: tuple[int, ...],
    adj: tuple[tuple[int, ...], ...],
) -> int:
    """Single-count edge sum across the grid; see spec §3.3."""
    total = 0
    for c, neighbours in enumerate(adj):
        tc = slots[pi[c]]
        for nb in neighbours:
            if c < nb:
                total += test_similarity(tc, slots[pi[nb]])
    return total


def _swap_delta(
    pi: list[int],
    a: int,
    b: int,
    slots: tuple[int, ...],
    adj: tuple[tuple[int, ...], ...],
) -> int:
    """Fitness change when swapping pi[a] and pi[b]; see spec §4.2."""
    ta = slots[pi[a]]
    tb = slots[pi[b]]
    if ta == tb:
        return 0
    d = 0
    for n in adj[a]:
        if n == b:
            continue
        tn = slots[pi[n]]
        d += test_similarity(tb, tn) - test_similarity(ta, tn)
    for n in adj[b]:
        if n == a:
            continue
        tn = slots[pi[n]]
        d += test_similarity(ta, tn) - test_similarity(tb, tn)
    return d


def _tournament(
    rng: random.Random,
    fit: list[int],
    k: int,
) -> int:
    """k-way tournament selection; returns winning chromosome index. Spec §3.4."""
    pop_size = len(fit)
    best = rng.randrange(pop_size)
    best_fit = fit[best]
    for _ in range(k - 1):
        candidate = rng.randrange(pop_size)
        if fit[candidate] > best_fit:
            best = candidate
            best_fit = fit[candidate]
    return best


def _pmx(
    p1: list[int],
    p2: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Partially-Mapped Crossover; produces two valid permutation children.

    Verbatim from spec §3.5.
    """
    M = len(p1)
    a, b = sorted(rng.sample(range(M), 2))
    c1 = p1[:]
    c2 = p2[:]
    seg1 = p1[a:b + 1]
    seg2 = p2[a:b + 1]
    c1[a:b + 1] = seg2
    c2[a:b + 1] = seg1
    map_12 = {seg2[i]: seg1[i] for i in range(len(seg1))}
    map_21 = {seg1[i]: seg2[i] for i in range(len(seg1))}
    for i in range(M):
        if a <= i <= b:
            continue
        v = c1[i]
        while v in map_12:
            v = map_12[v]
        c1[i] = v
        v = c2[i]
        while v in map_21:
            v = map_21[v]
        c2[i] = v
    return c1, c2


def _decode(
    pi: list[int],
    slots: tuple[int, ...],
    num_rows: int,
    num_cols: int,
    neighbors: int,
) -> EndState:
    """Decode a chromosome into an EndState (spec §3.1)."""
    M = num_rows * num_cols
    grid = [[0] * num_cols for _ in range(num_rows)]
    for c in range(M):
        r, col = divmod(c, num_cols)
        grid[r][col] = slots[pi[c]]
    return EndState(
        grid=tuple(tuple(row) for row in grid),
        neighbors=neighbors,
    )


def _packed_seed(
    types_order: list[int],
    slots: tuple[int, ...],
    N: int,
    M: int,
) -> list[int]:
    """Build a chromosome that places `types_order` in cells 0..N-1 row-major,
    with empty slots filling the rest. Helper for row-major-pack seeds."""
    slot_by_type: dict[int, list[int]] = {}
    for i in range(N):
        slot_by_type.setdefault(slots[i], []).append(i)
    pi = [-1] * M
    for c, t in enumerate(types_order):
        pi[c] = slot_by_type[t].pop(0)
    empty_iter = iter(range(N, M))
    for c in range(M):
        if pi[c] == -1:
            pi[c] = next(empty_iter)
    return pi


# ---------------------------------------------------------------------------
# Generator class.
# ---------------------------------------------------------------------------


class GeneticEndStateGenerator(Generator):
    name: ClassVar[str] = "genetic_endstate"
    description: ClassVar[str] = (
        "Time-budgeted GA over type-multiset permutations. "
        "Maximises scoring.score; deterministic given (problem, rng_state)."
    )

    def __init__(
        self,
        neighbors: int = 4,
        time_budget_s: float = 3.0,
        population_size: int = 192,
        mutation_rate: float = 0.02,
        tournament_size: int = 3,
        elite_count: int = 2,
        crossover_rate: float = 0.85,
        stagnation_limit: int = 200,
    ) -> None:
        self.neighbors = neighbors
        self.time_budget_s = time_budget_s
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.crossover_rate = crossover_rate
        self.stagnation_limit = stagnation_limit

    def generate_end_state(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateEndStateResult:
        # §6.4: multi-cell-item assertion (before any short-circuits).
        assert all(item.size == Size(1, 1) for item in problem.items), (
            "GeneticEndStateGenerator requires 1x1 items; "
            "got at least one multi-cell item."
        )

        num_rows = problem.num_rows
        num_cols = problem.num_cols
        M = num_rows * num_cols
        N = len(problem.items)
        neighbors = self.neighbors

        # §6.1: N=0 — all-zero grid.
        if N == 0:
            grid = tuple(
                tuple(0 for _ in range(num_cols)) for _ in range(num_rows)
            )
            return GenerateEndStateResult(
                EndState(grid=grid, neighbors=neighbors), Status.SOLVED,
            )

        # §6.2: N=1 — identity grid (single item at its pos).
        if N == 1:
            item = problem.items[0]
            grid_rows = [[0] * num_cols for _ in range(num_rows)]
            grid_rows[item.pos.row][item.pos.col] = item.type
            grid = tuple(tuple(row) for row in grid_rows)
            return GenerateEndStateResult(
                EndState(grid=grid, neighbors=neighbors), Status.SOLVED,
            )

        # §6.3: all items share a single type — short-circuit to identity grid.
        if len({item.type for item in problem.items}) <= 1:
            grid_rows = [[0] * num_cols for _ in range(num_rows)]
            for item in problem.items:
                grid_rows[item.pos.row][item.pos.col] = item.type
            grid = tuple(tuple(row) for row in grid_rows)
            return GenerateEndStateResult(
                EndState(grid=grid, neighbors=neighbors), Status.SOLVED,
            )

        # ----- GA setup (unbudgeted) -----
        # Slots: sorted ascending item types, then zeros for empty cells.
        slots: tuple[int, ...] = (
            tuple(sorted(item.type for item in problem.items))
            + (0,) * (M - N)
        )

        # Precomputed adjacency from self.neighbors (spec §4.1: self.neighbors wins).
        dirs = directions_for(neighbors)
        adj_list: list[tuple[int, ...]] = []
        for c in range(M):
            r, col = divmod(c, num_cols)
            nbs: list[int] = []
            for d in dirs:
                nr, nc = r + d.row, col + d.col
                if 0 <= nr < num_rows and 0 <= nc < num_cols:
                    nbs.append(nr * num_cols + nc)
            adj_list.append(tuple(nbs))
        adj: tuple[tuple[int, ...], ...] = tuple(adj_list)

        # Deadline (spec §5).
        if timeout is None:
            effective = self.time_budget_s
        else:
            effective = min(timeout, self.time_budget_s)
        deadline = time.monotonic() + effective

        # ----- Initial population (spec §3.2) -----
        pop_size = self.population_size
        population: list[list[int]] = []

        # 1. Identity seed: place each item at its current pos cell.
        items_sorted_by_type = sorted(problem.items, key=lambda it: it.type)
        pi_identity = [-1] * M
        for slot_i, item in enumerate(items_sorted_by_type):
            cell = item.pos.row * num_cols + item.pos.col
            pi_identity[cell] = slot_i
        empty_slot_iter = iter(range(N, M))
        for c in range(M):
            if pi_identity[c] == -1:
                pi_identity[c] = next(empty_slot_iter)
        population.append(pi_identity)

        # 2. Row-major-pack-desc seed.
        types_desc = sorted(
            (item.type for item in problem.items), reverse=True,
        )
        population.append(_packed_seed(types_desc, slots, N, M))

        # 3. Row-major-pack-asc seed.
        types_asc = sorted(item.type for item in problem.items)
        population.append(_packed_seed(types_asc, slots, N, M))

        # 4. Random shuffle seeds for the remainder.
        while len(population) < pop_size:
            pi = list(range(M))
            rng.shuffle(pi)
            population.append(pi)

        # Initial cached fitness.
        fit: list[int] = [_full_fitness(pi, slots, adj) for pi in population]

        # ----- GA loop (spec §3.7, §3.8) -----
        elite_count = self.elite_count
        crossover_rate = self.crossover_rate
        mutation_rate = self.mutation_rate
        tournament_size = self.tournament_size
        stagnation_limit = self.stagnation_limit

        best_fit_so_far = max(fit)
        stagnant_gens = 0

        while time.monotonic() < deadline:
            # Elitism: top `elite_count` survive unchanged.
            elite_idx = sorted(
                range(pop_size), key=lambda i: fit[i], reverse=True,
            )[:elite_count]
            new_pop: list[list[int]] = [population[i][:] for i in elite_idx]
            new_fit: list[int] = [fit[i] for i in elite_idx]

            # Breeding fills the remaining `pop_size - elite_count` slots.
            while len(new_pop) < pop_size:
                p1 = _tournament(rng, fit, tournament_size)
                p2 = _tournament(rng, fit, tournament_size)
                if rng.random() < crossover_rate:
                    c1, c2 = _pmx(population[p1], population[p2], rng)
                    # Children differ from parents at many positions; full recompute.
                    f1 = _full_fitness(c1, slots, adj)
                    f2 = _full_fitness(c2, slots, adj)
                else:
                    # Spec §4.3 bullet 3: inherit parent fitness, no recompute.
                    c1 = population[p1][:]
                    c2 = population[p2][:]
                    f1 = fit[p1]
                    f2 = fit[p2]

                # Per-cell mutation on child 1 with incremental fitness update.
                for c in range(M):
                    if rng.random() < mutation_rate:
                        cp = rng.randrange(M)
                        delta = _swap_delta(c1, c, cp, slots, adj)
                        c1[c], c1[cp] = c1[cp], c1[c]
                        f1 += delta
                new_pop.append(c1)
                new_fit.append(f1)
                if len(new_pop) >= pop_size:
                    # Second child dropped if it would overflow (spec §3.7).
                    break

                # Per-cell mutation on child 2.
                for c in range(M):
                    if rng.random() < mutation_rate:
                        cp = rng.randrange(M)
                        delta = _swap_delta(c2, c, cp, slots, adj)
                        c2[c], c2[cp] = c2[cp], c2[c]
                        f2 += delta
                new_pop.append(c2)
                new_fit.append(f2)

            population = new_pop
            fit = new_fit

            gen_best = max(fit)
            if gen_best > best_fit_so_far:
                best_fit_so_far = gen_best
                stagnant_gens = 0
            else:
                stagnant_gens += 1
                if stagnant_gens >= stagnation_limit:
                    break

        best = max(range(pop_size), key=lambda i: fit[i])
        end_state = _decode(population[best], slots, num_rows, num_cols, neighbors)
        return GenerateEndStateResult(end_state, Status.SOLVED)
