import argparse
import random
import time
from pathlib import Path

from ...grid import apply_moves
from ...io import load, save
from ...models import GenerateResult, Pos, Problem, Status, Swap
from ...scoring import score
from . import BoundSolver


class GeneticSolver(BoundSolver):
    """Genetic algorithm for grid inventory optimization."""

    name = "genetic"

    def __init__(
        self,
        num_moves: int = 10,
        population_size: int = 100,
        generations: int | None = 300,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        elite_count: int = 2,
    ) -> None:
        """*generations*: max GA iterations. Pass ``None`` for "run until
        the cooperative timeout fires" — in that mode, deadline-exit is
        the normal completion path and the run reports ``SOLVED`` with
        the best individual found so far. Requires the caller to pass a
        finite ``timeout`` to ``solve()`` (else the loop never terminates).
        """
        self.num_moves = num_moves
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count

    def generate_moves(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        _t0 = time.perf_counter()
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        population_size = self.population_size
        generations = self.generations
        mutation_rate = self.mutation_rate
        tournament_size = self.tournament_size
        elite_count = self.elite_count

        positions = _all_positions(problem.num_rows, problem.num_cols)

        if problem.neighbors == 4:
            directions: tuple[Pos, ...] = (Pos(0, -1), Pos(-1, 0), Pos(1, 0), Pos(0, 1))
        else:
            directions = (Pos(0, -1), Pos(-1, 0), Pos(1, 0), Pos(0, 1),
                          Pos(-1, -1), Pos(-1, 1), Pos(1, -1), Pos(1, 1))

        def evaluate(swaps: tuple[Swap, ...]) -> int:
            items = apply_moves(problem.items, swaps)
            return score(items, directions=directions)

        def random_individual() -> tuple[Swap, ...]:
            return tuple(
                _random_swap(positions, rng) for _ in range(self.num_moves)
            )

        population = [random_individual() for _ in range(population_size)]
        fitness = [evaluate(ind) for ind in population]

        _t1 = time.perf_counter()
        model_build_ms = (_t1 - _t0) * 1000.0
        timed_out = False
        unlimited = generations is None
        if unlimited and deadline is None:
            raise ValueError(
                "GeneticSolver(generations=None) requires a finite timeout "
                "passed to solve() — otherwise the loop never terminates."
            )
        gen_idx = 0
        while True:
            if not unlimited and gen_idx >= generations:
                break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            ranked = sorted(range(len(population)), key=lambda i: fitness[i], reverse=True)
            next_gen: list[tuple[Swap, ...]] = [
                population[ranked[i]] for i in range(elite_count)
            ]

            while len(next_gen) < population_size:
                p1 = _tournament(population, fitness, tournament_size, rng)
                p2 = _tournament(population, fitness, tournament_size, rng)
                c1, c2 = _crossover(p1, p2, rng)
                next_gen.append(_mutate(c1, positions, mutation_rate, rng))
                if len(next_gen) < population_size:
                    next_gen.append(
                        _mutate(c2, positions, mutation_rate, rng)
                    )

            population = next_gen
            fitness = [evaluate(ind) for ind in population]
            gen_idx += 1

        _t2 = time.perf_counter()
        engine_solve_ms = (_t2 - _t1) * 1000.0
        best_idx = max(range(len(population)), key=lambda i: fitness[i])
        moves = population[best_idx]
        readback_ms = (time.perf_counter() - _t2) * 1000.0
        self._timing = {
            'model_build_ms': model_build_ms,
            'engine_solve_ms': engine_solve_ms,
            'readback_ms': readback_ms,
            'generations_completed': gen_idx,
        }
        # In unlimited mode, deadline-exit IS the normal completion path —
        # report SOLVED with the best individual. In bounded-gens mode,
        # deadline-exit is an interruption — report TIMEOUT.
        if timed_out and not unlimited:
            return GenerateResult(moves, Status.TIMEOUT)
        if not moves:
            return GenerateResult(moves, Status.ALREADY_OPTIMAL)
        return GenerateResult(moves, Status.SOLVED)


# ---------------------------------------------------------------------------
# GA primitives
# ---------------------------------------------------------------------------


def _all_positions(num_rows: int, num_cols: int) -> tuple[Pos, ...]:
    return tuple(
        Pos(r, c) for r in range(num_rows) for c in range(num_cols)
    )


def _random_swap(positions: tuple[Pos, ...], rng: random.Random) -> Swap:
    p1, p2 = rng.sample(positions, 2)
    return Swap(p1, p2)


def _tournament(
    population: list[tuple[Swap, ...]],
    fitness: list[int],
    k: int,
    rng: random.Random,
) -> tuple[Swap, ...]:
    candidates = rng.sample(range(len(population)), k)
    winner = max(candidates, key=lambda i: fitness[i])
    return population[winner]


def _crossover(
    p1: tuple[Swap, ...],
    p2: tuple[Swap, ...],
    rng: random.Random,
) -> tuple[tuple[Swap, ...], tuple[Swap, ...]]:
    if len(p1) < 2:
        return p1, p2
    point = rng.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


def _mutate(
    individual: tuple[Swap, ...],
    positions: tuple[Pos, ...],
    rate: float,
    rng: random.Random,
) -> tuple[Swap, ...]:
    swaps = list(individual)
    for i in range(len(swaps)):
        if rng.random() < rate:
            swaps[i] = _random_swap(positions, rng)
    return tuple(swaps)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Genetic algorithm solver")
    parser.add_argument("input", type=str, help="Problem JSON file")
    parser.add_argument("-k", "--moves", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-o", "--output", type=str, default="solution")
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--mutation", type=float, default=0.1)
    args = parser.parse_args()

    problem = load(Path(args.input))
    solver = GeneticSolver(
        num_moves=args.moves,
        population_size=args.pop,
        generations=args.gens,
        mutation_rate=args.mutation,
    )
    result = solver.solve(problem.problem, seed=args.seed)
    save(Path(args.output), result)
    print(
        f"Score: {result.initial_score} -> {result.final_score} "
        f"({len(result.moves)} moves)"
    )


if __name__ == "__main__":
    main()
