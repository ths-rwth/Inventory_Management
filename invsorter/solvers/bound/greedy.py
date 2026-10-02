import copy
import random
import argparse
import time

from typing import Callable, override
from pathlib import Path

from ...io import load, save
from ...models import Problem, Item, Swap, Pos, Size, GenerateResult, Status
from . import BoundSolver
from .._utils import (
    get_neighbours,
    moore_neighbours,
    von_neumann_neighbours,
    test_similarity,
)


def solve_greedy(
        num_rows: int,
        num_cols: int,
        items: list[Item],
        num_iters,
        similarity: Callable[[int, int], int],
        neighbours: Callable[[int, int, int, int, int], list[tuple[int, int]]],
        deadline: float | None = None,
) -> tuple[list[Swap], list[list[list[Item | None]]], bool]:
    """Greedy local-search swap optimization.

    Args:
        num_rows, num_cols: grid dimensions.
        items: items currently placed on the grid.
        num_iters: maximum number of swap-iterations to perform.
        similarity: pairwise similarity callable.
        neighbours: neighbour-position callable.
        deadline: optional absolute ``time.monotonic()`` timestamp; if set,
            the loop exits early once ``time.monotonic() >= deadline``.

    Returns:
        A 3-tuple ``(swaps, history, timed_out)``. ``swaps`` is whatever the
        algorithm accumulated (possibly empty if the deadline fired before
        iteration 0 completed). ``history`` is the per-iteration inventory
        snapshots used for ILP warm-starts. ``timed_out`` is True iff the
        deadline triggered the early break.
    """
    _timing = {}
    _t0 = time.perf_counter()
    # maintain inventory as 2D Array
    history = []  # history of the inventory - can be used for solution construction (e.g. for the ILP)
    inventory = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for item in items:
        inventory[item.pos.row][item.pos.col] = item
    history.append(copy.deepcopy(inventory))

    # item of type 0 representing positions of the inventory where there is no actual item
    null_item = Item(id=0, type=0, pos=Pos(1, 1), size=Size(1, 1))
    swaps: list[Swap] = []
    timed_out = False

    _t1 = time.perf_counter()
    _timing['model_build_ms'] = (_t1 - _t0) * 1000.0
    for k in range(num_iters):
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break
        best_profit = 0
        best_swap = Swap(pos1=Pos(0, 0), pos2=Pos(0, 0))

        for row1 in range(num_rows):
            for col1 in range(num_cols):
                for (row2, col2) in [(row, col)
                                     for row in range(num_rows)
                                     for col in range(num_cols)
                                     if (row1, col1) <= (row, col)]:
                    old_reward = 0
                    item1 = null_item
                    if inventory[row1][col1] is not None:
                        item1 = inventory[row1][col1]

                    item2 = null_item
                    if inventory[row2][col2] is not None:
                        item2 = inventory[row2][col2]

                    for (neighbor_row, neighbor_col) in neighbours(num_rows, num_cols, row1, col1, 0):
                        neighbor_item = null_item
                        if inventory[neighbor_row][neighbor_col] is not None:
                            neighbor_item = inventory[neighbor_row][neighbor_col]
                        old_reward += similarity(item1.type, neighbor_item.type)

                    for (neighbor_row, neighbor_col) in neighbours(num_rows, num_cols, row2, col2, 0):
                        neighbor_item = null_item
                        if inventory[neighbor_row][neighbor_col] is not None:
                            neighbor_item = inventory[neighbor_row][neighbor_col]
                        old_reward += similarity(item2.type, neighbor_item.type)

                    new_reward = 0
                    for (neighbor_row, neighbor_col) in neighbours(num_rows, num_cols, row1, col1, 0):
                        neighbor_item = null_item
                        if inventory[neighbor_row][neighbor_col] is not None:
                            neighbor_item = inventory[neighbor_row][neighbor_col]
                        if (neighbor_row, neighbor_col) == (row2, col2):
                            neighbor_item = item1
                        new_reward += similarity(item2.type, neighbor_item.type)

                    for (neighbor_row, neighbor_col) in neighbours(num_rows, num_cols, row2, col2, 0):
                        neighbor_item = null_item
                        if inventory[neighbor_row][neighbor_col] is not None:
                            neighbor_item = inventory[neighbor_row][neighbor_col]
                        if (neighbor_row, neighbor_col) == (row1, col1):
                            neighbor_item = item2
                        new_reward += similarity(item1.type, neighbor_item.type)

                    profit = new_reward - old_reward
                    if profit > best_profit:
                        best_profit = profit
                        best_swap = Swap(pos1=Pos(row1, col1), pos2=Pos(row2, col2))

        inventory[best_swap.pos1.row][best_swap.pos1.col], inventory[best_swap.pos2.row][best_swap.pos2.col]\
            = inventory[best_swap.pos2.row][best_swap.pos2.col], inventory[best_swap.pos1.row][best_swap.pos1.col]
        history.append(copy.deepcopy(inventory))
        if best_profit > 0:
            swaps.append(best_swap)

    _t2 = time.perf_counter()
    _timing['engine_solve_ms'] = (_t2 - _t1) * 1000.0
    return swaps, history, timed_out, _timing


class Greedy(BoundSolver):
    """
    A greedy solver that just always does the next move that gives the highest improvement in the objective function.
    So it is as short-sighted as it can be and has no guarantee whatsoever to give optimal solutions.
    Can be used to provide an initial solution for the ILP solver which might give it a good incumbent,
    by which parts of the branch and bound tree can be cut by branch.
    """

    name = 'greedy'

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
    # TODO(fairness)[high]: zero/negative-profit iterations emit no-op default Swap; rely on effective_moves filter or fix loop
    def generate_moves(
        self,
        problem: Problem,
        # rng: unused — greedy is deterministic; signature kept for ABC compatibility
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        # Select neighbor function from problem config
        if problem.neighbors == 8:
            neighbour_fn = moore_neighbours
        else:
            neighbour_fn = von_neumann_neighbours
        swaps, _, timed_out, _timing = solve_greedy(
            problem.num_rows, problem.num_cols,
            list(problem.items), self.num_moves, test_similarity, neighbour_fn,
            deadline=deadline,
        )
        _t3 = time.perf_counter()
        moves = tuple(swaps)
        readback_ms = (time.perf_counter() - _t3) * 1000.0
        self._timing = {
            'model_build_ms': _timing.get('model_build_ms', 0.0),
            'engine_solve_ms': _timing.get('engine_solve_ms', 0.0),
            'readback_ms': readback_ms,
        }
        if timed_out:
            return GenerateResult(moves, Status.TIMEOUT)
        if not moves:
            return GenerateResult(moves, Status.ALREADY_OPTIMAL)
        return GenerateResult(moves, Status.SOLVED)


def main() -> None:
    parser = argparse.ArgumentParser(description='Greedy Solver 1')
    parser.add_argument('input', type=str, help='Problem JSON file')
    parser.add_argument('-k', '--moves', type=int, default=10)
    parser.add_argument('-o', '--output', type=str, default='solution')
    args = parser.parse_args()

    problem = load(Path(args.input))
    solver = Greedy(num_moves=args.moves)
    result = solver.solve(problem.problem)
    save(Path(args.output), result)
    print(
        f'Score: {result.initial_score} -> {result.final_score} '
        f'({len(result.moves)} moves)'
    )


if __name__ == '__main__':
    main()
