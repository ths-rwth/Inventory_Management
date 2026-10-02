"""GreedyReconstructor — heuristic swap reconstructor lifted from twolayer._reconstruct_greedy."""

from __future__ import annotations

import random
from collections import defaultdict

from . import UnboundSolver
from ...models import (
    EndState,
    GenerateResult,
    Item,
    Pos,
    Problem,
    Status,
    Swap,
    TypeGrid,
)


def _build_type_grid(
    items: tuple[Item, ...],
    num_rows: int,
    num_cols: int,
) -> TypeGrid:
    grid = [[0] * num_cols for _ in range(num_rows)]
    for it in items:
        grid[it.pos.row][it.pos.col] = it.type
    return tuple(tuple(row) for row in grid)


class GreedyReconstructor(UnboundSolver):
    name = "greedy_reconstructor"
    description = "Greedy swap reconstruction toward a target type-grid."

    def __init__(self) -> None:
        pass

    # TODO(fairness)[medium]: no shape validation between init grid and target.grid
    # TODO(fairness)[medium]: Item.size silently ignored (assumes 1×1)
    # TODO(fairness)[low]: open-chain cycles leave residual mismatch on anchor
    def generate_moves(
        self,
        problem: Problem,
        target: EndState,
        rng: random.Random,
    ) -> GenerateResult:
        init_grid = _build_type_grid(problem.items, problem.num_rows, problem.num_cols)
        goal_grid = target.grid

        H = len(init_grid)
        W = len(init_grid[0])

        mismatches: dict[tuple[int, int], list[Pos]] = defaultdict(list)
        for r in range(H):
            for c in range(W):
                if init_grid[r][c] != goal_grid[r][c]:
                    mismatches[init_grid[r][c], goal_grid[r][c]].append(Pos(r, c))

        swaps: list[Swap] = []

        # 2-cycle pairing: match (s, t) with (t, s).
        seen: set[tuple[int, int]] = set()
        for (s, t) in list(mismatches):
            if (s, t) in seen:
                continue
            seen.add((s, t))
            seen.add((t, s))
            if s == t:
                continue
            list_st = mismatches[s, t]
            list_ts = mismatches[t, s]
            while list_st and list_ts:
                swaps.append(Swap(list_st.pop(), list_ts.pop()))

        # Resolve remaining longer cycles.
        has_type: dict[int, list[Pos]] = defaultdict(list)
        for (s, _), positions in mismatches.items():
            for p in positions:
                has_type[s].append(p)

        visited: set[Pos] = set()
        need_at: dict[Pos, int] = {}
        for (s, t), positions in mismatches.items():
            for p in positions:
                need_at[p] = t

        for start in list(need_at):
            if start in visited:
                continue

            cycle = [start]
            visited.add(start)
            current_need = need_at[start]

            while True:
                found = None
                candidates = has_type[current_need]
                while candidates:
                    cand = candidates.pop()
                    if cand not in visited:
                        found = cand
                        break
                if found is None:
                    break
                cycle.append(found)
                visited.add(found)
                current_need = need_at[found]

            # Swap anchor with cycle members in reverse order.
            anchor = cycle[0]
            for i in range(len(cycle) - 1, 0, -1):
                swaps.append(Swap(anchor, cycle[i]))

        result_swaps = tuple(swaps)
        if not result_swaps:
            return GenerateResult(result_swaps, Status.ALREADY_OPTIMAL)
        return GenerateResult(result_swaps, Status.SOLVED)
