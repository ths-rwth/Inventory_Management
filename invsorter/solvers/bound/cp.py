import argparse
import random
import time
from pathlib import Path

from ortools.sat.python import cp_model

from ...io import load, save
from ...models import Item, Pos, Problem, Swap, GenerateResult, Status
from . import BoundSolver
from .._engine_utils import apply_timeout_cpsat, apply_seed_cpsat
from .._utils import test_similarity


def _greedy_swaps(
    init_grid: list[list[int]],
    num_rows: int,
    num_cols: int,
    num_moves: int,
    dirs: tuple[tuple[int, int], ...],
) -> tuple[list[tuple[int, int, int, int]], list[list[list[int]]]]:
    """Greedy swap sequence for warm-starting the CP solver.

    Returns (swaps, grids) where grids[k] is the type grid at step k.
    """
    H, W, K = num_rows, num_cols, num_moves

    def _sim(t1: int, t2: int) -> int:
        return test_similarity(t1, t2)

    def _cell_score(g: list[list[int]], r: int, c: int) -> int:
        t = g[r][c]
        s = 0
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                s += _sim(t, g[nr][nc])
        return s

    grid = [row[:] for row in init_grid]
    grids = [[row[:] for row in grid]]
    swaps: list[tuple[int, int, int, int]] = []

    for _ in range(K):
        best_delta = 0
        best = (0, 0, 0, 0)

        for r1 in range(H):
            for c1 in range(W):
                for r2 in range(r1, H):
                    c2_start = c1 + 1 if r2 == r1 else 0
                    for c2 in range(c2_start, W):
                        if grid[r1][c1] == grid[r2][c2]:
                            continue
                        old = (_cell_score(grid, r1, c1)
                               + _cell_score(grid, r2, c2))
                        # When the two cells are neighbors, their mutual
                        # contribution is counted in both _cell_score calls;
                        # subtract it to avoid double-counting.
                        neighbors = (abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1)
                        if neighbors:
                            old -= 2 * _sim(grid[r1][c1], grid[r2][c2])
                        grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]
                        new = (_cell_score(grid, r1, c1)
                               + _cell_score(grid, r2, c2))
                        if neighbors:
                            new -= 2 * _sim(grid[r1][c1], grid[r2][c2])
                        grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]

                        delta = new - old
                        if delta > best_delta:
                            best_delta = delta
                            best = (r1, c1, r2, c2)

        r1, c1, r2, c2 = best
        grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]
        grids.append([row[:] for row in grid])
        swaps.append(best)

    return swaps, grids


def solve_cpsat_cp(
    num_rows: int,
    num_cols: int,
    items: tuple[Item, ...],
    num_moves: int,
    dirs: tuple[tuple[int, int], ...],
    verbose: bool = False,
    use_warm_start: bool = False,
    timeout: float | None = None,
    seed: int | None = None,
) -> tuple[cp_model.CpSolverStatus, cp_model.CpSolver,
           list[cp_model.IntVar], list[cp_model.IntVar],
           dict[str, float]]:
    """Build and solve a CP-style model for grid inventory optimization.

    Cell variables store type labels directly ({0, ..., T}), eliminating
    artificial symmetry between items of the same type and removing the
    need for type-lookup element constraints.
    """
    _timing = {}
    _t0 = time.perf_counter()
    model = cp_model.CpModel()
    H, W = num_rows, num_cols
    K = num_moves
    n_cells = H * W

    T = max((it.type for it in items), default=0)

    type_count: dict[int, int] = {t: 0 for t in range(T + 1)}
    type_count[0] = n_cells
    for it in items:
        type_count[it.type] += 1
        type_count[0] -= 1

    init_grid = [[0] * W for _ in range(H)]
    for it in items:
        init_grid[it.pos.row][it.pos.col] = it.type

    # C8. Effective swap budget: any type-preserving permutation of N
    # non-empty cells needs at most N-1 swaps (cycle decomposition).
    # Instead of adding explicit no-op constraints for k >= N-1,
    # we reduce the unfolding depth so excess variables are never created.
    N = sum(1 for it in items if it.type != 0)
    K_eff = min(K, N - 1) if N > 0 else 0

    # ----- Greedy warm-start -----

    greedy_swaps: list[tuple[int, int, int, int]] | None = None
    greedy_grids: list[list[list[int]]] | None = None
    if use_warm_start:
        greedy_swaps, greedy_grids = _greedy_swaps(
            init_grid, H, W, K_eff, dirs,
        )

    # ----- Variables -----
    # Only create cell layers 0..K_eff (not 0..K).

    cell = [
        [
            [model.new_int_var(0, T, f'c_{r}_{c}_{k}')
             for k in range(K_eff + 1)]
            for c in range(W)
        ]
        for r in range(H)
    ]

    swap1 = [model.new_int_var(0, n_cells - 1, f's1_{k}')
             for k in range(K_eff)]
    swap2 = [model.new_int_var(0, n_cells - 1, f's2_{k}')
             for k in range(K_eff)]

    # ----- Hints from greedy -----
    if use_warm_start:
        for k in range(K_eff):
            r1, c1, r2, c2 = greedy_swaps[k]  # type: ignore[index]  # guarded by use_warm_start
            p1, p2 = r1 * W + c1, r2 * W + c2
            if p1 > p2:
                p1, p2 = p2, p1
            model.add_hint(swap1[k], p1)
            model.add_hint(swap2[k], p2)

        for k in range(K_eff + 1):
            for r in range(H):
                for c in range(W):
                    model.add_hint(cell[r][c][k], greedy_grids[k][r][c])

    # ----- Constraints -----

    # C1. Fix initial state
    for r in range(H):
        for c in range(W):
            model.add(cell[r][c][0] == init_grid[r][c])

    # C2. Symmetry breaking: ordered swap positions (s2 >= s1)
    for k in range(K_eff):
        model.add(swap2[k] >= swap1[k])

    # C3. Element constraints + same-type elimination + swap transition.
    #
    # Element: v1^k = cells[s1^k], v2^k = cells[s2^k] (indirect lookup).
    # Same-type elim: (s1 != s2) => (v1 != v2), since swapping identical
    #   types is a no-op and therefore dominated.
    # Transition (per cell, per step):
    #   α = (s1 == p):  cell gets v2  (effect axiom)
    #   β = (s2 == p):  cell gets v1  (effect axiom)
    #   ¬α ∧ ¬β:        cell persists (frame axiom, multi-literal enforce)
    for k in range(K_eff):
        v1 = model.new_int_var(0, T, f'v1_{k}')
        v2 = model.new_int_var(0, T, f'v2_{k}')
        cells_k = [cell[r][c][k] for r in range(H) for c in range(W)]
        model.add_element(swap1[k], cells_k, v1)
        model.add_element(swap2[k], cells_k, v2)

        is_noop = model.new_bool_var(f'noop_{k}')
        model.add(swap1[k] == swap2[k]).only_enforce_if(is_noop)
        model.add(swap1[k] != swap2[k]).only_enforce_if(is_noop.negated())
        model.add(v1 != v2).only_enforce_if(is_noop.negated())

        for r in range(H):
            for c in range(W):
                p = r * W + c
                is_s1 = model.new_bool_var(f'i1_{r}_{c}_{k}')
                model.add(swap1[k] == p).only_enforce_if(is_s1)
                model.add(swap1[k] != p).only_enforce_if(is_s1.negated())

                is_s2 = model.new_bool_var(f'i2_{r}_{c}_{k}')
                model.add(swap2[k] == p).only_enforce_if(is_s2)
                model.add(swap2[k] != p).only_enforce_if(is_s2.negated())

                model.add(cell[r][c][k + 1] == v2).only_enforce_if(is_s1)
                model.add(cell[r][c][k + 1] == v1).only_enforce_if(is_s2)
                model.add(
                    cell[r][c][k + 1] == cell[r][c][k]
                ).only_enforce_if(is_s1.negated(), is_s2.negated())

    # C4. No-op ordering: (s1^k == s2^k) => (s1^{k+1} == s2^{k+1}).
    # Pushes effective swaps to the front, breaking permutation symmetry.
    for k in range(K_eff - 1):
        noop_k = model.new_bool_var(f'noop_ord_{k}')
        model.add(swap1[k] == swap2[k]).only_enforce_if(noop_k)
        model.add(swap1[k] != swap2[k]).only_enforce_if(noop_k.negated())
        model.add(swap1[k + 1] == swap2[k + 1]).only_enforce_if(noop_k)

    # C5. Consecutive cancellation: forbid two adjacent identical non-trivial
    # swaps, since they cancel out. Encoded as: δ_k ∨ (s1^k ≠ s1^{k+1}) ∨ (s2^k ≠ s2^{k+1}).
    for k in range(K_eff - 1):
        same1 = model.new_bool_var(f'same1_{k}')
        model.add(swap1[k] == swap1[k + 1]).only_enforce_if(same1)
        model.add(swap1[k] != swap1[k + 1]).only_enforce_if(same1.negated())
        same2 = model.new_bool_var(f'same2_{k}')
        model.add(swap2[k] == swap2[k + 1]).only_enforce_if(same2)
        model.add(swap2[k] != swap2[k + 1]).only_enforce_if(same2.negated())
        nontrivial_k = model.new_bool_var(f'nt_{k}')
        model.add(swap1[k] != swap2[k]).only_enforce_if(nontrivial_k)
        model.add(swap1[k] == swap2[k]).only_enforce_if(nontrivial_k.negated())
        model.add_bool_or(
            [nontrivial_k.negated(), same1.negated(), same2.negated()]
        )

    # C6. Per-type cardinality invariants: |{(r,c) : c^k_{r,c} = t}| = n_t.
    # Logically redundant with C3 (swaps preserve the multiset), but the
    # explicit Boolean indicator encoding gives CP-SAT tighter LP bounds
    # and stronger clause learning. Removing this makes solving worse.
    for k in range(1, K_eff + 1):
        for t in range(T + 1):
            indicators: list[cp_model.IntVar] = []
            for r in range(H):
                for c in range(W):
                    b = model.new_bool_var(f'card_{r}_{c}_{k}_{t}')
                    model.add(cell[r][c][k] == t).only_enforce_if(b)
                    model.add(cell[r][c][k] != t).only_enforce_if(b.negated())
                    indicators.append(b)
            model.add(sum(indicators) == type_count[t])

    # C7. Table constraints for scoring on undirected neighbour edges.
    # For each edge {p,q} in E (lexicographic order), a Table constraint
    # maps (type_p, type_q) -> score contribution via the σ function.
    # Objective = 2 * Σ o_e (factor 2 converts undirected to directed).
    scoring_table: list[tuple[int, int, int]] = []
    for t1 in range(T + 1):
        for t2 in range(T + 1):
            scoring_table.append((t1, t2, test_similarity(t1, t2)))
    lo = min(s for _, _, s in scoring_table)
    hi = max(s for _, _, s in scoring_table)

    obj_vars: list[cp_model.IntVar] = []
    for r in range(H):
        for c in range(W):
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (r, c) < (nr, nc):
                    contrib = model.new_int_var(
                        lo, hi, f'o_{r}_{c}_{nr}_{nc}',
                    )
                    model.add_allowed_assignments(
                        [cell[r][c][K_eff], cell[nr][nc][K_eff],
                         contrib],
                        scoring_table,
                    )
                    obj_vars.append(contrib)

    model.maximize(2 * sum(obj_vars))

    # ----- Decision strategy: branch on swap variables first -----
    #model.add_decision_strategy(
    #    swap1 + swap2,
    #    cp_model.CHOOSE_FIRST,
    #    cp_model.SELECT_MIN_VALUE,
    #)

    # ----- Solve -----
    solver = cp_model.CpSolver()
    apply_timeout_cpsat(solver, timeout)
    apply_seed_cpsat(solver, seed)
    solver.parameters.num_workers = 1
    solver.parameters.linearization_level = 2 # adds all Boolean constraints to the LP
    solver.parameters.symmetry_level = 4 # [0-4], higher = more aggressive symmetry breaking 
    solver.parameters.use_symmetry_in_lp = True # Fold symmetric variables in LP (type-indistinguishable items)
    solver.parameters.add_objective_cut = True # Integer objective: round up fractional LP bound for free tightening
    solver.parameters.exploit_best_solution = True # When branching, prioritize the hint
    solver.parameters.max_cut_rounds_at_level_zero = 10 # Extra cut rounds at root node for tighter initial LP bounds
    if verbose:
        solver.parameters.log_search_progress = True
    _t1 = time.perf_counter()
    _timing['model_build_ms'] = (_t1 - _t0) * 1000.0
    status = solver.solve(model)
    _t2 = time.perf_counter()
    _timing['engine_solve_ms'] = (_t2 - _t1) * 1000.0

    return status, solver, swap1, swap2, _timing


def _extract_swaps(
    solver: cp_model.CpSolver,
    swap1: list[cp_model.IntVar],
    swap2: list[cp_model.IntVar],
    num_cols: int,
) -> tuple[Swap, ...]:
    """Extract swaps from the integer swap-position variables."""
    swaps: list[Swap] = []
    for k in range(len(swap1)):
        s1 = solver.value(swap1[k])
        s2 = solver.value(swap2[k])
        if s1 == s2:
            continue  # skip identity swap
        r1, c1 = divmod(s1, num_cols)
        r2, c2 = divmod(s2, num_cols)
        swaps.append(Swap(pos1=Pos(r1, c1), pos2=Pos(r2, c2)))
    return tuple(swaps)


# Scored on the correct adjacency graph via problem.neighbors (defaults to 4, matching scoring.DIRECTIONS=4).
class CPSolver(BoundSolver):
    """OR-Tools CP-SAT solver using a CP-style formulation."""

    name = 'cp'

    def __init__(
        self,
        num_moves: int = 10,
        verbose: bool = False,
        use_warm_start: bool = False,
    ) -> None:
        self.num_moves = num_moves
        self.verbose = verbose
        self.use_warm_start = use_warm_start

    def generate_moves(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        seed_val = rng.randint(0, 2**31 - 1)
        if problem.neighbors == 8:
            dirs = ((-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1))
        else:  # problem.neighbors == 4
            dirs = ((-1, 0), (0, -1), (0, 1), (1, 0))
        status, solver, swap1, swap2, _timing = solve_cpsat_cp(
            num_rows=problem.num_rows,
            num_cols=problem.num_cols,
            items=problem.items,
            num_moves=self.num_moves,
            dirs=dirs,
            verbose=self.verbose,
            use_warm_start=self.use_warm_start,
            timeout=timeout,
            seed=seed_val,
        )

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            _t3 = time.perf_counter()
            moves = _extract_swaps(solver, swap1, swap2, problem.num_cols)
            readback_ms = (time.perf_counter() - _t3) * 1000.0
            self._timing = {
                'model_build_ms': _timing.get('model_build_ms', 0.0),
                'engine_solve_ms': _timing.get('engine_solve_ms', 0.0),
                'readback_ms': readback_ms,
            }
            out_status = (
                Status.OPTIMAL if status == cp_model.OPTIMAL else Status.SOLVED
            )
            return GenerateResult(moves, out_status)
        else:
            self._timing = _timing
            if status == cp_model.INFEASIBLE:
                return GenerateResult((), Status.UNSAT)
            else:
                return GenerateResult((), Status.TIMEOUT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='OR-Tools CP-SAT inventory solver (CP-style model)'
    )
    parser.add_argument('input', type=str, help='Problem JSON file')
    parser.add_argument('-k', '--moves', type=int, default=10)
    parser.add_argument(
        '-t', '--time-limit', type=float, default=30.0,
        help='Solver time limit in seconds',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show CP-SAT search progress',
    )
    parser.add_argument(
        '--use-warm-start', action='store_true', default=False,
        help='Use internal greedy warm-start (unfair vs other solvers; '
             'disabled by default for benchmark fairness)',
    )
    parser.add_argument('-o', '--output', type=str, default='solution')
    args = parser.parse_args()

    problem = load(Path(args.input))
    solver = CPSolver(
        num_moves=args.moves,
        verbose=args.verbose,
        use_warm_start=args.use_warm_start,
    )
    result = solver.solve(problem.problem)
    save(Path(args.output), result)
    print(
        f'Score: {result.initial_score} -> {result.final_score} '
        f'({len(result.moves)} moves)'
    )


if __name__ == '__main__':
    main()
