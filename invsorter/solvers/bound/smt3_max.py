import argparse
import random
import sys
import time
from pathlib import Path

from z3 import Optimize, Int, And, Or, Implies, If, Sum, sat, unsat

from ...grid import apply_move, valid_targets
from ...io import load, save
from ...models import Move, Problem, Swap, Pos, GenerateResult, Status
from . import BoundSolver
from .._engine_utils import apply_timeout_z3, apply_seed_z3
from .._utils import test_similarity

def inv(h: int) -> int:
    return 3 - h
class SMT3maxSolver(BoundSolver):
    """Improved SMT encoding. Same approach as in smt3, but another implementation."""

    name = 'smt3_max'

    def __init__(self, num_moves: int = 10) -> None:
        self.num_moves = num_moves

    # TODO(fairness)[high]: multi-cell items mishandled
    # TODO(fairness)[medium]: unknown status indistinguishable from unsat; surface via GenerateResult
    def generate_moves(
        self,
        problem: Problem,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        items = problem.items
        types = list(set(item.type for item in items))
        if problem.neighbors != 4:
            self._timing = {'model_build_ms': 0.0, 'engine_solve_ms': 0.0, 'readback_ms': 0.0}
            return GenerateResult(moves=(), status=Status.UNSUPPORTED)
        swaps: list[Swap] = []
        
        _t0 = time.perf_counter()
        o = Optimize()
        apply_timeout_z3(o, timeout)
        apply_seed_z3(o, rng.randint(0, 2**31 - 1))

        # ----- Variables -----

        t_start = {}
        t_end = {}
        p_row = {}
        p_col = {}
        t = {}
        
        for r in range(problem.num_rows):
            for s in range(problem.num_cols):
                t_start[r, s] = Int('t_start_' + str(r) + '_' + str(s))
                t_end[r, s] = Int('t_end_' + str(r) + '_' + str(s))

        for k in range(self.num_moves):
            for n in range(1, 3):
                p_row[k, n] = Int('p_row_' + str(k) + '_' + str(n))
                p_col[k, n] = Int('p_col_' + str(k) + '_' + str(n))
                t[k, n] = Int('t_' + str(k) + '_' + str(n))


        # ----- Formula -----

        #1.
        for r in range(problem.num_rows):
            for s in range(problem.num_cols):
                found = False
                for i in items:
                    if i.pos.row == r and i.pos.col == s:
                        o.add(t_start[r, s] == i.type)
                        found = True
                if not found:
                    o.add(t_start[r, s] == 0)

        #2.
        for k in range(self.num_moves):
            for h in range(1, 3):
                o.add(p_row[k, h] >= 0)
                o.add(p_row[k, h] < problem.num_rows)
                o.add(p_col[k, h] >= 0)
                o.add(p_col[k, h] < problem.num_cols)

        #3.
        for k in range(self.num_moves):
            for h in range(1, 3):
                conjunction = []
                for k2 in range(k):
                    conjunction.append(
                        Or(
                            p_row[k, h] != p_row[k2, 1],
                            p_col[k, h] != p_col[k2, 1]
                        )
                    )
                    conjunction.append(
                        Or(
                            p_row[k, h] != p_row[k2, 2],
                            p_col[k, h] != p_col[k2, 2]
                        )
                    )
                disjunction = []
                for r in range(problem.num_rows):
                    for s in range(problem.num_cols):
                        disjunction.append(
                            And(
                                p_row[k, h] == r,
                                p_col[k, h] == s,
                                t[k, h] == t_start[r, s]
                            )
                        )
                o.add(
                    Implies(
                        And(conjunction),
                        Or(disjunction)
                    )
                )

        #4.
        for k in range(self.num_moves):
            for k2 in range(k+1, self.num_moves):
                for h in range(1, 3):
                    for h2 in range(1, 3):
                        conjunction = [p_row[k, h] == p_row[k2, h2], p_col[k, h] == p_col[k2, h2]]
                        for k3 in range(k+1, k2):
                            conjunction.append(Or(
                                p_row[k3, 1] != p_row[k, h],
                                p_col[k3, 1] != p_col[k, h],
                            ))
                            conjunction.append(Or(
                                p_row[k3, 2] != p_row[k, h],
                                p_col[k3, 2] != p_col[k, h],
                            ))
                        o.add(Implies(And(conjunction), t[k2, h2] == t[k, inv(h)]))

        #5.
        for r in range(problem.num_rows):
            for s in range(problem.num_cols):
                conjunction = []
                for k in range(self.num_moves):
                    conjunction.append(Or(
                        p_row[k, 1] != r,
                        p_col[k, 1] != s,
                    ))
                    conjunction.append(Or(
                        p_row[k, 2] != r,
                        p_col[k, 2] != s,
                    ))
                o.add(Implies(And(conjunction), t_end[r, s] == t_start[r, s]))

        #6.
        for r in range(problem.num_rows):
            for s in range(problem.num_cols):
                for h in range(1, 3):
                    for k in range(self.num_moves):
                        conjunction = [p_row[k, h] == r, p_col[k, h] == s]
                        for k2 in range(k+1, self.num_moves):
                            conjunction.append(Or(
                                p_row[k2, 1] != p_row[k, h],
                                p_col[k2, 1] != p_col[k, h]
                            ))
                            conjunction.append(Or(
                                p_row[k2, 2] != p_row[k, h],
                                p_col[k2, 2] != p_col[k, h]
                            ))
                        o.add(Implies(And(conjunction), t_end[r, s] == t[k, inv(h)]))

        
        # ----- objective -----

        objective = []

        for r in range(problem.num_rows):
            for s in range(problem.num_cols):
                if r - 1 >= 0:
                    objective.append(Sum([If(And(t_end[r, s] == t1, t_end[r - 1, s] == t2), test_similarity(t1, t2), 0)
                                          for t1 in types for t2 in types]))
                if r + 1 < problem.num_rows:
                    objective.append(Sum([If(And(t_end[r, s] == t1, t_end[r + 1, s] == t2), test_similarity(t1, t2), 0)
                                          for t1 in types for t2 in types]))
                if s - 1 >= 0:
                    objective.append(Sum([If(And(t_end[r, s] == t1, t_end[r, s - 1] == t2), test_similarity(t1, t2), 0)
                                          for t1 in types for t2 in types]))
                if s + 1 < problem.num_cols:
                    objective.append(Sum([If(And(t_end[r, s] == t1, t_end[r, s + 1] == t2), test_similarity(t1, t2), 0)
                                          for t1 in types for t2 in types]))

        o.maximize(Sum(objective))

        # for i in items:
            # print(i.type)

        _t1 = time.perf_counter()
        _model_build_ms = (_t1 - _t0) * 1000.0
        z3_status = o.check()
        _t2 = time.perf_counter()
        _engine_solve_ms = (_t2 - _t1) * 1000.0
        if z3_status == unsat:
            self._timing = {'model_build_ms': _model_build_ms, 'engine_solve_ms': _engine_solve_ms, 'readback_ms': 0.0}
            return GenerateResult((), Status.UNSAT)
        if z3_status != sat:
            print(f"[smt3_max] no plan: status={z3_status}", file=sys.stderr)
            self._timing = {'model_build_ms': _model_build_ms, 'engine_solve_ms': _engine_solve_ms, 'readback_ms': 0.0}
            return GenerateResult((), Status.TIMEOUT)
        model = o.model()
        # print(model)

        # ----- recompute Swaps -----

        _t3 = time.perf_counter()
        for k in range(self.num_moves):
            swaps.append(Swap(
                pos1=Pos(
                    row=model.eval(p_row[k, 1]).as_long(),
                    col=model.eval(p_col[k, 1]).as_long()
                ),
                pos2=Pos(
                    row=model.eval(p_row[k, 2]).as_long(),
                    col=model.eval(p_col[k, 2]).as_long()
                )
            ))

        result_moves = tuple(swaps)
        readback_ms = (time.perf_counter() - _t3) * 1000.0
        self._timing = {
            'model_build_ms': _model_build_ms,
            'engine_solve_ms': _engine_solve_ms,
            'readback_ms': readback_ms,
        }
        if not result_moves:
            return GenerateResult(result_moves, Status.ALREADY_OPTIMAL)
        return GenerateResult(result_moves, Status.SOLVED)


def main() -> None:
    parser = argparse.ArgumentParser(description='improved smt encoding')
    parser.add_argument('input', type=str, help='Problem JSON file')
    parser.add_argument('-k', '--moves', type=int, default=10)
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-o', '--output', type=str, default='solution')
    args = parser.parse_args()

    problem = load(Path(args.input))
    solver = SMT3maxSolver(num_moves=args.moves)
    result = solver.solve(problem.problem, seed=args.seed)
    save(Path(args.output), result)
    print(
        f'Score: {result.initial_score} -> {result.final_score} '
        f'({len(result.moves)} moves)'
    )


if __name__ == '__main__':
    main()
