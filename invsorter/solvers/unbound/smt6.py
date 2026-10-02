"""SMT6Router — Z3 SMT routing from a target EndState to a swap sequence.

Body lifted from the post-findOptimal block of
invsorter/solvers/smt6_unbounded_k.py:SMT6Solver.generate_moves.
Stdout `print(...)` calls preserved (design §7).
"""

from __future__ import annotations

import random
import sys

from z3 import And, Bool, Implies, Int, Not, Optimize, Or, Sum, is_true, sat, unsat

from . import UnboundSolver
from ...models import EndState, GenerateResult, Item, Pos, Problem, Size, Status, Swap
from .._engine_utils import apply_seed_z3, apply_timeout_z3
from .._utils import add_placeholders


class SMT6Router(UnboundSolver):
    name = "smt6"
    description = "SMT6 routing — find a swap sequence reaching a target EndState."

    def __init__(self, reorder: bool = True) -> None:
        self.reorder = reorder

    # TODO(fairness)[high]: target type absent from source items silently misencoded
    def generate_moves(
        self,
        problem: Problem,
        target: EndState,
        rng: random.Random,
        timeout: float | None = None,
    ) -> GenerateResult:
        items = problem.items
        moves: list[Swap] = []

        extendedItems = add_placeholders(problem.num_rows, problem.num_cols, items)

        goal_items = []
        for r in range(problem.num_rows):
            for c in range(problem.num_cols):
                t = target.grid[r][c]
                goal_items.append(
                    Item(id=0, type=t, pos=Pos(row=r, col=c), size=Size(w=1, h=1))
                )

        print("found goal state")

        num_types = max({i.type for i in items})  # excluding the empty type

        type_frequency = {}
        for i in extendedItems:
            if i.type in type_frequency:
                type_frequency[i.type] += 1
            else:
                type_frequency[i.type] = 1

        print(type_frequency)

        if self.reorder:
            types = [
                t[0]
                for t in sorted(type_frequency.items(), key=lambda element: element[1])
            ]
        else:
            types = [
                t[0]
                for t in sorted(type_frequency.items(), key=lambda element: element[0])
            ]

        print(types)

        itemsOfType = {}
        for t in range(0, len(types)):
            itemsOfType[t + 1] = [i for i in extendedItems if i.type == types[t]]

        itemsOfLargerType = {}
        for t in range(0, len(types)):
            itemsOfLargerType[t + 1] = [
                i for i in extendedItems if i.type in types[t + 1 : len(types)]
            ]

        itemsOfSmallerType = {}
        for t in range(0, len(types)):
            itemsOfSmallerType[t + 1] = [
                i for i in extendedItems if (i.type in types[0:t])
            ]

        goal_itemsOfType = {}
        for t in range(num_types + 1):
            goal_itemsOfType[t] = [i for i in goal_items if i.type == t]

        o = Optimize()

        # ----- variables -----

        x = {}
        y = {}
        s = {}

        for k in range(len(types)):
            for i in extendedItems:
                x[i.id, k] = Int(f"x_id:{i.id}_k:{k}")
                y[i.id, k] = Int(f"y_id:{i.id}_k:{k}")

        for k in range(1, len(types)):
            for i1 in itemsOfType[k]:
                for i2 in itemsOfLargerType[k]:  # + itemsOfType0:
                    s[i1.id, i2.id, k] = Bool(f"s_{i1.id}_{i2.id}_k:{k}")

        # ----- formula -----

        # initial postions
        for i in extendedItems:
            o.add(x[i.id, 0] == i.pos.col)
            o.add(y[i.id, 0] == i.pos.row)

        # at most one swap per active item (fixed first item of swap)
        for k in range(1, len(types)):
            for i1 in itemsOfType[k]:
                for i2 in itemsOfLargerType[k]:  # + itemsOfType0:
                    swap1 = s[i1.id, i2.id, k]
                    for j2 in itemsOfLargerType[k]:  # + itemsOfType0:
                        if i2.id != j2.id:
                            swap2 = s[i1.id, j2.id, k]
                            o.add(
                                Or(Not(swap1), Not(swap2))
                            )  # TODO: remove duplicates/symmetry

        # at most one swap per non active item (fixed second item of swap)
        for k in range(1, len(types)):
            for i2 in itemsOfLargerType[k]:  # + itemsOfType0:
                for i1 in itemsOfType[k]:
                    swap1 = s[i1.id, i2.id, k]
                    for j1 in itemsOfType[k]:
                        if i1.id != j1.id:
                            swap2 = s[j1.id, i2.id, k]
                            o.add(Or(Not(swap1), Not(swap2)))

        # the end state is equal to the previous computed optimal
        for i in extendedItems:
            o.add(
                Or(
                    [
                        And(
                            x[i.id, len(types) - 1] == g.pos.col,
                            y[i.id, len(types) - 1] == g.pos.row,
                        )
                        for g in goal_itemsOfType[i.type]
                    ]
                )
            )

        # update positions of swapped items
        for k in range(1, len(types)):
            for i in itemsOfType[k]:
                for j in itemsOfLargerType[k]:
                    o.add(
                        Implies(
                            s[i.id, j.id, k],
                            And(
                                x[i.id, k] == x[j.id, k - 1],
                                y[i.id, k] == y[j.id, k - 1],
                                x[j.id, k] == x[i.id, k - 1],
                                y[j.id, k] == y[i.id, k - 1],
                            ),
                        )
                    )

        # ensure position consistency for non swapped active items
        for k in range(1, len(types)):
            for i in itemsOfType[k]:
                if len((itemsOfLargerType[k])) > 0:
                    o.add(
                        Implies(
                            And(
                                [Not(s[i.id, j.id, k]) for j in (itemsOfLargerType[k])]
                            ),
                            And(
                                x[i.id, k] == x[i.id, k - 1],
                                y[i.id, k] == y[i.id, k - 1],
                            ),
                        )
                    )

        # ensure position consistency for larger item tpyes
        for k in range(1, len(types)):
            for j in itemsOfLargerType[k]:
                if len(itemsOfType[k]) > 0:
                    o.add(
                        Implies(
                            And([Not(s[i.id, j.id, k]) for i in itemsOfType[k]]),
                            And(
                                x[j.id, k] == x[j.id, k - 1],
                                y[j.id, k] == y[j.id, k - 1],
                            ),
                        )
                    )

        # ensure position consistency for smaller item types
        for k in range(1, len(types)):
            for i in itemsOfSmallerType[k]:
                o.add(x[i.id, k] == x[i.id, k - 1])
                o.add(y[i.id, k] == y[i.id, k - 1])

        # ----- objective -----

        evaluation = []

        for k in range(1, len(types)):
            for i1 in itemsOfType[k]:
                for i2 in itemsOfLargerType[k]:
                    evaluation.append(s[i1.id, i2.id, k])

        # Z3 Sum() over BoolRef swap variables: z3-solver coerces Bool to Int (False=0, True=1),
        # effectively counting total swaps. This is intentional — minimizing swap count.
        o.minimize(Sum(evaluation))

        # ----- solving -----
        print("start solving")
        apply_timeout_z3(o, timeout)
        seed_val = rng.randint(0, 2**31 - 1)
        apply_seed_z3(o, seed_val)
        status = o.check()
        if status == sat:
            model = o.model()

            # ----- reconstruct moves -----

            for k in range(1, len(types)):
                for i1 in itemsOfType[k]:
                    for i2 in itemsOfLargerType[k]:
                        if is_true(
                            model.eval(s[i1.id, i2.id, k], model_completion=True)
                        ):
                            moves.append(
                                Swap(
                                    pos1=Pos(
                                        row=model.eval(y[i2.id, k]).as_long(),
                                        col=model.eval(x[i2.id, k]).as_long(),
                                    ),
                                    pos2=Pos(
                                        row=model.eval(y[i1.id, k]).as_long(),
                                        col=model.eval(x[i1.id, k]).as_long(),
                                    ),
                                )
                            )

            return GenerateResult(tuple(moves), Status.SOLVED)
        elif status == unsat:
            return GenerateResult((), Status.UNSAT)
        else:
            return GenerateResult((), Status.TIMEOUT)
