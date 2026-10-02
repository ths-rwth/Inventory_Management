import argparse
import random
from pathlib import Path

from .grid import build_occupancy
from .io import save
from .models import Item, Pos, Size, Problem, Solution, Status
from .scoring import score


def generate(num_rows: int = 10, num_cols: int = 10, num_items: int = 30, num_types: int = 5, seed: int = 42,
             neighbors: int = 4) -> Solution:
    """Generate a random inventory problem. Deterministic for a given seed."""
    rng = random.Random(seed)
    items: list[Item] = []
    occ: set[Pos] = set()

    for i in range(num_items):
        item_type = rng.randint(1, num_types)  # TODO: do we need a +1 here?
        size = Size(1, 1)

        candidates: list[Pos] = []
        for col in range(num_cols - size.w + 1):
            for row in range(num_rows - size.h + 1):
                if all(
                    Pos(row + d_row, col + d_col) not in occ
                    for d_col in range(size.w)
                    for d_row in range(size.h)
                ):
                    candidates.append(Pos(row, col))
        if not candidates:
            break

        pos = rng.choice(candidates)
        items.append(Item(id=i, type=item_type, pos=pos, size=size))
        for d_col in range(size.w):
            for d_row in range(size.h):
                occ.add(Pos(pos.row + d_row, pos.col + d_col))

    frozen = tuple(items)
    problem = Problem(
        num_cols=num_cols,
        num_rows=num_rows,
        items=frozen,
        neighbors=neighbors,
        seed=seed,
    )
    s = score(frozen)
    return Solution(
        problem=problem,
        moves=(),
        initial_items=frozen,
        final_items=frozen,
        initial_score=s,
        final_score=s,
        status=Status.ALREADY_OPTIMAL,
    )


def generate_batch(
    count: int = 100,
    seed: int = 42,
    **kwargs: int,
) -> list[Solution]:
    """Generate multiple problems. Each gets a sub-seed derived from the master seed."""
    rng = random.Random(seed)
    seeds = [rng.randrange(2**31) for _ in range(count)]
    return [generate(seed=s, **kwargs) for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate inventory problems')
    parser.add_argument('-W', '--width', type=int, default=10)
    parser.add_argument('-H', '--height', type=int, default=10)
    parser.add_argument('-n', '--items', type=int, default=30)
    parser.add_argument('-t', '--types', type=int, default=5)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-c', '--count', type=int, default=100)
    parser.add_argument('-o', '--output', type=str, default='problems')
    args = parser.parse_args()

    solutions = generate_batch(
        count=args.count,
        seed=args.seed,
        num_cols=args.width,
        num_rows=args.height,
        num_items=args.items,
        num_types=args.types,
    )
    out = Path(args.output)
    for i, sol in enumerate(solutions):
        save(out / f'p{i:04d}', sol)
    print(f'Generated {len(solutions)} problems in {out}/')


if __name__ == '__main__':
    main()
