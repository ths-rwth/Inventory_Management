import json
from pathlib import Path

from .grid import apply_moves, cells_of
from .models import Item, Move, Swap, Pos, Problem, Size, Solution
from .scoring import score


def save(path: Path, solution: Solution) -> None:
    """Write solution as .json and .txt side by side."""
    path = Path(path)
    stem = path.with_suffix('')
    stem.parent.mkdir(parents=True, exist_ok=True)

    data = _to_dict(solution)
    stem.with_suffix('.json').write_text(
        json.dumps(data, indent=2) + '\n', encoding='utf-8'
    )

    ascii_text = _build_ascii(solution)
    stem.with_suffix('.txt').write_text(ascii_text, encoding='utf-8')


def load(path: Path) -> Solution:
    """Load a solution (or problem with 0 moves) from JSON."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    num_cols = data['grid']['num_cols']
    num_rows = data['grid']['num_rows']
    items = tuple(
        Item(
            id=d['id'],
            type=d['type'],
            pos=Pos(d['row'], d['col']),
            size=Size(d['w'], d['h']),
        )
        for d in data['items']
    )
    moves = tuple(
        Move(item_id=d['item_id'], target=Pos(d['row'], d['col']))
        for d in data.get('moves', [])
    )
    swaps = tuple(
        Swap(pos1=Pos(row=d['row1'], col=d['col1']), pos2=Pos(row=d['row2'], col=d['col2']))
        for d in data.get('swaps', [])
    )
    if not moves:
        moves = swaps
        # TODO maybe create an own attribute 'swaps' for a solution so we don't need to make so many case distinctions

    neighbors = data.get('neighbors', 4)
    problem = Problem(
        num_cols=num_cols,
        num_rows=num_rows,
        items=items,
        neighbors=neighbors,
        seed=data.get('seed', 0),
    )
    final_items = apply_moves(items, moves)
    initial_score = score(items)
    final_score = score(final_items)
    return Solution(
        problem=problem,
        moves=moves,
        initial_items=problem.items,
        final_items=final_items,
        initial_score=initial_score,
        final_score=final_score,
    )


def to_ascii(num_rows: int, num_cols: int, items: tuple[Item, ...]) -> str:
    """Render grid state as ASCII string."""
    grid: list[list[str]] = [
        ['.' for _ in range(num_cols)] for _ in range(num_rows)
    ]
    for item in items:
        label = str(item.type)
        for cell in cells_of(item):
            if 0 <= cell.row < num_rows and 0 <= cell.col < num_cols:
                grid[cell.row][cell.col] = label
    lines: list[str] = []
    col_width = 2
    border = '+' + '-' * (num_cols * col_width + 1) + '+'
    lines.append(border)
    for row in grid:
        cells = ' '.join(f'{c}' for c in row)
        lines.append(f'| {cells} |')
    lines.append(border)
    return '\n'.join(lines)


def _to_dict(solution: Solution) -> dict[str, object]:
    p = solution.problem
    return {
        'version': 1,
        'grid': {'num_cols': p.num_cols, 'num_rows': p.num_rows},
        'seed': p.seed,
        'neighbors': p.neighbors,
        'items': [
            {
                'id': item.id,
                'type': item.type,
                'row': item.pos.row,
                'col': item.pos.col,
                'w': item.size.w,
                'h': item.size.h,
            }
            for item in p.items
        ],
        'moves': [
            {'item_id': m.item_id, 'row': m.target.row, 'col': m.target.col}
            for m in solution.moves if isinstance(m, Move)
        ],
        'swaps': [
            {'row1': m.pos1.row, 'col1': m.pos1.col, 'row2': m.pos2.row, 'col2': m.pos2.col}
            for m in solution.moves if isinstance(m, Swap)
        ],
        'initial_score': solution.initial_score,
        'final_score': solution.final_score,
    }


def _build_ascii(solution: Solution) -> str:
    p = solution.problem
    header = (
        f'Grid: {p.num_rows}x{p.num_cols}  '
        f'Seed: {p.seed}  '
        f'Items: {len(p.items)}  '
        f'Moves: {len(solution.moves)}\n'
        f'Score: {solution.initial_score} -> {solution.final_score}\n'
    )
    sections = [header, 'Initial state:', to_ascii(p.num_rows, p.num_cols, p.items)]
    if solution.moves:
        sections.extend([
            '\nFinal state:',
            to_ascii(p.num_rows, p.num_cols, solution.final_items),
        ])
    return '\n'.join(sections) + '\n'
