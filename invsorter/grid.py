from .models import Item, Move, Pos, Swap


def cells_of(item: Item) -> frozenset[Pos]:
    """All cells occupied by an item."""
    return frozenset(
        Pos(item.pos.row + d_row, item.pos.col + d_col)
        for d_row in range(item.size.h)
        for d_col in range(item.size.w)
    )


def build_occupancy(
        items: tuple[Item, ...],
        exclude: int | None = None,
) -> dict[Pos, int]:
    """Map each occupied cell to its item ID. Optionally exclude one item."""
    occ: dict[Pos, int] = {}
    for item in items:
        if item.id == exclude:
            continue
        for cell in cells_of(item):
            occ[cell] = item.id
    return occ


def fits(num_rows: int, num_cols: int, items: tuple[Item, ...], item_id: int, target: Pos) -> bool:
    """Check if moving item_id to target is valid. Vacated cells count as empty."""
    item = _find_item(items, item_id)
    if (
            target.row < 0
            or target.col < 0
            or target.row + item.size.h > num_rows
            or target.col + item.size.w > num_cols
    ):
        return False
    occ = build_occupancy(items, exclude=item_id)
    for d_row in range(item.size.h):
        for d_col in range(item.size.w):
            if Pos(target.row + d_row, target.col + d_col) in occ:
                return False
    return True


def valid_targets(num_rows: int, num_cols: int, items: tuple[Item, ...], item_id: int) -> list[Pos]:
    """All valid positions for an item given current state."""
    item = _find_item(items, item_id)
    occ = build_occupancy(items, exclude=item_id)
    results: list[Pos] = []
    for col in range(num_cols - item.size.w + 1):
        for row in range(num_rows - item.size.h + 1):
            if all(
                    Pos(row + d_row, col + d_col) not in occ
                    for d_col in range(item.size.w)
                    for d_row in range(item.size.h)
            ):
                results.append(Pos(row, col))
    return results


def apply_move(items: tuple[Item, ...], move: Move) -> tuple[Item, ...]:
    """Return new items tuple with one item relocated. No validation."""
    return tuple(
        item._replace(pos=move.target) if item.id == move.item_id else item
        for item in items
    )


def apply_swap(items: tuple[Item, ...], swap: Swap) -> tuple[Item, ...]:
    """Return new items tuple with two items swapped. No validation."""
    return tuple(
        item._replace(pos=swap.pos2) if item.pos == swap.pos1
        else (item._replace(pos=swap.pos1) if item.pos == swap.pos2 else item)
        for item in items
    )


def apply_moves(
        items: tuple[Item, ...],
        moves: tuple[Move | Swap, ...],
) -> tuple[Item, ...]:
    """Apply a sequence of moves or swaps, returning final items."""

    for move in moves:
        if isinstance(move, Move):
            items = apply_move(items, move)
        elif isinstance(move, Swap):
            items = apply_swap(items, move)
    return items


def all_valid_moves(num_rows: int, num_cols: int, items: tuple[Item, ...]) -> list[Move]:
    """All valid non-no-op moves for all items."""
    moves: list[Move] = []
    for item in items:
        for target in valid_targets(num_rows, num_cols, items, item.id):
            if target != item.pos:
                moves.append(Move(item.id, target))
    return moves


def _find_item(items: tuple[Item, ...], item_id: int) -> Item:
    for item in items:
        if item.id == item_id:
            return item
    raise ValueError(f'item {item_id} not found')
