from .grid import build_occupancy, cells_of
from .models import Item, Pos
from .solvers._utils import test_similarity

#DIRECTIONS: tuple[Pos, ...] = (
#    Pos(-1, -1), Pos(0, -1), Pos(1, -1),
#    Pos(-1,  0),             Pos(1,  0),
#    Pos(-1,  1), Pos(0,  1), Pos(1,  1),
#)


DIRECTIONS: tuple[Pos, ...] = (Pos(0, -1), Pos(-1,  0), Pos(1,  0), Pos(0,  1))

DIRECTIONS_8: tuple[Pos, ...] = (
    Pos(-1, -1), Pos(0, -1), Pos(1, -1),
    Pos(-1,  0),             Pos(1,  0),
    Pos(-1,  1), Pos(0,  1), Pos(1,  1),
)


def directions_for(neighbors: int) -> tuple[Pos, ...]:
    """Return the direction set for the given neighbor connectivity."""
    if neighbors == 8:
        return DIRECTIONS_8
    return DIRECTIONS


def score(
    items: tuple[Item, ...],
    directions: tuple[Pos, ...] = DIRECTIONS,
) -> int:
    """Adjacency score for entire state. Higher is better."""
    occ = build_occupancy(items)
    items_by_id = {item.id: item for item in items}
    total = 0
    for item in items:
        for cell in cells_of(item):
            for d in directions:
                neighbor = Pos(cell.row + d.row, cell.col + d.col)
                neighbor_id = occ.get(neighbor)
                if neighbor_id is None or neighbor_id == item.id:
                    continue
                total += test_similarity(item.type, items_by_id[neighbor_id].type)
    return total


def pair_scores(
    items: tuple[Item, ...],
    directions: tuple[Pos, ...] = DIRECTIONS,
) -> list[tuple[Pos, Pos, int]]:
    """Per-pair adjacency score breakdown.

    Returns one entry per undirected adjacent cell pair belonging to different
    items. Each entry is (pos_a, pos_b, contribution) where pos_a < pos_b
    lexicographically. The contribution is twice the test_similarity score
    for the pair (each directed edge counted once from each side).

    The sum of all contributions equals ``score(items, directions)``.
    """
    occ = build_occupancy(items)
    items_by_id = {item.id: item for item in items}
    seen: set[frozenset[Pos]] = set()
    result: list[tuple[Pos, Pos, int]] = []
    for item in items:
        for cell in cells_of(item):
            for d in directions:
                neighbor = Pos(cell.row + d.row, cell.col + d.col)
                neighbor_id = occ.get(neighbor)
                if neighbor_id is None or neighbor_id == item.id:
                    continue
                pair = frozenset((cell, neighbor))
                if pair in seen:
                    continue
                seen.add(pair)
                contribution = 2 * test_similarity(
                    item.type, items_by_id[neighbor_id].type
                )
                if cell < neighbor:
                    result.append((cell, neighbor, contribution))
                else:
                    result.append((neighbor, cell, contribution))
    return result
