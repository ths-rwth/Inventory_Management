import math
from argparse import ArgumentParser
from typing import Callable

from ..models import Item, Size, Pos, Swap


def von_neumann_neighbours(num_rows: int,
                           num_cols: int,
                           row: int,
                           col: int,
                           idx_start: int = 1) -> list[tuple[int, int]]:
    """
    Gives the 4 neighboring positions in the inventory with Manhattan distance 1
    with indices starting at idx_start.
    """
    res = []
    if row > idx_start:
        res.append((row - 1, col))
    if row < num_rows - (1 - idx_start):
        res.append((row + 1, col))
    if col > idx_start:
        res.append((row, col - 1))
    if col < num_cols - (1 - idx_start):
        res.append((row, col + 1))
    return res


def moore_neighbours(num_rows: int,
                     num_cols: int,
                     row: int,
                     col: int,
                     idx_start: int = 1) -> list[tuple[int, int]]:
    """
    Gives the 8 neighbouring positions in the inventory with Chebyshev distance 1 (a king's move)
    with indices starting at idx_start.
    """
    # getting those coordinates with Manhattan distance 1
    res = von_neumann_neighbours(num_rows, num_cols, row, col, idx_start)
    # adding the diagonal coordinates
    if row > idx_start and col > idx_start:
        res.append((row - 1, col - 1))
    if row > idx_start and col < num_cols - (1 - idx_start):
        res.append((row - 1, col + 1))
    if row < num_rows - (1 - idx_start) and col > idx_start:
        res.append((row + 1, col - 1))
    if row < num_rows - (1 - idx_start) and col < num_cols - (1 - idx_start):
        res.append((row + 1, col + 1))
    return res


def get_neighbours(diagonals: bool = False) -> Callable[[int, int, int, int, int], list[tuple[int, int]]]:
    if diagonals:
        return moore_neighbours
    else:
        return von_neumann_neighbours


def test_similarity(type1: int, type2: int) -> int:
    if type1 == 0 or type2 == 0:
        return 0
    if type1 == type2:
        return 10
    if (type1 == 1 and type2 == 2) or (type1 == 2 and type2 == 3):
        return 5
    if type1 > type2:
        return test_similarity(type2, type1)  # symmetric
    else:
        return -5


def add_placeholders(num_rows: int, num_cols: int, items: tuple[Item, ...]) -> tuple[Item, ...]:
    i = max([item.id for item in items]) + 1
    placeholder_type = 0
    items_w_placeholders = list(items)
    for row in range(num_rows):
        for col in range(num_cols):
            if (row, col) not in [(item.pos.row, item.pos.col) for item in items]:
                item = Item(i, placeholder_type, Pos(row=row, col=col), Size(1, 1))
                items_w_placeholders.append(item)
                i += 1
    return tuple(items_w_placeholders)


def get_k_iter_end(placeholder_items: bool, cutting_planes: bool, unfolding_depth: int) -> int:
    k_iter_end = 0
    if cutting_planes:
        k_iter_end = unfolding_depth + 1
    elif not placeholder_items:
        k_iter_end = 1
    return k_iter_end


def add_args_bit_blasting(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('input', type=str, help='Problem JSON file')
    parser.add_argument('-k', '--moves', type=int, default=10)

    parser.add_argument('--placeholder-items', dest='placeholder_items', action='store_true')
    parser.add_argument('--no-placeholder-items', dest='placeholder_items', action='store_false')
    parser.add_argument('--cutting-planes', dest='cutting_planes', action='store_true')
    parser.add_argument('--no-cutting-planes', dest='cutting_planes', action='store_false')

    parser.add_argument('-o', '--output', type=str, default='solution')
    parser.set_defaults(placeholder_items=False, cutting_planes=True)
    return parser


def inv(h: int) -> int:
    return -h + 3


def first_l_positions(num_rows: int, num_cols: int, swaps_per_step: int, idx_start: int = 1) -> list[tuple[int, int]]:
    if swaps_per_step > num_rows * num_cols:
        raise Exception("Number of swaps per step is too high for grid size!")
    res = []
    for row in range(1, math.floor(swaps_per_step / num_cols) + 1):
        for col in range(1, num_cols + 1):
            res.append((row + idx_start - 1, col + idx_start - 1))
    for col in range(1, (swaps_per_step % num_cols) + 1):
        res.append((math.ceil(swaps_per_step / num_cols) + idx_start - 1, col + idx_start - 1))
    return res


def get_max_type(items: list[Item]):
    max_type = 0
    for item in items:
        if item.type > max_type:
            max_type = item.type
    return max_type
