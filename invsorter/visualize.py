import argparse
import tkinter as tk
from pathlib import Path

from .grid import apply_moves, cells_of
from .io import load
from .models import Item, Move, Swap, Pos, Solution
from .scoring import directions_for, pair_scores

PALETTE = (
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
    '#59a14f', '#edc948', '#b07aa1', '#ff9da7',
    '#9c755f', '#bab0ac', '#6baed6', '#fd8d3c',
)
EMPTY_COLOR = '#2b2b2b'
GRID_LINE = '#555555'
HIGHLIGHT_SRC = '#ff0000'
HIGHLIGHT_DST = '#00ff00'


class Viewer:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution
        self.p = solution.problem
        self.step = 0
        self.max_step = len(solution.moves)
        self.playing = False
        self.cell_size = self._compute_cell_size()
        self.show_similarities = True

        self.root = tk.Tk()
        self.root.title('Inventory Viewer')
        self.root.configure(bg='#1e1e1e')

        self._build_ui()
        self._bind_keys()
        self._draw()

    def _compute_cell_size(self) -> int:
        target = 600
        return max(16, target // max(self.p.num_cols, self.p.num_rows))

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg='#1e1e1e')
        top.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.btn_prev = tk.Button(
            top, text='< Prev', command=self._prev, width=8,
        )
        self.btn_prev.pack(side=tk.LEFT)

        self.label_step = tk.Label(
            top, text='', fg='white', bg='#1e1e1e',
            font=('Consolas', 12),
        )
        self.label_step.pack(side=tk.LEFT, expand=True)

        self.btn_next = tk.Button(
            top, text='Next >', command=self._next, width=8,
        )
        self.btn_next.pack(side=tk.LEFT)

        self.btn_play = tk.Button(
            top, text='Play', command=self._toggle_play, width=8,
        )
        self.btn_play.pack(side=tk.LEFT, padx=(10, 0))

        info = tk.Frame(self.root, bg='#1e1e1e')
        info.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.label_score = tk.Label(
            info, text='', fg='#cccccc', bg='#1e1e1e',
            font=('Consolas', 11),
        )
        self.label_score.pack(side=tk.LEFT)

        self.label_move = tk.Label(
            info, text='', fg='#999999', bg='#1e1e1e',
            font=('Consolas', 10),
        )
        self.label_move.pack(side=tk.RIGHT)

        cw = self.p.num_cols * self.cell_size + 1
        ch = self.p.num_rows * self.cell_size + 1
        self.canvas = tk.Canvas(
            self.root, width=cw, height=ch, bg=EMPTY_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=(0, 10))

    def _bind_keys(self) -> None:
        self.root.bind('<Left>', lambda _: self._prev())
        self.root.bind('<Right>', lambda _: self._next())
        self.root.bind('<space>', lambda _: self._toggle_play())
        self.root.bind('<KeyPress-s>', lambda _: self._toggle_similarities())
        self.root.bind('<KeyPress-S>', lambda _: self._toggle_similarities())

    def _prev(self) -> None:
        if self.step > 0:
            self.step -= 1
            self._draw()

    def _next(self) -> None:
        if self.step < self.max_step:
            self.step += 1
            self._draw()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.btn_play.config(text='Pause' if self.playing else 'Play')
        if self.playing:
            self._auto_step()

    def _toggle_similarities(self) -> None:
        self.show_similarities = not self.show_similarities
        self._draw()

    def _auto_step(self) -> None:
        if not self.playing or self.step >= self.max_step:
            self.playing = False
            self.btn_play.config(text='Play')
            return
        self.step += 1
        self._draw()
        self.root.after(800, self._auto_step)

    def _current_items(self) -> tuple[Item, ...]:
        moves = self.solution.moves[:self.step]
        return apply_moves(self.p.items, moves)

    def _draw(self) -> None:
        self.canvas.delete('all')
        items = self._current_items()
        cs = self.cell_size

        for item in items:
            color = PALETTE[item.type % len(PALETTE)]
            for cell in cells_of(item):
                x0 = cell.col * cs
                y0 = cell.row * cs
                self.canvas.create_rectangle(
                    x0, y0, x0 + cs, y0 + cs,
                    fill=color, outline=GRID_LINE, width=1,
                )
                self.canvas.create_text(
                    x0 + cs // 2, y0 + cs // 2,
                    text=str(item.type), fill='white',
                    font=('Consolas', max(8, cs // 3)),
                )

        dirs = directions_for(self.p.neighbors)
        pairs = pair_scores(
            items, dirs,
        )
        self._draw_similarity_indicators(pairs)

        for x in range(self.p.num_cols + 1):
            self.canvas.create_line(
                x * cs, 0, x * cs, self.p.num_rows * cs, fill=GRID_LINE,
            )
        for y in range(self.p.num_rows + 1):
            self.canvas.create_line(
                0, y * cs, self.p.num_cols * cs, y * cs, fill=GRID_LINE,
            )

        if 0 < self.step <= self.max_step:
            move = self.solution.moves[self.step - 1]
            self._highlight_move(move, items)

        self.label_step.config(
            text=f'State {self.step}/{self.max_step}'
        )

        current_score = sum(contribution for _, _, contribution in pairs)
        self.label_score.config(text=f'Score: {current_score}')

        if 0 < self.step <= self.max_step:
            move = self.solution.moves[self.step - 1]
            if isinstance(move, Move):
                self.label_move.config(
                    text=f'Move: item {move.item_id} -> ({move.target.row}, {move.target.col})'
                )
            elif isinstance(move, Swap):
                self.label_move.config(
                    text=f'Swap: ({move.pos1.row}, {move.pos1.col}) <-> ({move.pos2.row}, {move.pos2.col})'
                )
        else:
            self.label_move.config(text='Initial state')

        self.btn_prev.config(state=tk.NORMAL if self.step > 0 else tk.DISABLED)
        self.btn_next.config(
            state=tk.NORMAL if self.step < self.max_step else tk.DISABLED,
        )

    def _highlight_move(self, move: Move, items: tuple[Item, ...]) -> None:
        cs = self.cell_size
        if isinstance(move, Move):
            for item in items:
                if item.id == move.item_id:
                    for cell in cells_of(item):
                        x0 = cell.col * cs
                        y0 = cell.row * cs
                        self.canvas.create_rectangle(
                            x0 + 1, y0 + 1, x0 + cs - 1, y0 + cs - 1,
                            outline=HIGHLIGHT_DST, width=2,
                        )
        elif isinstance(move, Swap):
            pass  # TODO highlighting Swaps

    def _draw_similarity_indicators(
        self, pairs: list[tuple[Pos, Pos, int]],
    ) -> None:
        if not self.show_similarities:
            return
        cs = self.cell_size
        if cs < 20:
            return
        for pos_a, pos_b, contribution in pairs:
            mid_x = (pos_a.col + pos_b.col + 1) * cs / 2
            mid_y = (pos_a.row + pos_b.row + 1) * cs / 2
            if cs >= 30:
                text = f'+{contribution}' if contribution > 0 else str(contribution)
                color = '#00cc00' if contribution > 0 else '#cc0000'
                font_size = max(6, cs // 4)
            else:  # 20 <= cs < 30
                text = '+' if contribution > 0 else '-'
                color = '#00cc00' if contribution > 0 else '#cc0000'
                font_size = max(6, cs // 3)
            self.canvas.create_text(
                mid_x, mid_y, text=text, fill=color,
                font=('Consolas', font_size, 'bold'),
            )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description='Inventory viewer')
    parser.add_argument('input', type=str, help='JSON file to visualize')
    args = parser.parse_args()

    solution = load(Path(args.input))
    viewer = Viewer(solution)
    viewer.run()


if __name__ == '__main__':
    main()
