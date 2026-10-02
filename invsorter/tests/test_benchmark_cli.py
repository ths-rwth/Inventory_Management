from __future__ import annotations

import pytest

import invsorter.benchmark.__main__ as cli_mod


def _invoke_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    seen: dict[str, object] = {}

    def recorder(args) -> None:
        seen["args"] = args

    monkeypatch.setattr(cli_mod, "_HANDLERS", {name: recorder for name in cli_mod._HANDLERS})
    monkeypatch.setattr(cli_mod.sys, "argv", argv)
    cli_mod.main()
    return seen["args"]


@pytest.mark.parametrize(
    ("argv", "expected_cmd"),
    [
        (["prog", "run", "--db", "runs.jsonl"], "run"),
        (["prog", "status", "--db", "runs.jsonl"], "status"),
        (["prog", "verify", "--db", "runs.jsonl"], "verify"),
    ],
)
def test_main_dispatches_each_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_cmd: str,
) -> None:
    args = _invoke_main(monkeypatch, argv)
    assert args.cmd == expected_cmd
