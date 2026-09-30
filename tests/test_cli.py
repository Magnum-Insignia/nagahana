"""The CLI runs without optional dependencies and reports undecided config values."""

from pathlib import Path

from nagahana.cli import main

CONF = Path(__file__).resolve().parents[1] / "conf"


def test_cli_commands(capsys):
    for cmd in (["decisions"], ["stages"], ["metrics"], ["access"]):
        assert main(cmd) == 0
    out = capsys.readouterr().out
    assert "D-12" in out and "pretrain-taaft" in out.replace("Self-supervised pretraining: TAAFT", "pretrain-taaft")


def test_check_config_lists_undecided(capsys):
    assert main(["check-config", str(CONF / "model" / "taaft" / "taaft.yaml")]) == 0
    out = capsys.readouterr().out
    assert "??? policy_coupling" in out
