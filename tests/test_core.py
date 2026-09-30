"""Roles (legacy names), run modes (no default), config (???), introspection hooks."""

import pytest
import torch

from nagahana.core.config import ConfigMissing, get_required, iter_missing
from nagahana.core.errors import ModeViolation
from nagahana.core.introspection import ActivationRecorder
from nagahana.core.modes import RunMode, current_mode, require_mode, run_mode
from nagahana.core.roles import Role, resolve


def test_legacy_role_names_resolve_with_warning():
    with pytest.warns(DeprecationWarning):
        assert resolve("renderer") is Role.FORECASTER
    with pytest.warns(DeprecationWarning):
        assert resolve("Planner") is Role.ADVISOR
    assert resolve("advisor") is Role.ADVISOR
    with pytest.raises(ValueError):
        resolve("oracle")


def test_run_mode_has_no_default():
    with pytest.raises(ModeViolation):
        current_mode()
    with run_mode(RunMode.TRAIN):
        assert current_mode() is RunMode.TRAIN
        require_mode(RunMode.TRAIN, component="x")
        with pytest.raises(ModeViolation):
            require_mode(RunMode.INFER_LIVE, component="x")
    with pytest.raises(ModeViolation):
        current_mode()


def test_config_missing_values_are_never_defaulted():
    cfg = {"a": {"b": 1, "c": "???"}, "d": "???"}
    assert get_required(cfg, "a.b") == 1
    with pytest.raises(ConfigMissing):
        get_required(cfg, "a.c")
    with pytest.raises(ConfigMissing):
        get_required(cfg, "a.zzz")
    assert sorted(iter_missing(cfg)) == ["a.c", "d"]


def test_activation_recorder_records_and_cleans_up():
    net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU())
    with ActivationRecorder(net, ["0", "1"], max_items=2) as rec:
        for _ in range(3):
            net(torch.ones(1, 2))
    assert len(rec.records["0"]) == 2 and rec.records["1"][0].shape == (1, 3)
    assert not net[0]._forward_hooks  # hooks removed on exit
    with pytest.raises(KeyError), ActivationRecorder(net, ["nope"]):
        pass
