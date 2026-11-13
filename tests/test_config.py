"""Config inheritance, overrides and hashing."""

from __future__ import annotations

import pytest

from eeg2img.config import apply_overrides, config_hash, deep_merge, load_config, require


def test_deep_merge_overrides_leaves_and_keeps_siblings():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1}


def test_base_chain_is_resolved_relative_to_the_file(tmp_path):
    (tmp_path / "base.yaml").write_text("model: {name: ridge, depth: 1}\nseed: 0\n")
    child = tmp_path / "sub" / "child.yaml"
    child.parent.mkdir()
    child.write_text("_base_: ../base.yaml\nmodel: {depth: 4}\n")
    config = load_config(child)
    assert config == {"model": {"name": "ridge", "depth": 4}, "seed": 0}


def test_circular_base_chain_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\n")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_overrides_are_typed_and_create_missing_parents():
    config = apply_overrides({"fit": {"alpha": 1.0}}, ["fit.alpha=10", "new.flag=true", "name=x"])
    assert config == {"fit": {"alpha": 10}, "new": {"flag": True}, "name": "x"}


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["no_equals_sign"])


def test_hash_ignores_key_order_and_tracks_values():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_require_names_the_missing_key():
    with pytest.raises(KeyError, match=r"model\.name"):
        require({"model": {}}, "model.name")


def test_shipped_experiment_configs_resolve():
    for name in ("smoke_mean", "smoke_ridge"):
        config = load_config(f"configs/experiment/{name}.yaml")
        assert require(config, "model.name") in {"mean", "ridge"}
        assert require(config, "data.name") == "synthetic"
        assert isinstance(require(config, "seed"), int)
