# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Weights on disk: what each checkpoint is, the disk per mount, deletion."""

from pathlib import Path

import pytest

from giq import storage
from giq.registry import reload_registry
from giq.storage import StorageError, delete_weights, disk_report, installed, weights_report


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Operator recipes over a split GGUF, a checkpoint directory two recipes
    load, and a plain GGUF."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "solo.gguf").write_bytes(b"a" * 1000)
    for i in (1, 2):
        (models / f"big-0000{i}-of-00002.gguf").write_bytes(b"g" * 600)
    # A checkpoint directory's own safetensors shards are not a GGUF shard set.
    ckpt = models / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_bytes(b"{}")
    (ckpt / "model-00001-of-00002.safetensors").write_bytes(b"s" * 3000)
    (ckpt / "model-00002-of-00002.safetensors").write_bytes(b"s" * 1000)

    operator = tmp_path / "recipes"
    operator.mkdir()
    for name, path in (
        ("solo", "solo.gguf"),
        ("big", "big-00001-of-00002.gguf"),
        ("ckpt-a", "ckpt"),
        ("ckpt-b", "ckpt"),
        ("missing", "missing.gguf"),
    ):
        (operator / f"{name}.yaml").write_text(
            f"name: {name}\nmodalities: [llm]\nengine: llama.cpp\n"
            f"weights: {{path: {path}}}\nvram: {{gb: 1.0}}\n"
        )
    monkeypatch.setenv("GIQ_MODELS_DIR", str(models))
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(operator))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    reload_registry()
    yield models
    monkeypatch.undo()
    reload_registry()


def _of(name: str) -> dict:
    return next(w for w in weights_report() if name in w["recipes"])


def test_every_shard_of_a_split_gguf_counts(layout):
    assert _of("big")["size_bytes"] == 1200 and _of("big")["on_disk"]


def test_a_checkpoint_directory_is_one_item_sized_whole(layout):
    ckpt = _of("ckpt-a")
    assert ckpt["recipes"] == ["ckpt-a", "ckpt-b"]
    assert ckpt["on_disk"] and ckpt["size_bytes"] == 4002


def test_missing_weights_are_reported_not_dropped(layout):
    missing = _of("missing")
    assert not missing["on_disk"] and missing["size_bytes"] == 0 and missing["mount"] is None
    assert not installed("missing") and installed("solo")


def test_the_disk_counts_a_shared_checkpoint_once(layout):
    mounts = {d["mount"]: d for d in disk_report()}
    mount = _of("solo")["mount"]
    ours = sum(w["size_bytes"] for w in weights_report() if w["mount"] == mount)
    assert mounts[mount]["models_bytes"] == ours
    assert mounts[mount]["other_bytes"] >= 0


def test_delete_on_a_read_only_store_is_a_refusal(layout, monkeypatch):
    def refuse(self):
        raise PermissionError(30, "Read-only file system")

    monkeypatch.setattr(Path, "unlink", refuse)
    with pytest.raises(StorageError) as exc:
        delete_weights(_of("solo")["id"], busy=set())
    assert exc.value.status == 409
    assert "read-only" in str(exc.value)
    assert (layout / "solo.gguf").exists()


def test_deleting_a_split_gguf_takes_every_shard(layout):
    result = delete_weights(_of("big")["id"], busy=set())
    assert result["freed_bytes"] == 1200 and len(result["deleted"]) == 2
    assert not list(layout.glob("big-*"))


def test_gguf_shard_expansion(tmp_path):
    shard1 = tmp_path / "big-00001-of-00002.gguf"
    shard2 = tmp_path / "big-00002-of-00002.gguf"
    shard1.write_bytes(b"x")
    shard2.write_bytes(b"y")
    assert storage._expand_gguf(shard1) == [shard1, shard2]


def test_the_real_catalog_reports_without_raising():
    """Against the built-in recipes: every one that loads something has its weights listed."""
    from giq.registry import all_recipes
    from giq.weights import locations

    listed = {r for w in weights_report() for r in w["recipes"]}
    assert listed == {r.name for r in all_recipes() if locations(r.name)}


# --- snapshots loaded by directory, under their environment overrides --------


def test_ocr_snapshot_resolves_by_directory(tmp_path, monkeypatch):
    """The OCR model is a local snapshot loaded by path, not an HF-cache repo:
    the catalog must find it where GIQ_OCR_MODEL_DIR says, or the dashboard
    reports a working model as absent."""
    snap = tmp_path / "baidu-Unlimited-OCR"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"o" * 1234)
    monkeypatch.setenv("GIQ_OCR_MODEL_DIR", str(snap))
    assert installed("unlimited-ocr") and _of("unlimited-ocr")["size_bytes"] == 1234

    monkeypatch.setenv("GIQ_OCR_MODEL_DIR", str(tmp_path / "missing"))
    assert not installed("unlimited-ocr")


def test_glm_ocr_needs_both_of_its_directories(tmp_path, monkeypatch):
    glm = tmp_path / "glm"
    layout = tmp_path / "layout"
    glm.mkdir()
    (glm / "model.safetensors").write_bytes(b"g" * 100)
    monkeypatch.setenv("GIQ_GLM_OCR_MODEL_DIR", str(glm))
    monkeypatch.setenv("GIQ_GLM_LAYOUT_DIR", str(layout))
    # The recognizer alone is not the model: without the layout stage it is not installed.
    assert not installed("glm-ocr")
    layout.mkdir()
    assert installed("glm-ocr")


def test_depth_snapshot_resolves_by_directory(tmp_path, monkeypatch):
    """Depth models are local snapshots under one root, loaded by path."""
    root = tmp_path / "models"
    snap = root / "depth-anything-Depth-Anything-V2-Small-hf"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"d" * 99)
    monkeypatch.setenv("GIQ_DEPTH_MODELS_DIR", str(root))
    assert installed("depth-anything-v2-small")
    assert _of("depth-anything-v2-small")["size_bytes"] == 99
