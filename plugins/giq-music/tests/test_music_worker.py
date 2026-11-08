# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""MusicAdapter: the child's flags come from the recipe, its results become
MusicResults in the parent.

The child is replaced by an inline script that answers every task with a
canned track, so this exercises the protocol, hydration and the error
envelope without a GPU or a model.
"""

import json
import os
import sys

import pytest

from giq_music.adapter import (
    MusicAdapter,
    MusicConfig,
    MusicResult,
    StableAudioAdapter,
    hydrate,
)

CHILD_SCRIPT = """
import sys, json
sys.stdout.write(json.dumps({"type": "ready"}) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "shutdown":
        break
    results = []
    for t in msg["tasks"]:
        if t["id"].startswith("bad"):
            results.append({"id": t["id"], "error": "task needs a 'prompt' describing the music"})
        else:
            results.append({"id": t["id"], "audio_b64": "ZkxhQw==", "format": "flac",
                "sample_rate": 48000, "duration": 30.0, "seed": 7, "bpm": 92,
                "key_scale": "A minor", "time_signature": "4/4", "caption": t["prompt"]})
    sys.stdout.write(json.dumps({"type": "results", "results": results}) + "\\n")
    sys.stdout.flush()
"""


class _StubMusicWorker(MusicAdapter):
    def _command(self) -> list[str]:
        return [sys.executable, "-u", "-c", CHILD_SCRIPT]

    def _spawn_env(self) -> dict[str, str]:
        return dict(os.environ)


def _recipe(tmp_path, monkeypatch, params: str, engine: str = "acestep"):
    (tmp_path / "ace.yaml").write_text(
        "name: ace-test\n"
        "modalities: [text2music]\n"
        f"engine: {engine}\n"
        "weights: {path: ACE-Step-Ace-Step1.5, format: safetensors, licence: mit}\n"
        f"{params}"
        "vram: {gb: 12.0}\n"
        "max_batch: 4\n"
    )
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path))
    from giq import recipes

    recipes.reload()
    return "ace-test"


@pytest.fixture()
def music_recipe(tmp_path, monkeypatch):
    yield _recipe(tmp_path, monkeypatch, "")
    monkeypatch.undo()
    from giq import recipes

    recipes.reload()


async def test_run_batch_returns_music_results(music_recipe):
    worker = _StubMusicWorker(MusicConfig(model=music_recipe))
    await worker.start()
    try:
        results = await worker.run_batch([{"id": "a", "prompt": "lo-fi"}, {"id": "bad-1"}])
    finally:
        await worker.stop()

    assert all(isinstance(r, MusicResult) for r in results)
    a, bad = results
    assert (a.format, a.sample_rate, a.duration, a.seed, a.bpm) == ("flac", 48000, 30.0, 7, 92)
    assert (a.key_scale, a.time_signature, a.caption) == ("A minor", "4/4", "lo-fi")
    assert a.error is None
    assert bad.error and bad.audio_b64 is None
    dumped = json.loads(json.dumps([r.model_dump() for r in results]))
    assert dumped[0]["audio_b64"] == "ZkxhQw=="


def test_the_child_loads_the_folders_the_recipe_names(tmp_path, monkeypatch):
    model = _recipe(
        tmp_path, monkeypatch, "params: {dit: acestep-v15-sft, lm: acestep-5Hz-lm-4B, steps: 50}\n"
    )
    args = MusicAdapter(MusicConfig(model=model)).child_args()
    assert args[args.index("--dit") + 1] == "acestep-v15-sft"
    assert args[args.index("--lm") + 1] == "acestep-5Hz-lm-4B"
    assert args[args.index("--steps") + 1] == "50"
    assert args[args.index("--weights") + 1].endswith("ACE-Step-Ace-Step1.5")


def test_a_recipe_without_a_planner_passes_no_lm(tmp_path, monkeypatch):
    model = _recipe(tmp_path, monkeypatch, "params: {lm: null}\n")
    args = MusicAdapter(MusicConfig(model=model)).child_args()
    assert "--lm" not in args
    assert args[args.index("--dit") + 1] == "acestep-v15-turbo"


def test_params_reject_names_that_are_not_acestep_folders(tmp_path):
    from giq.recipes import load_file

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nmodalities: [text2music]\nengine: acestep\n"
        "weights: {path: x, format: safetensors, licence: mit}\n"
        "params: {dit: ../../etc}\nvram: {gb: 12.0}\n"
    )
    with pytest.raises(ValueError, match="dit"):
        load_file(bad)


def test_hydrate_error_envelope():
    bad = hydrate({"id": "x", "error": "boom"})
    assert bad.id == "x" and bad.error == "boom"


def test_the_builtin_recipe():
    from giq.registry import get_recipe

    recipe = get_recipe("ace-step-1.5-turbo")
    assert recipe is not None and recipe.engine == "acestep"
    assert recipe.modalities == ("text2music",)
    assert recipe.params.dit == "acestep-v15-turbo"  # ty: ignore[unresolved-attribute]
    assert recipe.weights is not None and recipe.weights.source == "hf:ACE-Step/Ace-Step1.5"


def test_stable_audio_flags_come_from_the_recipe(tmp_path, monkeypatch):
    model = _recipe(
        tmp_path,
        monkeypatch,
        "params: {steps: 50, cfg_scale: 7, max_seconds: 60, default_seconds: 90}\n",
        engine="stable-audio",
    )
    adapter = StableAudioAdapter(MusicConfig(model=model))
    args = adapter.child_args()
    assert adapter.child_module == "giq_music_child._sa3_child"
    assert args[args.index("--steps") + 1] == "50"
    assert args[args.index("--cfg-scale") + 1] == "7.0"
    assert args[args.index("--max-seconds") + 1] == "60.0"
    # A default longer than the cap is the cap.
    assert args[args.index("--default-seconds") + 1] == "60.0"


async def test_stable_audio_results_hydrate(tmp_path, monkeypatch):
    model = _recipe(tmp_path, monkeypatch, "", engine="stable-audio")

    class Stub(StableAudioAdapter):
        _command = _StubMusicWorker._command
        _spawn_env = _StubMusicWorker._spawn_env

    worker = Stub(MusicConfig(model=model))
    await worker.start()
    try:
        (result,) = await worker.run_batch([{"id": "a", "prompt": "door slam"}])
    finally:
        await worker.stop()
    assert isinstance(result, MusicResult) and result.caption == "door slam"


@pytest.mark.parametrize(
    ("name", "modality", "seconds"),
    [
        ("stable-audio-3-medium", "text2music", 380),
        ("stable-audio-3-small-music", "text2music", 120),
        ("stable-audio-3-small-sfx", "text2sfx", 120),
    ],
)
def test_the_stable_audio_recipes(name, modality, seconds):
    from giq.registry import get_recipe

    recipe = get_recipe(name)
    assert recipe is not None and recipe.engine == "stable-audio"
    assert recipe.modalities == (modality,)
    assert recipe.params.max_seconds == seconds  # ty: ignore[unresolved-attribute]
    assert recipe.weights is not None and recipe.weights.source == f"hf:stabilityai/{name}"
