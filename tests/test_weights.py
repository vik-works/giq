# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Workers load the weights their recipe names.

The failure this guards against is a recipe file that adds a model to
the catalog which its worker then cannot load, because the worker looked its
weights up in a table of its own keyed by the built-in names.
"""

import logging

import pytest

from giq import weights
from giq.paths import models_dir
from giq.recipes import RecipeError, load_file


@pytest.fixture
def operator_dir(tmp_path, monkeypatch):
    """An operator recipes directory, with the catalog rebuilt around it."""
    from giq.registry import reload_registry

    directory = tmp_path / "recipes"
    directory.mkdir()
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(directory))
    reload_registry()
    yield directory
    monkeypatch.undo()
    reload_registry()


def _add(directory, filename: str, text: str) -> None:
    from giq.registry import reload_registry

    (directory / filename).write_text(text)
    reload_registry()


# --- new names load -----------------------------------------------------------


def test_an_ocr_instance_under_a_new_name_loads_its_own_weights(operator_dir, monkeypatch):
    from giq.registry import get_recipe
    from giq_ocr.adapter import OcrAdapter, OcrConfig

    # The built-in's override is scoped to the built-in's name.
    monkeypatch.setenv("GIQ_OCR_MODEL_DIR", "/elsewhere/unlimited")
    # Written for Unlimited-OCR's former interpreter: still loads, as transformers.
    _add(
        operator_dir,
        "ocr-ft.yaml",
        "name: unlimited-ocr-ft\nmodalities: [ocr]\nengine: transformers-4.57\n"
        "weights: {path: /srv/models/unlimited-ft}\nvram: {gb: 9.0}\n",
    )
    assert get_recipe("unlimited-ocr-ft").engine == "transformers"
    w = OcrAdapter(OcrConfig(model="unlimited-ocr-ft"))
    assert w.child_args() == ["--weights", "/srv/models/unlimited-ft"]
    assert OcrAdapter(OcrConfig()).child_args() == ["--weights", "/elsewhere/unlimited"]


def test_a_layout_ocr_instance_reads_its_layout_part(operator_dir):
    from giq_ocr.adapter import OcrAdapter, OcrConfig

    _add(
        operator_dir,
        "glm-ft.yaml",
        "name: glm-ocr-ft\nmodalities: [ocr]\nengine: transformers\n"
        "weights:\n  path: glm-ft\n  parts: {layout: layout-v4}\nvram: {gb: 4.0}\n",
    )
    w = OcrAdapter(OcrConfig(model="glm-ocr-ft"))
    assert w.child_args() == [
        "--weights",
        str(models_dir() / "glm-ft"),
        "--layout",
        str(models_dir() / "layout-v4"),
    ]


def test_a_layout_ocr_instance_without_its_layout_is_refused_at_start(operator_dir, tmp_path):
    """The checkpoint says it is GLM-OCR, which cannot run without its layout
    stage; known once the weights are read, which is at start."""
    import json

    from giq_ocr.adapter import OcrAdapter, OcrConfig

    ckpt = tmp_path / "glm-ft"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps({"architectures": ["GlmOcrForConditionalGeneration"]})
    )
    _add(
        operator_dir,
        "glm-ft.yaml",
        "name: glm-ocr-ft\nmodalities: [ocr]\nengine: transformers\n"
        f"weights: {{path: {ckpt}}}\nvram: {{gb: 4.0}}\n",
    )
    with pytest.raises(ValueError, match="weights.parts.layout"):
        OcrAdapter(OcrConfig(model="glm-ocr-ft"))._command()


def test_an_instance_without_weights_fails_at_construction(operator_dir):
    from giq_depth.adapter import DepthAdapter, DepthConfig

    _add(
        operator_dir,
        "d.yaml",
        "name: depth-nowhere\nmodalities: [depth]\nengine: transformers\nvram: {gb: 1.0}\n",
    )
    with pytest.raises(ValueError, match="no weights.path"):
        DepthAdapter(DepthConfig(model="depth-nowhere"))


def test_speech_models_load_the_repository_their_instance_names(operator_dir, monkeypatch):
    from giq_speech.audio import AudioAdapter, AudioConfig, EmbedAdapter, EmbedConfig
    from giq_speech.stt import model_ref
    from giq_speech.tts import TtsAdapter, TtsConfig

    assert model_ref("large-v3") == "Systran/faster-whisper-large-v3"
    _add(
        operator_dir,
        "stt.yaml",
        "name: whisper-de\nmodalities: [stt]\nengine: faster-whisper\n"
        "weights: {path: /srv/ct2/whisper-de}\nvram: {gb: 3.0}\n",
    )
    assert model_ref("whisper-de") == "/srv/ct2/whisper-de"
    # No recipe: faster-whisper's own size names still work.
    assert model_ref("distil-large-v3") == "distil-large-v3"

    monkeypatch.delenv("GIQ_AUDIO_WHISPER_MODEL", raising=False)
    monkeypatch.setenv("GIQ_AUDIO_DIAR_MODEL", "org/other-diarization")
    env = AudioAdapter(AudioConfig())._spawn_env()
    assert env["GIQ_AUDIO_WHISPER_MODEL"] == "Systran/faster-whisper-large-v3"
    assert env["GIQ_AUDIO_DIAR_MODEL"] == "org/other-diarization"  # the environment wins
    monkeypatch.delenv("GIQ_EMBED_MODEL", raising=False)
    assert (
        EmbedAdapter(EmbedConfig())._spawn_env()["GIQ_EMBED_MODEL"]
        == "speechbrain/spkrec-ecapa-voxceleb"
    )
    assert TtsAdapter(TtsConfig()).child_args()[-2:] == ["--repo", "hexgrad/Kokoro-82M"]


def test_the_storage_catalog_finds_repositories_in_the_hf_cache(tmp_path, monkeypatch):
    from giq.storage import installed

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    assert not installed("faster-whisper-large-v3")
    (tmp_path / "models--Systran--faster-whisper-large-v3").mkdir()
    assert installed("faster-whisper-large-v3")
    # The audio resident also needs the diarization pipeline.
    assert not installed("whisper-large-v3")
    (tmp_path / "models--pyannote--speaker-diarization-community-1").mkdir()
    assert installed("whisper-large-v3")
    assert not installed("kokoro")
    (tmp_path / "models--hexgrad--Kokoro-82M").mkdir()
    assert installed("kokoro")


# --- image models -------------------------------------------------------------


def test_image_models_take_their_files_from_the_instance():
    from giq_sdcpp.adapter import SdCppConfig
    from giq_sdcpp.files import image_files

    cfg = SdCppConfig(model="flux_klein", device="GPU-x", port=1)
    assert cfg.diffusion == str(models_dir() / "diffusion_models/flux-2-klein-4b.safetensors")
    assert cfg.vae == str(models_dir() / "vae/flux2-vae.safetensors")
    files = image_files("zimage")
    assert files.text_encoder == str(models_dir() / "text_encoders/qwen_3_4b.safetensors")
    assert files.lora is None


def test_config_image_models_are_no_longer_read(tmp_path, caplog):
    """An image recipe's files are its own (ADR-002); since ADR-003 config.yaml
    does not override them, and an operator who still has the block hears so."""
    from giq.config import GiqConfig

    path = tmp_path / "config.yaml"
    path.write_text(
        "image_models:\n  zimage: {diffusion: /old/z.safetensors, text_encoder: t, vae: v}\n"
    )
    with caplog.at_level(logging.WARNING, logger="giq.config"):
        cfg = GiqConfig.load(path)
    assert not hasattr(cfg, "image_models")
    assert any("image_models is no longer read" in r.message for r in caplog.records)


def test_an_image_instance_without_its_files_fails_at_construction(operator_dir):
    from giq_sdcpp.adapter import SdCppConfig

    _add(
        operator_dir,
        "img.yaml",
        "name: sketchy\nmodalities: [text2image]\nengine: sd.cpp\n"
        "weights: {parts: {diffusion: d.gguf}}\nvram: {gb: 8.0}\n",
    )
    with pytest.raises(ValueError, match="weights.parts.text_encoder, weights.parts.vae"):
        SdCppConfig(model="sketchy", device="GPU-x", port=1)


# --- resolution ---------------------------------------------------------------


def test_env_outranks_the_instance_and_roots_apply_to_relative_paths(monkeypatch):
    monkeypatch.setenv("GIQ_GLM_LAYOUT_DIR", "/pinned/layout")
    assert weights.path_of("glm-ocr", "layout") == "/pinned/layout"
    assert weights.path_of("glm-ocr") == str(models_dir() / "zai-GLM-OCR")
    monkeypatch.setenv("GIQ_DEPTH_MODELS_DIR", "/depth-root")
    assert weights.path_of("depth-anything-v2-small") == (
        "/depth-root/depth-anything-Depth-Anything-V2-Small-hf"
    )
    # An absolute path in the file is not moved by a root.
    assert weights.resolve_path("depth", "/abs/snap") == "/abs/snap"


def test_unknown_models_and_parts_resolve_to_nothing():
    assert weights.path_of("no-such-ocr") is None
    assert weights.path_of("unlimited-ocr", "layout") is None
    with pytest.raises(ValueError, match="no recipe file"):
        weights.require_path("no-such-ocr")


def test_hub_sources_are_read_as_repositories():
    assert weights.hub_repo("hf:org/repo") == "org/repo"
    assert weights.hub_repo("https://example.org/x") is None
    assert weights.hub_repo(None) is None


# --- the schema ---------------------------------------------------------------

DEPTH = "name: d\nmodalities: [{worker}]\nengine: {engine}\nweights:\n{weights}vram: {{gb: 1.0}}\n"


def _load(tmp_path, worker, engine, weights_yaml):
    path = tmp_path / "i.yaml"
    path.write_text(DEPTH.format(worker=worker, engine=engine, weights=weights_yaml))
    return load_file(path)


def test_a_part_may_be_a_bare_path_or_carry_provenance(tmp_path):
    recipe = _load(
        tmp_path,
        "text2image",
        "sd.cpp",
        "  parts:\n    diffusion: a.gguf\n"
        "    vae: {path: vae.safetensors, source: 'hf:org/klein', licence: apache-2.0}\n",
    )
    assert recipe.weights.parts["diffusion"].path == "a.gguf"
    assert recipe.weights.parts["vae"].licence == "apache-2.0"


@pytest.mark.parametrize(
    ("worker", "engine", "parts", "complaint"),
    [
        ("depth", "transformers", "{layout: x}", "reads no parts"),
        ("ocr", "transformers", "{lay0ut: x}", "reads only layout"),
        ("text2image", "sd.cpp", "{vae: {licence: mit}}", "a path or a source"),
        ("text2image", "sd.cpp", "{Vae: x}", "pattern"),
    ],
)
def test_parts_a_worker_does_not_read_are_refused(tmp_path, worker, engine, parts, complaint):
    with pytest.raises(RecipeError, match=complaint):
        _load(tmp_path, worker, engine, f"  parts: {parts}\n")


# --- one engine vocabulary ----------------------------------------------------


def test_the_old_sdcpp_spelling_is_read_as_sd_cpp_with_a_warning(tmp_path, caplog, monkeypatch):
    from giq import engines

    monkeypatch.setattr(engines, "_warned_aliases", set())
    recipe = _load(
        tmp_path, "text2image", "sdcpp", "  parts: {diffusion: d, text_encoder: t, vae: v}\n"
    )
    assert recipe.engine == "sd.cpp"
    assert any("'sdcpp' is deprecated, write 'sd.cpp'" in r.message for r in caplog.records)


def test_config_yaml_is_parsed_into_canonical_engine_names(tmp_path, monkeypatch):
    from giq import engines
    from giq.config import GiqConfig

    monkeypatch.setattr(engines, "_warned_aliases", set())
    path = tmp_path / "config.yaml"
    path.write_text("engines:\n  sdcpp: /opt/sd-server\n")
    cfg = GiqConfig.load(path)
    monkeypatch.setattr("giq.config.get_config", lambda: cfg)
    assert engines.reload_engines()["sd.cpp"].binary == "/opt/sd-server"
    monkeypatch.undo()
    engines.reload_engines()
