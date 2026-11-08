# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-audio child process: Stable Audio 3 (a DiT over the SAME
autoencoder, conditioned through T5Gemma).

Invoked by ``StableAudioAdapter`` via ``<stable-audio python> -u -m
giq_music_child._sa3_child --model <recipe> --weights <repo folder>
--steps <n> --cfg-scale <x> --max-seconds <s> --default-seconds <s>``.
Loads the model once.

Stable Audio 3's own loader fetches each model from the hub by name, and its
config names the text encoder by repo and subfolder. The child loads from
the recipe's folder instead: the config read from there, the text encoder
pointed at the folder's copy, the hub offline, so a missing part fails here
rather than being fetched.

Per task: a prompt, an optional negative prompt, a length, a seed; the track
comes back encoded at the model's 44.1 kHz stereo. The seed is always drawn
here when the task names none, so the result can report it.

Wire protocol: see giq_child (``run_ipc_child_loop``). This module is a
top-level package importing no giq; the parsing below is pure, so giq's
suite tests it without Stable Audio 3.
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from giq_child import (  # noqa: E402
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)
from giq_music_child._audio import encode, format_of, number, seed_of  # noqa: E402

MIN_SECONDS = 1.0
# The range the child draws an unset seed from (torch takes any of them).
SEED_RANGE = 2**31
TEXT_ENCODER = "t5gemma-b-b-ul2"
FILES = ("model.safetensors", "model_config.json", os.path.join(TEXT_ENCODER, "model.safetensors"))


def task_inputs(task: dict, max_seconds: float, default_seconds: float) -> dict:
    """A task's request, checked. Raises ValueError naming the wrong field."""
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("task needs a 'prompt' describing the audio")
    negative = task.get("negative_prompt")
    if negative is not None and not isinstance(negative, str):
        raise ValueError("'negative_prompt' must be a string")
    duration = task.get("duration")
    seconds = default_seconds if duration is None else number(duration, "duration")
    if not MIN_SECONDS <= seconds <= max_seconds:
        raise ValueError(f"duration must be {MIN_SECONDS:g}-{max_seconds:g} s, got {seconds:g}")
    return {
        "prompt": prompt.strip(),
        "negative_prompt": negative.strip() if negative and negative.strip() else None,
        "duration": seconds,
        "seed": seed_of(task),
        "format": format_of(task),
    }


def local_config(weights: str) -> dict:
    """The folder's model config, its text encoder pointed at the folder.

    The config names the encoder as a hub repo and subfolder; the conditioner
    loads ``model_path`` first when it is set."""
    import json

    for name in FILES:
        if not os.path.isfile(os.path.join(weights, name)):
            raise RuntimeError(f"{weights} lacks {name}; Stable Audio 3 would try to download it")
    with open(os.path.join(weights, "model_config.json")) as f:
        config = json.load(f)
    encoder = os.path.join(weights, TEXT_ENCODER)
    found = 0

    def walk(node):
        nonlocal found
        if isinstance(node, dict):
            if node.get("type") == "t5gemma" and isinstance(node.get("config"), dict):
                cfg = node["config"]
                cfg["model_path"] = encoder
                cfg.pop("repo_id", None)
                cfg.pop("subfolder", None)
                found += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(config)
    if not found:
        raise RuntimeError(f"{weights}/model_config.json names no T5Gemma text encoder")
    return config


def _load(weights: str):
    import torch
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.model import StableAudioModel

    if not torch.cuda.is_available():
        raise RuntimeError("Stable Audio 3 needs a CUDA device here")
    config = local_config(weights)
    model = load_diffusion_cond(
        config, os.path.join(weights, "model.safetensors"), device="cuda", model_half=True
    )
    # What StableAudioModel.from_pretrained sets after loading: no LoRAs.
    model.use_lora = False
    model.lora_names = []
    return StableAudioModel(model, config, "cuda", True)


def _run(sa3, args, task: dict) -> dict:
    import secrets

    model_seconds = sa3.model_config["sample_size"] / sa3.model.sample_rate
    inputs = task_inputs(task, min(args.max_seconds, model_seconds), args.default_seconds)
    seed = inputs["seed"] if inputs["seed"] is not None else secrets.randbelow(SEED_RANGE)
    audio = sa3.generate(
        prompt=inputs["prompt"],
        negative_prompt=inputs["negative_prompt"],
        duration=inputs["duration"],
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=seed,
        sample_size=sa3.model_config["sample_size"],
    )
    sample_rate = int(sa3.model.sample_rate)
    audio_b64, seconds = encode(audio[0], sample_rate, inputs["format"])
    return {
        "id": task.get("id", "unknown"),
        "audio_b64": audio_b64,
        "format": inputs["format"],
        "sample_rate": sample_rate,
        "duration": round(seconds, 2),
        "seed": seed,
        "caption": inputs["prompt"],
    }


def main() -> None:
    import argparse

    reserve_ipc_stdout()
    parser = argparse.ArgumentParser()
    # --model names the recipe for log lines; the rest is what loads.
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--default-seconds", type=float, default=30.0)
    args = parser.parse_args()
    try:
        sa3 = _load(args.weights)
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_run(sa3, args, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
