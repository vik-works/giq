# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-sound-effects child process: MOSS-SoundEffect v2.

Invoked by ``SfxAdapter`` via ``<mosssfx python> -u -m
giq_sfx_child._sfx_child --model <recipe> --weights <folder> --steps <n>
--guidance <x> --shift <x> [--compile]``. Loads the pipeline once — the
1.3B DiT, the DAC VAE and the Qwen3-1.7B text encoder — from the recipe's
weights, MOSS's diffusers-style layout, checked before MOSS reads it, with
the hub offline.

Per task: a prompt (English or Chinese) and a length in, one 48 kHz mono
sound out, encoded. MOSS always denoises its full 30 s latent and crops it
to the length asked for, so a task costs the same whatever its length; the
length also goes into the prompt, as the model was trained with it. Tasks
render one at a time: a batched call shares one seed and one length across
its prompts, and with the DiT compiled every new batch size would compile
again.

``--compile`` keeps upstream's torch.compile with CUDA graphs over the DiT
and compiles it at load with a short warm-up render; without it TorchDynamo
is off and the DiT runs eagerly (see giq_sfx.params for why that is the
default).

Wire protocol: see giq_child (``run_ipc_child_loop``). This module is a
top-level package importing no giq: the mosssfx interpreter carries torch
and MOSS-SoundEffect, never giq's venv. The parsing below is pure, so giq's
suite tests it without either.
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
from giq_sfx_child._audio import encode, format_of, number, seed_of  # noqa: E402

# Below this a sound is a click; MOSS rounds lengths to a tenth of a second.
MIN_SECONDS = 0.5
# The latent MOSS denoises, and so the longest sound (model_index.json's
# max_inference_seconds); the pipeline reports its own once loaded.
MAX_SECONDS = 30.0
DEFAULT_SECONDS = 10.0
MAX_STEPS = 200
MAX_GUIDANCE = 20.0
# A seed the browser's numbers carry exactly, for a task that names none.
SEED_RANGE = 2**31
# What MOSS reads, by its pipeline's from_pretrained: each path, or any one
# of a tuple's.
REQUIRED = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    ("text_encoder/model.safetensors", "text_encoder/model.safetensors.index.json"),
    "tokenizer/tokenizer_config.json",
    ("vae/vae_128d_48k.pth", "vae/diffusion_pytorch_model.safetensors"),
)
# Steps of the warm-up render that compiles the DiT: the graph does not
# depend on the step count, so two compile what a hundred would.
WARMUP_STEPS = 2


def task_inputs(task: dict, defaults: dict, max_seconds: float = MAX_SECONDS) -> dict:
    """A task's request, checked, as the values MOSS takes.

    Raises ValueError naming the field that is wrong. ``defaults`` are the
    recipe's ``steps`` and ``guidance``, which a task may override. The seed
    stays None when the task names none; the caller draws one.
    """
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("task needs a 'prompt' describing the sound")
    negative = task.get("negative_prompt") or ""
    if not isinstance(negative, str):
        raise ValueError("'negative_prompt' must be a string")

    duration = task.get("duration")
    seconds = DEFAULT_SECONDS if duration is None else number(duration, "duration")
    # MOSS rounds to a tenth itself; doing it here keeps the check honest.
    seconds = round(seconds, 1)
    if not MIN_SECONDS <= seconds <= max_seconds:
        raise ValueError(f"duration must be {MIN_SECONDS:g}-{max_seconds:g} s, got {seconds:g}")

    steps = task.get("steps", defaults["steps"])
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be an integer 1-{MAX_STEPS}")
    guidance = number(task.get("guidance", defaults["guidance"]), "guidance")
    if not 1.0 <= guidance <= MAX_GUIDANCE:
        raise ValueError(f"guidance must be 1-{MAX_GUIDANCE:g}, got {guidance:g}")

    return {
        "prompt": prompt.strip(),
        "negative_prompt": negative.strip(),
        "seconds": seconds,
        "steps": steps,
        "guidance": guidance,
        "seed": seed_of(task),
        "format": format_of(task),
    }


def missing_files(weights: str) -> list[str]:
    """What of MOSS's layout ``weights`` lacks, as relative paths."""
    missing = []
    for want in REQUIRED:
        options = want if isinstance(want, tuple) else (want,)
        if not any(os.path.isfile(os.path.join(weights, p)) for p in options):
            missing.append(" or ".join(options))
    return missing


def compile_env(enabled: bool) -> dict[str, str]:
    """The variables that switch the DiT's compile, set before MOSS is imported.

    MOSS decorates its DiT call with torch.compile at import, and TorchDynamo
    reads TORCHDYNAMO_DISABLE right then. A compiled run keeps its inductor
    cache beside the other JIT caches giq points at (XDG_CACHE_HOME), so a
    reload finds the kernels a previous one built; an operator's own
    TORCHINDUCTOR_CACHE_DIR wins."""
    if not enabled:
        return {"TORCHDYNAMO_DISABLE": "1"}
    env = {}
    cache = os.environ.get("XDG_CACHE_HOME")
    if cache and not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache, "torchinductor")
    return env


def _quiet(iterable):
    """MOSS's progress bar, silenced: the parent logs stderr line by line."""
    return iterable


def _load(weights: str, compiled: bool, shift: float):
    if missing := missing_files(weights):
        raise RuntimeError(f"{weights} lacks {', '.join(missing)}")
    os.environ.update(compile_env(compiled))

    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("MOSS-SoundEffect needs a CUDA device")
    pipe = MossSoundEffectPipeline.from_pretrained(
        weights, torch_dtype=torch.bfloat16, device="cuda"
    )
    if compiled:
        # The compile happens at the first DiT call; here, not in a task.
        pipe(
            prompt="warm-up",
            seconds=1.0,
            num_inference_steps=WARMUP_STEPS,
            sigma_shift=shift,
            progress_bar_cmd=_quiet,
        )
    return pipe


def render(pipe, args, inputs: dict, draw_seed=None) -> dict:
    """One sound for a checked task: MOSS's call, and the result it makes."""
    seed = inputs["seed"]
    if seed is None:
        import secrets

        seed = (draw_seed or secrets.randbelow)(SEED_RANGE)
    audio = pipe(
        prompt=inputs["prompt"],
        negative_prompt=inputs["negative_prompt"],
        seconds=inputs["seconds"],
        num_inference_steps=inputs["steps"],
        cfg_scale=inputs["guidance"],
        sigma_shift=args.shift,
        seed=seed,
        progress_bar_cmd=_quiet,
    )
    rate = int(pipe.sample_rate)
    audio_b64, seconds = encode(audio[0], rate, inputs["format"])
    return {
        "audio_b64": audio_b64,
        "format": inputs["format"],
        "sample_rate": rate,
        "duration": round(seconds, 2),
        "seed": seed,
        "steps": inputs["steps"],
        "guidance": inputs["guidance"],
    }


def run_tasks(pipe, args, tasks: list[dict]) -> list[dict]:
    """Every task of a batch, one after another; a failure is its own result."""
    defaults = {"steps": args.steps, "guidance": args.guidance}
    max_seconds = float(getattr(pipe, "max_inference_seconds", MAX_SECONDS))
    results = []
    for task in tasks:
        rid = task.get("id", "unknown")
        try:
            inputs = task_inputs(task, defaults, max_seconds)
            results.append({"id": rid, **render(pipe, args, inputs)})
        except Exception as e:  # noqa: BLE001 — per-task error envelope
            results.append({"id": rid, "error": str(e)})
    return results


def main() -> None:
    import argparse

    reserve_ipc_stdout()
    parser = argparse.ArgumentParser()
    # --model names the recipe for log lines; the rest is what loads.
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    try:
        pipe = _load(args.weights, args.compile, args.shift)
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    run_ipc_child_loop(lambda tasks, params: run_tasks(pipe, args, tasks))


if __name__ == "__main__":
    main()
