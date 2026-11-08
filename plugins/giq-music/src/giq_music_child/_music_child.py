# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-music child process: ACE-Step 1.5, DiT plus the 5Hz planner LM.

Invoked by ``MusicAdapter`` via ``<acestep python> -u -m
giq_music_child._music_child --model <recipe> --weights <checkpoint root>
--dit <folder> [--lm <folder>] --steps <n>``. Loads the DiT, its VAE and
text encoder, then the planner LM, once.

ACE-Step reads one checkpoint root with fixed folder names, downloads any
it finds missing, and copies its own model code into the DiT folder on
load. So the child loads from a scratch root instead: links to the
recipe's folders, the DiT folder's code as copies (ACE-Step's sync then
writes only there), the layout checked before ACE-Step sees it and the hub
offline, so a missing part fails here rather than being fetched.

Per task: the planner (``plan``, default on when the recipe loads one)
fills in what the request leaves open, the DiT renders, and the track comes
back encoded, with the tempo, key, time signature and duration it got.

Wire protocol: see giq_child (``run_ipc_child_loop``). This module is a
top-level package importing no giq: the acestep interpreter carries torch
and ACE-Step, never giq's venv. The parsing below is pure, so giq's suite
tests it without ACE-Step.
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("ACESTEP_DISABLE_TQDM", "1")
# ACE-Step joins each render's thread with this timeout (read at import);
# just under the adapter's batch timeout, so a stuck render reports itself.
os.environ.setdefault("ACESTEP_GENERATION_TIMEOUT", "840")

from giq_child import (  # noqa: E402
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)

# ACE-Step's documented range.
MIN_SECONDS, MAX_SECONDS = 10.0, 600.0
# Without the planner nothing picks a length, so a task that names none
# gets this.
UNPLANNED_SECONDS = 30.0
MIN_BPM, MAX_BPM = 30, 300
# ACE-Step's time signatures are the beats of a bar: 2, 3, 4 or 6 (6/8).
TIME_SIGNATURES = {"2": "2/4", "3": "3/4", "4": "4/4", "6": "6/8"}
FORMATS = {"flac": ("FLAC", "PCM_16"), "wav": ("WAV", "PCM_16"), "mp3": ("MP3", None)}
# The folders ACE-Step checks for before it loads anything, whichever it loads.
REQUIRED = ("acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B")
WEIGHT_FILES = ("model.safetensors", "diffusion_pytorch_model.safetensors")


def task_inputs(task: dict, planner: bool) -> dict:
    """A task's request, checked, as the values ACE-Step takes.

    Raises ValueError naming the field that is wrong. ``plan`` is only on
    when the recipe loaded a planner; without one the task renders as given.
    """
    caption = task.get("prompt") or task.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("task needs a 'prompt' describing the music")
    lyrics = task.get("lyrics") or ""
    if not isinstance(lyrics, str):
        raise ValueError("'lyrics' must be a string")
    plan = bool(task.get("plan", True)) and planner

    duration = task.get("duration")
    if duration is None:
        seconds = -1.0 if plan else UNPLANNED_SECONDS  # -1: the planner picks
    else:
        seconds = _number(duration, "duration")
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise ValueError(f"duration must be {MIN_SECONDS:g}-{MAX_SECONDS:g} s, got {seconds:g}")

    bpm = task.get("bpm")
    if bpm is not None:
        bpm = int(_number(bpm, "bpm"))
        if not MIN_BPM <= bpm <= MAX_BPM:
            raise ValueError(f"bpm must be {MIN_BPM}-{MAX_BPM}, got {bpm}")

    seed = task.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("'seed' must be a non-negative integer")

    fmt = str(task.get("format") or "flac").lower()
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, got {fmt!r}")

    return {
        "caption": caption.strip(),
        "lyrics": lyrics,
        # No lyrics means an instrumental unless the task says otherwise.
        "instrumental": bool(task.get("instrumental", not lyrics.strip())),
        "language": str(task.get("language") or "unknown"),
        "duration": seconds,
        "bpm": bpm,
        "key_scale": str(task.get("key_scale") or ""),
        "time_signature": _beats(task.get("time_signature")),
        "seed": seed,
        "plan": plan,
        "format": fmt,
    }


def _number(value, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"'{field}' must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"'{field}' must be a finite number")
    return number


def _beats(value) -> str:
    """A time signature as ACE-Step's beats per bar: "3/4" or "3" → "3"."""
    if value is None or value == "":
        return ""
    beats = str(value).split("/")[0].strip()
    if beats not in TIME_SIGNATURES or (
        "/" in str(value) and TIME_SIGNATURES[beats] != str(value).replace(" ", "")
    ):
        raise ValueError(f"time_signature must be one of {', '.join(TIME_SIGNATURES.values())}")
    return beats


def track_meta(used: dict, inputs: dict) -> dict:
    """What the track got: the request's values, else what the planner chose.

    ``used`` is the params ACE-Step reports for the track, where the
    planner's picks are its ``cot_*`` fields."""

    def pick(given, planned):
        return given if given not in (None, "", -1, -1.0) else (planned or None)

    bpm = pick(inputs["bpm"], used.get("cot_bpm"))
    beats = pick(inputs["time_signature"], used.get("cot_timesignature"))
    seed = used.get("seed")
    try:
        seed = int(str(seed).split(",")[0]) if seed not in (None, "") else inputs["seed"]
    except ValueError:
        seed = inputs["seed"]
    return {
        "bpm": int(bpm) if bpm not in (None, "") else None,
        "key_scale": pick(inputs["key_scale"], used.get("cot_keyscale")),
        "time_signature": TIME_SIGNATURES.get(str(beats).split("/")[0]) if beats else None,
        "caption": used.get("cot_caption") or inputs["caption"],
        "lyrics": used.get("cot_lyrics") or inputs["lyrics"] or None,
        "seed": seed,
    }


def _scratch_root(weights: str, dit: str) -> str:
    """A checkpoint root ACE-Step may write into, over the recipe's folders.

    Every folder of ``weights`` is linked, except the DiT's, which is a real
    folder of links with its model code copied: ACE-Step syncs its own code
    over that code on load, and a copy over a link would write through it.
    """
    import atexit
    import shutil
    import tempfile

    for name in (*REQUIRED, dit):
        folder = os.path.join(weights, name)
        if not any(os.path.isfile(os.path.join(folder, f)) for f in WEIGHT_FILES):
            raise RuntimeError(f"{folder} holds no weights; ACE-Step would try to download it")

    scratch = tempfile.mkdtemp(prefix="giq-acestep-")
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    root = os.path.join(scratch, "checkpoints")
    os.makedirs(root)
    for name in os.listdir(weights):
        src = os.path.join(weights, name)
        if name != dit:
            os.symlink(src, os.path.join(root, name))
            continue
        os.makedirs(os.path.join(root, name))
        for f in os.listdir(src):
            copy = f.endswith(".py") or f.endswith(".json")
            dst = os.path.join(root, name, f)
            shutil.copy2(os.path.join(src, f), dst) if copy else os.symlink(
                os.path.join(src, f), dst
            )
    # ACE-Step keeps a progress cache under its project root, and transformers
    # its copies of remote model code under HF_MODULES_CACHE: both in scratch.
    os.environ["ACESTEP_PROJECT_ROOT"] = scratch
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = root
    os.environ["HF_MODULES_CACHE"] = os.path.join(scratch, "modules")
    os.chdir(scratch)
    return root


def _load(weights: str, dit: str, lm: str | None):
    root = _scratch_root(weights, dit)

    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    if not torch.cuda.is_available():
        raise RuntimeError("ACE-Step needs a CUDA device")
    handler = AceStepHandler()
    try:
        import flash_attn  # noqa: F401

        flash = True
    except ImportError:
        flash = False
    status, ok = handler.initialize_service(
        project_root=os.path.dirname(root),
        config_path=dit,
        device="cuda",
        use_flash_attention=flash,
    )
    if not ok:
        raise RuntimeError(f"ACE-Step DiT did not load: {status}")
    planner = None
    if lm:
        # After the DiT: the LM sizes its share of the card from what is
        # already allocated.
        planner = LLMHandler()
        status, ok = planner.initialize(
            checkpoint_dir=root, lm_model_path=lm, backend="vllm", device="cuda"
        )
        if not ok or not getattr(planner, "llm_initialized", False):
            raise RuntimeError(f"ACE-Step planner LM did not load: {status}")
    return handler, planner


def _encode(tensor, sample_rate: int, fmt: str) -> tuple[str, float]:
    import base64
    import io

    import soundfile

    audio = tensor.detach().float().cpu().numpy().T  # [samples, channels]
    container, subtype = FORMATS[fmt]
    buf = io.BytesIO()
    soundfile.write(buf, audio, sample_rate, format=container, subtype=subtype)
    return base64.b64encode(buf.getvalue()).decode(), audio.shape[0] / sample_rate


def _run(models, steps: int, task: dict) -> dict:
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    handler, planner = models
    inputs = task_inputs(task, planner is not None)
    plan = inputs["plan"]
    params = GenerationParams(
        task_type="text2music",
        caption=inputs["caption"],
        lyrics=inputs["lyrics"],
        instrumental=inputs["instrumental"],
        vocal_language=inputs["language"],
        bpm=inputs["bpm"],
        keyscale=inputs["key_scale"],
        timesignature=inputs["time_signature"],
        duration=inputs["duration"],
        inference_steps=steps,
        seed=inputs["seed"] if inputs["seed"] is not None else -1,
        thinking=plan,
        use_cot_metas=plan,
        use_cot_caption=plan,
        use_cot_language=plan,
    )
    # ACE-Step draws fresh seeds unless told not to, whatever the params say.
    seeded = inputs["seed"] is not None
    config = GenerationConfig(
        batch_size=1, use_random_seed=not seeded, seeds=[inputs["seed"]] if seeded else None
    )
    result = generate_music(handler, planner if plan else None, params, config, save_dir=None)
    if not result.success or not result.audios:
        raise RuntimeError(result.error or result.status_message or "ACE-Step returned no audio")
    track = result.audios[0]
    audio_b64, seconds = _encode(track["tensor"], int(track["sample_rate"]), inputs["format"])
    return {
        "id": task.get("id", "unknown"),
        "audio_b64": audio_b64,
        "format": inputs["format"],
        "sample_rate": int(track["sample_rate"]),
        "duration": round(seconds, 2),
        **track_meta(track.get("params") or {}, inputs),
    }


def main() -> None:
    import argparse

    reserve_ipc_stdout()
    parser = argparse.ArgumentParser()
    # --model names the recipe for log lines; the rest is what loads.
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--lm", default=None)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    try:
        models = _load(args.weights, args.dit, args.lm)
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_run(models, args.steps, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
