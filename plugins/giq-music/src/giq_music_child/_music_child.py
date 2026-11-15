# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-music child process: ACE-Step 1.5, DiT plus the 5Hz planner LM.

Invoked by ``MusicAdapter`` via ``<acestep python> -u -m
giq_music_child._music_child --model <recipe> --weights <checkpoint root>
--dit <name> [--dit-path <folder>] [--lm <folder>] --steps <n> --guidance <x>
--tasks <cover,lego,...>``. Loads the DiT, its VAE and text encoder, then the
planner LM, once. ``--dit-path`` is a DiT that ships as a repository of its
own (the XL ones): it joins the root under its name.

ACE-Step reads one checkpoint root with fixed folder names, downloads any
it finds missing, and copies its own model code into the DiT folder on
load. So the child loads from a scratch root instead: links to the
recipe's folders, the DiT folder's code as copies (ACE-Step's sync then
writes only there), the layout checked before ACE-Step sees it and the hub
offline, so a missing part fails here rather than being fetched.

Per task: the planner (``plan``, default on when the recipe loads one)
fills in what the request leaves open, the DiT renders, and the track comes
back encoded, with the tempo, key, time signature and duration it got.

A task may bring audio of its own. ``reference_audio_b64`` lends the track
its sound — timbre, mix, playing style — averaged over the clip, so not its
melody or rhythm. ``source_audio_b64`` is what the track is built from, by
``mode``: ``cover`` keeps its structure, melody and groove and restyles it
(``cover_strength``, 0-1, how closely); ``lego`` keeps it and adds one
``track`` (a guitar over your drums); ``complete`` adds the ``tracks`` it
lacks. Which modes a DiT does is the recipe's (``--tasks``): lego and
complete are the base models'. With source audio the planner sits out —
ACE-Step skips it, since its codes would replace the source's — and the
source's length is the track's.

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
from giq_music_child._audio import audio_bytes, encode, format_of, number, seed_of  # noqa: E402

# ACE-Step's documented range.
MIN_SECONDS, MAX_SECONDS = 10.0, 600.0
# Without the planner nothing picks a length, so a task that names none
# gets this.
UNPLANNED_SECONDS = 30.0
MIN_BPM, MAX_BPM = 30, 300
# ACE-Step's time signatures are the beats of a bar: 2, 3, 4 or 6 (6/8).
TIME_SIGNATURES = {"2": "2/4", "3": "3/4", "4": "4/4", "6": "6/8"}
# The folders ACE-Step checks for before it loads anything, whichever it loads.
REQUIRED = ("acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B")
# A folder holds weights when it has one of these; the XL DiTs are sharded.
WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors",
)
# What a task's source audio is for; see the module docstring.
SOURCE_MODES = ("cover", "lego", "complete")
# ACE-Step's stems, the tracks lego adds and complete fills in.
TRACKS = (
    "woodwinds", "brass", "fx", "synth", "strings", "percussion",
    "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals",
)  # fmt: skip
# How closely a cover follows its source when the task does not say: ACE-Step's
# guide puts a change of genre at 0.3-0.5.
COVER_STRENGTH = 0.5


def task_inputs(task: dict, planner: bool, modes: frozenset[str] = frozenset()) -> dict:
    """A task's request, checked, as the values ACE-Step takes.

    Raises ValueError naming the field that is wrong. ``plan`` is only on
    when the recipe loaded a planner; without one the task renders as given.
    ``modes`` are the source-audio modes the recipe's DiT does. The audio
    stays base64 here; the caller writes it to files.
    """
    caption = task.get("prompt") or task.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("task needs a 'prompt' describing the music")
    lyrics = task.get("lyrics") or ""
    if not isinstance(lyrics, str):
        raise ValueError("'lyrics' must be a string")
    audio = _audio_inputs(task, modes)
    plan = bool(task.get("plan", True)) and planner and audio["mode"] == "text2music"

    duration = task.get("duration")
    if duration is None:
        # -1: the planner picks, or the source audio's length is the track's.
        seconds = -1.0 if plan or audio["source"] else UNPLANNED_SECONDS
    else:
        seconds = number(duration, "duration")
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise ValueError(f"duration must be {MIN_SECONDS:g}-{MAX_SECONDS:g} s, got {seconds:g}")

    bpm = task.get("bpm")
    if bpm is not None:
        bpm = int(number(bpm, "bpm"))
        if not MIN_BPM <= bpm <= MAX_BPM:
            raise ValueError(f"bpm must be {MIN_BPM}-{MAX_BPM}, got {bpm}")

    seed = seed_of(task)
    fmt = format_of(task)

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
        **audio,
    }


def _audio_inputs(task: dict, modes: frozenset[str]) -> dict:
    """The audio a task brings and what it is for, checked; see the docstring."""
    reference = task.get("reference_audio_b64") or None
    source = task.get("source_audio_b64") or None
    for field, value in (("reference_audio_b64", reference), ("source_audio_b64", source)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"'{field}' must be base64 audio")
    mode = task.get("mode")
    if source is None:
        if mode not in (None, "", "text2music"):
            raise ValueError(f"mode {mode!r} needs a 'source_audio_b64' to work on")
        mode = "text2music"
    else:
        mode = mode or "cover"
        if mode not in SOURCE_MODES:
            raise ValueError(f"mode must be one of {', '.join(SOURCE_MODES)}, got {mode!r}")
        if mode not in modes:
            does = ", ".join(m for m in SOURCE_MODES if m in modes) or "none"
            raise ValueError(f"this recipe's model does no {mode} (it does: {does})")

    strength = task.get("cover_strength")
    if strength is not None:
        if mode != "cover":
            raise ValueError("'cover_strength' is for mode cover")
        strength = number(strength, "cover_strength")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"cover_strength must be 0-1, got {strength:g}")

    track = task.get("track")
    if mode == "lego":
        if track not in TRACKS:
            raise ValueError(f"mode lego needs a 'track', one of {', '.join(TRACKS)}")
    elif track is not None:
        raise ValueError("'track' is for mode lego")
    tracks = task.get("tracks")
    if tracks is not None:
        if mode != "complete":
            raise ValueError("'tracks' is for mode complete")
        if not isinstance(tracks, list) or any(t not in TRACKS for t in tracks):
            raise ValueError(f"'tracks' must be a list of {', '.join(TRACKS)}")
    return {
        "mode": mode,
        "reference": reference,
        "source": source,
        "cover_strength": COVER_STRENGTH if mode == "cover" and strength is None else strength,
        "track": track,
        "tracks": list(tracks or ()),
    }


def instruction(inputs: dict, templates: dict[str, str]) -> str:
    """The DiT's task instruction from ACE-Step's ``TASK_INSTRUCTIONS``.

    generate_music takes the instruction as given, and its default is
    text2music's, so a task of another kind names its own."""
    mode = inputs["mode"]
    if mode == "lego":
        return templates["lego"].format(TRACK_NAME=inputs["track"].upper())
    if mode == "complete":
        if not inputs["tracks"]:
            return templates["complete_default"]
        return templates["complete"].format(
            TRACK_CLASSES=" | ".join(t.upper() for t in inputs["tracks"])
        )
    return templates[mode]


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


def _scratch_root(weights: str, dit: str, dit_path: str | None = None) -> str:
    """A checkpoint root ACE-Step may write into, over the recipe's folders.

    Every folder of ``weights`` is linked, except the DiT's, which is a real
    folder of links with its model code copied: ACE-Step syncs its own code
    over that code on load, and a copy over a link would write through it.
    ``dit_path`` is a DiT from elsewhere, taking the place of any ``dit``
    folder in ``weights``.
    """
    import atexit
    import shutil
    import tempfile

    folders = {name: os.path.join(weights, name) for name in (*REQUIRED, dit)}
    if dit_path:
        folders[dit] = dit_path
    for folder in folders.values():
        if not any(os.path.isfile(os.path.join(folder, f)) for f in WEIGHT_FILES):
            raise RuntimeError(f"{folder} holds no weights; ACE-Step would try to download it")

    scratch = tempfile.mkdtemp(prefix="giq-acestep-")
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    root = os.path.join(scratch, "checkpoints")
    os.makedirs(root)
    linked = {name: os.path.join(weights, name) for name in os.listdir(weights)}
    for name, src in {**linked, dit: folders[dit]}.items():
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


def _load(weights: str, dit: str, dit_path: str | None, lm: str | None):
    root = _scratch_root(weights, dit, dit_path)

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


def _audio_files(inputs: dict, folder: str) -> dict:
    """The task's audio as files in ``folder``, by GenerationParams field."""
    files = {}
    for key, field in (("reference", "reference_audio"), ("source", "src_audio")):
        if inputs[key] is None:
            continue
        data, suffix = audio_bytes(inputs[key], f"{key}_audio_b64")
        path = os.path.join(folder, key + suffix)
        with open(path, "wb") as f:
            f.write(data)
        files[field] = path
    return files


def _run(models, args, task: dict) -> dict:
    import tempfile

    handler, planner = models
    inputs = task_inputs(task, planner is not None, frozenset(args.tasks))
    # The task's audio lives as long as its render, in memory-backed scratch
    # where there is one.
    with tempfile.TemporaryDirectory(prefix="giq-acestep-task-", dir=_shm()) as folder:
        return _render(handler, planner, args, task, inputs, _audio_files(inputs, folder))


def _shm() -> str | None:
    return "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None


def _render(handler, planner, args, task: dict, inputs: dict, files: dict) -> dict:
    from acestep.constants import TASK_INSTRUCTIONS
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    plan = inputs["plan"]
    cover = (
        {}
        if inputs["cover_strength"] is None
        else {"audio_cover_strength": inputs["cover_strength"]}
    )
    params = GenerationParams(
        task_type=inputs["mode"],
        instruction=instruction(inputs, TASK_INSTRUCTIONS),
        caption=inputs["caption"],
        lyrics=inputs["lyrics"],
        instrumental=inputs["instrumental"],
        vocal_language=inputs["language"],
        bpm=inputs["bpm"],
        keyscale=inputs["key_scale"],
        timesignature=inputs["time_signature"],
        duration=inputs["duration"],
        inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=inputs["seed"] if inputs["seed"] is not None else -1,
        thinking=plan,
        use_cot_metas=plan,
        use_cot_caption=plan,
        use_cot_language=plan,
        **files,
        **cover,
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
    audio_b64, seconds = encode(track["tensor"], int(track["sample_rate"]), inputs["format"])
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
    parser.add_argument("--dit-path", default=None)
    parser.add_argument("--lm", default=None)
    parser.add_argument("--steps", type=int, default=8)
    # Classifier-free guidance; the turbo DiTs ignore it (ACE-Step sets 1.0).
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--tasks", type=lambda v: [m for m in v.split(",") if m], default=["cover"])
    args = parser.parse_args()
    try:
        models = _load(args.weights, args.dit, args.dit_path, args.lm)
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_run(models, args, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
