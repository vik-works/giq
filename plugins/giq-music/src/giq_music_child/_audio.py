# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""What every giq-music child shares: checking a task's numbers and formats,
and encoding a track. Stdlib at import; numpy and soundfile only in
``encode``, which runs on the engine's interpreter."""

import math

# Container and sample format per requested format; mp3 picks its own.
FORMATS = {"flac": ("FLAC", "PCM_16"), "wav": ("WAV", "PCM_16"), "mp3": ("MP3", None)}


def number(value, field: str) -> float:
    """A finite number, or ValueError naming the field."""
    if isinstance(value, bool):
        raise ValueError(f"'{field}' must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number") from None
    if not math.isfinite(result):
        raise ValueError(f"'{field}' must be a finite number")
    return result


def seed_of(task: dict) -> int | None:
    seed = task.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("'seed' must be a non-negative integer")
    return seed


def format_of(task: dict) -> str:
    fmt = str(task.get("format") or "flac").lower()
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, got {fmt!r}")
    return fmt


def encode(tensor, sample_rate: int, fmt: str) -> tuple[str, float]:
    """A [channels, samples] tensor → (base64 of the encoded file, seconds)."""
    import base64
    import io

    import soundfile

    audio = tensor.detach().float().cpu().numpy().T  # [samples, channels]
    container, subtype = FORMATS[fmt]
    buf = io.BytesIO()
    soundfile.write(buf, audio, sample_rate, format=container, subtype=subtype)
    return base64.b64encode(buf.getvalue()).decode(), audio.shape[0] / sample_rate
