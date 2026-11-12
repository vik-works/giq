# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The audio side of the giq-music child: checking a task's numbers and formats,
reading the audio a task brings, and encoding a track. Stdlib at import;
numpy and soundfile only in ``encode``, which runs on the engine's
interpreter."""

import base64
import binascii
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


def audio_bytes(value, field: str) -> tuple[bytes, str]:
    """A task's base64 audio → (its bytes, the file suffix its container has).

    The engines read audio from a file and pick the decoder by its suffix,
    so the container is told from the bytes, not trusted from a name."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{field}' must be base64 audio")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"'{field}' is not valid base64") from None
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return data, ".wav"
    if data[:4] == b"fLaC":
        return data, ".flac"
    if data[:4] == b"OggS":
        return data, ".ogg"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return data, ".mp3"
    raise ValueError(f"'{field}' must be WAV, FLAC, Ogg or MP3 audio")


def encode(tensor, sample_rate: int, fmt: str) -> tuple[str, float]:
    """A [channels, samples] tensor → (base64 of the encoded file, seconds)."""
    import io

    import soundfile

    audio = tensor.detach().float().cpu().numpy().T  # [samples, channels]
    container, subtype = FORMATS[fmt]
    buf = io.BytesIO()
    soundfile.write(buf, audio, sample_rate, format=container, subtype=subtype)
    return base64.b64encode(buf.getvalue()).decode(), audio.shape[0] / sample_rate
