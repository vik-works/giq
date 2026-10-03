# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Audio child process: faster-whisper ASR + pyannote diarization.

Runs under giq's own interpreter like every other child; killing it frees its
whole CUDA context (~4GB), which is exactly how eviction for image jobs works.

The stack does not use whisperx, and whisperx was never doing the
transcription — faster-whisper is, with native word timestamps. whisperx
contributed two things: a thin wrapper around ``pyannote.audio.Pipeline`` and
``assign_word_speakers``. Both are inlined below, so the only hard pin in the
stack (whisperx's ``torch~=2.8.0``) is gone and pyannote/speechbrain — whose
floors are ``torch>=2.8`` and ``>=2.1`` — run on giq's torch 2.11. Verified
bit-identical against the whisperx path: same segments, same diarization
turns, 40/40 words with the same speaker.

Audio is decoded once here (PyAV, via faster-whisper) and the same array is
handed to both the ASR and the diarizer as an in-memory waveform — pyannote
checks its waveform-dict path before its own reader. The old code wrote a temp
file and let the two models decode it separately. whisperx fed pyannote a
waveform dict too, which is why the outputs match exactly.

(That also makes giq independent of torchcodec, pyannote's declared reader,
which mattered on torch 2.10+cu128 where no published build was ABI-
compatible. On the cu130 line it works again — but decoding once is still
better than twice, so this does not go back.)

Wire protocol: see giq.adapters._subprocess. Transcribe task shape:
  {"id": str, "audio_b64": str, "language": str|null,
   "task": "transcribe", "diarize": bool}
Result shape (unchanged — this is a served contract):
  {"id", "text", "language", "duration", "speakers": [...],
   "segments": [{"start","end","text","speaker","words"}], "error": str|null}
"""

import os
import sys
from pathlib import Path

from giq_child import (
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)

# Tuned defaults (VAD and no-speech thresholds chosen to suppress whisper's
# hallucinations on silence). The GIQ_AUDIO_* env vars override them.
VAD_THRESHOLD = float(os.environ.get("GIQ_AUDIO_VAD_THRESHOLD", "0.3"))
VAD_MIN_SPEECH_MS = int(os.environ.get("GIQ_AUDIO_MIN_SPEECH_MS", "150"))
VAD_MIN_SILENCE_MS = int(os.environ.get("GIQ_AUDIO_MIN_SILENCE_MS", "300"))
NO_SPEECH_THRESHOLD = float(os.environ.get("GIQ_AUDIO_NO_SPEECH", "0.6"))
DIAR_BATCH = int(os.environ.get("GIQ_AUDIO_DIAR_BATCH", "8"))
WHISPER_MODEL = os.environ.get("GIQ_AUDIO_WHISPER_MODEL", "large-v3")
WHISPER_COMPUTE = os.environ.get("GIQ_AUDIO_WHISPER_COMPUTE", "float16")
# Pinned explicitly. This used to be whisperx's default, inherited silently —
# and it is *not* the speaker-diarization-3.1 you would assume. Changing it
# changes attribution, which downstream speaker binding depends on.
DIAR_MODEL = os.environ.get("GIQ_AUDIO_DIAR_MODEL", "pyannote/speaker-diarization-community-1")
SAMPLE_RATE = 16000


def _hf_token() -> str | None:
    tok = Path.home() / ".cache" / "huggingface" / "token"
    return tok.read_text().strip() if tok.exists() else None


def _load_models():
    import torch
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    asr = WhisperModel(WHISPER_MODEL, device="cuda", compute_type=WHISPER_COMPUTE)
    pipeline = Pipeline.from_pretrained(DIAR_MODEL, token=_hf_token())
    if pipeline is None:
        # from_pretrained returns None rather than raising when the model is
        # gated and the token is missing or unaccepted — the common failure
        # here. Say so, instead of an AttributeError on None two lines down.
        raise RuntimeError(
            f"pyannote returned no pipeline for {DIAR_MODEL!r} — the model is gated: "
            "accept its terms on huggingface.co and put a valid token in "
            "~/.cache/huggingface/token"
        )
    diar = pipeline.to(torch.device("cuda"))
    # Bound pyannote batch so long-audio diarization stays within VRAM.
    for attr in ("embedding_batch_size", "segmentation_batch_size"):
        if hasattr(diar, attr):
            setattr(diar, attr, DIAR_BATCH)
    return asr, diar


class _Turns:
    """Diarization turns, queryable by overlap.

    Inlined from whisperx (sorted array + binary search rather than a scan;
    it matters on hours-long recordings). Turns are (start, end, speaker).
    """

    def __init__(self, turns: list[tuple[float, float, str]]):
        import numpy as np

        ordered = sorted(turns, key=lambda t: t[0])
        self.starts = np.array([t[0] for t in ordered], dtype=np.float64)
        self.ends = np.array([t[1] for t in ordered], dtype=np.float64)
        self.speakers = [t[2] for t in ordered]

    def __len__(self) -> int:
        return len(self.speakers)

    def dominant(self, start: float, end: float) -> str | None:
        """The speaker holding the most of [start, end], or None if no overlap."""
        import numpy as np

        if not self.speakers:
            return None
        # Only turns beginning before `end` can overlap it.
        right = int(np.searchsorted(self.starts, end, side="left"))
        if right == 0:
            return None
        window = slice(0, right)
        hits = (self.starts[window] < end) & (self.ends[window] > start)
        totals: dict[str, float] = {}
        for idx in np.where(hits)[0]:
            overlap = min(self.ends[idx], end) - max(self.starts[idx], start)
            if overlap > 0:
                speaker = self.speakers[idx]
                totals[speaker] = totals.get(speaker, 0.0) + overlap
        if not totals:
            return None
        return max(totals.items(), key=lambda kv: kv[1])[0]


def _assign_speakers(turns: _Turns, segments: list[dict]) -> None:
    """Label each segment and word with the speaker who dominates it. In place."""
    if not len(turns):
        return
    for seg in segments:
        speaker = turns.dominant(seg.get("start", 0.0), seg.get("end", 0.0))
        if speaker is not None:
            seg["speaker"] = speaker
        for word in seg.get("words") or []:
            if word.get("start") is None:
                continue
            start = word["start"]
            word_speaker = turns.dominant(start, word.get("end", start))
            if word_speaker is not None:
                word["speaker"] = word_speaker


def _diarize(diar, audio) -> _Turns:
    """Run pyannote on an in-memory waveform and flatten it to turns."""
    import torch

    output = diar({"waveform": torch.from_numpy(audio[None, :]), "sample_rate": SAMPLE_RATE})
    return _Turns(
        [
            (segment.start, segment.end, speaker)
            for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True)
        ]
    )


def _transcribe(asr, diar, task: dict) -> dict:
    import base64
    import io

    from faster_whisper.audio import decode_audio

    language = task.get("language") or None
    diarize = task.get("diarize", True)

    # One decode for both models — the old path wrote a temp file and let
    # faster-whisper and pyannote each decode it separately.
    audio = decode_audio(io.BytesIO(base64.b64decode(task["audio_b64"])), sampling_rate=SAMPLE_RATE)

    fw_segments, info = asr.transcribe(
        audio,
        language=language,
        # Re-detect language per segment when not forced, so a
        # code-switching clip is transcribed each part in its own language.
        multilingual=language is None,
        task="transcribe",
        word_timestamps=True,  # native, in-script — no align model needed
        vad_filter=True,
        vad_parameters={
            "threshold": VAD_THRESHOLD,
            "min_speech_duration_ms": VAD_MIN_SPEECH_MS,
            "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
        },
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,
    )
    segments = []
    for s in fw_segments:  # generator — consumed here
        words = [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]
        segments.append({"start": s.start, "end": s.end, "text": s.text, "words": words})
    lang = info.language
    duration = round(info.duration, 3)
    if not segments:  # VAD gated everything (sneeze / silence)
        return {
            "id": task["id"],
            "text": "",
            "language": lang,
            "duration": duration,
            "speakers": [],
            "segments": [],
        }

    if diarize:
        _assign_speakers(_diarize(diar, audio), segments)

    out = [
        {
            "start": round(float(s.get("start", 0.0)), 3),
            "end": round(float(s.get("end", 0.0)), 3),
            "text": (s.get("text") or "").strip(),
            "speaker": s.get("speaker"),
            "words": s.get("words") or [],
        }
        for s in segments
    ]
    text = " ".join(s["text"] for s in out if s["text"]).strip()
    speakers = sorted({s["speaker"] for s in out if s["speaker"]})
    return {
        "id": task["id"],
        "text": text,
        "language": lang,
        "duration": duration,
        "speakers": speakers,
        "segments": out,
    }


def main() -> None:
    reserve_ipc_stdout()
    try:
        asr, diar = _load_models()
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_transcribe(asr, diar, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
