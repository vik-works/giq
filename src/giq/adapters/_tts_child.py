# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Child process for the Kokoro TTS worker.

Invoked by ``TtsAdapter`` (parent) via
``python -u -m giq.adapters._tts_child --lang-code a``.

Kokoro was the last CUDA worker still running inside giq's own process. That
meant its context was never destroyed: the runner would log "Kokoro TTS
stopped", ``active_worker`` would go None, and ~968 MiB stayed held until giq
restarted. On a 16GB card with the resident set loaded, that was the
difference between the next render working and OOMing. See
``giq.adapters._subprocess`` for the mechanism and the rest of the rationale.
"""

from __future__ import annotations

# FIRST: reserve stdout for JSON IPC — kokoro/torch print during import and
# would otherwise corrupt the channel before the parent sees {"type":"ready"}.
from giq_child import reserve_ipc_stdout

reserve_ipc_stdout()

import argparse  # noqa: E402
import base64  # noqa: E402
import io  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from typing import Any  # noqa: E402

from giq.adapters.tts import KOKORO_SAMPLE_RATE, KOKORO_VOICES, VOICE_MAP  # noqa: E402
from giq.models import JobResult  # noqa: E402
from giq_child import run_ipc_child_loop, write_startup_error  # noqa: E402

logger = logging.getLogger(__name__)


def _load_pipeline(lang_code: str, repo_id: str | None):
    """Load the Kokoro pipeline (blocking)."""
    from kokoro import KPipeline

    return KPipeline(lang_code=lang_code, repo_id=repo_id)


def _synthesize(pipeline, text: str, voice: str) -> bytes:
    """Synthesize ``text`` to WAV bytes."""
    import numpy as np
    import soundfile as sf

    kokoro_voice = VOICE_MAP.get(voice, voice)
    if kokoro_voice not in KOKORO_VOICES:
        kokoro_voice = "af_heart"

    chunks = [audio for _gs, _ps, audio in pipeline(text, voice=kokoro_voice)]
    if not chunks:
        raise RuntimeError("No audio generated")

    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(chunks), KOKORO_SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    return buffer.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-code", default="a", help="Kokoro language code (a = US English)")
    # The recipe's hf: source. Unset = kokoro's own default repository.
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        logger.info(f"Loading Kokoro TTS pipeline (lang_code={args.lang_code})")
        pipeline = _load_pipeline(args.lang_code, args.repo)
        logger.info("Kokoro TTS ready")
    except Exception as e:
        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(
        tasks: list[dict[str, Any]], params: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for task in tasks:
            task_id = task.get("id", "unknown")
            text = task.get("text", "")
            voice = task.get("voice", "alloy")
            try:
                wav = _synthesize(pipeline, text, voice)
                results.append(
                    JobResult(id=task_id, output=base64.b64encode(wav).decode()).model_dump()
                )
                logger.info(f"TTS task {task_id}: {len(text)} chars -> {len(wav)} bytes")
            except Exception as e:
                logger.error(f"TTS task {task_id} failed: {e}")
                results.append(JobResult(id=task_id, output="", error=str(e)).model_dump())
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
