# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Speaker-embedding child process: speechbrain ECAPA-TDNN.

speechbrain's floor is torch>=2.1, so it runs on giq's own torch 2.11.
Voiceprints are stable across torch versions: cosine 0.9999999 against torch
2.8 vectors, i.e. float noise, so prints enrolled under an older stack stay
valid.

Stateless: clip in, L2-normalized 192-dim voiceprint out — clustering and
name-binding are the client's job. Tiny model (~20MB) but it gets its own
child (not folded into _audio_child) so a ~5ms embed never queues behind a
minutes-long diarization on the serial IPC channel.

Task: {"id": str, "audio_b64": str}
Result: {"id", "embedding": [float], "dim": int, "error": str|null}
"""

import os
import sys

from giq_child import (
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)

# Overridable through the GIQ_EMBED_* env vars.
EMBED_MODEL = os.environ.get("GIQ_EMBED_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
EMBED_REVISION = os.environ.get("GIQ_EMBED_REVISION", "")


def _load_model():
    from speechbrain.inference.speaker import EncoderClassifier

    kwargs = {"source": EMBED_MODEL, "run_opts": {"device": "cuda"}}
    if EMBED_REVISION:
        kwargs["revision"] = EMBED_REVISION
    model = EncoderClassifier.from_hparams(**kwargs)
    model.eval()
    return model


def _embed(model, task: dict) -> dict:
    import base64
    import io

    import numpy as np
    import soundfile as sf
    import torch

    audio_bytes = base64.b64decode(task["audio_b64"])
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 — soundfile raises varied types
        return {"id": task["id"], "error": f"undecodable audio: {e}"}
    mono = data.mean(axis=1).astype(np.float32)
    sig = torch.from_numpy(mono).unsqueeze(0)
    if sr != 16000:
        import torchaudio

        sig = torchaudio.functional.resample(sig, sr, 16000)
    if sig.shape[-1] == 0:
        return {"id": task["id"], "error": "empty audio after decode"}
    sig = sig.to("cuda")
    with torch.no_grad():
        emb = model.encode_batch(sig).squeeze()
    # ECAPA output is NOT normalized; L2-normalize so cosine == dot product.
    emb = torch.nn.functional.normalize(emb, dim=0)
    vector = emb.detach().cpu().tolist()
    return {"id": task["id"], "embedding": vector, "dim": len(vector)}


def main() -> None:
    reserve_ipc_stdout()
    try:
        model = _load_model()
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_embed(model, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
