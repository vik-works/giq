# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""OcrAdapter: child results become documents in the parent.

The child is replaced by an inline script that answers every task with a
canned tagged text, so this exercises hydration, the per-task flags and the
error envelope without a GPU or a model.
"""

import json
import sys

import pytest

from giq.models import OCRResult
from giq_ocr.adapter import OcrAdapter, OcrConfig, hydrate

RAW = (
    "<PAGE>\n<|det|>header [100, 36, 400, 70]<|/det|>Example AG\n"
    "<|det|>text [100, 100, 900, 200]<|/det|>Body text.\n"
    "<|det|>page_number [900, 940, 950, 970]<|/det|>1\n"
)

# Answers each task with RAW, or an error when the task id says so.
CHILD_SCRIPT = f"""
import sys, json
RAW = {RAW!r}
sys.stdout.write(json.dumps({{"type": "ready"}}) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "shutdown":
        break
    results = []
    for t in msg["tasks"]:
        if t["id"].startswith("bad"):
            results.append({{"id": t["id"], "error": "no pages to parse"}})
        else:
            results.append(
                {{"id": t["id"], "raw": RAW, "pages": 1, "tokens_in": 260, "tokens_out": 12}}
            )
    sys.stdout.write(json.dumps({{"type": "results", "results": results}}) + "\\n")
    sys.stdout.flush()
"""


class _StubOCRWorker(OcrAdapter):
    def _command(self) -> list[str]:
        return [sys.executable, "-u", "-c", CHILD_SCRIPT]

    def _spawn_env(self) -> dict[str, str]:
        import os

        return dict(os.environ)


@pytest.mark.asyncio
async def test_run_batch_returns_documents_not_pages():
    worker = _StubOCRWorker(OcrConfig())
    await worker.start()
    try:
        results = await worker.run_batch(
            [{"id": "a", "pdf_b64": "x"}, {"id": "b", "pdf_b64": "x", "raw": True}, {"id": "bad-1"}]
        )
    finally:
        await worker.stop()

    assert all(isinstance(r, OCRResult) for r in results)
    a, b, bad = results
    assert a.html == '<p class="text" data-page="1" data-bbox="100,100,900,200">Body text.</p>'
    assert a.pages == 1 and a.tokens_in == 260 and a.tokens_out == 12 and a.error is None
    assert "Example AG" not in a.html  # header stripped
    assert a.raw is None  # only on request
    assert b.raw == RAW
    assert bad.error == "no pages to parse" and bad.html == ""
    # The runner stores results via model_dump(); make sure that round-trips.
    dumped = json.loads(json.dumps([r.model_dump() for r in results]))
    assert dumped[0]["blocks"][0]["label"] == "text"


def test_hydrate_honours_strip_and_merge_flags():
    result = {"id": "t", "raw": RAW, "pages": 1}
    kept = hydrate(result, {"strip": False})
    assert "Example AG" in kept.html
    assert [b["label"] for b in kept.blocks] == ["header", "text", "page_number"]
    default = hydrate(result, {})
    assert [b["label"] for b in default.blocks] == ["text"]


def test_hydrate_counts_pages_when_the_child_did_not():
    assert hydrate({"id": "t", "raw": RAW}, {}).pages == 1


def test_estimated_vram_comes_from_the_registry():
    assert OcrAdapter(OcrConfig()).estimated_vram_gb == 9.0


def _checkpoint(tmp_path, name, architecture):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    return d


def test_the_checkpoint_picks_the_child(tmp_path, monkeypatch):
    """Both OCR pipelines run on the transformers engine; which child reads a
    checkpoint is the architecture its config.json names."""
    unlimited = _checkpoint(tmp_path, "unlimited", "UnlimitedOCRForCausalLM")
    glm = _checkpoint(tmp_path, "glm", "GlmOcrForConditionalGeneration")
    monkeypatch.setenv("GIQ_OCR_MODEL_DIR", str(unlimited))
    monkeypatch.setenv("GIQ_GLM_OCR_MODEL_DIR", str(glm))
    monkeypatch.setenv("GIQ_GLM_LAYOUT_DIR", str(tmp_path / "layout"))

    cmd = OcrAdapter(OcrConfig(model="unlimited-ocr"))._command()
    assert cmd[1:4] == ["-u", "-m", "giq_ocr._ocr_child"]
    cmd = OcrAdapter(OcrConfig(model="glm-ocr"))._command()
    assert cmd[1:4] == ["-u", "-m", "giq_ocr._glm_ocr_child"]
    with pytest.raises(ValueError):
        OcrAdapter(OcrConfig(model="no-such-ocr"))


def test_an_unknown_architecture_is_refused_at_spawn(tmp_path, monkeypatch):
    other = _checkpoint(tmp_path, "other", "SomethingElseForCausalLM")
    monkeypatch.setenv("GIQ_OCR_MODEL_DIR", str(other))
    with pytest.raises(ValueError, match="SomethingElseForCausalLM"):
        OcrAdapter(OcrConfig(model="unlimited-ocr"))._command()


def test_glm_vram_comes_from_the_registry():
    assert OcrAdapter(OcrConfig(model="glm-ocr")).estimated_vram_gb == 4.0


def test_both_children_run_on_giqs_own_interpreter(tmp_path, monkeypatch):
    """No second interpreter and no PYTHONPATH: the child imports giq the way
    giq itself does, which is what lets a wheel install run it."""

    monkeypatch.setenv(
        "GIQ_OCR_MODEL_DIR", str(_checkpoint(tmp_path, "u", "UnlimitedOCRForCausalLM"))
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    w = OcrAdapter(OcrConfig(model="unlimited-ocr"))
    assert w._command()[0] == sys.executable
    assert "PYTHONPATH" not in w._spawn_env()
