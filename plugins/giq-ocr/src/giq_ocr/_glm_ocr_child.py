# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
# SPDX-FileCopyrightText: 2026 Zhipu AI
#
# SPDX-License-Identifier: Apache-2.0

"""OCR child process: GLM-OCR behind PP-DocLayoutV3, both native transformers.

The two-stage pipeline that GLM-OCR's benchmark numbers come from: the
layout model finds every region of a page with a label, a bbox and a reading
order; each region is then read by GLM-OCR with the prompt its label calls
for — ``Table Recognition:`` for tables (HTML out), ``Formula Recognition:``
for formulas, ``Text Recognition:`` for the rest. Regions that are page
furniture (running headers, footers, page numbers, margin notes) are never
read at all: they cost nothing and cannot leak into the document.

Neither model carries remote code; both load from local snapshots pinned to
reviewed revisions — the parent passes the recipe's ``weights.path`` as
``--weights`` and its ``weights.parts.layout`` as ``--layout`` — with the hub
disabled before transformers is imported.
zai-org's own ``glmocr`` SDK is deliberately not used: it defaults to
``maas.enabled: true``, which forwards documents to Zhipu's cloud API, and
depends on PyMuPDF (AGPL). The pipeline it implements is small and is
reproduced here.

The pipeline follows the glmocr SDK in zai-org/GLM-OCR
(https://github.com/zai-org/GLM-OCR), Apache-2.0, Copyright 2026 Zhipu AI.
It has been modified: rewritten against native transformers, without the
cloud path or PyMuPDF, and emitting giq's tagged-text result shape.

Output is the same tagged text the Unlimited-OCR child emits —
``<|det|>label [x1, y1, x2, y2]<|/det|>content`` on a 0-999 page box, pages
introduced by ``<PAGE>`` — so ``giq_ocr.ocrdoc`` strips, merges and renders
either model's output identically. Labels are mapped onto that vocabulary.

Wire protocol: see giq.adapters._subprocess. Task and result shapes match
``_ocr_child`` (``tokens_in``/``tokens_out`` are summed over regions).
"""

import os
import sys
from collections.abc import Iterator

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from giq_child import (  # noqa: E402
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)

DEFAULT_DPI = 200
MAX_PAGES = int(os.environ.get("GIQ_OCR_MAX_PAGES", "200"))
# Layout detections under this score are noise (the SDK's default).
LAYOUT_THRESHOLD = float(os.environ.get("GIQ_GLM_LAYOUT_THRESHOLD", "0.3"))
# Tables and dense text regions can run long; 8192 is the SDK's budget.
MAX_NEW_TOKENS = 8192
# Pixels of context around a region crop; the layout box is tight.
CROP_PAD = 4

PROMPTS = {
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "text": "Text Recognition:",
}

# PP-DocLayoutV3 label → (what to do, giq_ocr.ocrdoc label). "abandon" regions
# are emitted as empty tagged blocks so the furniture is visible in the raw
# output and stripped by label like Unlimited-OCR's; "skip" regions become
# image placeholders.
LABELS: dict[str, tuple[str, str]] = {
    "header": ("abandon", "header"),
    "header_image": ("abandon", "header"),
    "footer": ("abandon", "footer"),
    "footer_image": ("abandon", "footer"),
    "number": ("abandon", "page_number"),
    "aside_text": ("abandon", "aside_text"),
    "reference": ("abandon", "reference"),
    "image": ("skip", "image"),
    "chart": ("skip", "image"),
    "table": ("table", "table"),
    "display_formula": ("formula", "formula"),
    "inline_formula": ("formula", "formula"),
    "formula": ("formula", "formula"),
    "doc_title": ("text", "title"),
    "paragraph_title": ("text", "title"),
    "figure_title": ("text", "caption"),
    "vision_footnote": ("text", "caption"),
    "footnote": ("text", "footnote"),
    "formula_number": ("text", "formula_number"),
    "abstract": ("text", "text"),
    "algorithm": ("text", "text"),
    "content": ("text", "text"),
    "reference_content": ("text", "text"),
    "seal": ("text", "text"),
    "text": ("text", "text"),
    "vertical_text": ("text", "text"),
}


def _load(weights: str, layout_dir: str):
    import torch
    from transformers import (
        AutoProcessor,
        GlmOcrForConditionalGeneration,
        PPDocLayoutV3ForObjectDetection,
        PPDocLayoutV3ImageProcessor,
    )

    for d in (weights, layout_dir):
        if not os.path.isdir(d):
            raise RuntimeError(f"model directory does not exist: {d}")
    layout_proc = PPDocLayoutV3ImageProcessor.from_pretrained(layout_dir, local_files_only=True)
    layout = PPDocLayoutV3ForObjectDetection.from_pretrained(layout_dir, local_files_only=True)
    layout = layout.eval().cuda()
    proc = AutoProcessor.from_pretrained(weights, local_files_only=True)
    glm = GlmOcrForConditionalGeneration.from_pretrained(
        weights, dtype=torch.bfloat16, local_files_only=True
    )
    glm = glm.eval().cuda()
    return layout_proc, layout, proc, glm


def _pdf_pages(data: bytes, dpi: int, pages: list[int] | None) -> Iterator:
    """Yield rendered pages one at a time; the whole PDF is never in memory as bitmaps."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(data)
    try:
        n = len(doc)
        wanted = pages or list(range(1, n + 1))
        for p in wanted:
            if not 1 <= p <= n:
                raise ValueError(f"page {p} out of range 1..{n}")
        if len(wanted) > MAX_PAGES:
            raise ValueError(f"{len(wanted)} pages requested; the limit is {MAX_PAGES}")
        for p in wanted:
            yield doc[p - 1].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        doc.close()


def _image_pages(b64_list: list[str]) -> Iterator:
    import base64
    import io

    from PIL import Image, ImageOps

    if len(b64_list) > MAX_PAGES:
        raise ValueError(f"{len(b64_list)} images; the limit is {MAX_PAGES}")
    for b in b64_list:
        yield ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b)))).convert("RGB")


def _detect(layout_proc, layout, page) -> list[dict]:
    """Regions in reading order: the model predicts the order itself."""
    import torch

    inputs = layout_proc(images=page, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = layout(**inputs)
    (res,) = layout_proc.post_process_object_detection(
        out, threshold=LAYOUT_THRESHOLD, target_sizes=[page.size[::-1]]
    )
    regions = []
    for label_id, box in zip(res["labels"], res["boxes"], strict=True):
        x1, y1, x2, y2 = (int(v) for v in box.tolist())
        regions.append({"label": layout.config.id2label[int(label_id)], "box": (x1, y1, x2, y2)})
    return regions


def _recognize(proc, glm, crop, prompt: str) -> tuple[str, int, int]:
    import torch

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": crop}, {"type": "text", "text": prompt}],
        }
    ]
    inputs = proc.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    inputs.pop("token_type_ids", None)
    n_in = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        gen = glm.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new = gen[0][n_in:]
    return proc.decode(new, skip_special_tokens=True).strip(), n_in, int(new.shape[0])


def _page_to_tagged(models, page) -> tuple[str, int, int, bool]:
    """One page → tagged text, prompt tokens, generated tokens, hit-the-ceiling."""
    layout_proc, layout, proc, glm = models
    w, h = page.size
    lines, n_in_total, n_out_total, truncated = [], 0, 0, False
    for r in _detect(layout_proc, layout, page):
        action, label = LABELS.get(r["label"], ("text", r["label"]))
        x1, y1, x2, y2 = r["box"]
        bbox = f"[{999 * x1 // w}, {999 * y1 // h}, {999 * x2 // w}, {999 * y2 // h}]"
        tag = f"<|det|>{label} {bbox}<|/det|>"
        if action in ("abandon", "skip"):
            lines.append(tag)
            continue
        crop = page.crop(
            (
                max(0, x1 - CROP_PAD),
                max(0, y1 - CROP_PAD),
                min(w, x2 + CROP_PAD),
                min(h, y2 + CROP_PAD),
            )
        )
        if crop.width < 2 or crop.height < 2:
            continue
        text, n_in, n_out = _recognize(proc, glm, crop, PROMPTS[action])
        n_in_total += n_in
        n_out_total += n_out
        truncated = truncated or n_out >= MAX_NEW_TOKENS
        lines.append(tag + text)
    return "<PAGE>\n" + "\n".join(lines), n_in_total, n_out_total, truncated


def _run(models, task: dict) -> dict:
    import base64

    dpi = int(task.get("dpi") or DEFAULT_DPI)
    if not 50 <= dpi <= 400:
        raise ValueError("dpi must be between 50 and 400")
    if task.get("pdf_b64"):
        pages = _pdf_pages(base64.b64decode(task["pdf_b64"]), dpi, task.get("pages"))
    elif task.get("images_b64"):
        pages = _image_pages(task["images_b64"])
    else:
        raise ValueError("task needs pdf_b64 or images_b64")

    parts: list[str] = []
    count = tokens_in = tokens_out = 0
    truncated = False
    for page in pages:
        text, n_in, n_out, hit = _page_to_tagged(models, page)
        parts.append(text)
        count += 1
        tokens_in += n_in
        tokens_out += n_out
        truncated = truncated or hit
        del page
    if count == 0:
        raise ValueError("no pages to parse")
    return {
        "id": task["id"],
        "raw": "\n".join(parts),
        "pages": count,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "truncated": truncated,
    }


def main() -> None:
    import argparse

    reserve_ipc_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--layout", required=True)
    args = parser.parse_args()
    try:
        models = _load(args.weights, args.layout)
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_run(models, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
