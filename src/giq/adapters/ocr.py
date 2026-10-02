# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""OCR worker: PDF or page images → layout-tagged text → HTML.

The child — ``_ocr_child`` for baidu/Unlimited-OCR, ``_glm_ocr_child`` for
GLM-OCR — returns the model's raw tagged text, loading the weights the
recipe file names (``giq.weights``), which the parent passes on its
command line. Everything that turns that into a document — dropping
headers, footers and page numbers, re-joining a table or paragraph the page
break cut in two, and rendering HTML — happens here in the parent through
``giq.ocrdoc``, on plain strings, so it is testable without a GPU and without
a real document.

Sleepy model: loads on demand, evictable. A pass over a dozen pages runs a
few minutes at ~80 tok/s, so the batch timeout is generous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from giq import ocrdoc
from giq.adapters._subprocess import SubprocessAdapter
from giq.models import OCRResult
from giq.registry import vram_for
from giq.weights import recipe_of, require_path

# 8266 MiB per-process peak measured on an RTX 5090 (4-page pass at 1024px);
# 6.3 GiB idle after load. Declared above the peak, as a gate figure
# should be.
OCR_VRAM_GB = 9.0

# Which child reads a checkpoint is the checkpoint's own to say: the
# architecture its config.json declares. giq has two OCR pipelines and each
# child is written for one: Unlimited-OCR's remote code, and GLM-OCR behind a
# PP-DocLayoutV3 layout stage (weights.parts.layout). An operator's OCR recipe
# is another checkpoint of one of the two, and is read the same way. Both run
# in giq's own interpreter on its transformers.
CHILD_OF_ARCHITECTURE: dict[str, str] = {
    "UnlimitedOCRForCausalLM": "giq.adapters._ocr_child",
    "GlmOcrForConditionalGeneration": "giq.adapters._glm_ocr_child",
}
# The parts a child cannot run without, besides the main weights.
PARTS_OF_CHILD: dict[str, tuple[str, ...]] = {"giq.adapters._glm_ocr_child": ("layout",)}


def child_of(weights: str) -> str:
    """The child module for the checkpoint at ``weights``, by its architecture."""
    import json

    config = Path(weights).expanduser() / "config.json"
    try:
        architectures = json.loads(config.read_text(encoding="utf-8")).get("architectures")
    except FileNotFoundError:
        raise RuntimeError(f"{config} not found: are the OCR weights on disk?") from None
    architecture = (architectures or [None])[0]
    if architecture not in CHILD_OF_ARCHITECTURE:
        raise ValueError(
            f"{weights}: no OCR child reads {architecture!r} "
            f"(one of {', '.join(sorted(CHILD_OF_ARCHITECTURE))})"
        )
    return CHILD_OF_ARCHITECTURE[architecture]


def _instance(model: str):
    recipe = recipe_of(model)
    if recipe is None:
        raise ValueError(f"unknown OCR model {model!r}: no recipe file defines it")
    return recipe


@dataclass
class OcrConfig:
    model: str = "unlimited-ocr"


class OcrAdapter(SubprocessAdapter):
    """An OCR model in a child process; documents assembled in the parent."""

    child_module: ClassVar[str] = "giq.adapters._ocr_child"  # per checkpoint; see child_of
    modality: ClassVar[str] = "ocr"
    # A long document is several passes of a few minutes each.
    run_batch_timeout: ClassVar[float] = 3600.0

    def __init__(self, config: OcrConfig, device: str | None = None):
        super().__init__(config, device)
        # Unknown model or no weights in its recipe: fail here, not at spawn.
        # The parts are what the recipe declares; the schema admits only the
        # ones an OCR child reads.
        recipe = _instance(config.model)
        self.weights = require_path(config.model)
        declared = recipe.weights.parts if recipe.weights is not None else {}
        self.parts = {part: require_path(config.model, part) for part in declared}

    def child_args(self) -> list[str]:
        args = ["--weights", self.weights]
        for part, path in self.parts.items():
            args += [f"--{part}", path]
        return args

    @property
    def estimated_vram_gb(self) -> float:
        return vram_for(self.config.model, default=OCR_VRAM_GB)

    def _command(self) -> list[str]:
        # The weights have to be on disk by now, so this is where the
        # checkpoint can say which child reads it. Instance attribute shadows
        # the ClassVar the base class's command and log lines read.
        self.child_module = child_of(self.weights)
        if missing := [p for p in PARTS_OF_CHILD.get(self.child_module, ()) if p not in self.parts]:
            raise ValueError(
                f"ocr/{self.config.model}: its checkpoint needs weights.parts.{missing[0]}, "
                "which the recipe does not declare"
            )
        return super()._command()

    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[OCRResult]:
        raw_results = await super().run_batch(tasks, params)
        options = {t.get("id"): t for t in tasks}
        return [hydrate(r, options.get(r.get("id"), {})) for r in raw_results]


def hydrate(result: dict[str, Any], task: dict[str, Any]) -> OCRResult:
    """Child result dict → OCRResult, applying the task's post-processing flags."""
    rid = str(result.get("id", "unknown"))
    if result.get("error"):
        return OCRResult(id=rid, error=str(result["error"]))
    raw = result.get("raw") or ""
    html, blocks = ocrdoc.render(
        raw, strip=task.get("strip", True) is not False, merge=task.get("merge", True) is not False
    )
    return OCRResult(
        id=rid,
        html=html,
        pages=int(result.get("pages") or ocrdoc.page_count(raw)),
        blocks=blocks,
        raw=raw if task.get("raw") else None,
        tokens_in=result.get("tokens_in"),
        tokens_out=result.get("tokens_out"),
        truncated=bool(result.get("truncated", False)),
    )
