# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-ocr: documents to layout-tagged text and HTML (ADR-004).

GLM-OCR behind its layout model, and Unlimited-OCR, on the transformers
engine in child processes; POST /ocr; the built-in OCR recipes."""

from __future__ import annotations

from pathlib import Path

from giq.plugin import API_VERSION, AdapterContext, Modality, Plugin

__version__ = "0.6.0"


def _ocr(ctx: AdapterContext):
    from giq_ocr.adapter import OcrAdapter, OcrConfig

    return OcrAdapter(config=OcrConfig(model=ctx.recipe), device=ctx.device)


def _ocr_plugin() -> Plugin:
    return Plugin(
        name="giq-ocr",
        api_version=API_VERSION,
        version=__version__,
        modalities=(
            # A long PDF is several passes of a few minutes each (~80 tok/s,
            # up to 32k tokens a pass). Matches OcrAdapter.run_batch_timeout.
            Modality(
                "ocr",
                label="Document OCR",
                icon="file-text",
                job_timeout=3600.0,
                parts=frozenset({"layout"}),
                payload_keys=("pdf_b64", "images_b64"),
            ),
        ),
        adapters={("transformers", "ocr"): _ocr},
        recipes=Path(__file__).parent / "recipes",
        routers=("giq_ocr.api:router",),
    )


plugin = _ocr_plugin()
