# SPDX-FileCopyrightText: 2026 Mohit Garg
#
# SPDX-License-Identifier: Apache-2.0

"""imajev's PyTorch decision path, the inference half only.

Vendored from ``scripts/torch_decision.py`` of
https://github.com/mohit67890/imajev at 318e8dad55c12a1c6faddbec5f30489d2ee529ff
(Apache-2.0). imajev installs ``vision_decision`` as a package but keeps this
module under ``scripts/``, so the child carries its own copy instead of a
checkout on PYTHONPATH. Modified: trimmed to load, readout and single-request
scoring (training, the batched, CUDA-graph and thinking paths are dropped),
and reformatted. The arithmetic is unchanged; keep it so when re-vendoring,
since the shipped readout and calibration were fitted against it.
"""

import json
from pathlib import Path
from typing import Any

import torch
from vision_decision.scoring import (
    DEFAULT_PROMPT_LAYOUT,
    MAX_READOUT_CODES,
    check_prompt_layout,
    check_readout_codes,
    readout_codes,
    verified_label_ids,
)

# The decision is read right after Qwen3.5's empty non-thinking block.
DECISION_TAIL = "</think>\n\n"


class _LazyMultimodalProcessor:
    """Text requests need only the tokenizer; the vision processor loads on first image."""

    def __init__(self, path):
        from transformers import AutoTokenizer

        self.path = path
        # transformers types the result Optional; a checkpoint without a
        # tokenizer fails in from_pretrained, never as None here.
        self.tokenizer: Any = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self._multimodal = None

    def _vision(self):
        if self._multimodal is None:
            from transformers import AutoProcessor

            self._multimodal = AutoProcessor.from_pretrained(self.path, local_files_only=True)
            self._multimodal.tokenizer = self.tokenizer
        return self._multimodal

    def apply_chat_template(self, messages, **kwargs):
        has_media = any(
            item.get("type") in ("image", "video")
            for message in messages
            for item in message.get("content", [])
            if isinstance(item, dict)
        )
        if has_media:
            return self._vision().apply_chat_template(messages, **kwargs)
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, text, images=None, **kwargs):
        if images is not None:
            return self._vision()(text=text, images=images, **kwargs)
        kwargs.setdefault("add_special_tokens", False)
        return self.tokenizer(text, **kwargs)


class TorchDecision:
    """Qwen3.5 plus the decision readout: candidate logits at the decision position."""

    def __init__(self, path, device, dtype=torch.bfloat16, max_length=4096):
        from transformers import Qwen3_5ForConditionalGeneration

        self.processor = _LazyMultimodalProcessor(path)
        self.device = device
        self.max_length = max_length
        self.model = (
            Qwen3_5ForConditionalGeneration.from_pretrained(
                path, local_files_only=True, dtype=dtype
            )
            .to(device)
            .eval()
        )
        self.readout = None
        self._codebook = None
        # 255 shipped; the adapter's decision_readout.json may say 256.
        self.codes = MAX_READOUT_CODES
        self.prompt_layout = DEFAULT_PROMPT_LAYOUT

    def _base(self):
        """The Qwen3.5 model under a PEFT wrapper, or the model itself."""
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def _ensure_codebook(self, n_images=0):
        if self._codebook is None or len(self._codebook) != self.codes:
            self._codebook = readout_codes(
                self.processor.tokenizer, self.render("", n_images), self.codes, limit=self.codes
            )
        return self._codebook

    def labels(self, count, n_images=0):
        codebook = self._ensure_codebook(n_images)
        if count > len(codebook):
            raise ValueError(
                f"{count} candidates exceed the {len(codebook)}-code readout "
                f"(at most {len(codebook) - 1} options + unknown)"
            )
        return [code for code, _ in codebook[:count]]

    def enable_readout(self, adapter):
        """Install the adapter's trained readout, Linear[codes, hidden] in float32.

        Rows past the trained ones start from the matching LM-head rows, as at
        training. The adapter's prompt layout is adopted. An adapter without
        decision_readout.safetensors scores through vocabulary rows (False).
        """
        from safetensors.torch import load_file

        base = self._base()
        path = Path(adapter) / "decision_readout.safetensors"
        if not path.exists():
            return False
        trained = load_file(str(path), device=str(self.device))["weight"].float()
        manifest_path = Path(adapter) / "decision_readout.json"
        if not manifest_path.exists():
            raise ValueError("Trained readout is missing decision_readout.json tokenizer binding")
        manifest = json.loads(manifest_path.read_text())
        rows = trained.shape[0] if trained.ndim == 2 else None
        if rows not in (255, 256):
            raise ValueError("Decision readout must be finite with shape [255 or 256, hidden_size]")
        self.codes = check_readout_codes(rows)
        self._codebook = None
        codebook = self._ensure_codebook(0)
        actual = [{"code": c, "token_id": i} for c, i in codebook]
        bound = manifest.get("codes")
        if (
            manifest.get("version") != 1
            or not isinstance(bound, list)
            or len(bound) != rows
            or bound != actual[:rows]
        ):
            raise ValueError("Decision readout code/token binding does not match this tokenizer")
        self.prompt_layout = check_prompt_layout(
            manifest.get("prompt_layout", DEFAULT_PROMPT_LAYOUT)
        )
        hidden = base.lm_head.weight.shape[1]
        if tuple(trained.shape) != (self.codes, hidden) or not bool(torch.isfinite(trained).all()):
            raise ValueError(
                f"Decision readout must be finite with shape [{self.codes}, hidden_size]"
            )
        self.readout = torch.nn.Linear(
            hidden, self.codes, bias=False, device=self.device, dtype=torch.float32
        )
        self.readout.weight.data.copy_(trained)
        self.readout.weight.requires_grad_(False)
        return True

    def render(self, prompt, n_images):
        content = [{"type": "image"}] * n_images + [{"type": "text", "text": prompt}]
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        if not rendered.endswith("<think>\n\n" + DECISION_TAIL):
            raise ValueError("Unexpected Qwen non-thinking template boundary")
        return rendered

    def prepare(self, images, prompt, labels):
        images = images or []
        rendered = self.render(prompt, len(images))
        token_ids = verified_label_ids(self.processor.tokenizer, rendered, labels)
        inputs = self.processor(text=[rendered], images=images or None, return_tensors="pt")
        if inputs["input_ids"].shape[-1] > self.max_length:
            raise ValueError(f"Processed request exceeds the {self.max_length}-token limit")
        suffix = self.processor.tokenizer.encode(DECISION_TAIL, add_special_tokens=False)
        if inputs["input_ids"][0, -len(suffix) :].tolist() != suffix:
            raise ValueError("Processed decision-position suffix mismatch")
        return rendered, inputs, token_ids

    def candidate_logits(self, inputs, token_ids):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        base = self._base()
        hidden = base.model(**inputs).last_hidden_state[0, -1].float()
        indices = self._readout_indices(token_ids)
        if self.readout is not None and indices is not None:
            return self.readout(hidden)[indices]
        head = base.lm_head.weight[torch.tensor(token_ids, device=self.device)]
        return hidden @ head.float().T

    def _readout_indices(self, token_ids):
        if self._codebook is None:
            return None
        lookup = {token: i for i, (_, token) in enumerate(self._codebook)}
        return [lookup[x] for x in token_ids] if all(x in lookup for x in token_ids) else None
