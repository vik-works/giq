# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""OCR child process: baidu/Unlimited-OCR through transformers, offline.

The model's own ``infer_multi()`` is not called: it prints every generated
token to stdout through a streamer (which giq's parent would drain into its
DEBUG log), creates an output directory, and under ``save_results`` runs
``eval()`` on the model's output and writes the document to disk. Its tensor
preparation is reproduced below and ``generate()`` is called directly, with
no streamer and nothing written anywhere.

The reproduced tensor preparation — marked as a snippet in ``_generate`` —
is adapted from the model's remote code in baidu/Unlimited-OCR
(https://github.com/baidu/Unlimited-OCR), MIT-licensed, Copyright (c) 2026
Baidu; the MIT licence text is in LICENSES/MIT.txt.

Weights and remote code load from the local snapshot the parent names with
``--weights`` (the recipe's ``weights.path``, pinned to HF revision
07dea83; ``GIQ_OCR_MODEL_DIR`` overrides it), with the hub disabled before
transformers is imported, so a later change to the upstream repo cannot reach this process.
Verified with strace: the process opens no network sockets.

Memory is bounded by construction: pages are rasterized one pass at a time
(``PAGES_PER_PASS`` of them, ~12 MB each at 200 dpi) and released before the
next pass, and a document over ``GIQ_OCR_MAX_PAGES`` is refused before any
page is rendered. Nothing is written to disk, tmp included.

Wire protocol: see giq.adapters._subprocess. Task shape:
  {"id": str, "pdf_b64": str | "images_b64": [str], "dpi": int?, "pages": [int]?}
Result shape (raw; the parent turns it into html via giq_ocr.ocrdoc):
  {"id", "raw", "pages", "tokens_in", "tokens_out", "truncated", "error": str|null}
"""

import os
import sys
from collections.abc import Iterator

# Before any transformers import: no hub lookups, no telemetry, ever.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from giq_child import (  # noqa: E402
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)

# Multi-page mode is base-only: every page padded to a 1024 square, 257 image
# tokens each. (Single-image "gundam" tiling exists but is a different prompt
# and code path; one path is enough for a first worker.)
IMAGE_SIZE = 1024
MAX_LENGTH = 32768
DEFAULT_DPI = 200
# Pages per forward pass. Budget is MAX_LENGTH for prompt + output together;
# a dense page runs ~1,500 output tokens, so 12 pages stays under it with room.
# Longer documents go through in consecutive passes and are concatenated —
# every page still gets its own <PAGE> marker, so page numbers stay right.
PAGES_PER_PASS = int(os.environ.get("GIQ_OCR_PAGES_PER_PASS", "12"))
# Refused before rendering anything. 200 pages is ~17 passes, ~10 minutes.
MAX_PAGES = int(os.environ.get("GIQ_OCR_MAX_PAGES", "200"))
PROMPT = "<image>Multi page parsing."
NGRAM_SIZE, NGRAM_WINDOW = 35, 1024
STOP = "<｜end▁of▁sentence｜>"
IMAGE_TOKEN_ID = 128815
PATCH_SIZE, DOWNSAMPLE = 16, 4


def _reinit_buffers(model):
    """transformers 5 materializes modules on the meta device and only fills
    tensors present in the checkpoint; a buffer the checkpoint lacks is left
    as uninitialized memory. Two are rebuilt as their modules' __init__
    built them: CLIP's position_ids (4.57 warned it was "newly
    initialized"), and every rotary embedding's inv_freq — computed, never
    stored, and NaN here, which turned the decoder's first attention and
    every logit after it into NaN."""
    import torch

    for mod in model.modules():
        ids = getattr(mod, "position_ids", None)
        table = getattr(mod, "position_embedding", None)
        if isinstance(ids, torch.Tensor) and table is not None:
            n = table.num_embeddings
            mod.position_ids = torch.arange(n, device=ids.device).expand(1, -1)
        freq = getattr(mod, "inv_freq", None)
        if isinstance(freq, torch.Tensor) and hasattr(mod, "compute_default_rope_parameters"):
            init = getattr(mod, "rope_init_fn", None) or mod.compute_default_rope_parameters
            inv_freq, mod.attention_scaling = init(mod.config, freq.device)
            mod.inv_freq = inv_freq
            if isinstance(getattr(mod, "original_inv_freq", None), torch.Tensor):
                mod.original_inv_freq = inv_freq.clone()


def _complete_config(cfg, tok):
    """transformers 5 drops a custom config's defaulted attributes (they are not
    in __dict__ and there is no class-level fallback), and the remote code and
    the Llama attention base read them unconditionally. Re-apply the defaults
    the config's own __init__ declares, plus the two Llama needs."""
    import inspect

    for klass in type(cfg).__mro__:
        if klass.__name__ != "DeepseekV2Config":
            continue
        for name, p in inspect.signature(klass.__init__).parameters.items():
            if p.default is inspect.Parameter.empty or name in ("self", "kwargs"):
                continue
            try:
                getattr(cfg, name)
            except AttributeError:
                setattr(cfg, name, p.default)
    if not hasattr(cfg, "head_dim"):
        cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    # 5.x rotary embeddings read a rope_parameters dict; the 4.x-era config
    # carries rope_theta and a (None) rope_scaling instead.
    if getattr(cfg, "rope_parameters", None) is None:
        cfg.rope_parameters = {"rope_type": "default", "rope_theta": float(cfg.rope_theta)}
    cfg.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return cfg


def _generation_inputs_as_in_457(model):
    """Feed the remote code's ``prepare_inputs_for_generation`` what 4.57 fed it.

    Two things changed under it in transformers 5, and each breaks generation:

    - A cache with no maximum answers ``get_max_length() == -1``; 4.57 said
      ``None``, which is what the hook tests for. Seeing -1 it takes the cache
      for full and crops the attention mask by ``mask[:, -(-1):]``, one
      position short of ``input_ids`` (a 279-against-280 mismatch in
      prefill). For the length of the call the cache gives 4.57's answer;
      transformers' own code never sees it.
    - ``generate()`` now passes ``position_ids`` for the whole sequence. The
      hook only trims position ids it computed itself, so a decode step got
      one new token with every position, and the rotary embedding broadcast
      that token's key to the full length. The ids are trimmed to the tokens
      actually passed, as the hook trims its own.
    """
    import functools

    from transformers import Cache

    cls = type(model)
    original = cls.prepare_inputs_for_generation

    @functools.wraps(original)
    def prepare(self, input_ids, past_key_values=None, *args, **kwargs):
        patched = None
        if isinstance(past_key_values, Cache) and hasattr(past_key_values, "get_max_length"):
            try:
                unbounded = past_key_values.get_max_length() == -1
            except ValueError:  # no layers yet: nothing bounds it either
                unbounded = True
            if unbounded:
                patched = past_key_values
                patched.get_max_length = lambda: None
        try:
            inputs = original(self, input_ids, past_key_values, *args, **kwargs)
        finally:
            if patched is not None:
                del patched.get_max_length
        ids, positions = inputs.get("input_ids"), inputs.get("position_ids")
        if ids is not None and positions is not None and positions.shape[-1] > ids.shape[-1]:
            inputs["position_ids"] = positions[..., -ids.shape[-1] :]
        return inputs

    cls.prepare_inputs_for_generation = prepare


def _tokenizer(weights: str):
    """The snapshot's tokenizer.json exactly as it is, without a class rebuild.

    The config names LlamaTokenizerFast, and transformers 5 maps that to its
    unified LlamaTokenizer, which rebuilds the vocabulary with Llama's
    SentencePiece-style rules. DeepSeek's tokenizer is byte-level BPE with
    its own pre-tokenizer, so every text token came out different ("Multi
    page parsing." as 6 ids instead of 4) and the model answered with a run
    of begin-of-sentence tokens. Loaded from the file, it matches 4.57's
    ids token for token, special tokens included."""
    import json

    from transformers import PreTrainedTokenizerFast

    with open(os.path.join(weights, "tokenizer_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    def special(name):
        value = cfg.get(name)
        return value.get("content") if isinstance(value, dict) else value

    return PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(weights, "tokenizer.json"),
        bos_token=special("bos_token"),
        eos_token=special("eos_token"),
        pad_token=special("pad_token"),
    )


def _load(weights: str):
    import torch
    from transformers import AutoConfig, AutoModel

    if not os.path.isdir(weights):
        raise RuntimeError(f"model directory does not exist: {weights}")
    tok = _tokenizer(weights)
    cfg = _complete_config(
        AutoConfig.from_pretrained(weights, trust_remote_code=True, local_files_only=True), tok
    )
    model = AutoModel.from_pretrained(
        weights,
        config=cfg,
        trust_remote_code=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = model.eval().cuda()
    _reinit_buffers(model)
    _generation_inputs_as_in_457(model)
    model.disable_torch_init()
    return tok, model


def _chunks(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _pdf_passes(data: bytes, dpi: int, pages: list[int] | None) -> Iterator[list]:
    """Yield one pass worth of rendered pages at a time; never the whole PDF."""
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
        for group in _chunks(wanted, PAGES_PER_PASS):
            yield [doc[p - 1].render(scale=dpi / 72).to_pil().convert("RGB") for p in group]
    finally:
        doc.close()


def _image_passes(b64_list: list[str]) -> Iterator[list]:
    import base64
    import io

    from PIL import Image, ImageOps

    if len(b64_list) > MAX_PAGES:
        raise ValueError(f"{len(b64_list)} images; the limit is {MAX_PAGES}")
    for group in _chunks(b64_list, PAGES_PER_PASS):
        yield [
            ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b)))).convert("RGB")
            for b in group
        ]


def _generate(tok, model, images) -> tuple[str, int, int, bool]:
    """infer_multi()'s tensor prep, minus the streamer and the file writes.

    Returns (text, prompt_tokens, generated_tokens, hit_the_ceiling).
    """
    import math

    import torch
    from PIL import ImageOps

    # SPDX-SnippetBegin
    # SPDX-SnippetCopyrightText: 2026 Baidu
    # SPDX-License-Identifier: MIT
    # Adapted from infer_multi() in baidu/Unlimited-OCR's remote code.
    m = sys.modules[type(model).__module__]
    conversation = [
        {"role": "<|User|>", "content": PROMPT, "images": ["<in-memory>"] * len(images)},
        {"role": "<|Assistant|>", "content": ""},
    ]
    formatted = m.format_messages(conversations=conversation, sft_format="plain", system_prompt="")
    before, after = formatted.split("<image>")
    transform = m.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    num_queries = math.ceil((IMAGE_SIZE // PATCH_SIZE) / DOWNSAMPLE)
    pad = tuple(int(x * 255) for x in transform.mean)

    ids = m.text_encode(tok, before, bos=False, eos=False)
    mask = [False] * len(ids)
    tensors = []
    for img in images:
        view = ImageOps.pad(img, (IMAGE_SIZE, IMAGE_SIZE), color=pad)
        tensors.append(transform(view).to(torch.bfloat16))
        tile = ([IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]) * num_queries + [IMAGE_TOKEN_ID]
        ids += tile
        mask += [True] * len(tile)
    tail = m.text_encode(tok, after, bos=False, eos=False)
    ids += tail
    mask += [False] * len(tail)
    ids = [0] + ids  # bos
    mask = [False] + mask

    input_ids = torch.LongTensor(ids).unsqueeze(0).cuda()
    images_ori = torch.stack(tensors, dim=0).cuda()
    dummy_crop = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE)).cuda()
    spatial = torch.tensor([[1, 1]] * len(images), dtype=torch.long)
    # SPDX-SnippetEnd

    cfg = model.config
    orig_sw = getattr(cfg, "sliding_window_size", None) or getattr(cfg, "sliding_window", None)
    cfg._ring_window = orig_sw  # the ring buffer reads it from here
    cfg.sliding_window = None  # keep DynamicCache from truncating the prefill
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                images=[(dummy_crop, images_ori)],
                images_seq_mask=torch.tensor(mask, dtype=torch.bool).unsqueeze(0).cuda(),
                images_spatial_crop=spatial,
                do_sample=False,
                temperature=None,
                eos_token_id=tok.eos_token_id,
                max_length=MAX_LENGTH,
                use_cache=True,
                logits_processor=[m.SlidingWindowNoRepeatNgramProcessor(NGRAM_SIZE, NGRAM_WINDOW)],
            )
    finally:
        cfg.sliding_window = orig_sw
    n_in = int(input_ids.shape[1])
    gen = out_ids[0, n_in:]
    text = tok.decode(gen)
    truncated = not text.endswith(STOP) and int(out_ids.shape[1]) >= MAX_LENGTH
    if text.endswith(STOP):
        text = text[: -len(STOP)]
    return text.strip(), n_in, int(gen.shape[0]), truncated


def _run(tok, model, task: dict) -> dict:
    import base64

    dpi = int(task.get("dpi") or DEFAULT_DPI)
    if not 50 <= dpi <= 400:
        raise ValueError("dpi must be between 50 and 400")
    if task.get("pdf_b64"):
        passes = _pdf_passes(base64.b64decode(task["pdf_b64"]), dpi, task.get("pages"))
    elif task.get("images_b64"):
        passes = _image_passes(task["images_b64"])
    else:
        raise ValueError("task needs pdf_b64 or images_b64")

    parts: list[str] = []
    pages = tokens_in = tokens_out = 0
    truncated = False
    for images in passes:
        text, n_in, n_out, hit = _generate(tok, model, images)
        parts.append(text)
        pages += len(images)
        tokens_in += n_in
        tokens_out += n_out
        truncated = truncated or hit
        del images  # this pass's bitmaps go before the next pass renders
    if pages == 0:
        raise ValueError("no pages to parse")
    return {
        "id": task["id"],
        "raw": "\n".join(parts),
        "pages": pages,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "truncated": truncated,
    }


def main() -> None:
    import argparse

    reserve_ipc_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    try:
        tok, model = _load(args.weights)
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_run(tok, model, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
