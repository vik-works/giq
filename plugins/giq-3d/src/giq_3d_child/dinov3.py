# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 ViT-L/16 from timm's redistribution, loaded as transformers' model.

Pixal3D loads its encoder with ``DINOv3ViTModel.from_pretrained`` and
reaches into its layers, so it wants transformers' model and parameter
names. Meta's own repository for those is gated behind a form; timm
publishes the same weights ungated, under the same DINOv3 licence, with
timm's names. ``from_timm`` reads timm's file and renames the parameters as
it loads, so the recipe's encoder is timm's repository as published.

timm's ``_qkvb`` variant is the one to use: it keeps the query and value
biases of the original checkpoint, where the plain variant drops them.

Run as a module it checks the renaming: both implementations are given the
same image and must return the same tokens (measured: they differ by about
1e-5 at 224, 512 and 1024 pixels).

    <pixal3d python> -m giq_3d_child.dinov3 <timm folder>
"""

import sys
from pathlib import Path

# The two implementations may differ by this much in any token value.
TOLERANCE = 1e-3


def convert(src: dict, reference: dict) -> dict:
    """timm's state dict under transformers' names; ``reference`` gives the shapes."""
    import torch

    out = {
        "embeddings.cls_token": src["cls_token"],
        "embeddings.register_tokens": src["reg_token"],
        # Only read when patches are masked, which inference never does.
        "embeddings.mask_token": torch.zeros_like(reference["embeddings.mask_token"]),
        "embeddings.patch_embeddings.weight": src["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.bias": src["patch_embed.proj.bias"],
        "norm.weight": src["norm.weight"],
        "norm.bias": src["norm.bias"],
    }
    depth = 1 + max(int(k.split(".")[1]) for k in src if k.startswith("blocks."))
    for i in range(depth):
        b, layer = f"blocks.{i}.", f"layer.{i}."
        q, k, v = src[b + "attn.qkv.weight"].chunk(3, dim=0)
        out[layer + "attention.q_proj.weight"] = q
        out[layer + "attention.k_proj.weight"] = k
        out[layer + "attention.v_proj.weight"] = v
        out[layer + "attention.q_proj.bias"] = src[b + "attn.q_bias"]
        out[layer + "attention.v_proj.bias"] = src[b + "attn.v_bias"]
        out[layer + "attention.o_proj.weight"] = src[b + "attn.proj.weight"]
        out[layer + "attention.o_proj.bias"] = src[b + "attn.proj.bias"]
        out[layer + "layer_scale1.lambda1"] = src[b + "gamma_1"]
        out[layer + "layer_scale2.lambda1"] = src[b + "gamma_2"]
        for norm in ("norm1", "norm2"):
            out[layer + f"{norm}.weight"] = src[b + f"{norm}.weight"]
            out[layer + f"{norm}.bias"] = src[b + f"{norm}.bias"]
        out[layer + "mlp.up_proj.weight"] = src[b + "mlp.fc1.weight"]
        out[layer + "mlp.up_proj.bias"] = src[b + "mlp.fc1.bias"]
        out[layer + "mlp.down_proj.weight"] = src[b + "mlp.fc2.weight"]
        out[layer + "mlp.down_proj.bias"] = src[b + "mlp.fc2.bias"]
    return out


def _config():
    from transformers import DINOv3ViTConfig

    return DINOv3ViTConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        patch_size=16,
        num_register_tokens=4,
        image_size=224,
    )


def from_timm(folder: str):
    """transformers' DINOv3ViTModel holding the weights of a timm folder."""
    from safetensors.torch import load_file
    from transformers import DINOv3ViTModel

    model = DINOv3ViTModel(_config())
    src = load_file(str(Path(folder) / "model.safetensors"))
    model.load_state_dict(convert(src, model.state_dict()), strict=True)
    return model.eval()


class TimmDINOv3:
    """Stands where Pixal3D imports ``DINOv3ViTModel``: the one call it makes."""

    @staticmethod
    def from_pretrained(folder: str, *_, **__):
        return from_timm(folder)


def main(argv: list[str]) -> int:
    import timm
    import torch
    import torch.nn.functional as F

    folder = argv[0]
    hf = from_timm(folder)
    tm = timm.create_model(
        "vit_large_patch16_dinov3_qkvb",
        pretrained=True,
        pretrained_cfg_overlay={"file": str(Path(folder) / "model.safetensors")},
        num_classes=0,
        dynamic_img_size=True,
    ).eval()

    def tokens_hf(image):
        # What Pixal3D computes: every layer, then a parameter-free norm.
        hidden = hf.embeddings(image, bool_masked_pos=None)
        position = hf.rope_embeddings(image)
        for layer in hf.layer:
            hidden = layer(hidden, position_embeddings=position)
        return F.layer_norm(hidden, hidden.shape[-1:])

    def tokens_timm(image):
        final, tm.norm = tm.norm, torch.nn.Identity()
        try:
            hidden = tm.forward_features(image)
        finally:
            tm.norm = final
        return F.layer_norm(hidden, hidden.shape[-1:])

    worst = 0.0
    with torch.no_grad():
        for size in (224, 512):
            torch.manual_seed(size)
            image = torch.randn(1, 3, size, size)
            diff = (tokens_hf(image) - tokens_timm(image)).abs().max().item()
            worst = max(worst, diff)
            print(f"dinov3: {size} px, the two implementations differ by at most {diff:.1e}")
    return 0 if worst <= TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
