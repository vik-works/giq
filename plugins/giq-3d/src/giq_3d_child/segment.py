# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Part segmentation of a generated asset, by a generator taught to paint parts.

SegviGen (Li et al., arXiv:2603.16869, MIT) is TRELLIS.2's texture flow
model fine-tuned to do one thing: given an asset's shape and its real
texture, draw a second texture that is one flat colour a part. Nothing of
it is its own architecture, so it runs on the TRELLIS.2 code and decoders
the engine has loaded already; what is here is its sampling loop and the
way it lays the asset out for the model, after the project's
``inference_full.py``.

It is a sampler: asked twice it draws two different cuts, one parting an
arm from a cab, another the cab from its tracks. That is used, not fought:
``paint`` is called once a seed and the cuts are laid over one another
(``parts.overlay``).

The asset goes in as the stock export's textured mesh, voxelised at the
512 grid the model was trained at; the colours come back on voxels of that
grid, in the mesh's own frame moved to the middle and scaled to fill it.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

GRID = 512
# The base model the checkpoint was tuned from, and its companions.
FLOW = "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16"
SHAPE_ENCODER = "ckpts/shape_enc_next_dc_f16c32_fp16"
TEX_ENCODER = "ckpts/tex_enc_next_dc_f16c32_fp16"


class _Painter(nn.Module):
    """The flow model, shown the asset's own texture beside the one it draws.

    The two are one sequence of twice the length: the noise being turned
    into part colours, then the real texture, each over the same shape.
    Only the first half is the answer.
    """

    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x_t, texture, shape, t, cond):
        import trellis2.modules.sparse as sp

        both = sp.SparseTensor(
            torch.cat([x_t.feats, texture.feats]), torch.cat([x_t.coords, texture.coords])
        )
        twice = sp.SparseTensor(
            torch.cat([shape.feats, shape.feats]), torch.cat([shape.coords, shape.coords])
        )
        out = self.flow_model(both, t, cond, twice)
        n = x_t.feats.shape[0]
        return sp.SparseTensor(out.feats[:n], out.coords[:n])


class Segmenter:
    def __init__(self, weights: str, checkpoint: str, models, decoders: dict):
        """``weights`` is the TRELLIS.2 folder, ``checkpoint`` SegviGen's
        ``full_seg.ckpt``; ``models`` is ``trellis2.models`` and ``decoders``
        the pipeline's shape and texture decoders, which are shared."""
        with open(os.path.join(weights, "pipeline.json")) as f:
            self.args = json.load(f)["args"]
        self.painter = _Painter(models.from_pretrained(os.path.join(weights, FLOW)))
        state = torch.load(checkpoint, map_location="cpu")["state_dict"]
        self.painter.load_state_dict(
            OrderedDict((k.replace("gen3dseg.", ""), v) for k, v in state.items())
        )
        del state
        self.painter.eval()
        self.shape_encoder = models.from_pretrained(os.path.join(weights, SHAPE_ENCODER)).eval()
        self.tex_encoder = models.from_pretrained(os.path.join(weights, TEX_ENCODER)).eval()
        self.shape_decoder, self.tex_decoder = decoders["shape"], decoders["tex"]

    def _modules(self):
        return (
            self.painter,
            self.shape_encoder,
            self.tex_encoder,
            self.shape_decoder,
            self.tex_decoder,
        )

    def to(self, device: str) -> None:
        for module in self._modules():
            module.to(device)

    @torch.no_grad()
    def lay_out(self, asset):
        """A textured mesh as the model takes it: (shape, texture, guides), and its frame.

        The frame is (centre, scale): a point p of the mesh is at voxel
        ((p - centre) * scale + 0.5) * GRID.
        """
        import o_voxel
        import trellis2.modules.sparse as sp

        low, high = asset.bounding_box.bounds
        centre, scale = (low + high) / 2, 0.99999 / float((high - low).max())
        asset = asset.copy()
        asset.apply_translation(-centre)
        asset.apply_scale(scale)
        mesh = asset.to_mesh()
        box = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        where, dual, crossed = o_voxel.convert.mesh_to_flexible_dual_grid(
            torch.from_numpy(mesh.vertices).float(),
            torch.from_numpy(mesh.faces).long(),
            grid_size=GRID,
            aabb=box,
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=False,
        )
        order = torch.argsort(o_voxel.serialize.encode_seq(where))
        where, dual, crossed = where[order], dual[order], crossed[order]
        painted_where, attrs = o_voxel.convert.textured_mesh_to_volumetric_attr(
            asset, grid_size=GRID, aabb=box, timing=False
        )
        order = torch.argsort(o_voxel.serialize.encode_seq(painted_where))
        attrs = {k: v[order] for k, v in attrs.items()}

        # As the model's training data was stored: eight bits a value.
        dual = (torch.clamp(dual * GRID - where, 0, 1) * 255).round() / 255
        coords = torch.cat(
            [torch.zeros(len(where), 1, dtype=torch.int32), where.int()], dim=1
        ).cuda()
        shape = self.shape_encoder(
            sp.SparseTensor(dual.float().cuda(), coords),
            sp.SparseTensor(crossed.float().cuda(), coords),
        )
        shape = sp.SparseTensor(shape.feats.cuda(), shape.coords.cuda())
        self.shape_decoder.set_resolution(GRID)
        _, guides = self.shape_decoder(shape, return_subs=True)
        colour = torch.cat(
            [attrs[k].float() / 255 for k in ("base_color", "metallic", "roughness", "alpha")],
            dim=-1,
        )
        texture = self.tex_encoder(sp.SparseTensor(colour.cuda() * 2 - 1, coords))
        return (shape, texture, guides), (centre, scale)

    @torch.no_grad()
    def paint(self, laid_out, cond: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """One cut: voxels of the grid, and the part colour (0..1) the model gave each."""
        import trellis2.modules.sparse as sp

        shape, texture, guides = laid_out
        dev = shape.feats.device

        def scaled(slat, key):
            std = torch.tensor(self.args[key]["std"])[None].to(dev)
            mean = torch.tensor(self.args[key]["mean"])[None].to(dev)
            return (slat - mean) / std, std, mean

        shape_n, _, _ = scaled(shape, "shape_slat_normalization")
        texture_n, std, mean = scaled(texture, "tex_slat_normalization")
        generator = torch.Generator(device=dev).manual_seed(int(seed))
        noise = torch.randn(texture_n.feats.shape, generator=generator, device=dev)
        sample = sp.SparseTensor(noise.to(texture_n.feats.dtype), shape_n.coords)

        params = self.args["tex_slat_sampler"]["params"]
        steps, rescale = params["steps"], params["rescale_t"]
        strength, (first, last) = params["guidance_strength"], params["guidance_interval"]
        times = np.linspace(1, 0, steps + 1)
        times = (rescale * times / (1 + (rescale - 1) * times)).tolist()
        for t, t_next in zip(times[:-1], times[1:], strict=True):
            at = torch.tensor([t * 1000] * sample.shape[0], dtype=torch.float32, device=dev)
            velocity = self.painter(sample, texture_n, shape_n, at, cond["cond"])
            if first <= t <= last and strength != 1:
                without = self.painter(sample, texture_n, shape_n, at, cond["neg_cond"])
                velocity = strength * velocity + (1 - strength) * without
            sample = sample - (t - t_next) * velocity
        voxels = self.tex_decoder(sample * std + mean, guide_subs=guides) * 0.5 + 0.5
        return (
            voxels.coords[:, 1:].cpu().numpy().astype(np.int32),
            voxels.feats[:, :3].float().clamp(0, 1).cpu().numpy(),
        )


def condition(image_cond_model, image) -> dict:
    """The picture of the asset as the model's condition, as SegviGen reads it."""
    image_cond_model.image_size = 512
    cond = image_cond_model([image])
    return {"cond": cond, "neg_cond": torch.zeros_like(cond)}
