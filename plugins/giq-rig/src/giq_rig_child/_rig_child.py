# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-to-rig child process: SkinTokens' TokenRig, and the transfer of what
it makes back onto the mesh it was given.

Invoked by ``RigAdapter`` via ``<skintokens python> -u -m
giq_rig_child._rig_child --model <recipe> --weights <checkpoint root>
--checkpoint <path in it> --llm <config.json> --source <SkinTokens
checkout>``. Loads TokenRig, the skin tokenizer it names and the mesh
transform once.

SkinTokens is a checkout, not a package: its code is the package ``src``
of ``--source``, and it reads its skeleton definitions from ``./configs``,
so the child works from that directory. Two of the checkout's relative
paths are pointed at the recipe's weights before anything loads: the skin
tokenizer, which the checkpoint names relative to the repository it came
from (``weights.path`` is a snapshot of that repository), and the
transformer's config, which upstream expects beside the code
(``weights.parts.llm``). The hub is offline, so anything missing fails here
rather than being fetched.

Upstream's command line runs Blender's bpy in a server process of its own
on a fixed local port, all interfaces, and posts pickles to it: a
workaround for data-loader worker threads. Here the mesh is read on the
main thread with no workers, so bpy runs in this process and nothing
listens anywhere. Upstream also leaves the model in training mode, which
draws the shape encoder's point sample from an unseeded generator; in eval
mode that sample is fixed, so a task's ``seed`` alone picks the rig.

Per task: the GLB is read with bpy, TokenRig generates the skeleton and
the skin tokens in one sequence by beam search with sampling, the tokens
are decoded to per-vertex weights, and the skeleton and weights are carried
over onto the GLB as given (upstream's ``--use_transfer``): the same
meshes, materials, textures and scale, now bound to one armature, four
weights a vertex. ``keep_skeleton`` skins a GLB to the armature it already
has; ``voxel_skin`` trims weights that bleed across parts of the mesh that
do not touch (upstream's ``--use_postprocess``).

Wire protocol: see giq_child (``run_ipc_child_loop``). This module is a
top-level package importing no giq: the skintokens interpreter carries
torch, flash-attn and bpy, never giq's venv. The parsing below is pure, so
giq's suite tests it without SkinTokens.
"""

import base64
import binascii
import os
import sys
import tempfile
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from giq_child import (  # noqa: E402
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)

# A GLB starts with this; glTF as JSON with separate files cannot travel as
# one payload.
GLB_MAGIC = b"glTF"
MAX_MESH_BYTES = 64 * 1024 * 1024
SEED = 42
MAX_SEED = 2**32 - 1
# The sampler's settings a task may change: (default, lowest, highest).
# Defaults and ranges are upstream's own (its demo's sliders).
SAMPLING = {
    "top_k": (5, 1, 200),
    "top_p": (0.95, 0.1, 1.0),
    "temperature": (1.0, 0.1, 2.0),
    "repetition_penalty": (2.0, 0.5, 3.0),
    "num_beams": (10, 1, 20),
}
WHOLE = frozenset({"top_k", "num_beams"})
# The longest sequence TokenRig may generate, skeleton and skin together.
MAX_LENGTH = 2048
# The class token a mesh is generated under: the dataset the recommended
# checkpoint was refined on, as upstream's command line asks for it.
CLASS = "articulation"
# Bone influences a vertex keeps in the export, as upstream writes them;
# glTF's own skins carry four a vertex.
INFLUENCES = 4
# The voxel grid `voxel_skin` finds connected parts of the mesh on.
VOXEL_RESOLUTION = 196


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"'{name}' must be a whole number")
    return int(value)


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number")
    return float(value)


def _flag(task: dict, name: str) -> bool:
    value = task.get(name, False)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be true or false")
    return value


def _mesh(value) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("task needs a 'glb_b64': the mesh as a base64 GLB")
    try:
        data = base64.b64decode(value.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("'glb_b64' is not base64") from None
    if len(data) > MAX_MESH_BYTES:
        raise ValueError(f"'glb_b64' is larger than {MAX_MESH_BYTES // 2**20} MB")
    if data[:4] != GLB_MAGIC:
        raise ValueError("'glb_b64' is not a GLB (binary glTF)")
    return data


def task_inputs(task: dict) -> dict:
    """A task's request, checked. Raises ValueError naming the field that is wrong."""
    mesh = _mesh(task.get("glb_b64"))
    seed = task.get("seed")
    seed = SEED if seed is None else _integer(seed, "seed")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"'seed' must be 0 to {MAX_SEED}")
    sampling = {}
    for name, (default, low, high) in SAMPLING.items():
        value = task.get(name)
        if value is None:
            value = default
        value = _integer(value, name) if name in WHOLE else _number(value, name)
        if not low <= value <= high:
            raise ValueError(f"'{name}' must be {low}-{high}")
        sampling[name] = value
    return {
        "mesh": mesh,
        "seed": seed,
        "sampling": sampling,
        "keep_skeleton": _flag(task, "keep_skeleton"),
        "voxel_skin": _flag(task, "voxel_skin"),
    }


def _load(args):
    """TokenRig on the card, and what turns a GLB into its input."""
    import importlib
    from pathlib import Path
    from types import SimpleNamespace

    # The child changes directory below; every path it was given is fixed first.
    weights, llm, source = (os.path.abspath(p) for p in (args.weights, args.llm, args.source))
    checkpoint = os.path.join(weights, args.checkpoint)
    if not os.path.isfile(os.path.join(source, "demo.py")):
        raise RuntimeError(f"no SkinTokens checkout at {source}: run `giq prepare skintokens`")
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"no TokenRig checkpoint at {checkpoint}")
    if not os.path.isfile(llm):
        raise RuntimeError(f"no transformer config at {llm} (weights.parts.llm)")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SkinTokens needs a CUDA device: its attention is flash-attn's")

    sys.path.insert(0, source)
    os.chdir(source)
    tokenrig: Any = importlib.import_module("src.model.tokenrig")
    tokenrig.LLM_LOCAL_DIR = Path(os.path.dirname(llm))
    load_vae = tokenrig.SkinVAEModel.load_from_system_checkpoint
    tokenrig.SkinVAEModel.load_from_system_checkpoint = staticmethod(
        lambda checkpoint_path, **kw: load_vae(os.path.join(weights, checkpoint_path), **kw)
    )
    model = importlib.import_module("src.server.spec").get_model(checkpoint, hf_path=None)
    model.eval()
    tokenizer = importlib.import_module("src.tokenizer.parse").get_tokenizer(
        **model.tokenizer_config
    )
    transform = importlib.import_module("src.data.transform").Transform.parse(
        **model.transform_config["predict_transform"]
    )
    return SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        dataset=importlib.import_module("src.data.dataset"),
        voxel_skin=importlib.import_module("src.data.vertex_group").voxel_skin,
        # Imports bpy: a missing one fails the start, not the first task.
        transfer=importlib.import_module("src.rig_package.parser.bpy").transfer_rigging,
    )


def _predict(engine, path: str, inputs: dict):
    """The rig TokenRig generates for the mesh at ``path``, as upstream's Asset."""
    import torch

    dataset = engine.dataset
    config = dataset.DatasetConfig.parse(
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        datapath={"data_name": None, "loader": "bpy", "filepaths": {CLASS: [path]}},
    ).split_by_cls()
    module = dataset.RigDatasetModule(
        predict_dataset_config=config,
        predict_transform=engine.transform,
        tokenizer=engine.tokenizer,
        process_fn=engine.model._process_fn,
    )
    batch = next(iter(module.predict_dataloader()[CLASS]))
    batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    skeleton = None
    if inputs["keep_skeleton"]:
        tokens, mask = batch.get("skeleton_tokens"), batch.get("skeleton_mask")
        if tokens is None or mask is None or not bool((mask[0] == 1).any()):
            raise ValueError("'keep_skeleton' needs a GLB that carries a skeleton")
        skeleton = [tokens[0][mask[0] == 1].cpu().numpy()]
    else:
        batch.pop("skeleton_tokens", None)
        batch.pop("skeleton_mask", None)
    batch["generate_kwargs"] = {
        "max_length": MAX_LENGTH,
        "num_return_sequences": 1,
        "do_sample": True,
        **inputs["sampling"],
    }
    with torch.no_grad():
        results = engine.model.predict_step(batch, skeleton_tokens=skeleton, make_asset=True)
    asset = results["results"][0].asset
    if asset is None:
        raise RuntimeError("TokenRig returned no rig")
    return asset


def _run(engine, task: dict) -> dict:
    import random

    import numpy as np
    import torch

    inputs = task_inputs(task)
    seed = inputs["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory(prefix="giq-rig-") as folder:
        mesh, rigged = os.path.join(folder, "mesh.glb"), os.path.join(folder, "rigged.glb")
        with open(mesh, "wb") as f:
            f.write(inputs["mesh"])
        asset = _predict(engine, mesh, inputs)
        if inputs["voxel_skin"]:
            voxel = asset.voxel(resolution=VOXEL_RESOLUTION)
            asset.skin *= engine.voxel_skin(
                grid=0,
                grid_coords=voxel.coords,
                joints=asset.joints,
                vertices=asset.vertices,
                faces=asset.faces,
                mode="square",
                voxel_size=voxel.voxel_size,
            )
            asset.normalize_skin()
        engine.transfer(
            source_asset=asset, target_path=mesh, export_path=rigged, group_per_vertex=INFLUENCES
        )
        if not os.path.isfile(rigged):
            raise RuntimeError("the transfer onto the mesh wrote no GLB")
        with open(rigged, "rb") as f:
            data = f.read()
    torch.cuda.empty_cache()
    return {
        "id": task.get("id", "unknown"),
        "glb_b64": base64.b64encode(data).decode("ascii"),
        "format": "glb",
        "bones": int(len(asset.joints)),
        "seed": seed,
    }


def main() -> None:
    import argparse

    reserve_ipc_stdout()
    parser = argparse.ArgumentParser()
    # --model names the recipe for log lines; the rest is what loads.
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    try:
        engine = _load(args)
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_run(engine, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
