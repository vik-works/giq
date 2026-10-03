# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Depth child process: Depth Anything V2, native transformers.

Invoked by ``DepthAdapter`` via ``python -u -m giq_depth._depth_child
--model depth-anything-v2-small --weights <dir>``. Loads the local snapshot
the recipe names, with the hub disabled before transformers is imported,
so nothing is ever fetched.

Per task: decode the image, run the model at its own working resolution
(the processor scales the long side to 518 and keeps the aspect, a multiple
of 14), bilinearly resample the prediction back to the input size, and
return it as a 16-bit PNG with the range it was scaled from. The model's
output is relative inverse depth — larger is nearer, no unit — and it is
returned as-is; no inversion, no normalisation beyond the 16-bit mapping.

Wire protocol: see giq.adapters._subprocess.
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from giq_child import (  # noqa: E402
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)

# An image bigger than this on its long side is downscaled before the
# processor sees it: the model works at 518 anyway, and the only thing a
# 40-megapixel input costs is the resample back up to it.
MAX_INPUT_SIDE = int(os.environ.get("GIQ_DEPTH_MAX_INPUT_SIDE", "4096"))

# ColorBrewer Spectral, the 11-class ramp matplotlib's `Spectral` is built
# from. The visualisation paints near red and far blue, the convention
# Marigold's and Depth Anything's own demos use.
_SPECTRAL = (
    (0x9E, 0x01, 0x42),
    (0xD5, 0x3E, 0x4F),
    (0xF4, 0x6D, 0x43),
    (0xFD, 0xAE, 0x61),
    (0xFE, 0xE0, 0x8B),
    (0xFF, 0xFF, 0xBF),
    (0xE6, 0xF5, 0x98),
    (0xAB, 0xDD, 0xA4),
    (0x66, 0xC2, 0xA5),
    (0x32, 0x88, 0xBD),
    (0x5E, 0x4F, 0xA2),
)


def _load(path: str):
    import torch
    from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

    if not os.path.isdir(path):
        raise RuntimeError(f"model directory does not exist: {path}")
    proc = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    net = DepthAnythingForDepthEstimation.from_pretrained(
        path, dtype=torch.float32, local_files_only=True
    )
    return proc, net.eval().cuda()


def _decode(b64: str):
    import base64
    import io

    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b64)))).convert("RGB")
    longest = max(img.size)
    if longest > MAX_INPUT_SIDE:
        scale = MAX_INPUT_SIDE / longest
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS
        )
    return img


def _png_b64(array) -> str:
    """uint16 → 16-bit grey PNG, uint8 HxWx3 → RGB; PIL reads the dtype itself."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    # Lossless either way; the default level spends most of a task's wall
    # time deflating a multi-megapixel map for a few percent of size.
    Image.fromarray(array).save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _to_16bit(depth):
    """Prediction → (uint16 map, min, max). A flat prediction maps to zeros."""
    import numpy as np

    lo, hi = float(depth.min()), float(depth.max())
    if hi > lo:
        scaled = (depth - lo) / (hi - lo)
    else:
        scaled = np.zeros_like(depth)
    return (scaled * 65535.0 + 0.5).astype(np.uint16), lo, hi


def visualize(depth, near_is_high: bool = True):
    """Colour-map a prediction: near red, far blue, as an RGB uint8 array."""
    import numpy as np

    lo, hi = float(depth.min()), float(depth.max())
    t = (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)
    if near_is_high:
        t = 1.0 - t  # Spectral runs red → blue, so near (high) must map to 0
    lut = np.asarray(_SPECTRAL, dtype=np.float32)
    x = np.linspace(0.0, 1.0, len(lut))
    rgb = np.stack([np.interp(t, x, lut[:, c]) for c in range(3)], axis=-1)
    return (rgb + 0.5).astype(np.uint8)


def _run(models, task: dict) -> dict:
    import torch

    proc, net = models
    if not task.get("image_b64"):
        raise ValueError("task needs image_b64")
    image = _decode(task["image_b64"])
    inputs = proc(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = net(**inputs)
    (post,) = proc.post_process_depth_estimation(out, target_sizes=[image.size[::-1]])
    depth = post["predicted_depth"].float().cpu().numpy()
    map16, lo, hi = _to_16bit(depth)
    result = {
        "id": task["id"],
        "depth_b64": _png_b64(map16),
        "width": image.width,
        "height": image.height,
        "depth_min": lo,
        "depth_max": hi,
        "metric": False,
    }
    if task.get("visualize"):
        result["visualization_b64"] = _png_b64(visualize(depth))
    return result


def main() -> None:
    import argparse

    reserve_ipc_stdout()
    parser = argparse.ArgumentParser()
    # --model names the recipe for log lines; --weights is what loads.
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    try:
        models = _load(args.weights)
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
