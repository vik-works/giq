# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Shape, then paint, on real hardware: times and VRAM peaks against the one-job path.

A client of a running giq, not a test: it loads TRELLIS.2 and keeps the card
busy for some minutes, so it runs only in a window set aside for it, against
a giq that is not serving anyone (AGENTS.md: no GPU work unasked, no second
giq beside a live one). Run it from a checkout of the branch, against a giq
started from that checkout:

    uv run python plugins/giq-3d/scripts/smoke_shape_paint.py \\
        --url http://127.0.0.1:8085 --image character.png --out /tmp/shape-paint

One picture, one seed, one job at a time:

  1. ``shape-full``    image2mesh, ``texture: false``: the generated mesh, and its latent
  2. ``shape-game``    the same with ``mode: game``: the cage at ``--faces``
  3. ``paint-given``   mesh2texture on the cage, handed the latent back
  4. ``paint-encoded`` mesh2texture on the cage, without it (the mesh encoded)
  5. ``paint-decimated`` mesh2texture on a coarser copy of the full mesh, with the latent
  6. ``one-job``       the default image2mesh task: shape, texture and game export at once

The first job also loads the model unless the instance is up already; the
table says which. VRAM is the child's own peak, sampled from ``nvidia-smi``
every 100 ms, as the recipes' figures are. Every paint is checked for the
vertices and faces it was sent (needs trimesh, which the dev group has).
Everything lands in ``--out``: the GLBs, ``summary.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np


def call(url: str, path: str, body: dict | None = None, timeout: float = 1200.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Peaks(threading.Thread):
    """The most VRAM each process on one card held, sampled every 100 ms."""

    def __init__(self, gpu: str):
        super().__init__(daemon=True)
        self.gpu, self.peaks, self.running = gpu, {}, True

    def run(self) -> None:
        command = ["nvidia-smi", "-i", self.gpu, "--query-compute-apps=pid,used_memory"]
        command += ["--format=csv,noheader,nounits"]
        while self.running:
            try:
                out = subprocess.run(command, capture_output=True, text=True, timeout=5).stdout
            except (OSError, subprocess.TimeoutExpired):
                out = ""
            for line in out.strip().splitlines():
                pid, _, mib = line.partition(",")
                if pid.strip().isdigit() and mib.strip().isdigit():
                    p = int(pid)
                    self.peaks[p] = max(self.peaks.get(p, 0), int(mib))
            time.sleep(0.1)

    def take(self) -> dict[int, int]:
        peaks, self.peaks = self.peaks, {}
        return peaks


def child_pid(url: str, recipe: str) -> int | None:
    for inst in call(url, "/instances").get("instances", []):
        if inst.get("recipe") == recipe and inst.get("pid"):
            return int(inst["pid"])
    return None


def decimated(glb: bytes, cells: int) -> bytes:
    """A coarser mesh of another topology: vertices clustered on a grid of
    ``cells`` a side, the faces that collapse dropped."""
    import trimesh

    mesh = trimesh.load(io.BytesIO(glb), file_type="glb", process=False).to_geometry()
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    low, size = v.min(0), (v.max(0) - v.min(0)).max()
    key = np.floor((v - low) / size * (cells - 1e-6)).astype(np.int64)
    cell, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    centres = np.zeros((len(cell), 3))
    np.add.at(centres, inverse, v)
    centres /= np.bincount(inverse, minlength=len(cell))[:, None]
    faces = inverse[f]
    faces = faces[
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    ]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return trimesh.Trimesh(centres, faces[np.sort(first)], process=False).export(file_type="glb")


def same_mesh(sent: bytes, got: bytes, vertex_map: list[int] | None) -> str:
    """Whether a paint gave the mesh back as it was sent."""
    try:
        import trimesh
    except ImportError:
        return "unchecked (no trimesh)"

    def arrays(data: bytes):
        g = trimesh.load(io.BytesIO(data), file_type="glb", process=False).to_geometry()
        return np.asarray(g.vertices), np.asarray(g.faces)

    (sv, sf), (gv, gf) = arrays(sent), arrays(got)
    if vertex_map is not None:
        m = np.asarray(vertex_map)
        ok = np.array_equal(gv, sv[m]) and np.array_equal(m[gf], sf)
        return "same (unwrapped)" if ok else "DIFFERENT"
    return "same" if np.array_equal(gv, sv) and np.array_equal(gf, sf) else "DIFFERENT"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8085")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="trellis2")
    parser.add_argument("--gpu", default="0", help="the card the recipe is bound to")
    parser.add_argument("--faces", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cells", type=int, default=48, help="the decimated copy's grid")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    image = base64.b64encode(args.image.read_bytes()).decode()
    loaded_before = child_pid(args.url, args.model) is not None

    peaks = Peaks(args.gpu)
    peaks.start()
    rows: list[dict] = []

    def job(step: str, modality: str, task: dict) -> dict:
        peaks.take()
        start = time.monotonic()
        response = call(
            args.url,
            "/run?wait=true",
            {"modality": modality, "model": args.model, "tasks": [{"id": step, **task}]},
        )
        seconds = time.monotonic() - start
        pid = child_pid(args.url, args.model)
        sampled = peaks.take()
        result = (response.get("results") or [{}])[0]
        if response.get("status") != "completed" or result.get("error"):
            # A failed job's reason is in giq's log; a task's is in its result.
            raise SystemExit(f"{step}: {response.get('status')}: {result.get('error') or ''}")
        glb = base64.b64decode(result["glb_b64"])
        (args.out / f"{step}.glb").write_bytes(glb)
        row = {
            "step": step,
            "seconds": round(seconds, 1),
            "vram_mib": sampled.get(pid) if pid else None,
            "faces": result.get("faces"),
            "vertices": result.get("vertices"),
            "glb_mb": round(len(glb) / 2**20, 2),
            "loaded": not rows and not loaded_before,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        return result

    common = {"image_b64": image, "seed": args.seed}
    full = job("shape-full", "image2mesh", {**common, "texture": False})
    shape = full["shape_b64"]
    (args.out / "shape.b64").write_text(shape)
    game = job(
        "shape-game",
        "image2mesh",
        {**common, "texture": False, "mode": "game", "faces": args.faces},
    )
    rows[-1]["check"] = "same latent" if game["shape_b64"] == shape else "latent differs"
    cage = base64.b64decode(game["glb_b64"])

    for step, mesh, given in (
        ("paint-given", cage, True),
        ("paint-encoded", cage, False),
        ("paint-decimated", decimated(base64.b64decode(full["glb_b64"]), args.cells), True),
    ):
        task = {**common, "mesh_b64": base64.b64encode(mesh).decode()}
        if given:
            task["shape_b64"] = shape
        result = job(step, "mesh2texture", task)
        rows[-1]["check"] = same_mesh(
            mesh, base64.b64decode(result["glb_b64"]), result.get("vertex_map")
        )
        rows[-1]["uv"] = result.get("uv")

    job("one-job", "image2mesh", {**common, "faces": args.faces})
    peaks.running = False

    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    head = "| step | s | VRAM MiB | faces | vertices | GLB MB | check |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    for r in rows:
        step = r["step"] + (" (with load)" if r["loaded"] else "")
        lines.append(
            f"| {step} | {r['seconds']} | {r['vram_mib']} | {r['faces']} | {r['vertices']} "
            f"| {r['glb_mb']} | {r.get('check', '')} |"
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
