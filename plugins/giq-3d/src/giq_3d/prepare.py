# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""``giq prepare pixal3d``: what the engine needs that `uv sync` cannot give.

After ``cd envs/pixal3d && uv sync`` the environment has torch and
everything that installs from an index. This adds, each skipped when it is
already there:

  1. Pixal3D and TRELLIS.2 themselves, checkouts at pinned commits (neither
     is a package);
  2. MoGe, the camera estimator, without its dependencies;
  3. the compiled extensions, at pinned commits: CuMesh, FlexGEMM, O-Voxel
     (from TRELLIS.2) and NATTEN. Their own install scripts take whatever
     commit is newest, and FlexGEMM has since moved a module Pixal3D
     imports;
  4. the NAF feature upsampler, into torch.hub's cache.

nvdiffrast and nvdiffrec are deliberately not built: their licence allows
research and evaluation only. giq's own UV rasteriser stands in for the
first; the second only drives a preview renderer giq does not use.

The extensions compile against the environment's torch with the machine's
CUDA toolkit. A toolkit older than the card (Ubuntu 24.04 ships CUDA 12.0,
which does not know an RTX 50) still works, because the build asks for an
older architecture *with PTX*, which the driver compiles for the card at
load time. Verified with CUDA 12.0 and gcc 12 for an RTX 5090.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

PIXAL3D = ("https://github.com/TencentARC/Pixal3D.git", "f7cf38429b0bd264f1995f0f8743a88b1c728b94")
# The commit O-Voxel is built from, checked out for the trellis2 recipes too.
TRELLIS2 = (
    "https://github.com/microsoft/TRELLIS.2.git",
    "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
)
MOGE = "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"
EXTENSIONS = (
    (
        "cumesh",
        "git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f",
    ),
    (
        "flex_gemm",
        "git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e",
    ),
    (
        "o_voxel",
        "git+https://github.com/microsoft/TRELLIS.2.git@75fbf0183001ed9876c8dbb35de6b68552ee08bd"
        "#subdirectory=o-voxel",
    ),
)
NATTEN = (
    "https://files.pythonhosted.org/packages/46/c9/aa6bd171aa23de07f401131ce2aff0d1cfa036e22ad8e7e8faa9d66b8067/natten-0.21.0.tar.gz",
    "810aaf179e1a6fdb0a9279d7858d00412a0d3f045e546cada415205215ccd539",
)
# NATTEN asks CMake for machine code only ("89-real"). Without the suffix
# CMake emits PTX beside it, which is what a card newer than the toolkit runs.
NATTEN_REAL_ONLY = """    return (
        "-real;".join([str(x) if x not in [90, 100] else f"{x}a" for x in arch_list])
        + "-real"
    )"""
NATTEN_WITH_PTX = (
    """    return ";".join([str(x) if x not in [90, 100] else f"{x}a" for x in arch_list])"""
)
# The newest architecture CUDA 12.0 can target; its PTX runs on anything newer.
ARCH = "8.9"


def _say(message: str) -> None:
    print(f"prepare pixal3d: {message}", flush=True)


def _run(command: list[str], env: dict[str, str] | None = None, cwd: str | None = None) -> None:
    subprocess.run(command, check=True, env=env, cwd=cwd)


def _has(python: str, distribution: str) -> bool:
    """Whether a distribution is installed in the engine's environment.

    Asked of the package metadata, not by importing: O-Voxel imports
    nvdiffrast at the top, which is deliberately absent, so an import says
    "missing" for a build that is there and it would be rebuilt every time.
    """
    probe = subprocess.run(
        [python, "-c", "import importlib.metadata as m, sys; m.version(sys.argv[1])", distribution],
        capture_output=True,
    )
    return probe.returncode == 0


def build_env(python: str) -> dict[str, str]:
    """The environment the extensions compile in.

    ``GIQ_PIXAL3D_CUDA_ARCH`` replaces the architecture list (torch's
    TORCH_CUDA_ARCH_LIST form, "12.0" for a toolkit that knows the card),
    and CUDA_HOME, CC and CXX are respected when set.
    """
    env = dict(os.environ)
    site = subprocess.run(
        [
            python,
            "-c",
            "import nvidia.cuda_cccl, os; print(os.path.dirname(nvidia.cuda_cccl.__file__))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    nvcc = shutil.which("nvcc")
    if nvcc is None and "CUDA_HOME" not in env:
        raise RuntimeError("no nvcc on PATH: install a CUDA toolkit, or set CUDA_HOME to one")
    env.setdefault("CUDA_HOME", str(Path(nvcc).resolve().parents[1]) if nvcc else env["CUDA_HOME"])
    # CUDA 12.0 refuses gcc 13; gcc 12 builds for every toolkit this was tried on.
    host = env.get("CXX") or shutil.which("g++-12") or "g++"
    env.setdefault("CC", shutil.which("gcc-12") or "gcc")
    env.setdefault("CXX", host)
    env.setdefault("CUDAHOSTCXX", host)
    arch = env.get("GIQ_PIXAL3D_CUDA_ARCH", ARCH)
    env["TORCH_CUDA_ARCH_LIST"] = arch if "+PTX" in arch or arch != ARCH else f"{arch}+PTX"
    env["NATTEN_CUDA_ARCH"] = arch.replace("+PTX", "")
    # NATTEN's translation units take several gigabytes each to compile:
    # with 12 jobs the build passed 28 GB and was killed. Six fits in about
    # 20 GB; NATTEN_N_WORKERS and MAX_JOBS raise or lower it.
    jobs = str(max(1, min(6, (os.cpu_count() or 4) - 2)))
    env.setdefault("NATTEN_N_WORKERS", jobs)
    env.setdefault("MAX_JOBS", jobs)
    env["NVCC_PREPEND_FLAGS"] = f"-ccbin {host} -I{site}/include " + env.get(
        "NVCC_PREPEND_FLAGS", ""
    )
    return env


def _pip(python: str, *args: str, env: dict[str, str] | None = None) -> None:
    _run(["uv", "pip", "install", "--python", python, *args], env=env)


def _natten(python: str, env: dict[str, str]) -> None:
    url, sha256 = NATTEN
    with tempfile.TemporaryDirectory(prefix="giq-natten-") as folder:
        archive = Path(folder) / "natten.tar.gz"
        urllib.request.urlretrieve(url, archive)  # noqa: S310 — a pinned https URL, checksum below
        if hashlib.sha256(archive.read_bytes()).hexdigest() != sha256:
            raise RuntimeError("the NATTEN source archive does not match its pinned checksum")
        with tarfile.open(archive) as tar:
            tar.extractall(folder, filter="data")
        source = next(p for p in Path(folder).iterdir() if p.is_dir())
        setup = source / "setup.py"
        text = setup.read_text()
        if NATTEN_REAL_ONLY not in text:
            raise RuntimeError("NATTEN's setup.py is not the one this patch was written for")
        setup.write_text(text.replace(NATTEN_REAL_ONLY, NATTEN_WITH_PTX))
        _pip(python, "--no-build-isolation", "--no-deps", str(source), env=env)


def main(argv: list[str]) -> int:
    from giq.engines import require_binary
    from giq_3d.adapter import source_dir

    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    try:
        python = require_binary("pixal3d")
    except Exception as e:  # noqa: BLE001 — say how to get the interpreter
        _say(f"{e}\n  set it up first: cd envs/pixal3d && uv sync")
        return 1

    for pipeline, (url, commit), marker in (
        ("pixal3d", PIXAL3D, "inference.py"),
        ("trellis2", TRELLIS2, "trellis2"),
    ):
        source = Path(source_dir(pipeline))
        if not (source / marker).exists():
            _say(f"cloning {source.name} at {commit[:10]} into {source}")
            _run(["git", "clone", "--quiet", url, str(source)])
            _run(["git", "checkout", "--quiet", commit], cwd=str(source))

    if not _has(python, "moge"):
        _say("installing MoGe")
        _pip(python, "--no-deps", MOGE)

    env = None
    for module, requirement in EXTENSIONS:
        if _has(python, module):
            continue
        env = env or build_env(python)
        arch, host = env["TORCH_CUDA_ARCH_LIST"], env["CXX"]
        _say(f"building {module} (CUDA arch {arch}, host compiler {host})")
        # --no-deps: O-Voxel names CuMesh and FlexGEMM unpinned, and would
        # replace the pinned builds with whatever is newest.
        _pip(python, "--no-build-isolation", "--no-deps", requirement, env=env)
    if not _has(python, "natten"):
        env = env or build_env(python)
        _say("building NATTEN (about half an hour)")
        _natten(python, env)

    _say("caching the NAF upsampler")
    _run(
        [
            python,
            "-c",
            "import torch; "
            "torch.hub.load('valeoai/NAF', 'naf', pretrained=True, device='cpu', trust_repo=True)",
        ]
    )

    _say("done; the recipe's weights come with `giq add pixal3d`")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
