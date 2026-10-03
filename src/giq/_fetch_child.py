# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""One transfer of a fetch (ADR-005 D4), in a process of its own.

Run as ``python -m giq._fetch_child '<json>'`` by :mod:`giq.fetch`, which
measures progress by watching the files grow and cancels by ending this
process. Partial files stay where the Hub library keeps them, so the next
fetch resumes rather than starts over.

Files land under a hidden ``.<name>.partial`` sibling of their destination
and are moved into place when every one of them is there: a recipe never
sees half a checkpoint as installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

_SHARD_RE = re.compile(r"^(.*)-00001-of-(\d{5})\.gguf$")


def partial_dir(dest: Path) -> Path:
    """Where a transfer to ``dest`` downloads before it is moved into place."""
    return dest.parent / f".{dest.name}.partial"


def _with_shards(repo: str, revision: str | None, file: str) -> list[str]:
    """``file``, or every shard of the GGUF whose first shard it is."""
    m = _SHARD_RE.match(file)
    if m is None:
        return [file]
    from huggingface_hub import list_repo_files

    shard = re.compile(rf"^{re.escape(m.group(1))}-\d{{5}}-of-{m.group(2)}\.gguf$")
    return sorted(f for f in list_repo_files(repo, revision=revision) if shard.match(f))


def transfer(repo: str, revision: str | None, file: str | None, dest: str | None) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    if dest is None:
        # A repository the children load from the Hugging Face cache.
        snapshot_download(repo, revision=revision)
        return
    target = Path(dest)
    partial = partial_dir(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if file is None:
        snapshot_download(repo, revision=revision, local_dir=partial)
        os.replace(partial, target)
        return
    files = _with_shards(repo, revision, file)
    for name in files:
        hf_hub_download(repo, name, revision=revision, local_dir=partial)
    for name in files:
        os.replace(partial / name, target if name == file else target.parent / Path(name).name)
    shutil.rmtree(partial, ignore_errors=True)


def main() -> int:
    spec = json.loads(sys.argv[1])
    try:
        transfer(spec["repo"], spec.get("revision"), spec.get("file"), spec.get("dest"))
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
