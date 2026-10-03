# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""One transfer of a fetch (ADR-005 D4), in a process of its own.

Run as ``python -m giq._fetch_child '<json>'`` by :mod:`giq.fetch`, which
cancels by ending this process. Partial files stay where the Hub library
keeps them, so the next fetch resumes rather than starts over.

Progress is the Hub library's own: its byte counters, reported as JSON
lines on stdout (``{"bytes": n}``) when the spec asks for them. The files
themselves are no gauge: the Hub's Xet transfers write a file in one go
once all of it has arrived.

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
import threading
import time
from pathlib import Path
from typing import Any

_SHARD_RE = re.compile(r"^(.*)-00001-of-(\d{5})\.gguf$")


def partial_dir(dest: Path) -> Path:
    """Where a transfer to ``dest`` downloads before it is moved into place."""
    return dest.parent / f".{dest.name}.partial"


def _reporting_bars() -> Any:
    """A tqdm class that counts the bytes the Hub library downloads and
    prints them as JSON lines, at most twice a second; the bars themselves
    are drawn nowhere."""
    from tqdm.auto import tqdm

    bars: list[Any] = []
    lock = threading.Lock()

    def received() -> int:
        with lock:
            # Xet transfers report "downloading bytes" next to a
            # "reconstructing" bar over the same bytes; plain HTTP has one
            # bar per file.
            fetched = [b for b in bars if "downloading bytes" in str(b.desc).lower()]
            return int(sum(b.n for b in (fetched or [b for b in bars if b.unit == "B"])))

    def report() -> None:
        last = -1
        while True:
            time.sleep(0.5)
            now = received()
            if now != last:
                print(json.dumps({"bytes": now}), flush=True)
                last = now

    threading.Thread(target=report, daemon=True).start()
    devnull = open(os.devnull, "w")  # noqa: SIM115 (lives as long as the process)

    class Reporting(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["file"] = devnull
            super().__init__(*args, **kwargs)
            with lock:
                bars.append(self)

    return Reporting


def _with_shards(repo: str, revision: str | None, file: str) -> list[str]:
    """``file``, or every shard of the GGUF whose first shard it is."""
    m = _SHARD_RE.match(file)
    if m is None:
        return [file]
    from huggingface_hub import list_repo_files

    shard = re.compile(rf"^{re.escape(m.group(1))}-\d{{5}}-of-{m.group(2)}\.gguf$")
    return sorted(f for f in list_repo_files(repo, revision=revision) if shard.match(f))


def transfer(
    repo: str, revision: str | None, file: str | None, dest: str | None, report: bool = False
) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    # Without a report, the Hub library draws its own bars (`giq add`).
    bars = {"tqdm_class": _reporting_bars()} if report else {}
    if dest is None:
        # A repository the children load from the Hugging Face cache.
        snapshot_download(repo, revision=revision, **bars)
        return
    target = Path(dest)
    partial = partial_dir(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if file is None:
        snapshot_download(repo, revision=revision, local_dir=partial, **bars)
        os.replace(partial, target)
        return
    files = _with_shards(repo, revision, file)
    for name in files:
        hf_hub_download(repo, name, revision=revision, local_dir=partial, **bars)
    for name in files:
        os.replace(partial / name, target if name == file else target.parent / Path(name).name)
    shutil.rmtree(partial, ignore_errors=True)


def main() -> int:
    spec = json.loads(sys.argv[1])
    try:
        transfer(
            spec["repo"],
            spec.get("revision"),
            spec.get("file"),
            spec.get("dest"),
            report=bool(spec.get("report")),
        )
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
