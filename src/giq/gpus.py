# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Per-GPU telemetry: identity, VRAM, thermals, power, throttle state.

Read-only companion to vram.py. vram.py answers "can this model load?" on one
card for the scheduler (that card comes from here); this module answers
"which cards exist and how do they feel?" for the /gpus endpoint, the
stats sampler, and the dashboard. Cards are identified by their NVML UUID
(GPU-xxxx…), which is stable across reboots and PCI re-enumeration — the
anchor for per-device pinning: ``device_env`` hands children a
CUDA_VISIBLE_DEVICES set from one, and ``selected_device`` resolves the card
giq runs models on (config ``gpu.device``, by index or UUID).

Per-model binding to a card lives in ``giq.policy``. What this module
provides is the per-card view: telemetry, the VRAM gate and the eviction
planner each have to name which card they mean, or on a multi-GPU machine
they silently describe different ones.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from giq.paths import cache_env

logger = logging.getLogger(__name__)

# nvidia-smi clocks_throttle_reasons.active bitmask → (label, severity).
# severity: "info" is normal operation (idle clocks), "warning" is a soft cap,
# "critical" means the card is actively being slowed down (thermals / power
# brake) — the "why is this render slow?" signal.
_THROTTLE_BITS: tuple[tuple[int, str, str], ...] = (
    (0x1, "idle clocks", "info"),
    (0x2, "app clock limit", "info"),
    # Not a fault: the driver trimming clocks to hold the power limit the
    # operator configured (nvidia-smi -pl). A card at its limit under load is
    # the cap working, and it is on for most of any sustained job — as a
    # warning chip it was permanent decoration. The limit itself is already
    # on screen next to the draw.
    (0x4, "sw power cap", "info"),
    (0x8, "hw slowdown", "critical"),
    (0x10, "sync boost", "info"),
    (0x20, "sw thermal", "warning"),
    (0x40, "hw thermal", "critical"),
    (0x80, "power brake", "critical"),
)

_FULL_QUERY = (
    "uuid,index,name,memory.total,memory.used,temperature.gpu,"
    "power.draw,power.limit,utilization.gpu,fan.speed,"
    "clocks_throttle_reasons.active"
)
# Older drivers may not support every telemetry column; identity + memory
# always work (vram.py has queried them for the service's whole life).
_MINIMAL_QUERY = "uuid,index,name,memory.total,memory.used"

_CACHE_TTL_SECONDS = 2.0


@dataclass
class GpuTelemetry:
    """One GPU's identity and current readings.

    Telemetry fields are None when the driver doesn't report them ([N/A]);
    identity and VRAM are always present.
    """

    uuid: str
    index: int
    name: str
    vram_total_gb: float
    vram_used_gb: float
    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    utilization_pct: float | None = None
    fan_pct: float | None = None
    throttle: list[dict[str, str]] = field(default_factory=list)

    @property
    def vram_free_gb(self) -> float:
        return self.vram_total_gb - self.vram_used_gb

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "index": self.index,
            "name": self.name,
            "vram_total_gb": round(self.vram_total_gb, 2),
            "vram_used_gb": round(self.vram_used_gb, 2),
            "vram_free_gb": round(self.vram_free_gb, 2),
            "temperature_c": self.temperature_c,
            "power_draw_w": self.power_draw_w,
            "power_limit_w": self.power_limit_w,
            "utilization_pct": self.utilization_pct,
            "fan_pct": self.fan_pct,
            "throttle": self.throttle,
        }


def _num(value: str) -> float | None:
    """Parse an nvidia-smi numeric field; [N/A]/[Not Supported] → None."""
    value = value.strip()
    if not value or value.startswith("[") or value.upper() in ("N/A", "NOT SUPPORTED"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def decode_throttle(value: str) -> list[dict[str, str]]:
    """Decode the clocks_throttle_reasons.active hex bitmask."""
    value = value.strip()
    try:
        mask = int(value, 16)
    except ValueError:
        return []
    return [
        {"reason": label, "severity": severity}
        for bit, label, severity in _THROTTLE_BITS
        if mask & bit
    ]


def _run_query(fields: str) -> list[str] | None:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.strip().split("\n") if line.strip()]


def _parse_line(line: str) -> GpuTelemetry | None:
    parts = [p.strip() for p in line.split(", ")]
    # Names can contain commas in principle; the ", " split plus known column
    # count keeps this safe for every shipping GeForce/Quadro name.
    if len(parts) < 5:
        return None
    try:
        index = int(parts[1])
        total_mb = float(parts[3])
        used_mb = float(parts[4])
    except ValueError:
        return None
    gpu = GpuTelemetry(
        uuid=parts[0],
        index=index,
        name=parts[2],
        vram_total_gb=total_mb / 1024,
        vram_used_gb=used_mb / 1024,
    )
    if len(parts) >= 11:
        gpu.temperature_c = _num(parts[5])
        gpu.power_draw_w = _num(parts[6])
        gpu.power_limit_w = _num(parts[7])
        gpu.utilization_pct = _num(parts[8])
        gpu.fan_pct = _num(parts[9])
        gpu.throttle = decode_throttle(parts[10])
    return gpu


_cache_lock = threading.Lock()
_cache: tuple[float, list[GpuTelemetry]] | None = None


def get_gpus(max_age: float = _CACHE_TTL_SECONDS) -> list[GpuTelemetry]:
    """All GPUs with current telemetry, ordered by index.

    Results are cached briefly (default 2s): the dashboard and /status poll
    every few seconds and nvidia-smi costs ~30ms per invocation.
    """
    global _cache
    with _cache_lock:
        if _cache is not None and time.monotonic() - _cache[0] < max_age:
            return _cache[1]
    gpus: list[GpuTelemetry] = []
    try:
        lines = _run_query(_FULL_QUERY)
        if lines is None:
            # Driver rejected a telemetry column — degrade to identity + VRAM
            lines = _run_query(_MINIMAL_QUERY) or []
        for line in lines:
            gpu = _parse_line(line)
            if gpu is not None:
                gpus.append(gpu)
        gpus.sort(key=lambda g: g.index)
    except Exception as e:
        logger.warning(f"GPU telemetry query failed: {e}")
    with _cache_lock:
        _cache = (time.monotonic(), gpus)
    return gpus


def compute_apps() -> list[tuple[str, int, float]]:
    """(gpu_uuid, pid, GB) for every CUDA process, machine-wide.

    Note what this does NOT see: graphics contexts. Xorg, gnome-shell and a
    browser hold hundreds of megabytes each and appear in none of these rows.
    That is why third-party occupancy is computed as a remainder from the
    card's total rather than summed from here — a sum would report the
    desktop as free.

    Empty on any failure; callers degrade rather than fail.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        rows: list[tuple[str, int, float]] = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            rows.append((parts[0], int(parts[1]), int(parts[2]) / 1024))
        return rows
    except Exception as e:  # nvidia-smi missing, driver hiccup, odd output
        logger.warning(f"compute_apps failed: {e}")
        return []


def compute_app_memory(device: str | int | None = None) -> dict[int, float]:
    """pid -> GB of VRAM that process holds *on one card*, per nvidia-smi.

    This is what evicting it would actually free, which is not the same as the
    declared figure the VRAM gate uses. The gate wants a conservative upper
    bound ("is there room to load this?"); the eviction planner wants the
    truth ("how much do I get back?"). Same number for both meant the planner
    trusted padding: gemma is declared 9.5GB and holds 8.88.

    ``device`` is an index or UUID; None means the selected device. The
    per-card filter is not cosmetic: the query has always been machine-wide,
    so on a two-card rig it merged both cards' processes into one table while
    the VRAM gate was reading a single card. Planner and gate were describing
    different machines. Pass ``"all"`` for the unfiltered table.

    Empty on any failure — callers fall back to the declared figure.
    """
    want_uuid: str | None = None
    if device != "all":
        gpu = selected_device() if device is None else resolve_device(device)
        # No resolvable card (no driver, or a device string that matches
        # nothing) — return the unfiltered table rather than an empty one:
        # the pre-multi-GPU behaviour, and still better than no measurement.
        want_uuid = gpu.uuid.lower() if gpu else None
    return {
        pid: gb
        for uuid, pid, gb in compute_apps()
        if want_uuid is None or uuid.lower() == want_uuid
    }


def _children() -> dict[int, list[int]]:
    """ppid -> pids, from /proc. Empty where there is no /proc to read."""
    out: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as f:
                stat = f.read().decode(errors="replace")
        except OSError:  # exited between listdir and open, or not ours to read
            continue
        # The command name is in parentheses and may itself contain spaces
        # and parentheses; the fields after the last ")" are fixed: state, ppid.
        fields = stat.rpartition(")")[2].split()
        if len(fields) >= 2 and fields[1].isdigit():
            out.setdefault(int(fields[1]), []).append(int(entry))
    return out


def with_descendants(owned: dict[int, str]) -> dict[int, str]:
    """``owned`` plus every descendant of each pid, under its ancestor's label.

    The pid giq spawns is not always the one holding the VRAM: vllm's API
    server keeps none and its engine-core child holds the model, so counting
    spawned pids alone reported a card running a vllm instance as 0 GB ours.
    giq's own pid (in the map while STT runs in-process) is not expanded: its
    descendants are every engine giq runs, each labelled on its own already.
    """
    if not owned:
        return {}
    children = _children()
    out = dict(owned)
    todo = [pid for pid in owned if pid != os.getpid()]
    while todo:
        pid = todo.pop()
        for child in children.get(pid, ()):
            if child not in out:
                out[child] = out[pid]
                todo.append(child)
    return out


def attribute_vram(
    owned: dict[int, str] | None = None, gpus: list[GpuTelemetry] | None = None
) -> dict[str, dict[str, Any]]:
    """Per card: how much of the used VRAM is ours, and how much is not.

    ``owned`` maps a pid giq is responsible for to a label for it. Ours is
    summed from per-process measurements; **theirs is the remainder** of the
    card's used total, not a sum of the processes we can see — the desktop
    holds VRAM through graphics contexts that never appear in
    ``--query-compute-apps``, and reporting that as free is exactly the
    mistake that makes a gauge lie.

    ``giq_gb`` and ``other_gb`` always sum to the card's used total, because
    that is what a caller draws a bar from. Getting there needs a clamp: the
    per-process figures and the card total are two nvidia-smi queries a moment
    apart and do not reconcile exactly — observed live at 4.78 GB of processes
    on a card reporting 4.21 GB used, which as a bar would simply overflow.
    The unclamped per-process truth stays in ``giq``.

    Pass the same ``gpus`` list you are rendering, so the split describes the
    figures on screen rather than a fresher sample of a moving target.
    """
    owned = with_descendants(owned or {})
    mine: dict[str, float] = {}
    breakdown: dict[str, list[dict[str, Any]]] = {}
    for uuid, pid, gb in compute_apps():
        if pid not in owned:
            continue
        mine[uuid] = mine.get(uuid, 0.0) + gb
        breakdown.setdefault(uuid, []).append({"pid": pid, "label": owned[pid], "gb": round(gb, 2)})
    out: dict[str, dict[str, Any]] = {}
    for gpu in get_gpus() if gpus is None else gpus:
        ours = min(mine.get(gpu.uuid, 0.0), gpu.vram_used_gb)
        out[gpu.uuid] = {
            "giq_gb": round(ours, 2),
            "other_gb": round(max(gpu.vram_used_gb - ours, 0.0), 2),
            "free_gb": round(gpu.vram_free_gb, 2),
            "total_gb": round(gpu.vram_total_gb, 2),
            "giq": sorted(breakdown.get(gpu.uuid, []), key=lambda b: -b["gb"]),
        }
    return out


# --- Device selection --------------------------------------------------------
# One card is "the" device giq runs models on. That is a placeholder for real
# per-model binding, not a design position — but it has to exist first, because
# every VRAM number in the service is now ambiguous without it.

_selection_lock = threading.Lock()
_selected_uuid: str | None = None
_selection_warned = False


def resolve_device(value: str | int | None) -> GpuTelemetry | None:
    """Find a card by index ("1") or UUID ("GPU-xxxx…"). None if no match.

    Both forms are accepted because both are useful: an index is what an
    operator types and what nvidia-smi prints, a UUID is what survives a
    reboot, a PCI re-enumeration or a card swap. UUID is canonical — it is
    what gets stored and displayed, and what ``gpu_eras`` has always used to
    tell one card from another.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    gpus = get_gpus()
    if text.isdigit():
        index = int(text)
        return next((g for g in gpus if g.index == index), None)
    lowered = text.lower()
    return next((g for g in gpus if g.uuid.lower() == lowered), None)


def default_device(gpus: list[GpuTelemetry] | None = None) -> GpuTelemetry | None:
    """The card giq picks when config names none: biggest, then lowest index.

    Biggest rather than index 0 because VRAM is the binding constraint for
    every model in the registry, and because the small card in a mixed rig is
    typically the one driving the desktop (a GB or so already spoken for).
    """
    gpus = get_gpus() if gpus is None else gpus
    if not gpus:
        return None
    return min(gpus, key=lambda g: (-g.vram_total_gb, g.index))


def selected_device() -> GpuTelemetry | None:
    """The card giq loads models on. None when no GPU is visible.

    Resolved once from config ``gpu.device`` (or ``GIQ_GPU_DEVICE``) and then
    pinned by UUID for the life of the process: the choice must not drift
    under a running scheduler because a card's index moved or config was
    edited. Telemetry itself is re-read on every call.
    """
    global _selected_uuid, _selection_warned
    with _selection_lock:
        pinned = _selected_uuid
    if pinned is not None:
        gpu = resolve_device(pinned)
        if gpu is not None:
            return gpu
        # The pinned card vanished mid-run (driver reset, hot-unplug, ...).
        # Re-select rather than report a card that is no longer there.
        logger.warning(f"selected GPU {pinned} is gone — re-selecting")
        with _selection_lock:
            _selected_uuid = None

    configured = None
    try:
        from giq.config import get_config

        configured = get_config().gpu.device
    except Exception as e:  # config problems must not blind the VRAM gate
        logger.warning(f"gpu.device unreadable, falling back to default: {e}")

    gpu = resolve_device(configured)
    if gpu is None and configured:
        with _selection_lock:
            warned, _selection_warned = _selection_warned, True
        if not warned:
            logger.error(
                f"gpu.device={configured!r} matches no card "
                f"({', '.join(f'{g.index}:{g.uuid}' for g in get_gpus()) or 'none visible'}) "
                "— using the default card instead"
            )
    if gpu is None:
        gpu = default_device()
    if gpu is not None:
        with _selection_lock:
            _selected_uuid = gpu.uuid
        logger.info(f"giq device: GPU {gpu.index} {gpu.name} ({gpu.uuid})")
    return gpu


def reset_selected_device() -> None:
    """Drop the pinned selection so the next call re-resolves (tests)."""
    global _selected_uuid, _selection_warned
    with _selection_lock:
        _selected_uuid = None
        _selection_warned = False


# Spacing between one card's internal server ports and the next card's. The
# first card keeps the historical port (8086 llama, 8087 sd) so anything that
# hardcoded it still works; a second card's servers land 10 above. Wide enough
# that the llama and sd series never interleave.
DEVICE_PORT_STRIDE = 10


# Where each card's block of internal server ports begins: llama-server's
# historical port. A card owns DEVICE_PORT_STRIDE ports from here (8086-8095
# on the first): each engine's own port, and spares for a second server of
# the same engine on that card.
SERVER_PORT_BLOCK = 8086


def device_ports(device: GpuTelemetry | str | int | None = None) -> range:
    """Every internal server port of one card's block."""
    start = device_port(SERVER_PORT_BLOCK, device)
    return range(start, start + DEVICE_PORT_STRIDE)


def _bindable(port: int, host: str = "127.0.0.1") -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def server_port(preferred: int, device: GpuTelemetry | str | int | None = None) -> int:
    """The port a server about to start on ``device`` should bind.

    ``preferred`` — its engine's port on that card — when it is free, so a
    lone server keeps the address it always had. Otherwise the card holds a
    second server of the same engine (a pinned model and an on-demand one,
    or two models a big card fits at once), and it takes a free port of the
    card's block: the spares from the top down first, another engine's own
    port last. A fixed port per engine and card used to make the second
    server fail to bind, and a second vllm stopped the first one's scope.
    Starts are serialized, so the port found free here is still free when
    the server binds it.
    """
    if _bindable(preferred):
        return preferred
    for port in reversed(device_ports(device)):
        if port != preferred and _bindable(port):
            return port
    raise RuntimeError(f"no free internal port left on this card ({device_ports(device)})")


def device_port(base: int, device: GpuTelemetry | str | int | None = None) -> int:
    """The port a backend's server uses for one card.

    Two cards mean two llama-servers (one resident LLM each) and potentially
    two sd-servers, and every one of them used to bind the same constant. The
    card's index picks the offset — ports only need to be unique among the
    processes alive right now, so index is enough and stays readable in `ss`
    output.
    """
    gpu = device if isinstance(device, GpuTelemetry) else resolve_device(device)
    if gpu is None:
        gpu = selected_device()
    return base + DEVICE_PORT_STRIDE * (gpu.index if gpu else 0)


def device_env(device: str | int | None = None) -> dict[str, str]:
    """Child-process environment with CUDA_VISIBLE_DEVICES set to one card.

    Every CUDA worker giq spawns is a child process, so this one hook pins
    them all. It matters most for llama.cpp, whose default ``-sm layer``
    spreads a model's layers *and its KV cache* across every visible GPU: on a
    two-card rig gemma was silently living half on the card giq wasn't even
    measuring. One visible device makes the split modes moot.

    The UUID form is used rather than an index because CUDA accepts it and it
    cannot be invalidated by re-enumeration between giq's read and the child's
    exec. Falls through to an unmodified environment when no card resolves —
    setting an empty CUDA_VISIBLE_DEVICES would mean "no GPUs at all".

    It also carries giq's cache locations (``paths.cache_env``), so a child's
    HF downloads land where the storage catalog looks and its JIT caches land
    somewhere writable — whether or not the parent exported them itself.
    """
    env = {**cache_env(), **os.environ}
    gpu = selected_device() if device is None else resolve_device(device)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    return env


_capabilities: dict[str, str] | None = None


def compute_capabilities() -> dict[str, str]:
    """Every card's compute capability by UUID, asked once per process.

    A card's architecture does not change while giq runs, and the catalog
    asks on every poll. Empty, and asked again next time, when the query
    fails.
    """
    global _capabilities
    if _capabilities is not None:
        return _capabilities
    try:
        lines = _run_query("uuid,compute_cap")
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"compute capability query failed: {e}")
        return {}
    found = {}
    for line in lines or []:
        uuid, _, cap = (part.strip() for part in line.partition(","))
        if cap and not cap.startswith("["):
            found[uuid.lower()] = cap
    if found:
        _capabilities = found
    return found


def compute_capability(device: str | int | None = None) -> str | None:
    """A card's CUDA compute capability as nvidia-smi prints it ("12.0").

    Engines that compile kernels on first use (FlashInfer under vllm) need to
    be told the one architecture to build for; left to guess, they probe
    every visible card, and on a mixed rig that means building for a card the
    model will never run on. None when no card resolves or the driver is too
    old to report it.
    """
    gpu = selected_device() if device is None else resolve_device(device)
    if gpu is None:
        return None
    return compute_capabilities().get(gpu.uuid.lower())
