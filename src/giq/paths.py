# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Where giq looks for things on disk: config, model weights, engines, state.

One data root, ``GIQ_HOME``, holds everything giq reads or writes that is not
code::

    $GIQ_HOME/
      config.yaml   the one config
      models/       model weights
      recipes/    operator model recipes (ADR-002)
      engines/      engine builds: engines/<engine>/bin/<binary>
      state/        stats.db, inflight.log
      cache/        huggingface, torch, triton and CUDA caches

Each directory resolves, most specific first: its own environment variable,
then the ``paths:`` block of config.yaml, then the ``GIQ_HOME`` layout, then
the default giq has always had. Without ``GIQ_HOME`` and without a ``paths:``
block nothing changes from a plain checkout: models in ``~/models``, state in
``<repo>/data``, ``./config.yaml``, engines from PATH, library caches where
the libraries put them.

The config file's own location never comes from the config — it is
``GIQ_CONFIG``, else ``$GIQ_HOME/config.yaml``, else ``./config.yaml`` — so
resolving a path may read the config but reading the config never resolves a
path through it.

Every default is resolved when it is asked for, not at import, so an
environment variable set after ``import giq`` (tests, a launcher script) still
takes effect. Per-worker overrides (``GIQ_OCR_MODEL_DIR`` and friends) live
with their workers and fall back to paths built here. Code rather than data —
the side venvs under ``envs/`` and the dashboard build — stays relative to
the checkout.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = "~/models"
# ADR-002's default for operator recipe files outside a GIQ_HOME.
DEFAULT_RECIPES_DIR = "~/.config/giq/recipes"
LEGACY_CONFIG_FILE = "config.yaml"

# Keys of config.yaml's `paths:` block, and the GIQ_HOME subdirectory of each.
LAYOUT = ("models", "recipes", "engines", "state", "cache")


def home() -> Path | None:
    """The data root, ``$GIQ_HOME``; None when unset (a plain checkout)."""
    raw = os.environ.get("GIQ_HOME")
    return Path(raw).expanduser() if raw else None


def config_file() -> Path:
    """``$GIQ_CONFIG``, else ``$GIQ_HOME/config.yaml``, else ``./config.yaml``."""
    if explicit := os.environ.get("GIQ_CONFIG"):
        return Path(explicit).expanduser()
    if (root := home()) is not None:
        return root / "config.yaml"
    return Path(LEGACY_CONFIG_FILE)


def _from_env(var: str) -> Path | None:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else None


def _from_config(key: str) -> Path | None:
    """A directory from config.yaml's ``paths:`` block, made absolute.

    A relative entry is taken relative to GIQ_HOME, or to the config file's
    own directory when there is no GIQ_HOME — never to the working directory,
    which is whatever the service manager happened to start in.
    """
    try:
        from giq.config import get_config

        raw = (get_config().paths or {}).get(key)
    except Exception as e:  # a broken config must not stop path resolution
        logger.warning(f"paths: config.yaml `paths:` unreadable, ignoring it: {e}")
        return None
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    base = home() or config_file().absolute().parent
    return base / p


def _from_home(sub: str) -> Path | None:
    root = home()
    return root / sub if root is not None else None


def _systemd_dir(var: str) -> Path | None:
    # systemd exports StateDirectory=/CacheDirectory= as colon-separated
    # absolute paths; the first one is the unit's own.
    raw = os.environ.get(var)
    return Path(raw.split(":")[0]) if raw else None


def _resolve(env: str, key: str) -> Path | None:
    return _from_env(env) or _from_config(key) or _from_home(key)


def models_dir() -> Path:
    """Root of the local model store (``GIQ_MODELS_DIR`` … ``~/models``)."""
    return _resolve("GIQ_MODELS_DIR", "models") or Path(DEFAULT_MODELS_DIR).expanduser()


def model_path(path: str | os.PathLike[str]) -> str:
    """Resolve a model path as the registry writes it.

    ``~`` is expanded and an absolute path is returned as is; anything else is
    taken relative to :func:`models_dir`.
    """
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else models_dir() / p)


# Before ADR-003 the recipe files were called instance files, and lived here.
LEGACY_RECIPES_ENV = "GIQ_INSTANCES_DIR"
LEGACY_RECIPES_DIR = "~/.config/giq/instances"
_legacy_warned = False


def _legacy(path: Path, what: str) -> Path:
    global _legacy_warned
    if not _legacy_warned:
        _legacy_warned = True
        logger.warning(f"paths: reading recipe files from {path} ({what}); see ADR-003")
    return path


def recipes_dir() -> Path:
    """Operator recipe files (``GIQ_RECIPES_DIR`` … ``~/.config/giq/recipes``).

    An operator who set the old variable, or whose files sit in the old
    directory, keeps them: a rename must not make their recipes disappear.
    The new name wins wherever both exist.
    """
    if found := _from_env("GIQ_RECIPES_DIR") or _from_config("recipes"):
        return found
    if old := _from_env(LEGACY_RECIPES_ENV):
        return _legacy(old, f"{LEGACY_RECIPES_ENV} is now GIQ_RECIPES_DIR")
    root = home()
    default = root / "recipes" if root is not None else Path(DEFAULT_RECIPES_DIR).expanduser()
    old = root / "instances" if root is not None else Path(LEGACY_RECIPES_DIR).expanduser()
    # The old directory while the new one holds no recipe yet: `giq init`
    # moves it on a GIQ_HOME, but a plain checkout's ~/.config is the
    # operator's to rename.
    if old.is_dir() and not _has_recipes(default) and _has_recipes(old):
        return _legacy(old, f"rename it to {default}")
    return default


def _has_recipes(directory: Path) -> bool:
    return directory.is_dir() and any(
        p.suffix in (".yaml", ".yml") for p in directory.iterdir() if p.is_file()
    )


def engines_dir() -> Path | None:
    """Where engine builds live (``GIQ_ENGINES_DIR`` … ``$GIQ_HOME/engines``).

    None outside a GIQ_HOME: engines then come from config.yaml ``engines:``,
    their ``GIQ_*_BINARY`` variables, or PATH.
    """
    return _resolve("GIQ_ENGINES_DIR", "engines")


def repo_root() -> Path:
    """The giq checkout this package runs from (``src/giq/`` -> two up)."""
    return Path(__file__).resolve().parents[2]


def state_dir() -> Path:
    """giq's own state: stats DB and in-flight log.

    ``GIQ_DATA_DIR`` … ``$GIQ_HOME/state``, then systemd's
    ``$STATE_DIRECTORY`` for a unit that declares ``StateDirectory=`` without
    a GIQ_HOME, then ``<repo>/data``.
    """
    return (
        _resolve("GIQ_DATA_DIR", "state") or _systemd_dir("STATE_DIRECTORY") or repo_root() / "data"
    )


# The name every caller has used; state_dir() is the layout's word for it.
data_dir = state_dir


def stats_db() -> Path:
    """The stats database: ``$GIQ_STATS_DB``, else ``<state_dir>/stats.db``."""
    return _from_env("GIQ_STATS_DB") or state_dir() / "stats.db"


def apps_file() -> Path:
    """The apps list (ADR-006): ``$GIQ_APPS_FILE``, else config.yaml's
    ``paths: apps:``, else ``<state_dir>/apps.yaml``.

    State rather than beside config.yaml: giq writes it, and under the
    hardened unit the state directory is where the service may write.
    """
    return _from_env("GIQ_APPS_FILE") or _from_config("apps") or state_dir() / "apps.yaml"


def inflight_log() -> Path:
    """The in-flight job log: ``$GIQ_INFLIGHT_LOG``, else ``<state_dir>/inflight.log``."""
    return _from_env("GIQ_INFLIGHT_LOG") or state_dir() / "inflight.log"


def cache_dir() -> Path | None:
    """Download and JIT caches (``GIQ_CACHE_DIR`` … ``$GIQ_HOME/cache``).

    None when nothing names one — the libraries then keep their own defaults
    under ``~/.cache``, exactly as before GIQ_HOME existed. systemd's
    ``$CACHE_DIRECTORY`` is the last resort before that.
    """
    return _resolve("GIQ_CACHE_DIR", "cache") or _systemd_dir("CACHE_DIRECTORY")


def hf_home() -> Path:
    """The Hugging Face home: ``$HF_HOME``, else ``<cache>/huggingface``,
    else the library default ``~/.cache/huggingface``."""
    if explicit := _from_env("HF_HOME"):
        return explicit
    cache = cache_dir()
    if cache is not None:
        return cache / "huggingface"
    return Path("~/.cache/huggingface").expanduser()


def cache_env() -> dict[str, str]:
    """Cache variables giq hands to what it runs, when it owns a cache dir.

    The storage catalog finds HF snapshots through :func:`hf_home`; a child
    that downloaded into a different HF home would hold weights the catalog
    cannot see, so both must agree. The rest point JIT caches (torch, Triton,
    the CUDA driver's PTX cache, matplotlib's font cache) at a writable place:
    under a hardened unit the service user's home is read-only, and these
    libraries otherwise write to ``~``. A variable already set in the
    environment wins — the operator said so. Empty without a cache dir.
    """
    cache = cache_dir()
    if cache is None:
        return {}
    wanted = {
        "HF_HOME": hf_home(),
        "XDG_CACHE_HOME": cache,
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "nv" / "ComputeCache",
        "MPLCONFIGDIR": cache / "matplotlib",
    }
    return {k: str(v) for k, v in wanted.items() if not os.environ.get(k)}


def apply_cache_env() -> dict[str, str]:
    """Export :func:`cache_env` into this process, for in-process libraries.

    Called before the service imports anything that reads these at import
    time (huggingface_hub does), so the in-process workers and every child
    agree with the catalog; ``gpus.device_env`` adds them again for a child
    spawned by a process that never called this. Returns what it set.
    """
    added = cache_env()
    os.environ.update(added)
    return added


def env_python(env: str) -> str:
    """Interpreter of a side venv kept in the repo under ``envs/<env>``."""
    return str(repo_root() / "envs" / env / ".venv" / "bin" / "python")


def default_binary(name: str) -> str:
    """An engine executable found on PATH, else the bare name.

    The bare name is deliberate: it does not exist as a file, so
    ``engines.require_binary`` still refuses to spawn it and says how to
    declare the right path.
    """
    return shutil.which(name) or name


def engine_binary(engine: str, name: str) -> str:
    """An engine's built-in binary: ``<engines_dir>/<engine>/bin/<name>`` when
    that file exists, else :func:`default_binary` (PATH)."""
    root = engines_dir()
    if root is not None:
        candidate = root / engine / "bin" / name
        if candidate.exists():
            return str(candidate)
    return default_binary(name)


def resolved() -> dict[str, str | None]:
    """Every resolved location, for ``/storage`` and ``giq init``."""
    engines = engines_dir()
    cache = cache_dir()
    root = home()
    return {
        "home": str(root) if root is not None else None,
        "config": str(config_file().absolute()),
        "models": str(models_dir()),
        "recipes": str(recipes_dir()),
        "engines": str(engines) if engines is not None else None,
        "state": str(state_dir()),
        "stats_db": str(stats_db()),
        "cache": str(cache) if cache is not None else None,
        "hf_home": str(hf_home()),
    }
