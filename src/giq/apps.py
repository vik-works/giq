# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Apps: who sent a job, for counting (ADR-006).

An app is a name and the hash of a key. The key travels where clients
already put one (``Authorization: Bearer``, which is what the OpenAI SDKs send
as ``api_key``; ``X-Giq-Token``; ``?token=``), so nothing an app sends is
giq's own and the same app runs against any OpenAI-compatible server.

This is a label, not identity: no roles, no scopes, no audit trail. A request
without a valid key is still served and counted as :data:`ANONYMOUS`, so the
per-app totals add up to what the cards did. With ``access.token`` set, a
valid app key also admits a caller, in the same plain sense as the shared
token (``giq.api.access``).

The list lives in a YAML file, not in the stats database: it is a handful of
entries that change rarely, which the operator can read, back up and
provision by hand. giq rereads it when its mtime changes, so a hand edit
takes effect without a restart, and writes it by renaming a temporary file
over it, so a crash mid-write leaves the old list rather than half a new one.
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import logging
import os
import re
import secrets
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from giq import paths

logger = logging.getLogger(__name__)

# The bucket for every request without a valid app key. Reserved: no app may
# take the name, or its jobs would merge with everyone else's.
ANONYMOUS = "anonymous"
# What the dashboard shows for jobs recorded before apps existed (NULL in
# jobs.app). Reserved for the same reason.
UNKNOWN = "unknown"
_RESERVED = {ANONYMOUS, UNKNOWN}

KEY_PREFIX = "giq-"
# A slug: it names the app in the stats database for good, in URLs and on the
# command line.
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

# The app the current request is counted under. Set by the access middleware
# once per request and read by the orchestrator when it stamps a job, so
# routes and plugins attribute their jobs without knowing apps exist. It is
# never taken from the request body: a caller cannot name another app.
current_app: contextvars.ContextVar[str] = contextvars.ContextVar("giq_app", default=ANONYMOUS)


class AppError(ValueError):
    """A request about apps that cannot be honoured; the message says why."""


@dataclass
class App:
    name: str
    created: str
    key_sha256: str
    description: str | None = None
    revoked: str | None = None

    def public(self) -> dict:
        """What the API and CLI show: everything but the hash."""
        return {
            "name": self.name,
            "description": self.description,
            "created": self.created,
            "revoked": self.revoked,
        }


def hash_key(key: str) -> str:
    # A plain hash suffices: the key is 32 random bytes, so there is no
    # guessable input for a slow hash to protect.
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def new_key() -> str:
    raw = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    return KEY_PREFIX + raw


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class AppStore:
    def __init__(self, path: Path | None = None):
        self._path = path
        self._lock = threading.Lock()
        self._apps: dict[str, App] = {}
        self._by_hash: dict[str, str] = {}
        self._stamp: tuple[float, int] | None = None

    @property
    def path(self) -> Path:
        # Resolved when asked, like every giq path, so tests and launchers
        # that set GIQ_APPS_FILE after import still take effect.
        return self._path or paths.apps_file()

    # --- reading -----------------------------------------------------------

    def _stat(self) -> tuple[float, int] | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime, st.st_size)

    def _refresh(self) -> None:
        """Reread the file if it changed. Called under the lock."""
        stamp = self._stat()
        if stamp == self._stamp:
            return
        apps: dict[str, App] = {}
        if stamp is not None:
            try:
                data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as e:
                # Keep serving with the list as it was: a typo in a hand edit
                # must not turn every app anonymous at once.
                logger.warning(f"apps: {self.path} unreadable, keeping the previous list: {e}")
                return
            for name, entry in (data.get("apps") or {}).items():
                if not isinstance(entry, dict) or not entry.get("key_sha256"):
                    logger.warning(f"apps: entry {name!r} has no key_sha256, skipped")
                    continue
                apps[str(name)] = App(
                    name=str(name),
                    created=str(entry.get("created") or ""),
                    key_sha256=str(entry["key_sha256"]).lower(),
                    description=entry.get("description"),
                    revoked=str(entry["revoked"]) if entry.get("revoked") else None,
                )
        self._apps = apps
        self._by_hash = {a.key_sha256: a.name for a in apps.values() if not a.revoked}
        self._stamp = stamp

    def entries(self) -> list[App]:
        with self._lock:
            self._refresh()
            return sorted(self._apps.values(), key=lambda a: a.name)

    def get(self, name: str) -> App | None:
        with self._lock:
            self._refresh()
            return self._apps.get(name)

    def resolve(self, key: str | None) -> str | None:
        """The app a key belongs to, or None for no key, an unknown or a revoked one."""
        if not key or not key.startswith(KEY_PREFIX):
            return None
        with self._lock:
            self._refresh()
            return self._by_hash.get(hash_key(key))

    # --- writing -----------------------------------------------------------

    def _save(self) -> None:
        """Write the list atomically. Called under the lock."""
        doc = {"apps": {}}
        for app in sorted(self._apps.values(), key=lambda a: a.name):
            entry: dict = {"created": app.created, "key_sha256": app.key_sha256}
            if app.description:
                entry["description"] = app.description
            if app.revoked:
                entry["revoked"] = app.revoked
            doc["apps"][app.name] = entry
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".apps-", suffix=".yaml", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(
                    "# giq apps (ADR-006). Written by giq; hand edits are read on change.\n"
                    "# Keys are stored as their SHA-256 only.\n"
                )
                yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
            # The hashes admit a caller on a rig with a token set; nobody but
            # the service user needs to read them.
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._by_hash = {a.key_sha256: a.name for a in self._apps.values() if not a.revoked}
        self._stamp = self._stat()

    def create(self, name: str, description: str | None = None) -> tuple[App, str]:
        """A new app and its key. The key is returned here and never again."""
        if not _NAME.match(name):
            raise AppError(
                "An app name is lowercase letters, digits, '-' and '_', starting with a "
                "letter or digit, at most 63 characters."
            )
        if name in _RESERVED:
            raise AppError(f"{name!r} is reserved.")
        key = new_key()
        with self._lock:
            self._refresh()
            if name in self._apps:
                # A revoked entry blocks its name too: the stats count by
                # name, and a new app under it would inherit the old one's
                # history. Deleting the entry is how the operator says that
                # merge is what they want.
                raise AppError(f"An app named {name!r} already exists.")
            app = App(
                name=name,
                created=_now(),
                key_sha256=hash_key(key),
                description=(description or "").strip() or None,
            )
            self._apps[name] = app
            self._save()
        return app, key

    def revoke(self, name: str) -> App:
        with self._lock:
            self._refresh()
            app = self._apps.get(name)
            if app is None:
                raise KeyError(name)
            if not app.revoked:
                app.revoked = _now()
                self._save()
            return app

    def delete(self, name: str) -> None:
        with self._lock:
            self._refresh()
            if name not in self._apps:
                raise KeyError(name)
            del self._apps[name]
            self._save()


_store: AppStore | None = None


def get_apps() -> AppStore:
    global _store
    if _store is None:
        _store = AppStore()
    return _store


def presented_keys(headers, query_params) -> list[str]:
    """Every value a request offers as a key, in the places a client can put one."""
    out = []
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        out.append(auth[7:].strip())
    if v := headers.get("x-giq-token"):
        out.append(v)
    if v := query_params.get("token"):
        out.append(v)
    return out
