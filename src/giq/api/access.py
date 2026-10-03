# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Who may reach giq, and from where.

giq has no user model, and on a home rig it does not need one: the caller is
the person who plugged the card in. What it is *not* is a service the browser
may be talked into calling on someone else's behalf. A page the owner happens
to visit can aim requests at 127.0.0.1 without ever seeing the machine, and
nothing about that request looks unusual at the socket. So the checks here are
about the browser's own honest metadata rather than about proving identity:

``Host``
    Must name giq. A DNS-rebinding attack arrives as ``Host: evil.com``
    resolving to a local address; every legitimate client names the address it
    actually dialled. An IP literal is accepted whatever it is, because
    rebinding needs a *name* to re-point — a page that hardcodes the LAN IP is
    cross-origin and gets caught by the next rule instead.

``Origin``
    Present and foreign means a web page is doing the asking. curl, ESP32
    firmware and every other non-browser client omit it entirely, so this
    constrains exactly the caller it is aimed at and nobody else.

Both are free for honest clients, which is the whole point: the guarantee has
to hold on a rig whose operator is not going to configure anything. Note that
giq deliberately sends no CORS headers, so a foreign page cannot read a
response either — these rules close the two holes that survive that: requests
whose side effect is the attack, and rebinding, which makes the page
same-origin and hands it the reads too.

The token is the escape hatch for when loopback is left behind, and stays
opt-in. It is a shared secret, not a credential: everything on the LAN that
talks to giq carries the same one. Real per-client identity belongs to the
enterprise overlay, and calling this that would be worse than being plain
about what it is.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse

from giq.config import get_config

logger = logging.getLogger(__name__)

# Names that always mean "this machine", whatever giq is bound to.
_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", ""}

# Wildcard binds: giq is reachable at every address the host answers to, so
# the bound address itself tells us nothing about which Host to expect.
_WILDCARD = {"0.0.0.0", "::", "*"}

# What --host was given. The app object is built at import time and the bind
# address is only known once the CLI parses its arguments, so this is set
# rather than passed; the loopback default is the safe assumption for anything
# that imports the app without running the server (tests, tooling).
#
# It travels through the environment because uvicorn is handed an import
# string: under --reload the app is re-imported in a worker process that never
# runs cli(), and a module global would not survive the trip.
_bound_host = os.environ.get("GIQ_BOUND_HOST", "127.0.0.1")


def set_bound_host(host: str) -> None:
    global _bound_host
    _bound_host = host
    os.environ["GIQ_BOUND_HOST"] = host


def _hostname(value: str) -> str:
    """The host part of a Host header or an origin, lowercased, port dropped."""
    if value.startswith("["):  # [::1]:8084
        return value[1 : value.find("]")].lower() if "]" in value else value.lower()
    return value.rsplit(":", 1)[0].lower() if value.count(":") == 1 else value.lower()


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _own_names() -> set[str]:
    """Names this machine legitimately answers to."""
    names = set(_LOOPBACK_NAMES)
    try:
        hostname = socket.gethostname().lower()
        names |= {hostname, f"{hostname}.local"}
    except OSError:  # pragma: no cover — gethostname does not realistically fail
        pass
    return names


def host_allowed(host_header: str, *, bound_host: str, extra: list[str]) -> bool:
    """True when this Host header plausibly names giq itself."""
    host = _hostname(host_header)
    if _is_ip_literal(host):
        return True
    if host in _own_names():
        return True
    if bound_host.lower() not in _WILDCARD and host == _hostname(bound_host):
        return True
    return host in {_hostname(h) for h in extra}


def origin_allowed(origin: str, *, host_header: str, extra: list[str]) -> bool:
    """True when this Origin is giq's own page rather than a foreign one.

    Same-origin is judged against the Host the request arrived on, so it holds
    however the owner reached giq — localhost, the LAN IP, or a hostname —
    without giq having to be told its own address.
    """
    if origin in extra:
        return True
    parsed = urlsplit(origin)
    if not parsed.hostname:  # "null" — a sandboxed iframe or a file:// page
        return False
    return parsed.netloc.lower() == host_header.lower() or (
        parsed.hostname.lower() == _hostname(host_header) and not parsed.port
    )


def token_ok(request: Request, expected: str) -> bool:
    """Accept the shared secret in any form a local client can manage.

    A header is the right way and what scripts will use; the query parameter
    exists because firmware on a microcontroller often cannot set one, and a
    token nobody can send is a token that gets turned off.
    """
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and auth[7:].strip() == expected:
        return True
    if request.headers.get("x-giq-token", "") == expected:
        return True
    return request.query_params.get("token", "") == expected


def token_exempt(request: Request) -> bool:
    """The dashboard's own page and build assets, and plugins' dashboard
    modules, load without the token.

    A browser fetches `<script src>` and font files itself, with no way to
    attach a header, so a token-guarded rig would serve a page whose scripts
    all 401. What loads here is the same public code the wheel ships — no
    data, no action — and the page then sends the token (from ?token= or the
    tab's session) on every API call it makes. Host and Origin rules still
    apply to these paths like any other.
    """
    if request.method not in ("GET", "HEAD"):
        return False
    path = request.url.path
    if path in ("/dash", "/dash/") or path.startswith("/dash/assets/"):
        return True
    # A plugin's dashboard module and its strings, which the page import()s
    # the same way (ADR-004 D6): installed code, like the dashboard's own.
    parts = path.split("/")
    return len(parts) > 4 and parts[1] == "plugins" and parts[3] == "ui"


def _deny(reason: str, detail: str) -> JSONResponse:
    logger.warning("access denied (%s): %s", reason, detail)
    return JSONResponse(status_code=403, content={"detail": detail})


def install(app) -> None:
    """Attach the access rules to the app."""

    @app.middleware("http")
    async def _access_control(request: Request, call_next):
        access = get_config().access

        host_header = request.headers.get("host", "")
        if not host_allowed(host_header, bound_host=_bound_host, extra=access.allow_hosts):
            return _deny(
                "host",
                f"Host {host_header!r} does not name this server. If giq is behind a "
                "proxy, add the name to access.allow_hosts in config.yaml.",
            )

        origin = request.headers.get("origin")
        if origin and not origin_allowed(
            origin, host_header=host_header, extra=access.allow_origins
        ):
            return _deny(
                "origin",
                f"Requests from {origin} are refused: a web page on another site "
                "cannot use this API. Add it to access.allow_origins if you meant it.",
            )

        if access.token and not token_ok(request, access.token) and not token_exempt(request):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid token."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)


def posture(bound_host: str | None = None) -> dict:
    """How exposed giq is, in the terms the operator cares about."""
    bound_host = _bound_host if bound_host is None else bound_host
    access = get_config().access
    loopback = bound_host.lower() not in _WILDCARD and (
        _hostname(bound_host) in _LOOPBACK_NAMES
        or (
            _is_ip_literal(_hostname(bound_host))
            and ipaddress.ip_address(_hostname(bound_host)).is_loopback
        )
    )
    return {
        "bound_host": bound_host,
        "reachable": "this machine" if loopback else "this machine and your network",
        "loopback_only": loopback,
        "token_set": bool(access.token),
        "exposed": not loopback and not access.token,
    }


def log_posture(bound_host: str, port: int) -> None:
    """One line at boot saying who can reach this, because nothing else will.

    Being open on a home network is a legitimate choice — it is how the phone
    and the ESP32 reach giq. Being open without having decided to is not, and
    the difference is entirely whether anyone said it out loud.
    """
    p = posture(bound_host)
    if p["loopback_only"]:
        logger.info("giq on %s:%s — reachable from this machine only.", bound_host, port)
        return
    if p["token_set"]:
        logger.info("giq on %s:%s — reachable from your network; token required.", bound_host, port)
        return
    logger.warning(
        "giq on %s:%s — reachable from your whole network, and no token is set: any device "
        "on it can submit jobs, read past results and delete model weights. That is fine on "
        "a trusted LAN and is how phones and embedded clients reach giq. To close it, bind "
        "--host 127.0.0.1, or set GIQ_TOKEN (or access.token in config.yaml).",
        bound_host,
        port,
    )
