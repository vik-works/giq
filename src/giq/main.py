# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq - GPU Inference Queue service."""

import argparse
import logging
import os

import uvicorn
from fastapi import FastAPI

from giq import __version__, plugins
from giq.api import access
from giq.core.lifecycle import lifespan

logger = logging.getLogger(__name__)

app = FastAPI(
    title="giq",
    description="GPU Inference Queue - efficient GPU worker management for multimodal AI",
    version=__version__,
    lifespan=lifespan,
)

# Every route comes from a registration (ADR-004): core's own first, then
# each installed plugin's, so a plugin can add a route but never take one
# over.
plugins.mount(app)

# Outermost layer: nothing reaches a route without passing the Host/Origin
# rules. Installed at import so tooling and tests exercise the same app the
# server does, rather than a laxer one.
access.install(app)


# How long open responses (a streamed chat, a dashboard event stream) get to
# finish after SIGTERM before uvicorn cancels them and the lifespan teardown
# stops the workers. Without a bound, one idle stream holds a restart open
# until the service manager's stop timeout kills everything uncleanly; the
# systemd unit's TimeoutStopSec leaves room for this plus worker teardown.
GRACEFUL_SHUTDOWN_SECONDS = 30


def build_parser() -> argparse.ArgumentParser:
    """``giq serve`` options. Host and port default from ``GIQ_HOST`` and
    ``GIQ_PORT`` so a service manager can set them in an environment file
    without rewriting the command line."""
    parser = argparse.ArgumentParser(
        prog="giq serve",
        description="giq - GPU Inference Queue",
        epilog="`giq init` creates a GIQ_HOME data tree; see docs/deployment.md.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("GIQ_HOST") or "127.0.0.1",
        help="Host to bind to (env GIQ_HOST, default 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GIQ_PORT") or 8084),
        help="Port to bind to (env GIQ_PORT, default 8084)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    return parser


def cli(argv: list[str] | None = None) -> None:
    """Serve giq (``python -m giq.main`` and ``giq serve``)."""
    args = build_parser().parse_args(argv)

    access.set_bound_host(args.host)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    access.log_posture(args.host, args.port)

    uvicorn.run(
        "giq.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )


if __name__ == "__main__":
    cli()
