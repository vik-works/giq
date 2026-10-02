# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

.PHONY: sync ui ui-dev ui-dist ui-pack run test fmt check install-service uninstall-service

# Default port
PORT ?= 8084

# What the service binds to. Loopback by default: giq has no authentication
# out of the box, so reaching it from other machines is a choice you make
# rather than one you inherit. `make install-service HOST=0.0.0.0` opens it to
# your network — see docs/access-and-privacy.md.
HOST ?= 127.0.0.1

# Plain `uv sync` is now the whole story: every runtime dependency is a default
# dependency and the dev tools are in the default group, so a fresh clone needs
# no flags. It used to need `--extra tts --extra tts-qwen`, and since uv sync
# PRUNES to the lock, forgetting them removed the TTS stack from a working
# venv. That is why nothing giq needs lives in an extra any more.
#
# SKIP_UI=1 leaves the dashboard alone: deploy/install-debian.sh passes it
# because it installs the dashboard itself, from a tarball, a release or a
# local build, in that order of preference.
sync:
	uv sync
	cd envs/unlimited-ocr && uv sync
	@if [ -n "$(SKIP_UI)" ]; then :; \
	elif command -v npm >/dev/null 2>&1 && node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=12)?0:1)' 2>/dev/null; then $(MAKE) ui; \
	else echo "Node.js 22.12+ with npm not found: skipping the dashboard build. Install it and run 'make ui'."; fi

# The dashboard (frontend/, React + Vite) builds into src/giq/static/ui, which
# /dash serves. `npm ci` installs exactly the lockfile, so a build here is the
# build CI made.
ui:
	cd frontend && npm ci && npm run build

# The built dashboard as a tarball, for a host that cannot build it (Debian
# 13's Node.js is older than Vite needs): dist/giq-ui-<version>.tar.gz holds
# the contents of src/giq/static/ui/ with paths relative to it, next to its
# .sha256. `deploy/install-debian.sh --ui-tarball` installs it. ui-pack packs
# an existing build, which is what CI calls after its own `npm run build`.
#
# The archive is reproducible: sorted names, no owners, every mtime the
# commit's, gzip without a timestamp. The same build gives the same checksum.
VERSION = $(shell sed -n 's/^version = "\(.*\)"$$/\1/p' pyproject.toml | head -n1)
UI_TARBALL = dist/giq-ui-$(VERSION).tar.gz
UI_MTIME = $(shell git log -1 --format=%ct 2>/dev/null || echo 0)

ui-dist: ui
	$(MAKE) ui-pack

ui-pack:
	@test -n "$(VERSION)" || { echo "no version in pyproject.toml" >&2; exit 1; }
	@test -f src/giq/static/ui/index.html || { echo "no dashboard build in src/giq/static/ui: run 'make ui'" >&2; exit 1; }
	@test -f src/giq/static/ui/THIRD_PARTY_LICENSES.txt || { echo "the build has no THIRD_PARTY_LICENSES.txt" >&2; exit 1; }
	mkdir -p dist
	tar -C src/giq/static/ui --sort=name --owner=0 --group=0 --numeric-owner --mode=u+rwX,go=rX \
		--mtime=@$(UI_MTIME) -cf - . | gzip -n9 > $(UI_TARBALL).tmp
	mv $(UI_TARBALL).tmp $(UI_TARBALL)
	cd dist && sha256sum $(notdir $(UI_TARBALL)) > $(notdir $(UI_TARBALL)).sha256
	@cat $(UI_TARBALL).sha256

# Vite dev server with hot reload on http://localhost:5173/dash/, proxying the
# API to the giq running on $(PORT) (override with GIQ_URL).
ui-dev:
	cd frontend && npm install && GIQ_URL=http://127.0.0.1:$(PORT) npm run dev

run:
	uv run python -m giq.main --host $(HOST) --port $(PORT)

test:
	uv run pytest -v

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

check:
	uv run ruff check src tests
	uv run ty check src

# Systemd user service, for a desktop or development checkout: runs as you,
# from this directory. A server install is deploy/giq.service instead (a
# dedicated user, hardening, GIQ_HOME) — see docs/deployment.md. With
# GIQ_HOME set when you run this, the unit carries it, so the service finds
# the same config, models and state your shell does.
SERVICE_FILE = $(HOME)/.config/systemd/user/giq.service
PROJECT_DIR = $(shell pwd)

install-service:
	@mkdir -p $(HOME)/.config/systemd/user
	@echo "[Unit]" > $(SERVICE_FILE)
	@echo "Description=giq - GPU Inference Queue" >> $(SERVICE_FILE)
	@echo "After=network.target" >> $(SERVICE_FILE)
	@echo "" >> $(SERVICE_FILE)
	@echo "[Service]" >> $(SERVICE_FILE)
	@echo "Type=simple" >> $(SERVICE_FILE)
	@echo "WorkingDirectory=$(PROJECT_DIR)" >> $(SERVICE_FILE)
	@echo "ExecStart=$(PROJECT_DIR)/.venv/bin/python -m giq.main --host $(HOST) --port $(PORT)" >> $(SERVICE_FILE)
	@echo "Restart=on-failure" >> $(SERVICE_FILE)
	@echo "RestartSec=5" >> $(SERVICE_FILE)
	@echo "Environment=PYTHONUNBUFFERED=1" >> $(SERVICE_FILE)
	@if [ -n "$(GIQ_HOME)" ]; then echo "Environment=GIQ_HOME=$(abspath $(GIQ_HOME))" >> $(SERVICE_FILE); fi
	@echo "" >> $(SERVICE_FILE)
	@echo "[Install]" >> $(SERVICE_FILE)
	@echo "WantedBy=default.target" >> $(SERVICE_FILE)
	systemctl --user daemon-reload
	systemctl --user enable giq
	@echo "Service installed. Start with: systemctl --user start giq"

uninstall-service:
	-systemctl --user stop giq
	-systemctl --user disable giq
	rm -f $(SERVICE_FILE)
	systemctl --user daemon-reload
	@echo "Service uninstalled."

status:
	@curl -s http://127.0.0.1:$(PORT)/status | python -m json.tool 2>/dev/null || echo "giq not running"

logs:
	journalctl --user -u giq -f
