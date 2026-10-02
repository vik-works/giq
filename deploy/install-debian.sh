#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

# Install or update giq as a system service on Debian (13 "trixie" and later).
# Run as root. Safe to run again: it updates the checkout, re-syncs, replaces
# the unit and restarts, and never overwrites config.yaml, /etc/giq/giq.env or
# anything under GIQ_HOME. It never downloads models or builds engines.
# docs/deployment.md walks through what it does and what it expects.
#
# The dashboard needs Node.js 22.12+ to build, which Debian 13 does not ship,
# so it can also arrive prebuilt: --ui-tarball (from `make ui-dist` on another
# machine) or --ui-release (the tarball attached to a GitHub release).
# --ui-only does just that step, and runs without root against any checkout.

set -euo pipefail

GIQ_HOME=/projects/giq
PREFIX=/opt/giq
REPO=https://github.com/vik-works/giq.git
REF=main
BUILD_USER=giq-build
MIN_DRIVER=580
MAKE_TOKEN=1
START=1
UI_TARBALL=
UI_RELEASE=
UI_ONLY=0
WITH_VLLM=0
# Overridable so tests can stand in a local server for the GitHub API.
GITHUB_API=${GIQ_GITHUB_API:-https://api.github.com}

usage() {
    cat <<EOF
Usage: $0 [options]

  --home DIR         data root, GIQ_HOME (default $GIQ_HOME)
  --prefix DIR       where the checkout lives (default $PREFIX)
  --repo URL         git repository to clone (default $REPO)
  --ref REF          branch, tag or commit to install (default $REF)
  --build-user NAME  unprivileged user that owns the checkout and runs the
                     build (default $BUILD_USER)
  --no-token         do not generate GIQ_TOKEN in a new /etc/giq/giq.env
  --no-start         install everything but do not enable or start the unit
  --ui-tarball PATH  install the dashboard from this tarball (make ui-dist)
  --ui-release TAG   install the dashboard from the giq-ui tarball attached to
                     GitHub release TAG of --repo, checked against the
                     release's SHA256SUMS; a private repository needs a token
                     that can read it in GIQ_GITHUB_TOKEN
  --ui-only          install the dashboard into --prefix and nothing else;
                     runs without root (the files are then yours)
  --with-vllm        also sync envs/vllm (the second LLM engine, several GB)
                     and build its GPU kernels with giq prepare vllm
  -h, --help         this text

The dashboard comes from, first match wins: --ui-tarball, --ui-release, a
local build when Node.js 22.12+ is installed. Without any of them the API
works and /dash says "UI not built".
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --home) GIQ_HOME=${2:?--home needs a value}; shift 2 ;;
        --prefix) PREFIX=${2:?--prefix needs a value}; shift 2 ;;
        --repo) REPO=${2:?--repo needs a value}; shift 2 ;;
        --ref) REF=${2:?--ref needs a value}; shift 2 ;;
        --build-user) BUILD_USER=${2:?--build-user needs a value}; shift 2 ;;
        --no-token) MAKE_TOKEN=0; shift ;;
        --no-start) START=0; shift ;;
        --ui-tarball) UI_TARBALL=${2:?--ui-tarball needs a value}; shift 2 ;;
        --ui-release) UI_RELEASE=${2:?--ui-release needs a value}; shift 2 ;;
        --ui-only) UI_ONLY=1; shift ;;
        --with-vllm) WITH_VLLM=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

say() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

case "$GIQ_HOME" in /*) ;; *) die "--home must be an absolute path" ;; esac
case "$PREFIX" in /*) ;; *) die "--prefix must be an absolute path" ;; esac

BUILD_HOME=/var/lib/$BUILD_USER
UV=

# Temporary files and directories, removed however the script ends.
CLEANUP=()
# shellcheck disable=SC2317,SC2329 # invoked by the trap (older shellcheck says 2317)
cleanup() { [ ${#CLEANUP[@]} -eq 0 ] || rm -rf -- "${CLEANUP[@]}"; }
trap cleanup EXIT

# The build runs unprivileged with a fixed environment: uv's managed Pythons
# go under the prefix (the side envs need 3.11, which Debian does not ship),
# where the service user can read them — the default, the build user's home,
# is hidden from the service by ProtectHome and readable by nobody else.
# Without root (only --ui-only gets that far) the invoking user is the builder.
as_builder() {
    if [ "$(id -u)" -ne 0 ]; then
        "$@"
        return
    fi
    runuser -u "$BUILD_USER" -- env -i \
        HOME="$BUILD_HOME" \
        PATH="${UV:+$(dirname "$UV"):}/usr/local/bin:/usr/bin:/bin" \
        UV_PYTHON_INSTALL_DIR="$PREFIX/.uv-python" \
        UV_CACHE_DIR="$BUILD_HOME/uv-cache" \
        UV_COMPILE_BYTECODE=1 \
        "$@"
}

# --- the dashboard ------------------------------------------------------------------
# Functions only: they run after `make sync`, or alone under --ui-only.

checkout_version() {
    sed -n 's/^version = "\(.*\)"$/\1/p' "$PREFIX/pyproject.toml" 2>/dev/null | head -n1
}

# owner/name of --repo, for the REST API.
github_slug() {
    local slug
    case "$REPO" in
        https://github.com/*) slug=${REPO#https://github.com/} ;;
        git@github.com:*) slug=${REPO#git@github.com:} ;;
        ssh://git@github.com/*) slug=${REPO#ssh://git@github.com/} ;;
        *) return 1 ;;
    esac
    slug=${slug%/}
    slug=${slug%.git}
    case "$slug" in
        */*/* | /* | */) return 1 ;;
        ?*/?*) printf '%s\n' "$slug" ;;
        *) return 1 ;;
    esac
}

# The checks that can fail before anything is changed.
ui_preflight() {
    if [ -n "$UI_TARBALL" ]; then
        if [ ! -f "$UI_TARBALL" ] || [ ! -r "$UI_TARBALL" ]; then
            die "--ui-tarball $UI_TARBALL: no such readable file"
        fi
        UI_TARBALL=$(readlink -f "$UI_TARBALL")
        [ -z "$UI_RELEASE" ] || warn "--ui-tarball wins: ignoring --ui-release $UI_RELEASE"
        UI_RELEASE=
    elif [ -n "$UI_RELEASE" ]; then
        command -v curl >/dev/null || die "--ui-release needs curl (apt install curl)"
        github_slug >/dev/null || die "--ui-release: --repo $REPO is not a GitHub repository"
        [ -n "${GIQ_GITHUB_TOKEN:-}" ] || warn "GIQ_GITHUB_TOKEN is not set: fine for a public
  repository, a 404 for a private one. sudo drops the environment; pass it on
  with: sudo --preserve-env=GIQ_GITHUB_TOKEN $0 ..."
    fi
}

# One GitHub request. The token reaches curl on stdin rather than argv, where
# any local user could read it in ps. An asset request redirects to a storage
# host; curl does not forward the Authorization header to another host, and
# refuses to be redirected to anything but https.
github_get() { # url accept output
    local auth=""
    [ -z "${GIQ_GITHUB_TOKEN:-}" ] || auth="Authorization: Bearer $GIQ_GITHUB_TOKEN"
    printf '%s\n' "$auth" | curl -fsSL --retry 3 --connect-timeout 20 --proto-redir =https \
        -H @- -H "Accept: $2" -H "X-GitHub-Api-Version: 2022-11-28" -o "$3" "$1"
}

# The API URL of a release asset, read from the release JSON. That URL serves
# the file itself when asked for application/octet-stream, with the same
# token; browser_download_url would need a browser session for a private repo.
asset_url() { # python release.json name
    "$1" -c 'import json, sys
for a in json.load(open(sys.argv[1]))["assets"]:
    if a["name"] == sys.argv[2]:
        print(a["url"])
        break' "$2" "$3"
}

# Download the UI tarball of release $UI_RELEASE and check it against the
# release's SHA256SUMS; points UI_TARBALL at the verified file.
fetch_ui_release() {
    local slug dir asset python url sums_url
    slug=$(github_slug)
    asset="giq-ui-${UI_RELEASE#v}.tar.gz"
    dir=$(mktemp -d)
    CLEANUP+=("$dir")
    say "fetching $asset from release $UI_RELEASE of $slug"
    github_get "$GITHUB_API/repos/$slug/releases/tags/$UI_RELEASE" \
        application/vnd.github+json "$dir/release.json" ||
        die "release $UI_RELEASE of $slug not found (a private repository answers 404
  unless GIQ_GITHUB_TOKEN can read it)"

    python=$PREFIX/.venv/bin/python
    [ -x "$python" ] || python=$(command -v python3 || true)
    [ -n "$python" ] || die "--ui-release needs python3 to read the release (apt install python3)"
    url=$(asset_url "$python" "$dir/release.json" "$asset") ||
        die "the release JSON for $UI_RELEASE is not what GitHub sends"
    [ -n "$url" ] || die "release $UI_RELEASE has no asset $asset"
    sums_url=$(asset_url "$python" "$dir/release.json" SHA256SUMS)
    [ -n "$sums_url" ] || die "release $UI_RELEASE has no SHA256SUMS asset"

    github_get "$url" application/octet-stream "$dir/$asset" || die "downloading $asset failed"
    github_get "$sums_url" application/octet-stream "$dir/SHA256SUMS" ||
        die "downloading SHA256SUMS failed"
    awk -v f="$asset" '$2 == f || $2 == "*" f' "$dir/SHA256SUMS" >"$dir/expected"
    [ -s "$dir/expected" ] || die "SHA256SUMS of $UI_RELEASE does not list $asset"
    (cd "$dir" && sha256sum --check --status expected) ||
        die "$asset does not match the release's SHA256SUMS: not installing it"
    say "$asset matches SHA256SUMS"
    UI_TARBALL=$dir/$asset
}

# Unpack $UI_TARBALL into <prefix>/src/giq/static/ui. It is unpacked and
# checked in a sibling directory, then swapped in by rename: a bad or
# truncated tarball leaves the dashboard that was there untouched, and giq,
# which reads the files per request, never serves half of one.
install_ui_tarball() {
    local static=$PREFIX/src/giq/static ui owner tmp old name ver
    ui=$static/ui
    [ -d "$PREFIX/src/giq" ] || die "$PREFIX is not a giq checkout (no src/giq)"

    name=$(basename "$UI_TARBALL")
    ver=$(checkout_version)
    case "$name" in
        giq-ui-*.tar.gz)
            [ "$name" = "giq-ui-$ver.tar.gz" ] ||
                warn "$name is not the checkout's version ($ver): the dashboard may not match the API"
            ;;
    esac

    # Plain files and directories under relative paths only: no links, no
    # devices, nothing absolute or climbing out with "..".
    tar -tzf "$UI_TARBALL" >/dev/null 2>&1 || die "$UI_TARBALL is not a readable .tar.gz"
    if tar -tzf "$UI_TARBALL" | grep -qE '^/|(^|/)\.\.(/|$)'; then
        die "$UI_TARBALL has absolute or '..' paths"
    fi
    if tar -tvzf "$UI_TARBALL" | cut -c1 | grep -qv '[-d]'; then
        die "$UI_TARBALL holds something other than plain files and directories"
    fi

    # The dashboard belongs to whoever owns the checkout: the build user on a
    # server this script set up.
    owner=$(stat -c %u:%g "$PREFIX/src/giq")
    if [ ! -d "$static" ]; then
        mkdir -m 0755 "$static"
        [ "$(id -u)" -ne 0 ] || chown "$owner" "$static"
    fi
    tmp=$(mktemp -d "$static/.ui.new.XXXXXX")
    CLEANUP+=("$tmp")
    (umask 022 && tar -xzf "$UI_TARBALL" -C "$tmp" --no-same-owner --no-same-permissions) ||
        die "unpacking $UI_TARBALL failed"
    [ -f "$tmp/index.html" ] ||
        die "$UI_TARBALL has no index.html at its top level (make ui-dist packs the build's contents, not its directory)"
    [ -f "$tmp/THIRD_PARTY_LICENSES.txt" ] ||
        die "$UI_TARBALL has no THIRD_PARTY_LICENSES.txt, whose licence texts must ship with the bundled code"
    chmod -R u+rwX,go+rX,go-w "$tmp"
    [ "$(id -u)" -ne 0 ] || chown -R "$owner" "$tmp"

    if [ -e "$ui" ]; then
        old=$(mktemp -d "$static/.ui.old.XXXXXX")
        CLEANUP+=("$old")
        mv -T "$ui" "$old/ui"
        if ! mv -T "$tmp" "$ui"; then
            mv -T "$old/ui" "$ui"
            die "could not move the new dashboard into $ui; the old one is back"
        fi
    else
        mv -T "$tmp" "$ui"
    fi
    say "dashboard installed in $ui from $name"
}

node_ok() {
    # shellcheck disable=SC2016 # the JavaScript is meant literally
    as_builder sh -c 'command -v npm >/dev/null && node -e '\''
        const [a, b] = process.versions.node.split(".").map(Number);
        process.exit(a > 22 || (a === 22 && b >= 12) ? 0 : 1)'\' >/dev/null 2>&1
}

# The dashboard, first match wins: --ui-tarball, --ui-release, a local build.
install_ui() {
    local ver
    [ -z "$UI_RELEASE" ] || fetch_ui_release
    if [ -n "$UI_TARBALL" ]; then
        install_ui_tarball
        return
    fi
    if node_ok; then
        say "building the dashboard (make ui)"
        as_builder make -C "$PREFIX" ui
        return
    fi
    ver=$(checkout_version)
    if [ -f "$PREFIX/src/giq/static/ui/index.html" ]; then
        warn "no --ui-tarball, --ui-release or Node.js 22.12+: keeping the dashboard
  already in $PREFIX/src/giq/static/ui, which may be from another version."
    else
        warn "no --ui-tarball, --ui-release or Node.js 22.12+: NO DASHBOARD INSTALLED."
    fi
    cat >&2 <<EOF

  ##########################################################################
  #  /dash does not match this install (the API works regardless). Fix:
  #
  #  on a machine with Node.js 22.12+, in a checkout of the same commit:
  #      make ui-dist
  #      scp dist/giq-ui-$ver.tar.gz $(uname -n):/tmp/
  #  then here:
  #      sudo $0 --ui-only --prefix $PREFIX --ui-tarball /tmp/giq-ui-$ver.tar.gz
  #
  #  or, for a tagged release, with a token that can read the repository:
  #      sudo GIQ_GITHUB_TOKEN=... $0 --ui-only --prefix $PREFIX --ui-release v$ver
  ##########################################################################

EOF
}

ui_preflight

if [ "$UI_ONLY" = 1 ]; then
    [ -d "$PREFIX/src/giq" ] || die "$PREFIX is not a giq checkout (no src/giq)"
    [ -n "$UI_TARBALL$UI_RELEASE" ] || node_ok ||
        die "--ui-only needs --ui-tarball or --ui-release here: there is no Node.js 22.12+ to build with"
    install_ui
    say "done: giq serves it from the next page load, no restart needed"
    exit 0
fi

# --- prerequisites --------------------------------------------------------------
# All checked before anything is changed, so a missing piece costs nothing.

[ "$(id -u)" -eq 0 ] || die "run as root (sudo $0)"
[ -r /etc/debian_version ] || warn "not a Debian system; continuing, but untested"
command -v systemctl >/dev/null || die "systemd is required"

missing=()
command -v git >/dev/null || missing+=(git)
command -v make >/dev/null || missing+=(make)
command -v curl >/dev/null || missing+=(curl)
# giq sweeps stale engine processes at startup with pkill and fuser.
command -v pkill >/dev/null || missing+=(procps)
command -v fuser >/dev/null || missing+=(psmisc)
[ ${#missing[@]} -eq 0 ] || die "missing packages: apt install ${missing[*]}"

command -v nvidia-smi >/dev/null ||
    die "nvidia-smi not found: install the NVIDIA driver (>= $MIN_DRIVER) first"
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1) ||
    die "nvidia-smi fails: the driver is installed but not working (reboot after installing it?)"
[ -n "$driver" ] || die "nvidia-smi reports no GPU"
# torch's cu130 wheels need a CUDA 13 driver.
[ "${driver%%.*}" -ge "$MIN_DRIVER" ] ||
    die "NVIDIA driver $driver is too old: giq needs >= $MIN_DRIVER (CUDA 13)"
say "NVIDIA driver $driver"

# uv must be runnable by the build user, so not a copy under /root.
UV=$(command -v uv || true)
[ -n "$UV" ] || die "uv not found. Install it system-wide, e.g.
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"
UV=$(readlink -f "$UV")
case "$UV" in /root/*) die "uv at $UV is private to root; install it system-wide (see above)" ;; esac
say "uv $("$UV" --version | cut -d' ' -f2)"

if [ ! -e /dev/nvidia-uvm ]; then
    warn "/dev/nvidia-uvm is missing. The service cannot load the module itself
  (NoNewPrivileges): load it at boot with
    echo nvidia-uvm > /etc/modules-load.d/nvidia-uvm.conf && modprobe nvidia-uvm
  and enable nvidia-persistenced so the device nodes exist before giq starts."
fi

# --- users ----------------------------------------------------------------------

if ! id giq >/dev/null 2>&1; then
    say "creating system user giq"
    useradd --system --user-group --home-dir "$GIQ_HOME" --no-create-home \
        --shell /usr/sbin/nologin --comment "giq GPU inference queue" giq
fi
for group in video render; do
    getent group "$group" >/dev/null && usermod -aG "$group" giq
done

if ! id "$BUILD_USER" >/dev/null 2>&1; then
    say "creating build user $BUILD_USER"
    useradd --system --user-group --home-dir "$BUILD_HOME" --create-home \
        --shell /usr/sbin/nologin --comment "giq build" "$BUILD_USER"
fi

# --- checkout ---------------------------------------------------------------------

# A branch name means the remote's branch as just fetched, not a local branch
# left behind by an earlier run; anything else (a tag, a commit) as given.
checkout_ref() {
    as_builder git -C "$PREFIX" checkout --quiet "$@" --detach "origin/$REF" 2>/dev/null ||
        as_builder git -C "$PREFIX" checkout --quiet "$@" --detach "$REF"
}

if [ -d "$PREFIX/.git" ]; then
    say "updating $PREFIX to $REF"
    as_builder git -C "$PREFIX" fetch --quiet --tags --force origin
    checkout_ref --force
else
    [ ! -e "$PREFIX" ] || [ -z "$(ls -A "$PREFIX")" ] ||
        die "$PREFIX exists and is not a git checkout; move it away or pick --prefix"
    say "cloning $REPO into $PREFIX"
    install -d -o "$BUILD_USER" -g "$BUILD_USER" -m 0755 "$PREFIX"
    as_builder git clone --quiet "$REPO" "$PREFIX"
    checkout_ref
fi
say "installed commit $(as_builder git -C "$PREFIX" rev-parse --short HEAD)"

# The dashboard is installed separately, from whichever source wins.
say "syncing environments (make sync)"
as_builder make -C "$PREFIX" sync SKIP_UI=1
[ -x "$PREFIX/.venv/bin/python" ] || die "make sync did not produce $PREFIX/.venv"
install_ui

# --- data root --------------------------------------------------------------------

say "preparing $GIQ_HOME"
install -d -o giq -g giq -m 0755 "$GIQ_HOME"
runuser -u giq -- env -i GIQ_HOME="$GIQ_HOME" PATH=/usr/bin:/bin \
    "$PREFIX/.venv/bin/python" -m giq init
# Job history and the in-flight log are the operator's business only.
[ ! -d "$GIQ_HOME/state" ] || chmod 0750 "$GIQ_HOME/state"

# --- secrets and overrides ----------------------------------------------------------

install -d -o root -g giq -m 0750 /etc/giq
if [ ! -e /etc/giq/giq.env ]; then
    say "writing /etc/giq/giq.env"
    umask 027
    cp "$PREFIX/deploy/giq.env.example" /etc/giq/giq.env
    if [ "$MAKE_TOKEN" = 1 ]; then
        token=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')
        sed -i "s/^GIQ_TOKEN=\$/GIQ_TOKEN=$token/" /etc/giq/giq.env
        say "generated an access token: see GIQ_TOKEN in /etc/giq/giq.env"
    fi
    umask 022
fi
chown root:giq /etc/giq/giq.env
chmod 0640 /etc/giq/giq.env

# --- engines ------------------------------------------------------------------------

if [ -x "$GIQ_HOME/engines/llama.cpp/bin/llama-server" ]; then
    say "llama-server: $GIQ_HOME/engines/llama.cpp/bin/llama-server"
elif command -v llama-server >/dev/null; then
    say "llama-server: $(command -v llama-server) (PATH)"
else
    warn "no llama-server in $GIQ_HOME/engines/llama.cpp/bin or on PATH: LLM models
  will fail to load until one is built (docs/deployment.md, Engines)."
fi

# --- vllm (optional) ------------------------------------------------------------------
# The interpreter is synced like the other side envs, by the build user under
# the same uv rules. The kernels are built as giq, because they land in
# GIQ_HOME/cache, where the service compiles the small ones it still needs on
# first use and must be able to write. The build gets a RAM ceiling from a
# scope root creates — the giq user has no user manager to ask for one — so
# giq prepare is told not to make its own.

if [ "$WITH_VLLM" = 1 ]; then
    say "syncing envs/vllm (uv sync)"
    as_builder sh -c "cd '$PREFIX/envs/vllm' && '$UV' sync --quiet"
    [ -x "$PREFIX/envs/vllm/.venv/bin/vllm" ] || die "uv sync did not produce envs/vllm/.venv/bin/vllm"
    command -v systemd-run >/dev/null || die "systemd-run is required for --with-vllm"
    say "building vllm's GPU kernels (giq prepare vllm; minutes, up to 40 GB of RAM)"
    systemd-run --scope --quiet --collect -p MemoryMax=40G -p MemorySwapMax=0 -- \
        runuser -u giq -- env -i GIQ_HOME="$GIQ_HOME" PATH=/usr/bin:/bin HOME="$GIQ_HOME" \
        "$PREFIX/.venv/bin/python" -m giq prepare vllm --memory-max none
fi

# --- unit ---------------------------------------------------------------------------

say "installing /etc/systemd/system/giq.service"
sed -e "s|/projects/giq|$GIQ_HOME|g" -e "s|/opt/giq|$PREFIX|g" \
    "$PREFIX/deploy/giq.service" >/etc/systemd/system/giq.service.new
chmod 0644 /etc/systemd/system/giq.service.new
mv /etc/systemd/system/giq.service.new /etc/systemd/system/giq.service
systemctl daemon-reload

if [ "$START" = 0 ]; then
    say "done (not started: systemctl enable --now giq)"
    exit 0
fi
systemctl enable giq >/dev/null
systemctl restart giq

# --- first answer ------------------------------------------------------------------------

env_value() { sed -n "s/^$1=//p" /etc/giq/giq.env | tail -n1; }
port=$(env_value GIQ_PORT)
port=${port:-8084}
host=$(env_value GIQ_HOST)
case "$host" in "" | 0.0.0.0 | "::") host=127.0.0.1 ;; *:*) host="[$host]" ;; esac
token=$(env_value GIQ_TOKEN)
auth=()
[ -z "$token" ] || auth=(-H "Authorization: Bearer $token")

say "waiting for http://$host:$port/status"
for _ in $(seq 60); do
    if curl -fsS --max-time 2 "${auth[@]}" "http://$host:$port/status" >/dev/null 2>&1; then
        say "giq is up on $host:$port"
        exit 0
    fi
    systemctl is-active --quiet giq || break
    sleep 2
done
systemctl --no-pager --lines=30 status giq || true
die "giq did not answer on /status; see journalctl -u giq"
