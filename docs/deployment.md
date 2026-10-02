<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# Deployment

Running giq as a system service on a dedicated Debian GPU host: a system
user, one data root (`GIQ_HOME`), a hardened systemd unit, and optionally a
TLS reverse proxy. For a desktop or development checkout, `make
install-service` is enough ([below](#desktop-and-development)).

The files are in [`deploy/`](../deploy):

| File | What |
|------|------|
| `install-debian.sh` | Idempotent installer and updater, run as root |
| `giq.service` | The system unit |
| `giq.env.example` | Template for `/etc/giq/giq.env`: token and overrides |
| `nginx-giq.conf.example` | TLS reverse proxy, SSE-friendly |

## Prerequisites

On Debian 13 (trixie), x86_64:

- **NVIDIA driver ≥ 580.** giq's torch line is CUDA 13 (cu130), which needs a
  580-series driver. Debian's own `nvidia-driver` package is older, so use
  NVIDIA's repository: install `cuda-keyring` from
  `https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/`,
  then `apt install nvidia-open` (RTX 20xx and newer) and reboot.
  `nvidia-smi` must work before you go on.
- **Device nodes at boot.** The service runs with `NoNewPrivileges`, so the
  setuid `nvidia-modprobe` cannot load `nvidia-uvm` the first time CUDA asks
  for it. Load it at boot and keep the driver initialised:

  ```bash
  echo nvidia-uvm > /etc/modules-load.d/nvidia-uvm.conf
  modprobe nvidia-uvm
  systemctl enable --now nvidia-persistenced
  ```

- **Packages:** `apt install git make curl procps psmisc`. giq sweeps stale
  engine processes at startup with `pkill` (procps) and `fuser` (psmisc).
- **uv**, installed system-wide so the unprivileged build user can run it:
  `curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh`.
  uv fetches the Python versions giq needs (3.11 for the side environments,
  which Debian 13 does not ship).
- **The dashboard**, built with Node.js 22.12+. Debian 13's `nodejs` is
  older, so on a server it usually comes prebuilt instead; see
  [Dashboard on the server](#dashboard-on-the-server). Without one the API
  works and `/dash` says "UI not built".
- **CUDA 13 toolkit**, only to build the engines. `nvcc` must be the toolkit
  whose headers and libraries it builds against — mixing an `nvcc` from one
  install with headers from another produces binaries that fail at load.
  Debian's `nvidia-cuda-toolkit` is too old; install `cuda-toolkit-13-0` (or
  newer) from the same NVIDIA repository and put `/usr/local/cuda/bin` first
  on `PATH` while building.

## Layout: `GIQ_HOME`

```
$GIQ_HOME/              /projects/giq in the unit
  config.yaml           the one config
  models/               model weights (GGUFs, OCR/depth/multiview snapshots)
  recipes/              your recipe files (ADR-002, ADR-003)
  engines/              engine builds: engines/<engine>/bin/<binary>
  state/                stats.db, inflight.log
  cache/                huggingface/, torch, Triton and CUDA JIT caches
```

`python -m giq init [--home PATH]` (or `giq init`) creates the tree and a
commented `config.yaml` from the packaged template. It never overwrites a
file, so running it again is harmless; it prints what it created and every
resolved location.

Each location resolves, most specific first: its environment variable, the
`paths:` block of `config.yaml`, the `GIQ_HOME` layout, the plain-checkout
default. Entries in `paths:` are absolute or relative to `GIQ_HOME`, and `~`
is expanded.

| Location | Environment | `config.yaml` | With `GIQ_HOME` | Without |
|----------|-------------|---------------|-----------------|---------|
| Config file | `GIQ_CONFIG` | — | `$GIQ_HOME/config.yaml` | `./config.yaml` |
| Models | `GIQ_MODELS_DIR` | `paths.models` | `$GIQ_HOME/models` | `~/models` |
| Recipes | `GIQ_RECIPES_DIR` | `paths.recipes` | `$GIQ_HOME/recipes` | `~/.config/giq/recipes` |
| Engines | `GIQ_ENGINES_DIR` | `paths.engines` | `$GIQ_HOME/engines` | — (PATH) |
| State | `GIQ_DATA_DIR` | `paths.state` | `$GIQ_HOME/state` | `$STATE_DIRECTORY`, else `<checkout>/data` |
| Stats DB | `GIQ_STATS_DB` | — | `<state>/stats.db` | `<state>/stats.db` |
| In-flight log | `GIQ_INFLIGHT_LOG` | — | `<state>/inflight.log` | `<state>/inflight.log` |
| Caches | `GIQ_CACHE_DIR` | `paths.cache` | `$GIQ_HOME/cache` | `$CACHE_DIRECTORY`, else the libraries' own |
| HF home | `HF_HOME` | — | `<cache>/huggingface` | `~/.cache/huggingface` |

The config file's location never comes from the config itself. The
per-modality weight variables (`GIQ_OCR_MODEL_DIR` and friends) still
outrank the recipe files' weight paths for their built-ins (see
[Weights](configuration.md#weights)).

When giq has a cache directory it exports `HF_HOME`, `XDG_CACHE_HOME`,
`TRITON_CACHE_DIR`, `CUDA_CACHE_PATH` and `MPLCONFIGDIR` under it — to its
own process and to every engine and child it spawns — unless you set them
yourself. That keeps downloads where the inventory looks and JIT caches
somewhere the service can write. `$STATE_DIRECTORY` and `$CACHE_DIRECTORY`
(set by systemd for units with `StateDirectory=`/`CacheDirectory=`) are
used only when neither `GIQ_HOME` nor a more specific setting names one.

`GET /storage` includes a `paths` object with every resolved location, so
you can check what a running giq uses.

The side environments (`envs/*/.venv`) and the dashboard build are code,
not data: they stay in the checkout.

## Engines

giq does not build engines. Put each build under `$GIQ_HOME/engines`, where
giq finds it without configuration; `engines:` in `config.yaml` or
`GIQ_LLAMA_BINARY`/`GIQ_SDCPP_BINARY` still outrank it, and PATH is the last
resort. Building statically keeps each engine one self-contained file that
can be copied into place (a shared build's RUNPATH points into its build
tree):

```bash
export PATH=/usr/local/cuda/bin:$PATH
E=/projects/giq/engines

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j --target llama-server
install -D build/bin/llama-server $E/llama.cpp/bin/llama-server

git clone --recursive https://github.com/leejet/stable-diffusion.cpp && cd stable-diffusion.cpp
cmake -B build -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
install -D build/bin/sd-server $E/sd.cpp/bin/sd-server
```

`native` targets the cards in the build machine; name architectures
explicitly (`120` for RTX 50) when building elsewhere. Then check that the
CUDA libraries resolve: `ldd $E/llama.cpp/bin/llama-server | grep 'not found'`
must print nothing. If `libcublas`/`libcudart` are missing at runtime, add
the toolkit's `lib64` to `/etc/ld.so.conf.d/` and run `ldconfig` (or set
`LD_LIBRARY_PATH` in `/etc/giq/giq.env`). `GET /engines` shows the binary
and version giq resolved. Details on engines: [engines.md](engines.md).

## Install

```bash
sudo deploy/install-debian.sh            # defaults below
sudo deploy/install-debian.sh --home /srv/giq --prefix /opt/giq --ref v0.5.3
```

| Option | Default |
|--------|---------|
| `--home` | `/projects/giq` |
| `--prefix` | `/opt/giq` |
| `--repo` | `https://github.com/vik-works/giq.git` |
| `--ref` | `main` |
| `--build-user` | `giq-build` |
| `--no-token`, `--no-start` | generate a token; start the unit |
| `--ui-tarball PATH`, `--ui-release TAG`, `--ui-only` | see [Dashboard on the server](#dashboard-on-the-server) |
| `--with-vllm` | off: see [vllm](#vllm-optional) |

It checks the prerequisites before changing anything, then:

1. creates the system user `giq` (home `GIQ_HOME`, no shell, in `video` and
   `render`) and the build user `giq-build`;
2. clones or updates the checkout at `--prefix`, owned by the build user and
   readable by everyone — the service can read its code but not change it;
3. runs `make sync` as the build user, with uv's Pythons installed under
   `<prefix>/.uv-python` so the service user can execute them, and installs
   the dashboard ([below](#dashboard-on-the-server));
4. runs `giq init` as `giq`;
5. creates `/etc/giq/giq.env` (root:giq, 0640) from `giq.env.example` with a
   generated `GIQ_TOKEN`, if the file does not exist yet;
6. installs `/etc/systemd/system/giq.service` with your paths substituted,
   enables and (re)starts it, and waits for `/status` to answer.

It never downloads models and never overwrites `config.yaml` or `giq.env`.
It warns when no `llama-server` is found in `engines/` or on PATH.

Models go under `$GIQ_HOME/models` (paths as the recipe files name them;
see [configuration.md](configuration.md#recipes)). Weights loaded by library name —
faster-whisper, pyannote, speechbrain, kokoro — live in
`$GIQ_HOME/cache/huggingface`. The unit sets `HF_HUB_OFFLINE=1`, so fetch
them beforehand as a user who can write there, with
`HF_HOME=$GIQ_HOME/cache/huggingface`.

## vllm (optional)

The second LLM engine ([engines.md](engines.md#vllm)) is not installed by
default: its interpreter is several GB of wheels and its GPU kernels take
minutes and tens of GB of RAM to build. `--with-vllm` adds both steps to an
install or update:

```bash
sudo deploy/install-debian.sh --with-vllm
```

1. `uv sync` in `envs/vllm`, as the build user with uv's Pythons under
   `<prefix>/.uv-python`, like `make sync`;
2. `giq prepare vllm` as `giq`, so the kernels land in `$GIQ_HOME/cache`
   where the service can also write the small ones it compiles on first use.
   It builds once per distinct compute capability among the cards (two RTX
   PRO 6000 Blackwell are one build, `12.0f`), inside a scope root creates
   with `MemoryMax=40G` and no swap — two compile jobs peak near 15 GB each.
   Run it again after updating vllm; it only rebuilds what changed.

Then, by hand once the weights are in place, warm each vllm recipe up once
so its first real start does not spend tens of minutes compiling (see
[engines.md](engines.md#prepare-once-giq-prepare-vllm)) — with giq paused,
since it uses the card:

```bash
T=$(sudo sed -n 's/^GIQ_TOKEN=//p' /etc/giq/giq.env)
curl -X POST localhost:8084/control/pause -H "Authorization: Bearer $T" \
    -H 'content-type: application/json' -d '{}'
sudo systemd-run --scope -p MemoryMax=40G -p MemorySwapMax=0 -- \
    runuser -u giq -- env GIQ_HOME=/projects/giq HOME=/projects/giq \
    /opt/giq/.venv/bin/python -m giq prepare vllm --memory-max none \
    --recipe qwen3.8-27b-nvfp4
curl -X POST localhost:8084/control/resume -H "Authorization: Bearer $T"
```

Under the unit, giq cannot give vllm a scope of its own (the `giq` user has
no user manager); the unit's `MemoryMax=90%` is then the ceiling for giq and
its engines together, which is why the kernels are built beforehand rather
than inside the service. Make sure the machine's RAM holds the checkpoint
once more than its size while vllm loads it.

The built-in vllm recipes expect their weights under `$GIQ_MODELS_DIR`
(`nvidia-Qwen3.8-27B-NVFP4`); fetch them beforehand, e.g.
`hf download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f --local-dir
$GIQ_HOME/models/nvidia-Qwen3.8-27B-NVFP4`. Pin one resident from the
dashboard or with `residents:` in `config.yaml` once it runs: its start takes
minutes, which a user's first request should not pay.

## Dashboard on the server

The dashboard is static files in `<prefix>/src/giq/static/ui/`, built from
`frontend/` with Node.js 22.12+ and read by giq on every request, so
replacing them needs no restart. The installer takes them from the first of:

1. **`--ui-tarball PATH`**: a `giq-ui-<version>.tar.gz` made by `make
   ui-dist` on any machine with Node.js 22.12+ (the contents of the build,
   paths relative to it, plus a `.sha256` beside it).
2. **`--ui-release TAG`**: the `giq-ui-<version>.tar.gz` attached to that
   GitHub release of `--repo`, fetched through the REST API and checked
   against the release's `SHA256SUMS`. While the repository is private this
   needs a token that can read it (a fine-grained token with read access to
   the repository's contents) in `GIQ_GITHUB_TOKEN`; `sudo` drops the
   environment, so pass it with `sudo GIQ_GITHUB_TOKEN=… ` or
   `sudo --preserve-env=GIQ_GITHUB_TOKEN`. The token goes to curl on stdin,
   not on the command line.
3. **A local build** (`make ui`) as the build user, when Node.js 22.12+ is on
   its `PATH` (`/usr/local/bin`, `/usr/bin`).

With none of them it keeps whatever dashboard is already there, warns that it
may not match, and prints the commands below. Either kind of tarball is
checked (relative paths, plain files, `index.html` and
`THIRD_PARTY_LICENSES.txt` present), unpacked next to the current dashboard
and renamed into place, owned by the build user and readable by `giq`; a bad
tarball leaves the old dashboard untouched. A tarball whose version differs
from the checkout's installs with a warning.

`--ui-only` does only this step: no other change, no restart, and no root
needed if you own the checkout.

Recommended while the repository is private: build on a workstation and copy
the tarball over, so the server needs no GitHub credentials. In a checkout of
the commit the server runs:

```bash
make ui-dist                     # dist/giq-ui-<version>.tar.gz and .sha256
scp dist/giq-ui-0.5.3.tar.gz dist/giq-ui-0.5.3.tar.gz.sha256 server:/tmp/
```

On the server:

```bash
cd /tmp && sha256sum -c giq-ui-0.5.3.tar.gz.sha256
sudo /opt/giq/deploy/install-debian.sh --ui-only --ui-tarball /tmp/giq-ui-0.5.3.tar.gz
# or as part of an install or update, with the same options as before:
sudo deploy/install-debian.sh --ref v0.5.3 --ui-tarball /tmp/giq-ui-0.5.3.tar.gz
```

From a tagged release instead:

```bash
sudo GIQ_GITHUB_TOKEN=github_pat_… deploy/install-debian.sh --ref v0.5.3 --ui-release v0.5.3
```

Re-run with the matching tarball after every update that changes the
dashboard; `git checkout` leaves the old build (it is gitignored) in place.

## The unit

`deploy/giq.service` runs `python -m giq.main` as `giq` from the checkout,
with `GIQ_HOME` set and `/etc/giq/giq.env` read on top. Every directive
carries its reason in the file; the parts that matter operationally:

- **Read-only except state, cache and recipes** (`ProtectSystem=strict`,
  `ReadWritePaths=`). Models and engines are read-only to the service, so
  deleting weights from the dashboard is refused; do it as the operator. A
  directory you move elsewhere with `paths:` or an environment variable
  must be added to `ReadWritePaths` if giq writes to it (`systemctl edit
  giq`).
- **Not `PrivateDevices`** (the GPU is `/dev/nvidia*`) and **not
  `MemoryDenyWriteExecute`** (CUDA, torch and Triton generate code at
  runtime).
- **`MemoryHigh=80%`, `MemoryMax=90%`** of RAM: a runaway load hits giq, not
  the host. Tune them to the machine.
- **Stopping** gives open streams 30 s, then unloads every model;
  `TimeoutStopSec=90` covers that, and `KillMode=mixed` SIGKILLs any engine
  still alive at the end.
- The startup sweep (`pkill`, `fuser`) runs as `giq`, so it can only reach
  giq's own leftover engines.
- An optional `IPAddressDeny=any` block turns "no outbound connections" into
  a property of the unit.

Change the unit with a drop-in (`systemctl edit giq`) so an update can
replace the file.

## Access

The token lives in `/etc/giq/giq.env` as `GIQ_TOKEN`, which outranks
`access.token` in `config.yaml`. The environment file is readable by root
and the service only, while `config.yaml` is world-readable by default and
travels with every backup of `GIQ_HOME`. Clients
send it as `Authorization: Bearer <token>`, `X-Giq-Token: <token>` or
`?token=<token>`. The full rules: [access-and-privacy.md](access-and-privacy.md).

Two ways to reach giq from other machines:

- **Reverse proxy (recommended).** giq stays on `127.0.0.1:8084`; nginx
  terminates TLS ([`nginx-giq.conf.example`](../deploy/nginx-giq.conf.example):
  unbuffered for streaming, long read timeouts, large bodies for OCR and
  audio). Add the public name to `access.allow_hosts`, since nginx forwards
  it as Host. giq is on loopback here, so its startup banner does not ask
  for a token — set one anyway: everything that reaches nginx reaches giq.
- **Direct LAN bind.** Set `GIQ_HOST=0.0.0.0` (or an interface address) in
  `giq.env`, keep the token, and allow the port from your LAN only:
  `nft add rule inet filter input ip saddr 192.168.1.0/24 tcp dport 8084 accept`
  (or the ufw equivalent). No TLS: the token crosses the network in clear.

## Updates

Re-run the installer with the same options: it fetches `--ref`, re-syncs,
installs the dashboard, replaces the unit and restarts. Pass a
`--ui-tarball` or `--ui-release` for the new version if the server cannot
build the dashboard. By hand, the same steps are:

```bash
sudo -u giq-build git -C /opt/giq fetch origin
sudo -u giq-build git -C /opt/giq checkout --detach origin/main
sudo -u giq-build env UV_PYTHON_INSTALL_DIR=/opt/giq/.uv-python make -C /opt/giq sync
sudo systemctl restart giq
```

A restart stops every instance; the resident set reloads on its own.

## Backups

Back up `$GIQ_HOME/config.yaml`, `$GIQ_HOME/recipes/`, `$GIQ_HOME/state/`
and `/etc/giq/giq.env`. `stats.db` is SQLite in WAL mode: copy it with
`sqlite3 stats.db ".backup /path/to/copy.db"`, or stop giq first. Models,
engines and caches can be re-fetched or rebuilt; back them up only if that
is slower than restoring.

## Troubleshooting

- **Logs:** `journalctl -u giq -f`. Status: `curl -H "Authorization: Bearer $TOKEN"
  http://127.0.0.1:8084/status`; resolved directories: `/storage` → `paths`.
- **No GPU / `CUDA_ERROR_NO_DEVICE` / `cudaGetDeviceCount` fails:** check
  `ls -l /dev/nvidia*` (world read-write by default) and that
  `/dev/nvidia-uvm` exists (see Prerequisites); `sudo -u giq nvidia-smi`
  must work.
- **`Read-only file system` in the log:** something wrote outside state,
  cache and recipes — often a library's cache under `~`. Point its
  variable at `$GIQ_HOME/cache` in `giq.env`, or add the path to
  `ReadWritePaths`.
- **Killed with `oom-kill` in the journal:** the unit hit `MemoryMax`.
  `systemctl show giq -p MemoryPeak` shows how high it went; raise the limit
  with a drop-in or load fewer models at once.
- **`engine 'llama.cpp' binary not found`:** build it into
  `$GIQ_HOME/engines/llama.cpp/bin/` (see Engines) or declare it in
  `config.yaml`.
- **`status=226/NAMESPACE`:** a `ReadWritePaths` directory does not exist.
  Run `sudo -u giq GIQ_HOME=/projects/giq /opt/giq/.venv/bin/python -m giq init`.
- **403 "does not name this server":** add the name clients use to
  `access.allow_hosts`.

## Desktop and development

`make install-service` writes a systemd *user* unit that runs giq as you
from the checkout — no dedicated user, no hardening. With `GIQ_HOME` set in
your shell, the unit carries it, so the service sees the same config, models
and state you do; without it, giq uses the checkout defaults (`./config.yaml`,
`~/models`, `data/`). `make run` serves in the foreground the same way.
