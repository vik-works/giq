<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# Engines

giq executes models through declared engines — one binary per runtime. Out of
the box that is `llama-server` / `sd-server` as found on PATH; declaring an
absolute path pins the exact build, and a declared binary that is missing
fails loudly instead of resolving to whatever else is installed. Set one in
`config.yaml` or with `GIQ_LLAMA_BINARY` / `GIQ_SDCPP_BINARY`:

```yaml
engines:
  llama.cpp: /path/to/llama.cpp/build/bin/llama-server
  sd.cpp: /path/to/stable-diffusion.cpp/build/bin/sd-server
```

`GET /engines` reports the build each one identifies as. That matters more
than it sounds: a llama.cpp binary can outlive the source tree it was built
from, and nothing else in the system would notice.

Note that recent llama.cpp builds link shared libraries with an absolute
RUNPATH into the build directory — point the config at the build, don't copy
the binary out of it.

Besides the two servers, two interpreters are declared as engines, so every
model in the catalog names what executes it: `python` (giq's own venv — the
audio stack, Kokoro, both OCR models and depth) and `vllm` (`envs/vllm`, the
second LLM engine, [below](#vllm)); `GIQ_VLLM_PYTHON` points elsewhere.

## Engine names

Each engine has one name, used alike in recipe files (`engine:`),
`config.yaml` (`engines:`), the catalog (`/recipes`: `engine`) and the dashboard:

| Engine | Runs | Executed by (`runtime`) |
|--------|------|-------------------------|
| `llama.cpp` | LLMs | `llama-server` |
| `vllm` | LLMs from vendor checkpoints (NVFP4, FP8), many users at once | `envs/vllm` |
| `sd.cpp` | `flux_klein`, `zimage` | `sd-server` |
| `transformers` | Unlimited-OCR, GLM-OCR, depth | `python` |
| `faster-whisper`, `faster-whisper+pyannote`, `speechbrain`, `kokoro` | speech to text, the audio stack, voiceprints, text to speech | `python` |

The old spelling `sdcpp` is still accepted wherever an engine is named and
read as `sd.cpp`, with a deprecation warning.

A recipe's runtime is its `engine`. A recipe's VRAM figure is measured
under its engine; switching engine means measuring again.

## vllm

A second LLM engine next to llama.cpp, for two things llama-server does not
do: vendor checkpoints that run natively only on vllm (NVIDIA's ModelOpt
NVFP4 and FP8), and many users at once — vllm's paged KV cache and
continuous batching serve up to `max_num_seqs` requests from one shared pool
where llama-server hands out fixed slots. Measured on an RTX 5090 with
NVIDIA's NVFP4 checkpoint of Qwen3.8-27B, 256 generated tokens per request:

| Parallel requests | vllm, fp8 KV | llama.cpp Q6_K through giq (one slot) |
|---|---|---|
| 1 | 72 tok/s, first token 0.1 s | 56 tok/s |
| 4 | 242 tok/s in total | 56 tok/s in total, first token 9.5 s (median) |
| 16 | 937 tok/s in total, first token 0.2 s | 56 tok/s in total, first token 37 s (median) |

On batch extraction — a ~3.1k-token document in, ~490 tokens of
schema-constrained JSON out, thinking off — the work is mostly reading, and
the card saturates early. Same card, same checkpoint:

| `qwen3.8-27b-nvfp4` | Documents/min | Input tok/s | Output tok/s | Slowest 5% |
|---|---|---|---|---|
| 16 parallel, no speculation | 71 | 3,640 | 580 | 27 s |
| 64 parallel, no speculation | 65–71 | 3,350–3,670 | 530–580 | ~140 s |
| 16 parallel, MTP 1 draft token | 47 | 2,400 | 380 | 112 s |
| 16 parallel, MTP 2 draft tokens | 66 | 3,400 | 540 | 70 s |

More than 16 parallel only lengthens the queue, and a larger
`max_num_batched_tokens` bought nothing (at 16k it ran out of memory under
load). One request at a time is a different matter — speculation is nearly
free there, and JSON drafts well:

| Single request, tok/s | none | MTP 1 | MTP 2 | MTP 3 |
|---|---|---|---|---|
| vllm, NVFP4 | 69 | 108 | 138 | 162 |
| llama.cpp, Q6_K (`--spec-type draft-mtp`) | 52 | 72 | 82 | 89 |

The two files are the same size (20.4 GiB), so a single request reads the
same bytes per token on both; the gap is the engine. Hence two recipes over
one checkpoint: `qwen3.8-27b-nvfp4` for batch, `qwen3.8-27b-nvfp4-chat`
(MTP, 3 draft tokens) for the fastest single answer.

The power limit matters to batch only: 72 documents a minute at 575 W
(481 W drawn), 67 at 400 W (396 W) — 13% more documents per kWh for 7%
fewer per minute. A single request drew ~410 W and ran at 162 tok/s at
every limit from 400 to 575 W.

What it costs: a start of minutes rather than seconds (192 s the first time,
64 s of it CUDA graph capture; 82 s once its compile cache is warm), and
memory claimed up front. A vllm recipe is meant to be pinned resident, not
loaded on some user's first request; the built-ins are not resident by
default so that pinning is the operator's decision.

### The interpreter: `envs/vllm`

vllm pins its own torch, transformers and fastapi ranges, so it has its own
uv project. giq's venv runs the same torch line (2.13 on CUDA 13). It is not
part of `make sync` —
several GB of wheels most installs do not need:

```bash
cd envs/vllm && uv sync     # or: deploy/install-debian.sh --with-vllm
```

`GIQ_VLLM_PYTHON` points at a different interpreter; `vllm serve` is the
console script next to it. The CUDA compiler parts in that env — `nvcc`,
`crt`, `nvvm` (cicc) and `nvjitlink` — are held on 13.0, the minor version of
torch's CUDA runtime headers: FlashInfer compiles kernels on first use with
that nvcc against those headers, libcudacxx refuses a compiler newer than
its headers, and a newer cicc emits PTX the pinned ptxas rejects. The system
toolkit plays no part — giq points `CUDA_HOME` at the env's `nvidia/cu13`.

### Prepare once: `giq prepare vllm`

FlashInfer's NVFP4 and FP8 GEMM modules take minutes to build and more than
20 GB of RAM, and vllm would build them inside the first start. Build them
ahead of time, once per machine and vllm version:

```bash
giq prepare vllm             # every card, once per distinct compute capability
giq prepare vllm --gpu 0     # one card
```

It runs in a transient systemd scope capped at `--memory-max` (default 40G,
no swap) with two compile jobs — one `cicc` per job peaks near 10 GB, so the
cap is sized at ~15 GB per job. The build goes through ninja every time, so
what is current is left alone and an interrupted build resumes where it
stopped. Kernels are tied to the env they were built from — the build files
carry its absolute paths — so a second checkout, or a moved one, builds
again.

The GEMMs are the big part, not all of it: a first start still compiles
FlashInfer's attention and sampling modules for the model's shapes, runs
torch.compile and captures CUDA graphs — tens of minutes from an empty
cache (vllm 0.30 on an RTX 5090: 819 s in its warm-up run alone), longer
than a start may take. `--recipe` does one full start of an
instance with an hour to spare and stops it again, so every later start
finds the caches warm (82 s on an RTX 5090). It uses the card while it runs;
pause giq first if it is serving:

```bash
giq prepare vllm --gpu 0 --recipe qwen3.8-27b-nvfp4 --recipe qwen3.8-27b-nvfp4-chat
``` Compute capability 12.0 (RTX 50,
RTX PRO 6000 Blackwell) builds the `12.0f` family target; 10.x (B200, B300)
the `sm_100` modules. The kernels land in `$GIQ_HOME/cache/.cache/flashinfer`
(without a cache dir, `~/.cache/flashinfer`), where the server finds them.

### How giq runs it

Per instance, giq spawns `vllm serve <weights> --served-model-name <name>` on
a loopback port (8088, +10 per card index; a second vllm on the same card
takes a free port of that card's block, 8086-8095 on the first) with the
card's `CUDA_VISIBLE_DEVICES`, and proxies it like llama-server. The process
runs:

- **under a RAM ceiling** — `systemd-run --user --scope -p MemoryMax=<memory_max>
  -p MemorySwapMax=0`, so if anything balloons (a kernel compile the prepare
  step missed) the OOM killer can only take the engine. A process without a
  user systemd — giq as a system service — runs without a scope of its own
  and says so in the log; there the unit's own `MemoryMax` is the ceiling.
  (`RLIMIT_AS` is no substitute: a CUDA process maps far more address space
  than it touches.)
- **offline and quiet** — `HF_HUB_OFFLINE=1`, and vllm's usage statistics off.
- **with its caches under `$GIQ_HOME/cache`** — FlashInfer, vllm's compile
  cache and config, Triton.
- **with its output in `<state>/logs/vllm-<name>.log`**; a start that fails
  reports the log's last lines.

Ready means `/health` answers and `/v1/models` lists the name, within
`ready_timeout` (600 s). Stop sends SIGTERM to the process group, SIGKILL
after 30 s, then stops the scope, so the engine core that holds the card goes
with the API server. A giq that died without stopping it leaves the scope
behind; the next start (and giq's startup sweep) stops `giq-vllm-<port>.scope`.

vllm names the thinking channel `reasoning`; giq relays it as
`reasoning_content`, the name its llama.cpp path and API use. giq's loop
guard needs llama-server's control endpoint and does nothing on vllm.

### Parameters

| Parameter | Meaning |
|---|---|
| `kv_cache_memory` | KV cache size, e.g. `6G` (binary units) or bytes. **The preferred budget** |
| `gpu_memory_utilization` | The alternative: a fraction of the card for weights + activations + KV. Exactly one of the two is required |
| `max_model_len` | Context per request. Required — the native window may not fit the pool |
| `max_num_seqs` | Requests scheduled at once; giq dispatches exactly this many (the lane width is derived from it) |
| `max_num_batched_tokens` | Tokens per scheduler step; larger admits a long prefill sooner |
| `kv_cache_dtype` | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` |
| `speculative` | `{method: mtp, tokens: N}` — drafts with the checkpoint's own MTP head; refused for weights without one |
| `enforce_eager` | Skip CUDA graphs: faster start, slower decoding |
| `reasoning_parser`, `tool_call_parser` | vllm's parser names (`qwen3`, `qwen3_coder`); a tool parser turns on automatic tool choice |
| `chat_template_file` | Jinja file under `GIQ_MODELS_DIR` replacing the checkpoint's own template (a fixed upstream Qwen file, say); unset = the checkpoint's own |
| `structured_outputs` | Constrained decoding for `json_schema` requests: `{backend: xgrammar\|guidance\|auto, disable_any_whitespace: bool}`. Defaults to `{xgrammar, true}` — vllm's own default (`auto`, free whitespace) lets a large schema diverge into an unbounded whitespace run that never closes the object |
| `memory_max` | RAM ceiling of the engine process (default `40G`; `null` = none) |
| `ready_timeout` | Seconds a start may take (default 600) |

A multimodal checkpoint whose recipe does not list the `vision`
capability is started with `--language-model-only`.

**Why a KV size rather than a fraction.** `gpu_memory_utilization: 0.93` is
29 GB on a 32 GB card and 89 GB on a 96 GB one: on a big card shared with
other models it takes everything. A KV budget means the same on any card,
and giq's VRAM figure for the recipe is then the sum of three measured
parts — `vram.weights_gb` (vllm logs "Model loading took …") +
`kv_cache_memory` + `vram.overhead_gb` (CUDA context, activations, graphs,
the vision encoder's profile) — which the recipe file may state instead of
`vram.gb`. vllm still refuses to start unless free memory covers
`gpu_memory_utilization × card` (0.9 by default) even with a byte budget, so
giq passes the recipe's own figure over the card's size, and the check
means what it should.

### Profiles

A recipe names a profile and overrides single values under `params`;
the profile never sets the budget.

| Profile | Speculation | `max_num_seqs` | `max_num_batched_tokens` | KV |
|---|---|---|---|---|
| `interactive` | MTP, 2 draft tokens | 4 | 8192 | fp8 |
| `throughput` | none | 32 | vllm's default | fp8 |

The split is measured, not a preference. On the RTX 5090 with the NVFP4
Qwen3.8-27B, MTP with three draft tokens took one request from 72 to
117 tok/s, but the total at 16 parallel requests from 937 down to 750,
accepting 0.66, 0.37 and 0.17 drafts by position — the third is mostly
wasted, so `interactive` drafts two. MTP also lengthened the first token
under load through vllm's default scheduling budget, hence 8192. 16 parallel
requests was the measured point for `throughput`; 32 is only a ceiling over
the paged pool, so it suits a large card without costing a small one memory.

### The built-in recipes

`qwen3.8-27b-nvfp4` (throughput) and `qwen3.8-27b-nvfp4-chat`
(interactive) serve `nvidia/Qwen3.8-27B-NVFP4` from
`$GIQ_MODELS_DIR/nvidia-Qwen3.8-27B-NVFP4`. Their KV budgets keep them
loadable on a 32 GB card through giq's gate; on a larger card, replace the
file in your recipes directory with a larger `kv_cache_memory` (the VRAM
figure follows) — see [configuration.md](configuration.md#recipes):

```yaml
# ~/.config/giq/recipes/qwen3.8-27b-nvfp4.yaml — on a 96 GB card
name: qwen3.8-27b-nvfp4
modalities: [llm]
engine: vllm
weights:
  path: nvidia-Qwen3.8-27B-NVFP4
  format: modelopt
capabilities: [chat, vision]
profile: throughput
params:
  kv_cache_memory: 24G       # ~720k fp8 tokens
  max_model_len: 131072
  max_num_seqs: 64
  reasoning_parser: qwen3
  tool_call_parser: qwen3_coder
vram:
  weights_gb: 19.92          # measured, as in the built-in
  overhead_gb: 3.5           # measured, as in the built-in
  measured: false            # the sum was not observed at this budget
max_batch: 32
```
