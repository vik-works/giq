<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# Access and privacy

## Privacy

giq records **that** a job ran, never **what** it said. Prompts, system
messages, chat bodies, images and generated output are held in memory for the
life of the request and are not written anywhere.

What is persisted:

- `data/stats.db` — timestamps, modality, recipe, status, queue wait, duration,
  task count, token counts in/out, the card it ran on with its share of that
  card's time and energy, and the [app](#apps) the job is counted under; per
  card and hour, the energy drawn with no job running and while instances load and unload. The OpenAI `user` field is not recorded. There is no column a prompt could go
  in, and a test asserts that.
- The in-flight log (`GIQ_INFLIGHT_LOG`) — one line per job start
  and end, so a hard power cut can be traced to the job that was running. It
  records the *shape* of a request: input character count, message count,
  attachment count, token budget. Never the text.

This was not always true. Earlier versions of the in-flight log wrote the
full request in plaintext — every prompt and system message — because it was
built for post-mortem replay after a power cut. Identifying an interrupted
job needs a job id, a model and a size; it never needed the content. The log
was wiped and the code now measures rather than records. `tests/test_privacy.py`
pushes a canary string through every logging path and fails if it survives —
including through fields that do not exist yet, since measurement is additive
rather than a list of things to remember to redact.

llama-server runs with `--log-disable`, so the engine does not log requests
either.

Nothing leaves the machine. Every model giq advertises on `/v1/models` is one
it serves itself, from local weights, over a loopback socket — there is no
path that forwards a request to a hosted API.

That was also not always true. Earlier versions let `model: "codex"`
shell out to the Codex CLI with `--full-auto`, which handed the prompt to
OpenAI and let the reply run shell commands here. It sat before the
orchestrator, so it skipped the pause check, the queue, the stats row and the
in-flight log alike: the one kind of request that left the machine was the one
kind giq kept no record of. It is gone, along with its entry in `/v1/models`.
The tests now assert the general shape of that mistake rather than its name —
no API handler may spawn a process, and every advertised model must be
locally served.

## Access

giq has no user model, and on a home rig it does not need one: the caller is
the person who plugged the card in. What it is *not* is a service the browser
may be talked into calling on someone else's behalf. A page you happen to be
reading can aim requests at `127.0.0.1` without ever seeing your machine, and
at the socket that request looks exactly like yours. Two checks close that,
and neither costs an honest client anything:

- **`Host` must name giq.** A DNS-rebinding attack arrives as
  `Host: evil.example` resolving to a local address. An IP literal is always
  accepted — rebinding needs a *name* to re-point — so a phone or an ESP32
  dialling the LAN IP is unaffected.
- **A foreign `Origin` is refused.** Present-and-foreign means a web page is
  asking. curl, scripts and embedded firmware send no `Origin` at all, so the
  rule constrains exactly the caller it is aimed at. giq also sends no CORS
  headers, so a foreign page cannot read a response either.

Behind a reverse proxy, name it:

```yaml
access:
  allow_hosts: [giq.lan]
  allow_origins: ["https://console.example"]
```

## Leaving loopback

`--host 0.0.0.0` puts giq on your network. That is a legitimate thing to
want — it is how a phone or an ESP32 reaches it — but with no token, every
device on that network can submit jobs, read past results and delete model
weights. giq will not stop you; it will not let you not know, either. It says
so at startup:

```
giq on 0.0.0.0:8084 — reachable from your whole network, and no token is set:
any device on it can submit jobs, read past results and delete model weights.
```

and the dashboard header carries an **open to your network** chip while it is
true. `/status.access` reports the same thing for anything else that wants it.

To close it, bind loopback — or set a token:

```bash
GIQ_TOKEN=$(openssl rand -hex 16)   # or access.token in config.yaml
```

With one set, every request needs it, as `Authorization: Bearer …`, an
`X-Giq-Token` header, or `?token=…` for firmware that cannot set headers.
Open the dashboard once as `/dash?token=…` and it moves the value into
`sessionStorage` and attaches it from then on. The page itself and its build
assets (`/dash`, `/dash/assets/…`) load without the token, because a browser
cannot attach a header to a `<script src>`; they are the same public code the
wheel ships, and every API call the page makes carries the token.

This is deliberately a shared secret, not a credential: everything on the LAN
carries the same one, and it identifies nobody. Per-client identity, roles and
an audit trail belong to the enterprise overlay. Calling this that would be
worse than being plain about what it is.

## Apps

An app is a name and a key, so the Usage view can say which app used which
recipes, when and how much ([ADR-006](ADR-006-apps.md)). It is a label for
counting, not an identity: no roles, no scopes, no audit trail.

```bash
giq apps create photo-sorter -d "sorts the photo library"   # prints the key, once
giq apps list
giq apps revoke photo-sorter
```

or the dashboard's **Apps** page, or `POST /apps {"name": "photo-sorter"}`,
whose response is the only place the key appears. giq keeps its SHA-256 in `apps.yaml` (`GIQ_APPS_FILE`, default
in the state directory, mode 0600) and rereads the file when it changes, so a
hand edit needs no restart.

The app sends its key where an API key goes — `api_key` in the OpenAI SDKs,
which is `Authorization: Bearer giq-…` — or as `X-Giq-Token` or `?token=`.
Nothing about it is giq's own: the same app runs against any
OpenAI-compatible server by changing the base URL and the key.

| The caller sends | No token set | Token set |
|---|---|---|
| nothing | served, counted as `anonymous` | 401 |
| the shared token | served, counted as `anonymous` | served, counted as `anonymous` |
| a valid app key | served, counted as the app | served, counted as the app |
| a revoked or unknown key | served, counted as `anonymous` | 401 |

So with a token set, an app key admits a caller on its own, and admits it to
exactly what the shared token does, `/apps` included. Every response carries
`X-Giq-App`, the bucket it was counted under: without a token, that is how a
mistyped key shows up, and a key that never reached the client (an
environment file the process was started without) looks just the same, so
an app whose usage lands in `anonymous` is worth checking for that first.
Jobs recorded before apps existed have no app and show
as `unknown`, never folded into `anonymous`.
