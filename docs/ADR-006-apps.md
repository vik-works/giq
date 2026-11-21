<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# ADR-006: Apps — who used the GPU, for what, and how much it cost

**Status:** Accepted
**Date:** 2026-11-20
**Authors:** giq maintainers

## Context

giq records every job: its recipe, modality, queue wait, run time, tokens
and card (`jobs` in the stats database). It does not record *who sent it*.
On a rig that serves a chat client, a game's asset pipeline, a home
automation box and a few scripts, the question the operator asks is per
caller: which app uses which recipes, when, how much, and what that costs in
GPU time and energy.

What giq has for callers today is the shared token (`access.token`,
`GIQ_TOKEN`; `docs/access-and-privacy.md`). It is a secret every client on
the LAN shares, and it identifies nobody. That document says per-client
identity belongs to an enterprise overlay. This ADR does not add identity in
that sense: no users, no roles, no scopes, no audit trail. It adds a label
for counting, carried the way clients already carry a key.

Two constraints shape it:

- **Nothing an app sends is giq's own.** An app talks to giq the way it
  talks to any OpenAI-compatible server: a base URL and an API key. Swapping
  giq for OpenAI, vLLM, LiteLLM or OpenRouter means changing those two
  values and nothing else.
- **Unlabelled traffic keeps working.** Today's clients (curl, firmware,
  scripts) send no key or the shared token, and they must not break.

Energy is the hard part. `gpu_samples` records each card's power draw every
30 s, too coarse for jobs that finish in seconds. Several jobs can share a
card at once (parallel LLM slots, several instances on one card). And a card
draws power while recipes sit resident and idle, which no job caused.
`jobs.gpu_uuid` also records the first card, not the card the job ran on.

## Decision

### D1: An app is a name and a key, kept in a plain file

An app has a **name** (a slug, its identity, never changed; a rename is a
new app), an optional description, a creation time, the hash of its key,
and an optional revocation time. That is all.

Apps live in `apps.yaml` in the state directory by default, overridable by
`GIQ_APPS_FILE` and `paths: apps:` through `giq.paths`. It is a file of its
own, not a block of `config.yaml`, because giq writes it (D6) and must not
rewrite a file the operator edits by hand and comments; and it sits in the
state directory because that is where the hardened unit lets the service
write. The file is mode 0600: with a token set, the hashes stand for keys
that admit a caller. Writes go to a
temporary file renamed over the old one; giq rereads the file when its mtime
changes, so a hand edit takes effect without a restart.

```yaml
apps:
  kompostwars:
    description: asset pipeline
    created: 2026-11-20T10:12:00Z
    key_sha256: 9f2c…
  home-assistant:
    created: 2026-11-20T10:15:00Z
    key_sha256: 41ab…
    revoked: 2026-11-21T08:00:00Z
```

The list is small and changes rarely, like configuration, not like
telemetry. A file can be read, backed up, kept in git and provisioned
without the dashboard. Usage is telemetry and goes to the stats database
(D4).

### D2: The key travels the way keys already do

A key is `giq-` followed by 32 random bytes, base64url. It is shown once,
when the app is created; giq keeps only its SHA-256. A plain hash suffices:
the key is random and long, so there is nothing for a slow hash to protect
against.

giq reads the key from the places `access.token_ok` already reads the
shared token: `Authorization: Bearer` (what the OpenAI SDKs send as
`api_key`), `X-Giq-Token`, and `?token=` for firmware that cannot set a
header. An app needs no giq-specific header and no giq-specific field.

### D3: Who is counted as what

| The caller sends | No `access.token` set | `access.token` set |
|---|---|---|
| nothing | served, counted as `anonymous` | 401, as today |
| the shared token | served, counted as `anonymous` | served, counted as `anonymous` |
| a valid app key | served, counted as the app | served, counted as the app |
| a revoked or unknown key | served, counted as `anonymous` | 401 |

`anonymous` is a reserved name, a bucket like any app's, so the per-app
totals add up to what the cards did.

With a token set, a valid app key also gets a caller in, so an app carries
one secret, not two. That makes an app key a credential on such a rig, in
the same plain sense as the shared token: it admits, it does not grant
anything the shared token does not. Revoking it shuts the app out.

Without a token, a mistyped key quietly lands in `anonymous`. To make that
visible, every response carries `X-Giq-App: <name>`, the bucket the request
was counted under. Other clients ignore the header; it is information, not
protocol.

### D4: Attribution goes through the chokepoint

The access middleware resolves the app once per request and sets it in a
context variable. `Orchestrator.submit_job` and `submit_streaming_job` stamp
it onto the `Job`. It is never read from the request body, so a caller
cannot claim another app's name. Every route, the job API, `/v1/*` and the
plugins' routes alike, is attributed without knowing apps exist.

The stats database gets one column, `jobs.app` (text, the app's name).
Rows from before this ADR stay NULL, shown as "unknown", not folded into
`anonymous`. Since the name is never reused, a revoked or deleted app's
history stays readable.

The OpenAI `user` field is not recorded. It is the caller's own label for
its end user, often personal data, and giq's privacy rules
(`docs/access-and-privacy.md`) keep request content out of what outlives the
request.

### D5: Energy is measured per card and split by overlap

Each job records the card it ran on (fixing `jobs.gpu_uuid`), its
**GPU-seconds**, and its **energy** in watt-hours.

- **Source.** The card's cumulative energy counter
  (`nvmlDeviceGetTotalEnergyConsumption`, millijoules, Volta and later) is
  read at job start and end. nvidia-smi does not expose it, so this takes
  NVML bindings (`nvidia-ml-py`). Where the counter is unavailable, giq
  integrates the power samples over the job's run and marks the figure as an
  estimate.
- **Sharing a card.** When jobs overlap on one card, each interval's energy
  is split evenly between the jobs running in it. Splitting by tokens was
  considered and rejected: tokens mean nothing for image, audio and depth
  jobs.
- **Idle and residency.** Energy a card draws with no job running is
  attributed to nobody. Usage reports it as its own line, *idle/residency*,
  per card: the card's total over the period minus what jobs were charged.
  Spreading it over the apps would charge an app for recipes it never asked
  to keep loaded.

Every energy figure giq shows is labelled as attributed by this rule. It is
a fair split, not a meter reading per job.

### D6: Onboarding and reporting

- `POST /apps {name, description?}` returns the app and its key, once.
  `GET /apps` lists apps without keys. `POST /apps/{name}/revoke` revokes;
  a revoked entry keeps its name taken. `DELETE /apps/{name}` removes the
  entry: the history stays (D4), and an app created again under the name
  adds to it, which is what deleting rather than revoking says. The CLI
  mirrors them: `giq apps create|list|revoke|delete`, working on the file
  with or without a running giq.
- `GET /stats/usage` rolls up jobs, tokens and run time per app and recipe
  (`apps`), and `?app=` restricts every figure to one app (`unknown` for
  jobs from before apps). GPU-seconds, energy and the idle/residency line
  join it with D5.
- The dashboard's Usage view gets a breakdown by app, and an Apps page to
  create, revoke and copy a key once.

## Alternatives considered

- **A proxy in front (LiteLLM's virtual keys).** It counts requests and
  tokens per key, but it cannot see GPU-seconds, cards or energy, which only
  the scheduler knows. Both work together: a proxy in front of giq with one
  app key of its own is still possible.
- **Apps in SQLite.** It works, but it gives up a file the operator can read,
  edit and provision by hand, for a list that holds a handful of rows.
- **A giq-specific header (`X-Giq-App: name`).** Free to spoof, and an app
  that sends it is tied to giq. The key is what clients already send.
- **Not counting unauthenticated traffic.** The totals would then not add up
  to what the cards did, and "what does this rig burn" would have no answer.

## Consequences

- `docs/access-and-privacy.md` changes: app keys are labels for counting
  that also admit a caller when a token is set; identity, roles and audit
  stay out of scope.
- `docs/configuration.md` gains `GIQ_APPS_FILE` and `paths: apps:`.
- Core gains a dependency on `nvidia-ml-py` for D5's energy counter, with
  the power-sample estimate as the fallback where the counter is missing.
- D1–D4 and D6's API and CLI are built first; D5 and the dashboard pages
  follow.
- Quotas and rate limits per app become possible on top of this. They are
  not part of this ADR.
