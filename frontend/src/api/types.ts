// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* Response shapes of every endpoint the dashboard calls, transcribed from
   giq's routers (src/giq/api/*.py) and pydantic models (src/giq/models.py).
   Nullable where the server can send null — nvidia-smi reports [N/A] for
   telemetry a card does not expose, and SQL aggregates over no rows are
   NULL — so the type checker makes views handle the missing reading. */

export type Modality =
  | "llm"
  | "text2image"
  | "image_edit"
  | "tts"
  | "stt"
  | "audio"
  | "embed"
  | "ocr"
  | "depth";

export const WORKER_TYPES: readonly Modality[] = [
  "llm",
  "text2image",
  "image_edit",
  "audio",
  "embed",
  "tts",
  "stt",
  "ocr",
  "depth",
];

export type JobStatus = "pending" | "running" | "completed" | "failed";
export type ServiceState = "idle" | "ready" | "running" | "blocked" | "paused" | "error";
export type Policy = "pinned" | "auto" | "off";
export type Fit = "loaded" | "fits_now" | "fits_after_eviction" | "wont_fit_now" | "never";

// --- GET /status -------------------------------------------------------------

export interface AccessPosture {
  bound_host: string;
  reachable: string;
  loopback_only: boolean;
  token_set: boolean;
  /** Reachable beyond loopback with no token. */
  exposed: boolean;
}

export interface ActiveSlot {
  /** GPU UUID the on-demand instance sits on. */
  device: string | null;
  modality: string;
  recipe: string;
  ready: boolean;
}

export interface StatusGpu {
  uuid: string;
  index: number;
  name: string;
  selected: boolean;
  vram_used_gb: number;
  vram_total_gb: number;
  vram_free_gb: number;
  vram_giq_gb: number | null;
  vram_other_gb: number | null;
}

export interface Status {
  state: ServiceState;
  state_message: string | null;
  active_modality: Modality | null;
  active_recipe: string | null;
  active: ActiveSlot[];
  /** The ONE card the vram_* figures below describe (giq's default card). */
  gpu: { uuid: string; index: number; name: string } | null;
  vram_used_gb: number;
  vram_total_gb: number;
  vram_free_gb: number;
  vram_giq_gb: number | null;
  vram_other_gb: number | null;
  /** False while a pending job waits for room on its own card. */
  vram_ok: boolean;
  vram_message: string | null;
  /** The card (a `gpus` uuid) a blocked job is waiting on. */
  vram_blocked_gpu: string | null;
  /** Every card, in the terms of the one-card vram_* fields above. */
  gpus: StatusGpu[];
  queue_depth: number;
  /** Job ids, queue order. */
  jobs_pending: string[];
  jobs_running: string[];
  access: AccessPosture | null;
  paused: boolean;
  /** ISO 8601. */
  paused_since: string | null;
  pause_reason: string | null;
  /** giq's package version. */
  version: string;
  /** Seconds since the process started. */
  uptime_s: number;
}

// --- GET /gpus ---------------------------------------------------------------

export type ThrottleSeverity = "info" | "warning" | "critical";

export interface GpuProcess {
  pid: number;
  /** worker/model, or giq itself. */
  label: string;
  gb: number;
}

export interface Gpu {
  uuid: string;
  index: number;
  name: string;
  vram_total_gb: number;
  vram_used_gb: number;
  vram_free_gb: number;
  temperature_c: number | null;
  power_draw_w: number | null;
  power_limit_w: number | null;
  utilization_pct: number | null;
  fan_pct: number | null;
  throttle: { reason: string; severity: ThrottleSeverity }[];
  /** giq's default card. */
  selected: boolean;
  /** Held by giq's processes; with vram_other_gb sums to vram_used_gb. */
  vram_giq_gb: number;
  /** Everything else on the card (desktop, other CUDA apps). */
  vram_other_gb: number;
  giq: GpuProcess[];
}

export interface GpusResponse {
  selected: string | null;
  gpus: Gpu[];
}

// --- GET /recipes -------------------------------------------------------------

/** A recipe's residency: why it is (or is not) kept loaded. */
export interface Residency {
  policy: Policy;
  source: "override" | "default";
  reason: string | null;
  /** Kept loaded by default (the recipes' own residency or config.yaml's), whatever the operator set since. */
  default_resident: boolean;
}

/** The card a recipe runs on. */
export interface RecipeCard {
  /** Explicit binding (GPU UUID), null when unbound. */
  device: string | null;
  source: "override" | "config" | "default";
  /** The card it lands on either way. */
  effective: string | null;
  index: number | null;
  name: string | null;
}

/** One recipe: weights + engine + params, by the name a client sends as `model`. */
export interface RecipeEntry {
  name: string;
  label: string;
  detail: string;
  /** Every modality the recipe serves (flux_klein renders and edits); the first picks the defaults. */
  modalities: Modality[];
  engine: string;
  /** Engine binary that executes it (key into /engines). */
  runtime: string;
  aliases: string[];
  capabilities: string[];
  vision: boolean;
  /** LLMs: whether giq launches it with thinking on. */
  reasoning: "on" | "off" | "template" | null;
  vram_gb: number;
  measured: boolean;
  /** vram_gb plus margin and the card's reserve: what a load is gated on. */
  needed_gb: number;
  lanes: number;
  /** One figure, or one per modality. */
  max_batch: number | Partial<Record<Modality, number>> | null;
  voices: string[];
  /** Every file it loads is on disk. */
  installed: boolean;
  /** Whether it runs on this machine (ADR-005), and the checks it is judged on. */
  availability: Availability;
  checks: PlanCheck[];
  /** ids into /weights. */
  weights: string[];
  residency: Residency;
  card: RecipeCard;
  fit: Fit;
  /** When a job for it last completed (epoch seconds), across its modalities. */
  last_used: number | null;
  /** The instance running it now, if any. */
  instance: { id: string; state: InstanceState; residency: InstanceResidency } | null;
}

/** ready: runs here · fetchable: giq can fetch its weights · manual: place them by hand · unfit: cannot run here. */
export type Availability = "ready" | "fetchable" | "manual" | "unfit";

export interface PlanCheck {
  /** engine | card | compute | weights | disk | access | licence | pinned */
  check: string;
  status: "ok" | "warn" | "fail";
  message: string;
}

// --- GET /recipes/{name}/plan ------------------------------------------------------

export interface PlanTransfer {
  part: string | null;
  repo: string;
  revision: string | null;
  file: string | null;
  dest: string | null;
  files: number;
  bytes: number;
}

export interface RecipePlan {
  recipe: string;
  availability: Availability;
  /** Nothing but the missing weights stands in the way. */
  can_fetch: boolean;
  /** The service may write where the files go; else the operator runs `command`. */
  service_can_fetch: boolean;
  command: string;
  download_bytes: number;
  checks: PlanCheck[];
  transfers: PlanTransfer[];
}

// --- /downloads ----------------------------------------------------------------------

export type DownloadState = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Download {
  id: string;
  recipe: string;
  state: DownloadState;
  bytes_total: number;
  bytes_done: number;
  current: string | null;
  error: string | null;
  queued_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface DownloadsResponse {
  downloads: Download[];
}

// --- GET /plugins ---------------------------------------------------------------------

export interface PluginEntry {
  name: string;
  package: string;
  summary: string;
  engines: string[];
  modalities: string[];
  recipes: string[];
  needs: string;
  curated: boolean;
  installed: boolean;
  status: { loaded: boolean; reason: string | null; version: string; source: string } | null;
  /** The command that installs it; null once installed. */
  install: string | null;
}

export interface RemoveRecipeWeightsResult {
  recipe: string;
  deleted: string[];
  /** Weights another recipe also loads, left in place. */
  kept: { id: string; used_by: string[] }[];
  freed_bytes: number;
}

export interface PluginsResponse {
  plugins: PluginEntry[];
}

export interface CardBudget {
  uuid: string;
  index: number;
  name: string;
  total_gb: number;
  free_gb: number;
  reserve_gb: number;
  pinned_gb: number;
  pinned_needed_gb: number;
  pinned_fits: boolean;
  /** Recipes pinned to this card, in reload order. */
  pinned: string[];
  default: boolean;
}

export interface RecipesResponse {
  recipes: RecipeEntry[];
  cards: CardBudget[];
  /** Reload priority order. */
  pinned: string[];
}

/** PUT /recipes/{name}/residency, PUT /recipes/{name}/card. */
export interface RecipeWriteResponse {
  recipe: RecipeEntry;
  cards: CardBudget[];
  warnings: string[];
}

// --- GET /instances -----------------------------------------------------------

export type InstanceState = "ready" | "starting" | "stopped";
export type InstanceResidency = "resident" | "on_demand";

/** A recipe running on a card. */
export interface InstanceEntry {
  /** recipe@card */
  id: string;
  recipe: string;
  modalities: Modality[];
  engine: string | null;
  residency: InstanceResidency;
  state: InstanceState;
  device: string | null;
  device_index: number | null;
  device_name: string | null;
  /** Loopback port of a server engine; null for in-process and child adapters. */
  port: number | null;
  pid: number | null;
  lanes: number;
  in_flight: number;
  vram_gb: number | null;
  started_at: number;
}

export interface InstancesResponse {
  instances: InstanceEntry[];
}

// --- GET /weights, DELETE /weights/{id} --------------------------------------

/** One checkpoint, however many recipes load it. */
export interface WeightsItem {
  id: string;
  /** Exactly one of path (a file or directory) and repo (in the HF cache). */
  path: string | null;
  repo: string | null;
  format: string | null;
  source: string | null;
  revision: string | null;
  licence: string | null;
  recipes: string[];
  /** "recipe" for main weights, "recipe:part" for a part. */
  used_by: string[];
  on_disk: boolean;
  size_bytes: number;
  mount: string | null;
}

export interface WeightsResponse {
  weights: WeightsItem[];
}

export interface DeleteWeightsResult {
  id: string;
  recipes: string[];
  deleted: string[];
  missing: string[];
  freed_bytes: number;
}

// --- GET /storage ------------------------------------------------------------

export interface Disk {
  mount: string;
  total_bytes: number;
  free_bytes: number;
  models_bytes: number;
  other_bytes: number;
}

/** An operator recipe file that is serving. */
export interface RecipeFile {
  file: string;
  name: string;
  modalities: Modality[];
  /** Replaces the built-in recipe of the same name. */
  replaces_builtin: boolean;
}

/** An operator recipe file giq left out, and why. */
export interface RecipeLoadError {
  /** Null when the problem is not one file's (two files defining one recipe). */
  file: string | null;
  message: string;
}

/** The operator's recipe files, as the running snapshot read them. */
export interface RecipesInfo {
  dir: string | null;
  builtin_dir: string;
  files: RecipeFile[];
  /** The name of each built-in an operator file replaces. */
  overrides: string[];
  errors: RecipeLoadError[];
}

export interface StorageResponse {
  disks: Disk[];
  /** Every data directory giq resolved: models, recipes, engines, state, caches. */
  paths?: Record<string, string | null>;
  /** Absent from a giq older than recipe files. */
  recipes?: RecipesInfo;
}

// --- GET /engines --------------------------------------------------------------

export interface Engine {
  name: string;
  binary: string | null;
  detail?: string;
  present: boolean;
  version: string | null;
  error: string | null;
}

export interface EnginesResponse {
  engines: Engine[];
}

// --- /control -----------------------------------------------------------------

export interface PauseRequest {
  force?: boolean;
  reason?: string | null;
}

export interface PauseResponse {
  paused: boolean;
  since: string | null;
  reason: string | null;
  forced: boolean;
  drained: boolean;
  warnings: string[];
  vram_free_gb: number;
  vram_total_gb: number;
}

// --- jobs ------------------------------------------------------------------------

export interface JobRequest {
  modality: Modality;
  model: string;
  params?: Record<string, unknown>;
  tasks: Record<string, unknown>[];
}

/** POST /run (without wait). */
export interface JobSubmitted {
  job_id: string;
  position: number;
}

/** GET /jobs/{id}, POST /run?wait=true. Results are per worker (image_b64/seed/error, output, …). */
export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  modality: Modality;
  model: string;
  results: Record<string, unknown>[] | null;
  duration_ms: number | null;
}

/** DELETE /jobs/{id}: only pending jobs cancel. 404 when unknown. */
export interface CancelResponse {
  cancelled: boolean;
  reason?: string;
}

// --- /stats --------------------------------------------------------------------

/** GET /stats/summary?hours= */
export interface StatsSummary {
  hours: number;
  evictions: number;
  recipes: {
    modality: Modality;
    recipe: string;
    jobs: number;
    failed: number;
    avg_run_ms: number | null;
    max_run_ms: number | null;
    avg_queue_ms: number | null;
    tasks: number | null;
  }[];
}

/** GET /stats/timeline?hours=&bucket_s= */
export interface StatsTimeline {
  bucket_s: number;
  points: {
    /** Bucket start, epoch seconds. */
    t: number;
    modality: Modality;
    jobs: number;
    failed: number;
    avg_run_ms: number | null;
  }[];
}

/** GET /stats/vram?hours= (the default card). */
export interface StatsVram {
  total_gb: number;
  samples: { t: number; used: number; active: string | null; ready: number }[];
}

/** GET /stats/gpus?hours= */
export interface StatsGpus {
  hours: number;
  gpus: {
    uuid: string;
    index: number | null;
    name: string;
    total_gb: number;
    power_limit: number | null;
    samples: {
      t: number;
      used: number;
      temp: number | null;
      power: number | null;
      util: number | null;
    }[];
  }[];
}

/** GET /stats/gpus/eras */
export interface GpuEra {
  uuid: string;
  name: string;
  total_gb: number;
  first_seen: number;
  last_seen: number;
  jobs: number;
  failed: number;
  tokens_in: number | null;
  tokens_out: number | null;
  avg_run_ms: number | null;
  first_job: number | null;
  last_job: number | null;
}

export interface GpuErasResponse {
  eras: GpuEra[];
}

export type UsagePeriod = "day" | "week" | "month" | "all";

/** GET /stats/usage?period=&gpu=&since=&until= */
export interface StatsUsage {
  period: UsagePeriod;
  gpu: string | null;
  since: number;
  until: number | null;
  totals: { jobs: number; failed: number; tokens_in: number; tokens_out: number };
  recipes: {
    modality: Modality;
    recipe: string;
    jobs: number;
    failed: number;
    tasks: number | null;
    tokens_in: number | null;
    tokens_out: number | null;
    last_ts: number | null;
  }[];
  /** `b` is a local-time bucket key: "YYYY-MM-DD HH:00" (day), "YYYY-MM-DD" (week/month), "YYYY-MM" (all). */
  series: {
    b: string;
    modality: Modality;
    recipe: string;
    jobs: number;
    tokens_in: number | null;
    tokens_out: number | null;
  }[];
}

/** GET /stats/jobs?limit=&gpu= — newest first. */
export interface JobRecord {
  t: number;
  job_id: string;
  modality: Modality;
  recipe: string;
  status: JobStatus | string;
  queue_ms: number | null;
  run_ms: number | null;
  tasks: number | null;
  error: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
}

/** GET /stats/events?hours=&limit= */
export interface StatsEvent {
  t: number;
  kind: string;
  detail: string;
}

// --- GET /capabilities -------------------------------------------------------------

export interface ModalityCapability {
  /** Engines of the ready recipes, comma-separated. */
  engine: string;
  /** The recipes that run here now, kept warm first (ADR-005 D7). */
  recipes: string[];
  default: string | null;
  available: { name: string; availability: Availability; verdict: string }[];
  max_batch: number | null;
  voices: string[] | null;
  /** How the dashboard names and draws it, as its plugin registered it. */
  label: string;
  icon: string;
}

export interface Capabilities {
  /** In the server's registration order. */
  modalities: Record<string, ModalityCapability>;
  constraints: Record<string, unknown>;
}

// --- /v1 (OpenAI-compatible) ---------------------------------------------------------

export interface V1Models {
  object: "list";
  data: { id: string; object: "model"; created: number; owned_by: string }[];
}

export type ChatContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ChatContentPart[] | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  reasoning_content?: string;
}

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** One SSE chunk of POST /v1/chat/completions with stream: true. */
export interface ChatChunk {
  id?: string;
  model?: string;
  choices?: {
    index: number;
    delta?: { role?: string; content?: string | null; reasoning_content?: string | null };
    finish_reason?: string | null;
  }[];
  usage?: ChatUsage | null;
}

/** Non-streaming POST /v1/chat/completions. */
export interface ChatCompletion {
  id: string;
  model: string;
  choices: {
    index: number;
    message: ChatMessage;
    finish_reason: string | null;
  }[];
  usage?: ChatUsage;
}

/** POST /v1/audio/transcriptions (json / verbose_json). */
export interface Transcription {
  task?: "transcribe";
  text: string;
  language: string | null;
  duration: number | null;
  speakers: string[];
  segments: {
    start: number;
    end: number;
    text: string;
    speaker: string | null;
    language?: string | null;
    words?: unknown[];
  }[];
}

/** POST /v1/audio/embeddings. */
export interface AudioEmbedding {
  embedding: number[];
  dim: number;
  model: string;
  normalized: boolean;
}
