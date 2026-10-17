// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI-compatible HTTP transport for the upstream in-process text engine.
//! Public handlers, protocol conversion, aggregation and usage belong to Dynamo.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{bail, ensure};
use async_stream::stream;
use async_trait::async_trait;
use dynamo_runtime::engine::{AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::engines::StreamingEngine;
use crate::http::service::error::{HttpError, invalid_argument};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::protocols::openai::completions::{
    NvCreateCompletionRequest, NvCreateCompletionResponse,
};

const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_PREFILL_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

pub struct OpenAiHttpEngine {
    client: reqwest::Client,
    workers: Vec<reqwest::Url>,
    next: AtomicUsize,
    pd: Option<PdRoutes>,
}

/// One SGLang prefill worker: its HTTP origin and disaggregation bootstrap port.
pub struct PdPrefillOrigin {
    pub origin: String,
    pub bootstrap_port: u16,
}

struct PdPrefill {
    endpoint: reqwest::Url,
    bootstrap_host: String,
    bootstrap_port: u16,
    // Prompt bytes of requests whose prefill has not finished.
    outstanding: Arc<AtomicUsize>,
}

struct PdDecode {
    endpoint: reqwest::Url,
    // Requests whose decode stream is still open.
    outstanding: Arc<AtomicUsize>,
}

/// SGLang PD disaggregation: every request goes to one prefill and one decode
/// worker with a shared bootstrap room. The prefix KV cache lives on the
/// prefill, so prefill follows the request's session key. A session whose last
/// turn ran on a prefill costs only its new prompt bytes there and its whole
/// prompt anywhere else; it goes where it finishes first (outstanding work plus
/// its own cost). A session without that history follows its rendezvous order
/// and spills when its first choice is `PD_AFFINITY_SLACK` above the least
/// loaded prefill. Requests without a key, and decode, go to the least loaded
/// worker. Outstanding prefill work counts the estimated uncached bytes.
struct PdRoutes {
    prefills: Vec<PdPrefill>,
    decodes: Vec<PdDecode>,
    next: AtomicUsize,
    sessions: parking_lot::Mutex<lru::LruCache<Vec<u8>, SessionPlacement>>,
}

/// Where a session's last turn ran and its prompt size in bytes.
#[derive(Clone, Copy)]
struct SessionPlacement {
    prefill: usize,
    decode: usize,
    bytes: usize,
}

/// Open decode streams a session's last decode may carry above the least
/// loaded decode before the session moves. Staying lets a decode radix cache
/// reuse the session's history instead of receiving all of it again.
const PD_DECODE_SLACK: usize = 4;

/// Sessions whose placement the frontend remembers (least recently used first out).
const PD_SESSIONS: usize = 65_536;

/// Prompt work (serialized message bytes, ~4 per token) a session's preferred
/// prefill may carry above the least loaded one before the session spills.
/// About 250K tokens, i.e. roughly 20 s of PP8 prefill.
const PD_AFFINITY_SLACK: usize = 1_000_000;

impl PdRoutes {
    /// Returns (prefill, decode, estimated prefill work) for a request whose
    /// prompt is `bytes` long.
    fn select(&self, session: Option<&[u8]>, bytes: usize) -> (usize, usize, usize) {
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        let bytes = bytes.max(1);
        let loads: Vec<usize> = self
            .prefills
            .iter()
            .map(|p| p.outstanding.load(Ordering::Relaxed))
            .collect();
        let decode_loads: Vec<usize> = self
            .decodes
            .iter()
            .map(|d| d.outstanding.load(Ordering::Relaxed))
            .collect();
        let least_loaded_decode = least_loaded(decode_loads.iter().copied(), start);
        let (prefill, decode, work) = match session {
            Some(key) => {
                let mut sessions = self.sessions.lock();
                // A shorter prompt than last turn rewrote the history: no reuse.
                let placed = sessions
                    .get(key)
                    .copied()
                    .filter(|p| p.prefill < loads.len() && p.bytes <= bytes);
                let choice = match placed {
                    Some(placed) => {
                        let cost = |i: usize| {
                            loads[i].saturating_add(if i == placed.prefill {
                                bytes - placed.bytes
                            } else {
                                bytes
                            })
                        };
                        let n = loads.len();
                        let other = (0..n)
                            .map(|k| (start + k) % n)
                            .filter(|&i| i != placed.prefill)
                            .min_by_key(|&i| cost(i));
                        match other {
                            Some(i) if cost(i) < cost(placed.prefill) => i,
                            _ => placed.prefill,
                        }
                    }
                    None => {
                        let floor = loads.iter().copied().min().unwrap();
                        let mut order: Vec<usize> = (0..self.prefills.len()).collect();
                        order.sort_by_key(|&i| {
                            std::cmp::Reverse(rendezvous_score(
                                key,
                                self.prefills[i].endpoint.as_str(),
                            ))
                        });
                        order
                            .into_iter()
                            .find(|&i| loads[i] <= floor.saturating_add(PD_AFFINITY_SLACK))
                            .unwrap()
                    }
                };
                let work = match placed {
                    Some(placed) if placed.prefill == choice => (bytes - placed.bytes).max(1),
                    _ => bytes,
                };
                let floor = decode_loads[least_loaded_decode];
                let decode = placed
                    .map(|p| p.decode)
                    .filter(|&d| {
                        d < decode_loads.len()
                            && decode_loads[d] <= floor.saturating_add(PD_DECODE_SLACK)
                    })
                    .unwrap_or(least_loaded_decode);
                sessions.put(
                    key.to_vec(),
                    SessionPlacement { prefill: choice, decode, bytes },
                );
                (choice, decode, work)
            }
            None => (least_loaded(loads.into_iter(), start), least_loaded_decode, bytes),
        };
        (prefill, decode, work)
    }
}

fn rendezvous_score(key: &[u8], worker: &str) -> u64 {
    let mut hash = Sha256::new();
    hash.update((key.len() as u64).to_be_bytes());
    hash.update(key);
    hash.update((worker.len() as u64).to_be_bytes());
    hash.update(worker.as_bytes());
    u64::from_be_bytes(hash.finalize()[..8].try_into().unwrap())
}

/// JSON text with object keys sorted at every level. Fields such as
/// `chat_template_kwargs` are filled from hash maps, so their key order changes
/// per request; hashing the raw text would give every turn a different key.
fn canonical_json(value: &Value) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                out.push('{');
                for (i, (key, item)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(key.clone()).to_string());
                    out.push(':');
                    write(item, out);
                }
                out.push('}');
            }
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// Stable key for the KV prefix a request shares with its session: an explicit
/// `prompt_cache_key`, otherwise the prompt head that stays fixed across turns
/// (tools and template controls plus messages through the first user turn).
/// Scoped by model and cache namespace. None when the request has no prompt.
fn session_key(body: &Value) -> Option<Vec<u8>> {
    let mut hash = Sha256::new();
    let mut field = |name: &str, value: &str| {
        hash.update((name.len() as u64).to_be_bytes());
        hash.update(name.as_bytes());
        hash.update((value.len() as u64).to_be_bytes());
        hash.update(value.as_bytes());
    };
    for name in ["model", "cache_salt"] {
        field(name, body.get(name).and_then(Value::as_str).unwrap_or_default());
    }
    if let Some(key) = body
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .filter(|key| !key.is_empty())
    {
        field("prompt_cache_key", key);
        return Some(hash.finalize().to_vec());
    }
    let messages = body.get("messages")?.as_array().filter(|m| !m.is_empty())?;
    let head = messages
        .iter()
        .position(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .map_or(messages.len(), |i| i + 1);
    for name in ["tools", "chat_template", "chat_template_kwargs"] {
        if let Some(value) = body.get(name) {
            field(name, &canonical_json(value));
        }
    }
    field("messages", &canonical_json(&Value::from(messages[..head].to_vec())));
    Some(hash.finalize().to_vec())
}

/// Index of the smallest load; ties rotate with `start` so equal workers share traffic.
fn least_loaded(loads: impl ExactSizeIterator<Item = usize>, start: usize) -> usize {
    let n = loads.len();
    let loads: Vec<usize> = loads.collect();
    (0..n)
        .map(|offset| (start + offset) % n)
        .min_by_key(|&index| loads[index])
        .unwrap()
}

/// Load accounted to a worker until the request part it tracks finishes.
struct LoadHold(Arc<AtomicUsize>, usize);

impl LoadHold {
    fn new(counter: &Arc<AtomicUsize>, amount: usize) -> Self {
        counter.fetch_add(amount, Ordering::Relaxed);
        Self(counter.clone(), amount)
    }
}

impl Drop for LoadHold {
    fn drop(&mut self) {
        self.0.fetch_sub(self.1, Ordering::Relaxed);
    }
}

/// Cancels the prefill request when the client request ends first.
struct AbortOnDrop(tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Why a prefill did not finish successfully.
#[derive(Debug)]
enum PrefillError {
    /// The worker answered with this non-success status and body.
    Status(u16, String),
    /// The worker could not be reached.
    Unavailable,
    Failed(&'static str),
}

/// Keep the worker's status instead of reporting an internal error. The HTTP
/// service forwards client errors and sanitizes the message.
fn status_error(code: u16, body: &str) -> Error {
    HttpError {
        code,
        message: body.chars().take(2048).collect(),
    }
    .into()
}

fn unavailable(worker: &str) -> Error {
    HttpError {
        code: 503,
        message: format!("{worker} worker unavailable"),
    }
    .into()
}

fn send_error(error: reqwest::Error, worker: &str) -> Error {
    if error.is_connect() {
        unavailable(worker)
    } else {
        error.into()
    }
}

/// The prefill half of one PD request and the decode load it holds.
struct PdInFlight {
    task: tokio::task::JoinHandle<Result<(), PrefillError>>,
    _abort: AbortOnDrop,
    _decode: LoadHold,
}

fn prefill_outcome(
    result: Result<Result<(), PrefillError>, tokio::task::JoinError>,
) -> anyhow::Result<()> {
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(PrefillError::Status(code, body))) => Err(status_error(code, &body)),
        Ok(Err(PrefillError::Unavailable)) => Err(unavailable("prefill")),
        Ok(Err(PrefillError::Failed(error))) => bail!("prefill failed: {error}"),
        Err(_) => bail!("prefill task ended unexpectedly"),
    }
}

/// Send the request to prefill and drain its response. Decode owns the
/// client-visible output; prefill only has to finish successfully.
async fn run_prefill(
    client: reqwest::Client,
    endpoint: reqwest::Url,
    body: Value,
    _hold: LoadHold,
) -> Result<(), PrefillError> {
    let response = client
        .post(endpoint)
        .json(&body)
        .send()
        .await
        .map_err(|error| {
            if error.is_connect() {
                PrefillError::Unavailable
            } else {
                PrefillError::Failed("transport failed")
            }
        })?;
    let status = response.status();
    if status != reqwest::StatusCode::OK {
        let body = response.text().await.unwrap_or_default();
        return Err(PrefillError::Status(status.as_u16(), body));
    }
    let mut bytes = response.bytes_stream();
    let mut total = 0usize;
    while let Some(part) = bytes.next().await {
        let part = part.map_err(|_| PrefillError::Failed("response transport failed"))?;
        total += part.len();
        if total > MAX_PREFILL_RESPONSE_BYTES {
            return Err(PrefillError::Failed("response too large"));
        }
    }
    Ok(())
}

fn worker_origin(origin: &str) -> anyhow::Result<reqwest::Url> {
    let url = reqwest::Url::parse(origin)?;
    ensure!(
        matches!(url.scheme(), "http" | "https")
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.path() == "/"
            && url.query().is_none()
            && url.fragment().is_none(),
        "worker must be an HTTP origin without credentials, path or query"
    );
    Ok(url.join("v1/chat/completions")?)
}

impl OpenAiHttpEngine {
    fn worker_index(&self, body: &Value) -> usize {
        let explicit_key = body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .filter(|key| !key.is_empty());
        let messages = body.get("messages").filter(|messages| {
            messages
                .as_array()
                .is_some_and(|messages| !messages.is_empty())
        });
        if explicit_key.is_none() && messages.is_none() {
            return self.next.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        }
        let derived_key = explicit_key.is_none().then(|| {
            let mut hash = Sha256::new();
            for field in [
                "messages",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "chat_template",
                "chat_template_kwargs",
            ] {
                let Some(value) = body.get(field) else {
                    continue;
                };
                let value = canonical_json(value);
                hash.update((field.len() as u64).to_be_bytes());
                hash.update(field.as_bytes());
                hash.update((value.len() as u64).to_be_bytes());
                hash.update(value.as_bytes());
            }
            hash.finalize()
        });

        // Rendezvous affinity is a routing hint, not evidence of resident KV.
        // An explicit OpenAI prompt_cache_key remains authoritative. When a
        // client omits it, derive best-effort affinity from the prompt-forming
        // fields so identical C1 requests do not round-robin away from warm KV.
        // Length framing separates every field and worker identity; scores are
        // stable across frontend restart and worker-list reordering.
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let salt = body
            .get("cache_salt")
            .and_then(Value::as_str)
            .unwrap_or_default();
        self.workers
            .iter()
            .enumerate()
            .max_by_key(|(_, worker)| {
                let mut hash = Sha256::new();
                for value in [model, salt] {
                    hash.update((value.len() as u64).to_be_bytes());
                    hash.update(value.as_bytes());
                }
                if let Some(key) = explicit_key {
                    hash.update((key.len() as u64).to_be_bytes());
                    hash.update(key.as_bytes());
                } else if let Some(key) = &derived_key {
                    hash.update((key.len() as u64).to_be_bytes());
                    hash.update(key);
                }
                let worker = worker.as_str();
                hash.update((worker.len() as u64).to_be_bytes());
                hash.update(worker.as_bytes());
                let digest = hash.finalize();
                u64::from_be_bytes(digest[..8].try_into().unwrap())
            })
            .unwrap()
            .0
    }

    pub fn new(workers: Vec<String>) -> anyhow::Result<Self> {
        ensure!(
            !workers.is_empty() && workers.len() <= 64,
            "configure 1..64 worker origins"
        );
        let workers = workers
            .iter()
            .map(|origin| worker_origin(origin))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(600))
            .build()?;
        Ok(Self {
            client,
            workers,
            next: AtomicUsize::new(0),
            pd: None,
        })
    }

    /// Route through SGLang PD disaggregation instead of independent workers.
    pub fn new_pd(prefills: Vec<PdPrefillOrigin>, decodes: Vec<String>) -> anyhow::Result<Self> {
        ensure!(
            !prefills.is_empty() && prefills.len() <= 64 && !decodes.is_empty() && decodes.len() <= 64,
            "configure 1..64 prefill and 1..64 decode origins"
        );
        let prefills = prefills
            .iter()
            .map(|prefill| {
                let endpoint = worker_origin(&prefill.origin)?;
                ensure!(prefill.bootstrap_port != 0, "prefill bootstrap_port cannot be 0");
                let bootstrap_host = endpoint
                    .host_str()
                    .unwrap()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .to_owned();
                Ok(PdPrefill {
                    endpoint,
                    bootstrap_host,
                    bootstrap_port: prefill.bootstrap_port,
                    outstanding: Arc::new(AtomicUsize::new(0)),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let decodes = decodes
            .iter()
            .map(|origin| {
                Ok(PdDecode {
                    endpoint: worker_origin(origin)?,
                    outstanding: Arc::new(AtomicUsize::new(0)),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut origins = BTreeSet::new();
        for url in prefills.iter().map(|p| &p.endpoint).chain(decodes.iter().map(|d| &d.endpoint)) {
            ensure!(origins.insert(url.as_str()), "duplicate PD worker origin {url}");
        }
        // A long prefill can queue for minutes before decode emits its first
        // token, so bound idle reads rather than the whole request.
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .read_timeout(Duration::from_secs(3600))
            .build()?;
        Ok(Self {
            client,
            workers: Vec::new(),
            next: AtomicUsize::new(0),
            pd: Some(PdRoutes {
                prefills,
                decodes,
                next: AtomicUsize::new(0),
                sessions: parking_lot::Mutex::new(lru::LruCache::new(
                    std::num::NonZeroUsize::new(PD_SESSIONS).unwrap(),
                )),
            }),
        })
    }
}

fn worker_request(mut request: NvCreateChatCompletionRequest) -> anyhow::Result<Value> {
    // Dynamo owns canonical/legacy salt precedence; HTTP workers own the
    // top-level cache_salt field. Preserve this isolation boundary explicitly.
    let cache_salt =
        crate::protocols::common::extensions::request_cache_salt(&request).map(str::to_owned);
    if request
        .unsupported_fields
        .get("prompt_cache_retention")
        .is_some_and(|value| !value.is_null())
    {
        return Err(crate::http::service::error::invalid_argument(
            "prompt_cache_retention is not supported by this HTTP backend; automatic prefix caching remains available",
        ).into());
    }
    request.normalize_reasoning_template_args()?;
    let mut body = serde_json::to_value(request)?;
    // Dynamo keeps one assistant reasoning segment per tool call (Anthropic
    // thinking blocks); OpenAI HTTP workers take reasoning_content as a string.
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages.iter_mut().filter_map(Value::as_object_mut) {
            let Some(Value::Array(segments)) = message.get("reasoning_content") else {
                continue;
            };
            let text = segments
                .iter()
                .filter_map(Value::as_str)
                .filter(|segment| !segment.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() {
                message.remove("reasoning_content");
            } else {
                message.insert("reasoning_content".into(), json!(text));
            }
        }
    }
    if let Some(cache_salt) = cache_salt {
        body["cache_salt"] = json!(cache_salt);
    }
    // Dynamo owns canonical chat_template_args; OpenAI HTTP workers consume
    // chat_template_kwargs. Normalize once at this transport boundary.
    if let Some(arguments) = body.as_object_mut().unwrap().remove("chat_template_args") {
        body["chat_template_kwargs"] = arguments;
    }
    body["stream"] = json!(true);
    // Cumulative usage on every chunk feeds the frontend's token metrics;
    // UsageRelay returns only what the public client asked for.
    body["stream_options"] = json!({"include_usage": true, "continuous_usage_stats": true});
    Ok(body)
}

/// Turns the worker's per-chunk cumulative usage into the frontend's internal
/// token metrics and passes on only the usage the request asked for.
struct UsageRelay {
    requested: bool,
    continuous: bool,
    output_tokens: usize,
}

impl UsageRelay {
    fn new(request: &NvCreateChatCompletionRequest) -> Self {
        let options = request.inner.stream_options.as_ref();
        // A unary request is folded into one response, whose usage comes from
        // the usage-only chunk.
        let streaming = request.inner.stream.unwrap_or(false);
        Self {
            requested: !streaming || options.is_some_and(|o| o.include_usage),
            continuous: options.is_some_and(|o| o.continuous_usage_stats),
            output_tokens: 0,
        }
    }

    /// Returns false when the chunk carried only usage nobody asked for.
    fn relay(&mut self, chunk: &mut NvCreateChatCompletionStreamResponse) -> bool {
        let Some(usage) = chunk.inner.usage.as_ref() else {
            return true;
        };
        let output_tokens = usage.completion_tokens as usize;
        chunk.llm_metrics = Some(crate::protocols::common::metrics::LLMMetricAnnotation {
            input_tokens: usage.prompt_tokens as usize,
            output_tokens,
            chunk_tokens: output_tokens.saturating_sub(self.output_tokens),
            cached_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .map(|cached| cached as usize),
            ..Default::default()
        });
        self.output_tokens = self.output_tokens.max(output_tokens);
        if chunk.inner.choices.is_empty() {
            return self.requested;
        }
        if !self.continuous {
            chunk.inner.usage = None;
        }
        true
    }
}

fn decode_event(data: &str) -> anyhow::Result<NvCreateChatCompletionStreamResponse> {
    let mut wire: Value = serde_json::from_str(data)?;
    let prompt_tokens = wire.pointer("/usage/prompt_tokens").and_then(Value::as_u64);
    if let Some(details) = wire
        .pointer_mut("/usage/prompt_tokens_details")
        .and_then(Value::as_object_mut)
    {
        // vLLM 0.26 HTTP-worker compatibility within the supported N-2 window.
        // Retire this input-only alias when those worker revisions leave the window.
        // The canonical representation wins, including an explicit null.
        if let Some(written) = details.remove("created_cache_tokens") {
            details.entry("cache_write_tokens").or_insert(written);
        }
        if let Some(written) = details.get("cache_write_tokens").filter(|v| !v.is_null()) {
            ensure!(
                written
                    .as_u64()
                    .zip(prompt_tokens)
                    .is_some_and(|(written, prompt)| written <= prompt),
                "invalid native cache-write token count"
            );
        }
    }
    let chunk: NvCreateChatCompletionStreamResponse = serde_json::from_value(wire)?;
    // Required identity comes from the backend, never a synthesized template.
    ensure!(
        !chunk.inner.id.is_empty() && !chunk.inner.model.is_empty(),
        "missing backend identity"
    );
    Ok(chunk)
}

#[async_trait]
impl StreamingEngine for OpenAiHttpEngine {
    async fn handle_chat(
        &self,
        req: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        if req
            .get::<Vec<crate::http::service::anthropic_cache_control::CacheControlTarget>>(
                crate::http::service::anthropic_cache_control::CACHE_CONTROL_CONTEXT_KEY,
            )
            .is_ok()
        {
            return Err(crate::http::service::error::invalid_argument(
                "Anthropic cache_control is not supported by the OpenAI HTTP worker transport",
            )
            .into());
        }

        let (request, context) = req.into_parts();
        let ctx = context.context();
        let mut usage = UsageRelay::new(&request);
        // Obtain native usage for upstream unary aggregation and Messages,
        // independent of the public client's stream_options.
        let mut body = worker_request(request)?;
        let choices = body.get("n").and_then(Value::as_u64).unwrap_or(1) as usize;
        let mut pd_flight = None;
        let mut prefill_done = false;
        let response = if let Some(pd) = &self.pd {
            if choices != 1 {
                return Err(invalid_argument("PD disaggregation serves one choice per request (n must be 1)").into());
            }
            let bytes = body.get("messages").map_or(1, |m| m.to_string().len());
            let (prefill_index, decode_index, work) =
                pd.select(session_key(&body).as_deref(), bytes);
            let (prefill, decode) = (&pd.prefills[prefill_index], &pd.decodes[decode_index]);
            body["bootstrap_host"] = json!(prefill.bootstrap_host);
            body["bootstrap_port"] = json!(prefill.bootstrap_port);
            body["bootstrap_room"] = json!(rand::random::<u64>() >> 1);
            let mut prefill_body = body.clone();
            prefill_body["stream"] = json!(false);
            prefill_body.as_object_mut().unwrap().remove("stream_options");
            let task = tokio::spawn(run_prefill(
                self.client.clone(),
                prefill.endpoint.clone(),
                prefill_body,
                LoadHold::new(&prefill.outstanding, work),
            ));
            let mut flight = PdInFlight {
                _abort: AbortOnDrop(task.abort_handle()),
                task,
                _decode: LoadHold::new(&decode.outstanding, 1),
            };
            let send = self.client.post(decode.endpoint.clone()).json(&body).send();
            tokio::pin!(send);
            let response = loop {
                tokio::select! {
                    _ = ctx.stopped() => bail!("request cancelled before native HTTP headers"),
                    result = &mut flight.task, if !prefill_done => {
                        prefill_done = true;
                        prefill_outcome(result)?;
                    },
                    response = &mut send => break response.map_err(|e| send_error(e, "decode"))?,
                }
            };
            pd_flight = Some(flight);
            response
        } else {
            let worker = &self.workers[self.worker_index(&body)];
            tokio::select! {
                _ = ctx.stopped() => bail!("request cancelled before native HTTP headers"),
                response = self.client.post(worker.clone()).json(&body).send() => {
                    response.map_err(|e| send_error(e, "backend"))?
                }
            }
        };
        if response.status() != reqwest::StatusCode::OK {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(status_error(status, &body));
        }
        ensure!(
            response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.starts_with("text/event-stream")),
            "backend did not return an SSE stream"
        );
        let stream_context = ctx.clone();
        let output = stream! {
            // Dropping the stream aborts an unfinished prefill and releases the
            // decode load; a failed prefill ends the decode stream with an error.
            let mut pd_flight = pd_flight;
            let mut prefill_done = prefill_done || pd_flight.is_none();
            let mut bytes = response.bytes_stream();
            let mut buffer = Vec::new();
            let mut completed = BTreeSet::new();
            loop {
                let part = tokio::select! {
                    _ = stream_context.stopped() => {
                        yield Annotated::from_error("backend request cancelled");
                        return;
                    },
                    // select! evaluates disabled branches too, so the prefill
                    // future must not assume a PD flight exists.
                    Some(result) = futures::future::OptionFuture::from(
                        pd_flight.as_mut().map(|flight| &mut flight.task)
                    ), if !prefill_done => {
                        prefill_done = true;
                        if let Err(error) = prefill_outcome(result) {
                            yield Annotated::from_error(&error.to_string());
                            return;
                        }
                        continue;
                    },
                    part = bytes.next() => part,
                };
                let Some(part) = part else {
                    yield Annotated::from_error("backend SSE closed before DONE");
                    return;
                };
                let part = match part {
                    Ok(part) => part,
                    Err(_) => { yield Annotated::from_error("backend SSE transport failed"); return; }
                };
                buffer.extend_from_slice(&part);
                loop {
                    let boundary = buffer.windows(2).position(|v| v == b"\n\n").map(|p| (p, 2))
                        .into_iter()
                        .chain(buffer.windows(4).position(|v| v == b"\r\n\r\n").map(|p| (p, 4)))
                        .min_by_key(|(position, _)| *position);
                    let Some((end, separator)) = boundary else { break; };
                    if end > MAX_EVENT_BYTES {
                        yield Annotated::from_error("backend SSE event too large"); return;
                    }
                    let frame = match std::str::from_utf8(&buffer[..end]) {
                        Ok(frame) => frame,
                        Err(_) => { yield Annotated::from_error("backend SSE is not UTF-8"); return; }
                    };
                    let data = frame.lines().filter_map(|line| line.strip_prefix("data:")
                        .map(|v| v.strip_prefix(' ').unwrap_or(v))).collect::<Vec<_>>().join("\n");
                    if data == "[DONE]" {
                        if completed.len() != choices {
                            yield Annotated::from_error("backend SSE ended without every choice terminal");
                        }
                        return;
                    }
                    if !data.is_empty() {
                        let mut chunk = match decode_event(&data) {
                            Ok(chunk) => chunk,
                            Err(_) => { yield Annotated::from_error("invalid backend completion chunk"); return; }
                        };
                        for choice in &chunk.inner.choices {
                            if choice.index as usize >= choices || (choice.finish_reason.is_some() && !completed.insert(choice.index)) {
                                yield Annotated::from_error("invalid backend choice terminal"); return;
                            }
                        }
                        if usage.relay(&mut chunk) {
                            yield Annotated::from_data(chunk);
                        }
                    }
                    buffer.drain(..end + separator);
                }
                if buffer.len() > MAX_EVENT_BYTES {
                    yield Annotated::from_error("backend SSE event too large"); return;
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }

    async fn handle_completion(
        &self,
        _: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        bail!("HTTP text adapter supports Chat Completions, not legacy completions")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pd_selects_least_loaded_workers_and_rotates_ties() {
        assert_eq!(least_loaded([5, 1, 3].into_iter(), 0), 1);
        assert_eq!(least_loaded([5, 1, 3].into_iter(), 2), 1);
        let picks: BTreeSet<_> = (0..4).map(|s| least_loaded([0, 0, 0, 0].into_iter(), s)).collect();
        assert_eq!(picks.len(), 4);

        let engine = OpenAiHttpEngine::new_pd(
            vec![
                PdPrefillOrigin { origin: "http://p0:1/".into(), bootstrap_port: 8998 },
                PdPrefillOrigin { origin: "http://p1:1/".into(), bootstrap_port: 8998 },
            ],
            vec!["http://d0:1/".into()],
        )
        .unwrap();
        let pd = engine.pd.as_ref().unwrap();
        let _busy = LoadHold::new(&pd.prefills[0].outstanding, 100_000);
        for _ in 0..3 {
            assert_eq!(pd.select(None, 10), (1, 0, 10));
        }
        drop(_busy);
        assert_eq!(pd.prefills[0].outstanding.load(Ordering::Relaxed), 0);
        assert!(
            OpenAiHttpEngine::new_pd(
                vec![PdPrefillOrigin { origin: "http://p0:1/".into(), bootstrap_port: 0 }],
                vec!["http://d0:1/".into()],
            )
            .is_err()
        );
    }

    fn pd_engine(prefills: usize) -> OpenAiHttpEngine {
        OpenAiHttpEngine::new_pd(
            (0..prefills)
                .map(|i| PdPrefillOrigin { origin: format!("http://p{i}:1/"), bootstrap_port: 8998 })
                .collect(),
            vec!["http://d0:1/".into()],
        )
        .unwrap()
    }

    #[test]
    fn affinity_keys_ignore_object_key_order() {
        let a = json!({"model": "glm", "messages": [{"role": "user", "content": "x"}],
                       "chat_template_kwargs": {"thinking": true, "reasoning_effort": "low"}});
        let b = json!({"model": "glm", "messages": [{"content": "x", "role": "user"}],
                       "chat_template_kwargs": {"reasoning_effort": "low", "thinking": true}});
        assert_eq!(session_key(&a), session_key(&b));
        assert_eq!(canonical_json(&a), canonical_json(&b));
        let engine = OpenAiHttpEngine::new(
            (0..8).map(|i| format!("http://w{i}:1/")).collect(),
        )
        .unwrap();
        assert_eq!(engine.worker_index(&a), engine.worker_index(&b));
    }

    #[test]
    fn pd_session_keeps_one_prefill_across_turns_and_spills_when_overloaded() {
        let engine = pd_engine(8);
        let pd = engine.pd.as_ref().unwrap();
        let turn = |n: usize| {
            let mut messages = vec![json!({"role": "system", "content": "long shared system"}),
                                    json!({"role": "user", "content": "first question"})];
            for i in 0..n {
                messages.push(json!({"role": "assistant", "content": format!("answer {i}")}));
                messages.push(json!({"role": "user", "content": format!("follow-up {i}")}));
            }
            json!({"model": "glm", "messages": messages, "tools": [{"type": "function"}]})
        };
        let key = session_key(&turn(0)).unwrap();
        let home = pd.select(Some(&key), 100).0;
        for n in 1..4 {
            assert_eq!(session_key(&turn(n)).unwrap(), key, "multi-turn keeps the key");
            assert_eq!(pd.select(session_key(&turn(n)).as_deref(), 100 + n).0, home);
        }
        // Namespace and explicit keys separate sessions.
        let mut salted = turn(0);
        salted["cache_salt"] = json!("tenant-b");
        assert_ne!(session_key(&salted).unwrap(), key);
        let mut explicit = turn(2);
        explicit["prompt_cache_key"] = json!("k");
        let mut explicit_other_prompt = json!({"model": "glm", "messages": [{"role": "user", "content": "x"}]});
        explicit_other_prompt["prompt_cache_key"] = json!("k");
        assert_eq!(session_key(&explicit), session_key(&explicit_other_prompt));
        assert_eq!(session_key(&json!({"model": "glm"})), None);
        // A new session spills from an overloaded first choice to the same
        // second choice every time.
        let fresh = b"fresh".to_vec();
        let first = pd.select(Some(&fresh), 1).0;
        pd.sessions.lock().pop(&fresh);
        let busy = LoadHold::new(&pd.prefills[first].outstanding, PD_AFFINITY_SLACK + 1);
        let spill = pd.select(Some(&fresh), 1).0;
        assert_ne!(spill, first);
        pd.sessions.lock().pop(&fresh);
        assert_eq!(pd.select(Some(&fresh), 1).0, spill);
        drop(busy);
        // Distinct sessions spread over the prefills.
        let homes: BTreeSet<_> = (0..64)
            .map(|i| pd.select(Some(format!("s{i}").as_bytes()), 1).0)
            .collect();
        assert!(homes.len() >= 6, "{homes:?}");
    }

    #[test]
    fn pd_sessions_pay_only_new_bytes_where_their_prefix_is_cached() {
        let engine = pd_engine(8);
        let pd = engine.pd.as_ref().unwrap();
        let key = b"agent".to_vec();
        // First turn: whole prompt.
        let (home, _, work) = pd.select(Some(&key), 300_000);
        assert_eq!(work, 300_000);
        // Later turns: only the appended bytes, on the same prefill, even when
        // the home carries more raw work than the old byte-count slack allowed
        // but less than the cost of recomputing the prompt elsewhere.
        let _busy = LoadHold::new(&pd.prefills[home].outstanding, 250_000);
        let (stay, _, work) = pd.select(Some(&key), 310_000);
        assert_eq!((stay, work), (home, 10_000));
        // Recomputing elsewhere wins once the home backlog exceeds the prompt.
        let _busier = LoadHold::new(&pd.prefills[home].outstanding, 200_000);
        let (moved, _, work) = pd.select(Some(&key), 320_000);
        assert_ne!(moved, home);
        assert_eq!(work, 320_000);
        // The session stays where its prefix now lives.
        let (next, _, work) = pd.select(Some(&key), 330_000);
        assert_eq!((next, work), (moved, 10_000));
        // A rewritten (shorter) history is not treated as cached.
        assert_eq!(pd.select(Some(&key), 50_000).2, 50_000);
    }

    #[test]
    fn pd_sessions_keep_their_decode_until_it_is_overloaded() {
        let engine = OpenAiHttpEngine::new_pd(
            vec![PdPrefillOrigin { origin: "http://p0:1/".into(), bootstrap_port: 8998 }],
            (0..4).map(|i| format!("http://d{i}:1/")).collect(),
        )
        .unwrap();
        let pd = engine.pd.as_ref().unwrap();
        let key = b"agent".to_vec();
        let home = pd.select(Some(&key), 100).1;
        // The session's decode stays while within the slack of the least loaded.
        let _held = LoadHold::new(&pd.decodes[home].outstanding, PD_DECODE_SLACK);
        for n in 1..4 {
            assert_eq!(pd.select(Some(&key), 100 + n).1, home);
        }
        // Beyond the slack it moves, and then stays on the new decode.
        let _busy = LoadHold::new(&pd.decodes[home].outstanding, 1);
        let moved = pd.select(Some(&key), 200).1;
        assert_ne!(moved, home);
        assert_eq!(pd.select(Some(&key), 210).1, moved);
        // Requests without a session still go to the least loaded decode.
        let picks: BTreeSet<_> = (0..8).map(|_| pd.select(None, 1).1).collect();
        assert!(!picks.contains(&home));
    }

    #[test]
    fn usage_relay_feeds_metrics_and_returns_only_requested_usage() {
        let chunk = |choices: bool, out: u32| {
            let mut wire = json!({"id": "c", "object": "chat.completion.chunk", "created": 1,
                "model": "m", "choices": [],
                "usage": {"prompt_tokens": 100, "completion_tokens": out, "total_tokens": 100 + out,
                          "prompt_tokens_details": {"cached_tokens": 64}}});
            if choices {
                wire["choices"] = json!([{"index": 0, "delta": {"content": "x"}, "finish_reason": null}]);
            }
            decode_event(&wire.to_string()).unwrap()
        };
        let request = |options: Value| -> NvCreateChatCompletionRequest {
            let mut raw = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": true});
            if !options.is_null() {
                raw["stream_options"] = options;
            }
            serde_json::from_value(raw).unwrap()
        };
        // No usage requested: metrics on every chunk, usage stripped, usage-only chunk dropped.
        let mut relay = UsageRelay::new(&request(Value::Null));
        let mut first = chunk(true, 3);
        assert!(relay.relay(&mut first));
        assert!(first.inner.usage.is_none());
        let m = first.llm_metrics.unwrap();
        assert_eq!((m.input_tokens, m.output_tokens, m.chunk_tokens, m.cached_tokens), (100, 3, 3, Some(64)));
        let mut second = chunk(true, 7);
        assert!(relay.relay(&mut second));
        assert_eq!(second.llm_metrics.unwrap().chunk_tokens, 4);
        let mut tail = chunk(false, 7);
        assert!(!relay.relay(&mut tail));
        // include_usage: final usage chunk kept, per-chunk usage stripped.
        let mut relay = UsageRelay::new(&request(json!({"include_usage": true})));
        let mut first = chunk(true, 3);
        assert!(relay.relay(&mut first) && first.inner.usage.is_none());
        let mut tail = chunk(false, 3);
        assert!(relay.relay(&mut tail) && tail.inner.usage.is_some());
        assert_eq!(tail.llm_metrics.unwrap().chunk_tokens, 0);
        // Unary: the usage-only chunk always reaches the aggregator.
        let mut unary = request(Value::Null);
        unary.inner.stream = Some(false);
        let mut relay = UsageRelay::new(&unary);
        let mut first = chunk(true, 3);
        assert!(relay.relay(&mut first) && first.inner.usage.is_none());
        let mut tail = chunk(false, 3);
        assert!(relay.relay(&mut tail) && tail.inner.usage.is_some());
        // continuous_usage_stats: usage kept on every chunk.
        let mut relay = UsageRelay::new(&request(json!({"include_usage": true, "continuous_usage_stats": true})));
        let mut first = chunk(true, 3);
        assert!(relay.relay(&mut first) && first.inner.usage.is_some());
    }

    #[test]
    fn worker_request_joins_reasoning_segments_into_one_string() {
        let raw = json!({"model": "glm-5.3", "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "reasoning_content": ["Need the tool.", ""],
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "12C"},
            {"role": "assistant", "reasoning_content": ["", ""], "content": "12C."},
            {"role": "assistant", "reasoning_content": "plain", "content": "ok"}
        ]});
        let body = worker_request(serde_json::from_value(raw).unwrap()).unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[1]["reasoning_content"], "Need the tool.");
        assert!(messages[3].get("reasoning_content").is_none());
        assert_eq!(messages[4]["reasoning_content"], "plain");
    }

    #[test]
    fn prefill_failures_keep_worker_status() {
        let status = |result| {
            prefill_outcome(Ok(result))
                .unwrap_err()
                .downcast::<HttpError>()
                .map(|e| e.code)
                .ok()
        };
        assert_eq!(status(Err(PrefillError::Status(400, "bad".into()))), Some(400));
        assert_eq!(status(Err(PrefillError::Status(500, "boom".into()))), Some(500));
        assert_eq!(status(Err(PrefillError::Unavailable)), Some(503));
        assert_eq!(status(Err(PrefillError::Failed("x"))), None);
        assert!(prefill_outcome(Ok(Ok(()))).is_ok());
    }

    #[test]
    fn worker_request_maps_canonical_and_legacy_template_controls() {
        for extra in [
            json!({"chat_template_args": {"enable_thinking": false, "test_custom": 7}}),
            json!({"chat_template_kwargs": {"enable_thinking": false, "test_custom": 7}}),
            json!({"thinking": {"type": "disabled"}}),
        ] {
            let mut raw =
                json!({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "OK"}]});
            raw.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let body = worker_request(serde_json::from_value(raw).unwrap()).unwrap();
            assert!(body.get("chat_template_args").is_none());
            assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
            if extra.get("thinking").is_none() {
                assert_eq!(body["chat_template_kwargs"]["test_custom"], 7);
            }
            assert_eq!(body["stream"], true);
            assert_eq!(body["stream_options"]["include_usage"], true);
            assert_eq!(body["stream_options"]["continuous_usage_stats"], true);
        }
    }

    #[test]
    fn worker_request_preserves_canonical_cache_salt_and_legacy_precedence() {
        for (extra, expected) in [
            (json!({}), None),
            (json!({"cache_salt": "legacy"}), Some("legacy")),
            (
                json!({"nvext": {"cache_salt": "canonical"}}),
                Some("canonical"),
            ),
            (
                json!({"cache_salt": "legacy", "nvext": {"cache_salt": "canonical"}}),
                Some("canonical"),
            ),
            (
                json!({"cache_salt": "legacy", "nvext": {"cache_salt": ""}}),
                Some("legacy"),
            ),
        ] {
            let mut raw = json!({"model": "glm-5.3-flash",
                "messages": [{"role": "user", "content": "OK"}]});
            raw.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let body = worker_request(serde_json::from_value(raw).unwrap()).unwrap();
            assert_eq!(body.get("cache_salt").and_then(Value::as_str), expected);
        }
    }

    #[test]
    fn cache_affinity_is_stable_for_explicit_keys_and_identical_prompts() {
        let engine = OpenAiHttpEngine::new(vec![
            "http://127.0.0.1:28100".into(),
            "http://127.0.0.1:28101".into(),
        ])
        .unwrap();
        let raw = json!({"model":"glm-5.3-flash", "messages":[{"role":"user","content":"OK"}],
            "prompt_cache_key":"prefix-key", "nvext":{"cache_salt":"tenant-A"}});
        let body = worker_request(serde_json::from_value(raw).unwrap()).unwrap();
        assert_eq!(body["prompt_cache_key"], "prefix-key");
        let explicit = engine.worker_index(&body);
        for _ in 0..16 {
            assert_eq!(engine.worker_index(&body), explicit);
        }

        let automatic = json!({
            "model":"glm-5.3-flash",
            "messages":[{"role":"user","content":"stable prompt"}],
            "max_tokens":1,
        });
        let first = engine.worker_index(&automatic);
        for _ in 0..16 {
            assert_eq!(engine.worker_index(&automatic), first);
        }
        let different_generation_controls = json!({
            "model":"glm-5.3-flash",
            "messages":[{"role":"user","content":"stable prompt"}],
            "max_tokens":128,
            "temperature":0.7,
        });
        assert_eq!(engine.worker_index(&different_generation_controls), first);

        let no_prompt = json!({"model":"glm-5.3-flash"});
        assert_ne!(
            engine.worker_index(&no_prompt),
            engine.worker_index(&no_prompt)
        );
    }

    #[test]
    fn native_write_usage_normalizes_legacy_counts_without_inventing_them() {
        for (details, expected) in [
            (json!({"created_cache_tokens": 24}), Some(24)),
            (json!({"created_cache_tokens": 0}), Some(0)),
            (
                json!({"cache_write_tokens": 7, "created_cache_tokens": 24}),
                Some(7),
            ),
            (
                json!({"cache_write_tokens": null, "created_cache_tokens": 24}),
                None,
            ),
            (json!({}), None),
        ] {
            let raw = json!({"id":"native", "object":"chat.completion.chunk",
                "created":7, "model":"glm-5.3-flash", "choices":[],
                "usage":{"prompt_tokens":100,"completion_tokens":7,"total_tokens":107,
                    "prompt_tokens_details":details}});
            let chunk = decode_event(&raw.to_string()).unwrap();
            let encoded = serde_json::to_value(chunk).unwrap();
            assert_eq!(
                encoded
                    .pointer("/usage/prompt_tokens_details/cache_write_tokens")
                    .and_then(Value::as_i64),
                expected
            );
            assert!(
                encoded
                    .pointer("/usage/prompt_tokens_details/created_cache_tokens")
                    .is_none()
            );
        }
        for invalid in [json!(-1), json!(101), json!("24"), json!(true)] {
            let raw = json!({"id":"native", "object":"chat.completion.chunk",
                "created":7, "model":"glm-5.3-flash", "choices":[],
                "usage":{"prompt_tokens":100,"completion_tokens":7,"total_tokens":107,
                    "prompt_tokens_details":{"created_cache_tokens":invalid}}});
            assert!(decode_event(&raw.to_string()).is_err());
        }
    }

    #[test]
    fn native_usage_only_chunk_retains_backend_envelope() {
        let raw = json!({"id":"chatcmpl-native", "object":"chat.completion.chunk",
            "created":7, "model":"glm-5.3-flash", "choices":[],
            "usage":{"prompt_tokens":100,"completion_tokens":7,"total_tokens":107,
                "prompt_tokens_details":{"cached_tokens":60}}});
        let chunk = decode_event(&raw.to_string()).unwrap();
        let roundtrip = serde_json::to_value(chunk).unwrap();
        for key in ["id", "object", "created", "model", "usage"] {
            assert_eq!(roundtrip[key], raw[key]);
        }
        for key in ["id", "object", "created", "model"] {
            let mut invalid = raw.clone();
            invalid.as_object_mut().unwrap().remove(key);
            assert!(decode_event(&invalid.to_string()).is_err());
        }
        assert!(decode_event(&json!({"choices":[], "usage":raw["usage"]}).to_string()).is_err());
    }
}
