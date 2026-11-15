// Free-model aggregator entrypoint.
//
// Reads the brief, probes the ten providers, builds one catalog, and
// serves OpenAI clients (POST /v1/chat/completions) and Anthropic
// clients (POST /v1/messages) from the best free model that meets each
// request, failing over across providers inside their budget floors.
// Routing is policy v4: candidates ordered by expected value
// (rating * weight - 300 * failure odds - 50 for a first call,
// planWalk in lib/plan.ts), with a per-provider attempt cap, a
// three-strike dead rule on refusals, and exactly one successful
// upstream call per request. Every choice is a function of the brief
// plus live provider answers; there is no clock and no randomness
// anywhere in this file.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  anthropicError,
  anthropicModelList,
  buildUpstream,
  openaiError,
  openaiModelList,
  translateAnswer,
  translateStream,
  type Auth,
  type Wire,
} from './lib/wire.ts';
import {
  buildCatalog,
  candidateLists,
  CALLS_MIN,
  parseBrief,
  planWalk,
  readBriefFile,
  requirementsFromBody,
  type BriefProvider,
  type Candidate,
  type Catalog,
  type ManifestRow,
  type NeedSet,
  type ParsedBrief,
  type ProbeModel,
  type RatingRow,
} from './lib/plan.ts';

const PROTOCOL_VERSION = 1;
const POLL_EVERY_MS = 100;
const PROBE_LIMIT_MS = 5000;
const REGISTER_LIMIT_MS = 5000;
const WORK_LIMIT_MS = 2000;
const UPSTREAM_LIMIT_MS = 25000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type ClientWire = 'openai' | 'anthropic';

interface DialInfo {
  baseUrl: string;
  key: string;
}

interface ProviderBook {
  brief: BriefProvider;
  dial: DialInfo;
  probeOk: boolean;
  state: { attempts: number; streak: number; dead: boolean };
}

interface InstanceDoc {
  providers: Array<{ id: string; base_url: string; key: string }>;
  manifest: ManifestRow[];
}

let brief: ParsedBrief = { providers: [], ratings: [], manifest: [] };
let catalog: Catalog = { providers: [] };
let candidatesByRequest = new Map<string, Candidate[]>();
let books = new Map<string, ProviderBook>();
let manifestCursor = 0;

function note(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function textOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** Join the env origin to a brief path; an absolute base stands alone. */
function joinOrigin(origin: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  if (trimmed === '') return origin;
  return `${origin}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}

function authHeaders(scheme: Auth['scheme'], key: string): Record<string, string> {
  if (key === '') return {};
  if (scheme === 'bearer') return { authorization: `Bearer ${key}` };
  if (scheme === 'x-api-key') return { 'x-api-key': key };
  if (scheme === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function boolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** A listing's free flag: its own boolean, else true when every price is zero, else unknown. */
function freeFromListing(row: Record<string, unknown>): boolean | null {
  const direct = boolOrNull(row.free);
  if (direct !== null) return direct;
  if (!isRecord(row.pricing)) return null;
  const amounts = Object.values(row.pricing).flatMap((v) => {
    if (typeof v === 'number') return [v];
    if (typeof v === 'string' && v.trim() !== '') return [Number(v)];
    return [];
  });
  if (amounts.length === 0) return null;
  return amounts.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a catalog route spelled it, in that route's wire. */
function listingModel(wire: Wire, row: unknown): ProbeModel | null {
  if (!isRecord(row)) return null;
  const id = wire === 'gemini' ? textOf(row.name).replace(/^models\//, '') : textOf(row.id);
  if (id === '') return null;
  const context = finiteOrNull(row.context_length) ?? finiteOrNull(row.context_window) ?? finiteOrNull(row.inputTokenLimit);
  return { id, free: freeFromListing(row), context };
}

function listingModels(wire: Wire, body: unknown): ProbeModel[] {
  if (!isRecord(body)) return [];
  const rows = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(rows)) return [];
  return rows.flatMap((row) => {
    const model = listingModel(wire, row);
    return model === null ? [] : [model];
  });
}

/** GET /instance: absolute base URLs and keys, plus the live manifest. */
async function fetchInstance(): Promise<InstanceDoc | null> {
  if (ENV_URL === '') return null;
  try {
    const res = await fetch(`${ENV_URL}/instance`, { signal: AbortSignal.timeout(PROBE_LIMIT_MS) });
    if (!res.ok) return null;
    const body: unknown = await res.json();
    if (!isRecord(body)) return null;
    const providers = Array.isArray(body.providers)
      ? body.providers.flatMap((p) => {
          if (!isRecord(p) || textOf(p.id) === '' || textOf(p.base_url) === '') return [];
          return [{ id: textOf(p.id), base_url: textOf(p.base_url), key: textOf(p.key) }];
        })
      : [];
    const workload = isRecord(body.workload) && Array.isArray(body.workload.manifest) ? body.workload.manifest : [];
    const manifest = workload.flatMap((row) => {
      if (!isRecord(row) || textOf(row.id) === '') return [];
      const needs = isRecord(row.requires) ? row.requires : {};
      const requires: NeedSet = {};
      if (needs.tools === true) requires.tools = true;
      if (needs.vision === true) requires.vision = true;
      if (needs.json === true) requires.json = true;
      if (typeof needs.min_context === 'number' && Number.isFinite(needs.min_context)) requires.min_context = needs.min_context;
      return [
        {
          id: textOf(row.id),
          api: row.api === 'anthropic' ? 'anthropic' : 'openai',
          stream: row.stream === true,
          weight: row.weight === 2 ? 2 : 1,
          requires,
        } satisfies ManifestRow,
      ];
    });
    return { providers, manifest };
  } catch {
    return null;
  }
}

/** Probe one provider's catalog route in its own wire; failure drops it. */
async function probeOne(entry: BriefProvider, dial: DialInfo): Promise<{ ok: boolean; models: ProbeModel[] }> {
  const route = entry.wire === 'gemini' ? `${dial.baseUrl}/v1beta/models` : `${dial.baseUrl}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(entry.auth, dial.key) };
  if (entry.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(route, { headers, signal: AbortSignal.timeout(PROBE_LIMIT_MS) });
    if (!res.ok) return { ok: false, models: [] };
    return { ok: true, models: listingModels(entry.wire, await res.json()) };
  } catch {
    return { ok: false, models: [] };
  }
}

/** Read the brief, the instance, probe every provider, build the catalog. */
async function load(): Promise<void> {
  brief = readBriefFile(PROMPT_FILE);
  if (PROMPT_FILE !== '' && brief.providers.length === 0) note('the brief held no provider directory');
  const instance = await fetchInstance();
  const override = new Map(instance?.providers.map((p) => [p.id, p]) ?? []);
  if (instance !== null && instance.manifest.length > 0) brief = { ...brief, manifest: instance.manifest };
  const dialOf = (entry: BriefProvider): DialInfo => {
    const hit = override.get(entry.id);
    if (hit !== undefined) return { baseUrl: hit.base_url.replace(/\/+$/, ''), key: hit.key };
    return { baseUrl: joinOrigin(ENV_URL, entry.path), key: entry.key };
  };
  const probes = await Promise.all(
    brief.providers.map(async (entry) => {
      const dial = dialOf(entry);
      const probe = await probeOne(entry, dial);
      if (!probe.ok) note(`probe failed for ${entry.id}`);
      return { provider: entry, ok: probe.ok, models: probe.models };
    }),
  );
  catalog = buildCatalog(probes, brief.ratings as unknown[]);
  candidatesByRequest = candidateLists(brief.ratings, brief.manifest, brief.providers);
  books = new Map<string, ProviderBook>();
  for (let i = 0; i < brief.providers.length; i++) {
    const entry = brief.providers[i];
    const probe = probes[i];
    books.set(entry.id, {
      brief: entry,
      dial: dialOf(entry),
      probeOk: probe.ok,
      state: { attempts: 0, streak: 0, dead: !probe.ok },
    });
  }
}

/** The free models of the registered catalog, brief order, ids sorted. */
function servedModels(): Array<{ id: string; provider: string }> {
  const out: Array<{ id: string; provider: string }> = [];
  for (const provider of catalog.providers) {
    for (const model of [...provider.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (model.free) out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

/** The request now arriving: its manifest row by arrival order, else the body's own hints. */
function needsFor(body: Record<string, unknown>): { needs: NeedSet; list: Candidate[]; weight: number } {
  const row = brief.manifest[manifestCursor];
  manifestCursor += 1;
  if (row !== undefined) {
    return { needs: row.requires, list: candidatesByRequest.get(row.id) ?? [], weight: row.weight };
  }
  const needs = requirementsFromBody(body);
  const fallback: ManifestRow = { id: '__live__', api: 'openai', stream: false, weight: 1, requires: needs };
  const list = candidateLists(brief.ratings, [fallback], brief.providers).get('__live__') ?? [];
  return { needs, list, weight: 1 };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readBody(req: IncomingMessage): Promise<Record<string, unknown> | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = typeof body === 'string' ? body : JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendFault(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openaiError(message, type));
}

/** GET /v1/models, shaped by the auth header the client sent. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openaiModelList(list));
}

/**
 * Walk one request's candidate list in the order planWalk gives it
 * (highest expected value first: rating times the request weight,
 * minus 300 times the odds of the provider's current failure risk,
 * minus 50 when the call would spend a fresh provider's first call).
 * Providers that are already dead, or have spent their budget floor, are passed over without a call; the first HTTP 200
 * finishes the request; a 402, 429 or 5xx earns the provider a strike
 * (a third strike retires it for the rest of the trial); a 401 or 400
 * steps past that candidate with no strike, since every call we place
 * is well-formed by construction.
 */
async function serveRequest(clientWire: ClientWire, body: Record<string, unknown>, res: ServerResponse): Promise<void> {
  const { needs, list, weight } = needsFor(body);
  const wantsJson = needs.json === true;
  const streaming = body.stream === true;
  const live: Candidate[] = [];
  const attempts = new Map<string, number>();
  for (const candidate of list) {
    const book = books.get(candidate.provider);
    if (book === undefined || book.state.dead) continue;
    const cap = CALLS_MIN[candidate.provider] ?? Number.POSITIVE_INFINITY;
    if (book.state.attempts >= cap) continue;
    live.push(candidate);
    attempts.set(candidate.provider, book.state.attempts);
  }
  const walk = planWalk(live, attempts, weight);
  for (const candidate of walk) {
    const book = books.get(candidate.provider);
    if (book === undefined || book.state.dead) continue;
    const cap = CALLS_MIN[candidate.provider] ?? Number.POSITIVE_INFINITY;
    if (book.state.attempts >= cap) continue;
    const auth: Auth = { scheme: book.brief.auth, key: book.dial.key };
    let upstream: { url: string; headers: Record<string, string>; body: string };
    try {
      upstream = buildUpstream(clientWire, book.brief.wire, book.dial.baseUrl, candidate.model, body, wantsJson, auth);
    } catch {
      continue;
    }
    book.state.attempts += 1;
    let answer: Response;
    try {
      answer = await fetch(upstream.url, {
        method: 'POST',
        headers: upstream.headers,
        body: upstream.body,
        signal: AbortSignal.timeout(UPSTREAM_LIMIT_MS),
      });
    } catch {
      book.state.streak += 1;
      if (book.state.streak >= 3) book.state.dead = true;
      continue;
    }
    if (answer.status === 200) {
      book.state.streak = 0;
      if (streaming) {
        const sse = await answer.text();
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(translateStream(clientWire, book.brief.wire, sse, candidate.model));
        return;
      }
      const translated = translateAnswer(clientWire, book.brief.wire, answer.status, await answer.text(), candidate.model);
      const payload = typeof translated.body === 'string' ? translated.body : JSON.stringify(translated.body);
      res.writeHead(translated.status, { 'content-type': translated.contentType, 'content-length': Buffer.byteLength(payload) });
      res.end(payload);
      return;
    }
    if (answer.status === 401 || answer.status === 400 || answer.status === 404) {
      await answer.text().catch(() => '');
      continue;
    }
    await answer.text().catch(() => '');
    book.state.streak += 1;
    if (book.state.streak >= 3) book.state.dead = true;
  }
  if (clientWire === 'anthropic') {
    sendJson(res, 503, anthropicError('api_error', 'no free model on the directory meets this request'));
  } else {
    sendJson(res, 503, openaiError('no free model on the directory meets this request', 'no_free_model'));
  }
}

async function chatRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) {
    sendFault(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  await serveRequest('openai', body, res);
}

async function messagesRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) {
    sendFault(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  await serveRequest('anthropic', body, res);
}

/** Requests run one at a time: a plain promise chain is the whole queue. */
let tail: Promise<void> = Promise.resolve();
function serially(job: () => Promise<void>): void {
  const run = tail.then(job, job);
  tail = run.then(
    () => {},
    () => {},
  );
}

async function dispatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chatRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messagesRoute(req, res);
    return;
  }
  const wire: ClientWire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
  sendFault(res, wire, 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listenOn(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; any failure is fatal. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(REGISTER_LIMIT_MS),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json().catch(() => null);
  if (isRecord(reply) && typeof reply.requests === 'number') note(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(WORK_LIMIT_MS) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_EVERY_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    serially(() =>
      dispatch(req, res).catch((err: unknown) => {
        note((err as Error).stack ?? String(err));
        if (!res.headersSent) {
          sendFault(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
        }
      }),
    );
  });
  await listenOn(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  note(`listening on 0.0.0.0:${address.port} with ${catalog.providers.length} catalog providers and ${servedModels().length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  note((err as Error).stack ?? String(err));
  process.exit(1);
}
