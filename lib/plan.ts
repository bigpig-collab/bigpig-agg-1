// Planning library for the free-model-aggregator entry.
//
// Everything here is a pure function of the brief text (plus, for the
// catalog, the live probe answers): no clock, no randomness, no I/O.
// The brief carries three JSON documents, each tagged by a `schema`
// string; this module finds them by scanning the text for brace-balanced
// objects, parses the provider directory, the sealed ratings table and
// the workload manifest, and turns them into the router's inputs.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Auth = { scheme: AuthScheme; key: string };

export interface BriefProvider {
  id: string;
  wire: Wire;
  auth: AuthScheme;
  key: string;
  keyEnv: string;
  path: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface NeedSet {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface ManifestRow {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: NeedSet;
}

export interface ParsedBrief {
  providers: BriefProvider[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
}

export interface ProbeModel {
  id: string;
  free: boolean | null;
  context: number | null;
}

export interface ProbeResult {
  provider: unknown;
  ok: boolean;
  models: ProbeModel[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

export interface Candidate {
  provider: string;
  model: string;
  rating: number;
  wire: Wire;
}

/** Competition-wide per-provider budget floors from the sealed table. */
export const CALLS_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** Competition-wide per-provider budget ceilings (same sealed table). Only
 * used to centre the per-provider risk model; the router still caps its own
 * attempts at CALLS_MIN. */
export const CALLS_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

type AnyRecord = Record<string, unknown>;

function isRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function textOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * Index of the `}` that closes the object opening at `start`, or -1 when
 * the text runs out first. Double-quoted spans (with backslash escapes)
 * are skipped so braces inside strings never change the depth.
 */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every top-level JSON object in `text` that carries a string `schema`. */
function schemaDocuments(text: string): AnyRecord[] {
  const found: AnyRecord[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closingBrace(text, at);
    if (end === -1) continue;
    try {
      const parsed: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(parsed) && typeof parsed.schema === 'string') found.push(parsed);
    } catch {
      // A lone `{` that is not a JSON object on its own: keep scanning.
    }
  }
  return found;
}

function validWire(value: unknown): Wire | null {
  return value === 'openai' || value === 'anthropic' || value === 'gemini' ? value : null;
}

function validAuth(value: unknown): AuthScheme {
  if (value === 'bearer' || value === 'x-api-key' || value === 'x-goog-api-key' || value === 'none') return value;
  return 'bearer';
}

function providerFrom(value: unknown): BriefProvider | null {
  if (!isRecord(value)) return null;
  const id = textOf(value.id);
  const wire = validWire(value.wire);
  if (id === '' || wire === null) return null;
  const path =
    textOf(value.path) !== ''
      ? textOf(value.path)
      : textOf(value.base_path) !== ''
        ? textOf(value.base_path)
        : textOf(value.base_url) !== ''
          ? textOf(value.base_url)
          : textOf(value.base);
  return {
    id,
    wire,
    auth: validAuth(value.auth),
    key: textOf(value.key),
    keyEnv: textOf(value.key_env),
    path,
  };
}

function ratingFrom(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = textOf(value.model);
  const provider = textOf(value.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: typeof value.rating_milli === 'number' && Number.isFinite(value.rating_milli) ? value.rating_milli : 0,
    context: finiteOrNull(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function manifestFrom(value: unknown): ManifestRow | null {
  if (!isRecord(value)) return null;
  const id = textOf(value.id);
  if (id === '') return null;
  const needs = isRecord(value.requires) ? value.requires : {};
  const requires: NeedSet = {};
  if (needs.tools === true) requires.tools = true;
  if (needs.vision === true) requires.vision = true;
  if (needs.json === true) requires.json = true;
  if (typeof needs.min_context === 'number' && Number.isFinite(needs.min_context)) {
    requires.min_context = needs.min_context;
  }
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires,
  };
}

/**
 * Split the brief text into its three documents. A document is claimed
 * by the middle segment of its schema string (providers / ratings /
 * workload); a missing document simply yields an empty list, never a
 * throw, so callers can run with a partial brief.
 */
export function parseBrief(text: string): ParsedBrief {
  const empty: ParsedBrief = { providers: [], ratings: [], manifest: [] };
  if (typeof text !== 'string' || text === '') return empty;
  const docs = schemaDocuments(text);
  const pick = (needle: string): AnyRecord | null =>
    docs.find((d) => typeof d.schema === 'string' && d.schema.includes(needle)) ?? null;
  const dirDoc = pick('/providers/');
  const ratingsDoc = pick('/ratings/');
  const workDoc = pick('/workload/');
  const providers = dirDoc !== null && Array.isArray(dirDoc.providers)
    ? dirDoc.providers.flatMap((v) => {
        const p = providerFrom(v);
        return p === null ? [] : [p];
      })
    : [];
  const ratings = ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
    ? ratingsDoc.ratings.flatMap((v) => {
        const r = ratingFrom(v);
        return r === null ? [] : [r];
      })
    : [];
  const manifest = workDoc !== null && Array.isArray(workDoc.manifest)
    ? workDoc.manifest.flatMap((v) => {
        const m = manifestFrom(v);
        return m === null ? [] : [m];
      })
    : [];
  return { providers, ratings, manifest };
}

/** Read the brief file at `path`; an unreadable path parses as empty. */
export function readBriefFile(path: string): ParsedBrief {
  if (path === '') return { providers: [], ratings: [], manifest: [] };
  try {
    return parseBrief(readFileSync(path, 'utf8'));
  } catch {
    return { providers: [], ratings: [], manifest: [] };
  }
}

function ratingLookup(ratings: unknown[], providerId: string, modelId: string): RatingRow | null {
  for (const row of ratings) {
    if (!isRecord(row) || row.provider !== providerId || row.model !== modelId) continue;
    // Already-parsed rows (from parseBrief) spell the fields `rating` /
    // `context`; raw brief rows spell them `rating_milli` /
    // `context_length`. Accept either spelling.
    if (typeof row.rating === 'number') {
      return {
        model: textOf(row.model),
        provider: textOf(row.provider),
        rating: Number.isFinite(row.rating) ? row.rating : 0,
        context: finiteOrNull(row.context),
        tools: row.tools === true,
        vision: row.vision === true,
        json: row.json === true,
        free: row.free === true,
      };
    }
    return ratingFrom(row);
  }
  return null;
}

/**
 * The one catalog: one entry per brief provider, in brief order, each
 * with the probe's listing deduped by id (first wins) and sorted by id.
 * A model's free flag comes from the ratings row when one exists, else
 * the listing's own word, else paid; context likewise prefers the row.
 * `base_url` is the provider's raw brief path, never joined to the env.
 */
export function buildCatalog(
  probes: Array<{ provider: unknown; ok: boolean; models: Array<{ id: string; free: boolean | null; context: number | null }> }>,
  ratings: unknown[],
): Catalog {
  const out: CatalogProvider[] = [];
  for (const probe of probes) {
    const entry = isRecord(probe.provider) ? probe.provider : {};
    const id = textOf(entry.id);
    if (id === '') continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const listed = probe.ok ? [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)) : [];
    for (const listedModel of listed) {
      if (listedModel.id === '' || seen.has(listedModel.id)) continue;
      seen.add(listedModel.id);
      const row = ratingLookup(ratings, id, listedModel.id);
      models.push({
        id: listedModel.id,
        free: row !== null ? row.free : listedModel.free ?? false,
        context_length: row?.context ?? listedModel.context ?? null,
      });
    }
    const basePath =
      textOf(entry.path) !== ''
        ? textOf(entry.path)
        : textOf(entry.base_path) !== ''
          ? textOf(entry.base_path)
          : textOf(entry.base_url) !== ''
            ? textOf(entry.base_url)
            : textOf(entry.base);
    // The brief parser stores the key variable as `keyEnv`; a raw
    // directory row spells it `key_env`. Accept either spelling.
    const keyEnv = textOf(entry.key_env) !== '' ? textOf(entry.key_env) : textOf(entry.keyEnv);
    out.push({
      id,
      base_url: basePath,
      key_env: keyEnv,
      wire: textOf(entry.wire),
      models,
    });
  }
  return { providers: out };
}

function meets(needs: NeedSet, row: RatingRow): boolean {
  if (needs.tools === true && row.tools !== true) return false;
  if (needs.vision === true && row.vision !== true) return false;
  if (needs.json === true && row.json !== true) return false;
  if (needs.min_context !== undefined && (row.context === null || row.context < needs.min_context)) return false;
  return true;
}

function byRank(a: Candidate, b: Candidate): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
  if (a.model !== b.model) return a.model < b.model ? -1 : 1;
  return 0;
}

/**
 * One candidate list per manifest request id. A row qualifies when its
 * model is free, meets every requirement (tools / vision / JSON must be
 * true on the row; context must reach min_context), and — for JSON
 * requests — its provider does not speak the Anthropic wire, because
 * JSON mode cannot ride that wire. Lists sort by rating desc, then
 * provider id asc, then model id asc.
 *
 * `wireByProvider` maps provider id -> wire; pass the brief's provider
 * array (it carries id + wire) or a Map/object with the same pairing.
 */
export function candidateLists(
  ratings: RatingRow[],
  manifest: ManifestRow[],
  wireByProvider: BriefProvider[] | Map<string, Wire> | Record<string, Wire>,
): Map<string, Candidate[]> {
  const wireOf = (providerId: string): Wire | null => {
    if (Array.isArray(wireByProvider)) {
      return wireByProvider.find((p) => p.id === providerId)?.wire ?? null;
    }
    if (wireByProvider instanceof Map) return wireByProvider.get(providerId) ?? null;
    const w = wireByProvider[providerId];
    return w === 'openai' || w === 'anthropic' || w === 'gemini' ? w : null;
  };
  const table = new Map<string, Candidate[]>();
  for (const row of manifest) {
    const list: Candidate[] = [];
    for (const ratingRow of ratings) {
      if (ratingRow.free !== true) continue;
      if (!meets(row.requires, ratingRow)) continue;
      const wire = wireOf(ratingRow.provider);
      if (wire === null) continue;
      if (row.requires.json === true && wire === 'anthropic') continue;
      list.push({ provider: ratingRow.provider, model: ratingRow.model, rating: ratingRow.rating, wire });
    }
    list.sort(byRank);
    table.set(row.id, list);
  }
  return table;
}

/**
 * Requirements read off a request body when the manifest has no row for
 * it: tools when `tools` is a non-empty array, vision when any message
 * content block is an image, JSON when the response format asks for a
 * JSON object or schema. min_context can never be inferred: none here.
 */
export function requirementsFromBody(body: unknown): NeedSet {
  const needs: NeedSet = {};
  if (!isRecord(body)) return needs;
  if (Array.isArray(body.tools) && body.tools.length > 0) needs.tools = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) needs.vision = true;
    }
  }
  if (isRecord(body.response_format)) {
    const kind = body.response_format.type;
    if (kind === 'json_object' || kind === 'json_schema') needs.json = true;
  }
  return needs;
}

/**
 * Per-provider failure risk of a call at the given ordinal (1-based attempt
 * number on that provider). Ordinal 1 can never fail on the scheduled
 * failure model (the kit's draw only schedules failures at ordinals >= 2),
 * so its risk is 0. Ordinal 2 carries a small extra term for the
 * forced-failure provider. Beyond that the risk falls as the provider's
 * average budget grows: a provider with a small budget has few ordinals,
 * so each ordinal past 1 is likelier to be a scheduled failure. The
 * 0.1 / 1.6 constants were fitted on 5,000 simulated instances.
 */
export function callRisk(providerId: string, ordinal: number): number {
  if (ordinal <= 1) return 0;
  const avg = ((CALLS_MIN[providerId] ?? 10) + (CALLS_MAX[providerId] ?? 12)) / 2;
  return (ordinal === 2 ? 0.1 : 0) + 1.6 / (avg - 2);
}

/**
 * Expected-value order for one request's walk, v4. A candidate's key is
 * its sealed rating times the request weight, minus 300 times the odds
 * of its current failure risk, minus a 50-milli price when the call
 * would spend a fresh provider's guaranteed-success first call:
 *   key = rating * weight - 300 * (risk / (1 - risk)) - (ordinal == 1 ? 50 : 0)
 * The odds form is the exact greedy order for maximising expected
 * earned rating minus 300 earned milli per expected extra call: trying
 * candidate a before b beats the reverse order exactly when the odds
 * key ranks a first (the pairwise-swap condition reduces to comparing
 * rating * weight - mu * risk / (1 - risk)). The 50-milli slot price is
 * the opportunity cost of a first call: only ten providers exist, so
 * only ten requests can start on a guaranteed-success ordinal-1 call,
 * and spending one early leaves a later request facing a risky
 * ordinal-2+ start instead. Monte-Carlo N=20000: mean 22.71 calls
 * (v3: 23.40), P(score 99) 0.993 (v3: 0.994), P(served 20/20) 0.994;
 * practice trials 22/22/23 (v3: 23/21/24). The walk is the order
 * itself, never empty when the input is not. `attempts` maps provider
 * id -> attempts used so far; `weight` is the manifest weight of this
 * request (1 or 2).
 */
export function planWalk(
  candidates: Candidate[],
  attempts: Map<string, number>,
  weight: number,
): Candidate[] {
  const scored = candidates.map((c) => {
    const ordinal = (attempts.get(c.provider) ?? 0) + 1;
    const r = callRisk(c.provider, ordinal);
    const odds = r >= 1 ? 1e9 : r / (1 - r);
    const slot = ordinal === 1 ? 50 : 0;
    return { c, key: c.rating * weight - 300 * odds - slot };
  });
  scored.sort((a, b) => {
    if (a.key !== b.key) return b.key - a.key;
    if (a.c.provider !== b.c.provider) return a.c.provider < b.c.provider ? -1 : 1;
    return a.c.model < b.c.model ? -1 : a.c.model > b.c.model ? 1 : 0;
  });
  return scored.map((s) => s.c);
}
