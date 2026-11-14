// wire.ts — translation between the two client wires (OpenAI, Anthropic)
// and the three upstream wires (OpenAI, Anthropic, Gemini).
//
// Every function here is pure: same arguments in, same bytes out. There is
// no clock and no randomness anywhere — the timestamps baked into replies
// are fixed constants, and every id is derived from the provider's answer
// or from a fixed literal, so a replay produces identical output.
//
// The design funnels every client body through one neutral request shape
// first (system text, a list of messages made of small parts, tool
// definitions, sampling knobs), then renders that shape into whichever
// upstream wire was routed. That keeps the six client-to-provider paths
// consistent: a field the client sent — system prompt, user text, an
// image, tool names in order, max_tokens, stop strings in order, JSON
// mode — reaches the provider untouched in its own wire's spelling, which
// is what the simulator echoes back for the driver to compare.
//
// JSON mode is only ever requested on the OpenAI and Gemini wires. The
// Anthropic wire has no such field, and JSON-mode work is never routed
// there, so the Gemini and OpenAI renderers add it when `needsJson` says
// so while the Anthropic renderer simply has nothing to add.

export type Wire = "openai" | "anthropic" | "gemini";
export type ClientWire = "openai" | "anthropic";
export type Auth = { scheme: "bearer" | "x-api-key" | "x-goog-api-key" | "none"; key: string };

// Fixed instants stamped into replies: nothing here reads the clock.
const OPENAI_CREATED = 1767225600;
const ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z";
// Streaming text is re-emitted in slices of this many characters.
const STREAM_SLICE = 13;

// ---------------------------------------------------------------------------
// Small value helpers
// ---------------------------------------------------------------------------

function isObj(v: unknown): v is Record<string, any> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function asArray(v: unknown): any[] {
  return Array.isArray(v) ? v : [];
}

function asText(v: unknown, fallback: string): string {
  return typeof v === "string" ? v : fallback;
}

function asCount(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

// The text carried by a string, or by the text blocks of a block array.
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const bits: string[] = [];
  for (const b of content) {
    if (isObj(b) && b.type === "text" && typeof b.text === "string") bits.push(b.text);
  }
  return bits.join("\n");
}

// A tool-call arguments value as an object: the provider's JSON string
// parsed, an object passed through, anything else an empty object.
function argsObject(v: unknown): Record<string, any> {
  if (isObj(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    try {
      const parsed: unknown = JSON.parse(v);
      if (isObj(parsed)) return parsed;
    } catch {
      // Not JSON: fall through to the empty input.
    }
  }
  return {};
}

// A stop value (string or array) as an ordered list of strings.
function stopList(v: unknown): string[] {
  if (typeof v === "string") return v === "" ? [] : [v];
  if (!Array.isArray(v)) return [];
  return v.filter((s): s is string => typeof s === "string");
}

// ---------------------------------------------------------------------------
// The neutral request shape
// ---------------------------------------------------------------------------

type NormPart =
  | { kind: "text"; text: string }
  | { kind: "image"; mime: string; data: string; url: string }
  | { kind: "call"; id: string; name: string; args: Record<string, any> }
  | { kind: "result"; id: string; text: string };

type NormMsg = { role: "user" | "assistant" | "tool"; parts: NormPart[]; toolId: string };

type NormTool = { name: string; description: string; parameters: Record<string, any> };

type NormChoice =
  | { kind: "auto" }
  | { kind: "any" }
  | { kind: "none" }
  | { kind: "one"; name: string };

type NormReq = {
  system: string;
  messages: NormMsg[];
  tools: NormTool[];
  choice: NormChoice | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  streaming: boolean;
  json: boolean;
};

function emptyReq(): NormReq {
  return {
    system: "",
    messages: [],
    tools: [],
    choice: null,
    maxTokens: null,
    temperature: null,
    topP: null,
    stops: [],
    streaming: false,
    json: false,
  };
}

function textParts(parts: NormPart[]): string {
  const bits: string[] = [];
  for (const p of parts) if (p.kind === "text") bits.push(p.text);
  return bits.join("\n");
}

// ---------------------------------------------------------------------------
// Client bodies into the neutral shape
// ---------------------------------------------------------------------------

// One OpenAI content part (or plain string) as neutral parts.
function openaiParts(content: unknown): NormPart[] {
  if (typeof content === "string") return content === "" ? [] : [{ kind: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const out: NormPart[] = [];
  for (const p of content) {
    if (!isObj(p)) continue;
    if (p.type === "text" && typeof p.text === "string") {
      out.push({ kind: "text", text: p.text });
    } else if (p.type === "image_url" && isObj(p.image_url) && typeof p.image_url.url === "string") {
      const m = /^data:([^;]+);base64,(.*)$/.exec(p.image_url.url);
      if (m !== null) out.push({ kind: "image", mime: m[1], data: m[2], url: "" });
      else out.push({ kind: "image", mime: "image/png", data: "", url: p.image_url.url });
    }
  }
  return out;
}

function openaiChoice(v: unknown): NormChoice | null {
  if (v === "auto") return { kind: "auto" };
  if (v === "required" || v === "any") return { kind: "any" };
  if (v === "none") return { kind: "none" };
  if (isObj(v) && v.type === "function" && isObj(v.function) && typeof v.function.name === "string") {
    return { kind: "one", name: v.function.name };
  }
  return null;
}

function normalizeOpenAI(body: any): NormReq {
  const req = emptyReq();
  req.json =
    isObj(body.response_format) &&
    (body.response_format.type === "json_object" || body.response_format.type === "json_schema");
  for (const m of asArray(body.messages)) {
    if (!isObj(m)) continue;
    if (m.role === "system") {
      if (req.system === "") req.system = blocksText(m.content);
      continue;
    }
    if (m.role === "tool") {
      const id = asText(m.tool_call_id, "");
      req.messages.push({ role: "tool", toolId: id, parts: [{ kind: "result", id, text: blocksText(m.content) }] });
      continue;
    }
    if (m.role !== "user" && m.role !== "assistant") continue;
    const parts = openaiParts(m.content);
    if (m.role === "assistant") {
      for (const c of asArray(m.tool_calls)) {
        if (!isObj(c) || !isObj(c.function)) continue;
        parts.push({
          kind: "call",
          id: asText(c.id, ""),
          name: asText(c.function.name, ""),
          args: argsObject(c.function.arguments),
        });
      }
    }
    req.messages.push({ role: m.role, toolId: "", parts });
  }
  for (const t of asArray(body.tools)) {
    if (!isObj(t) || !isObj(t.function) || typeof t.function.name !== "string") continue;
    req.tools.push({
      name: t.function.name,
      description: asText(t.function.description, ""),
      parameters: isObj(t.function.parameters) ? t.function.parameters : { type: "object", properties: {} },
    });
  }
  req.choice = openaiChoice(body.tool_choice);
  req.maxTokens = typeof body.max_tokens === "number" ? body.max_tokens : null;
  req.temperature = typeof body.temperature === "number" ? body.temperature : null;
  req.topP = typeof body.top_p === "number" ? body.top_p : null;
  req.stops = stopList(body.stop);
  req.streaming = body.stream === true;
  return req;
}

// One Anthropic content block as a neutral part, or null when the block
// carries nothing the neutral shape can hold.
function anthropicPart(block: any): NormPart | null {
  if (!isObj(block)) return null;
  if (block.type === "text" && typeof block.text === "string") return { kind: "text", text: block.text };
  if (block.type === "image" && isObj(block.source)) {
    const s = block.source;
    if (s.type === "base64" && typeof s.data === "string") {
      return { kind: "image", mime: asText(s.media_type, "image/png"), data: s.data, url: "" };
    }
    if (s.type === "url" && typeof s.url === "string") {
      return { kind: "image", mime: "image/png", data: "", url: s.url };
    }
  }
  if (block.type === "tool_use") {
    return {
      kind: "call",
      id: asText(block.id, ""),
      name: asText(block.name, ""),
      args: isObj(block.input) ? block.input : {},
    };
  }
  if (block.type === "tool_result") {
    return { kind: "result", id: asText(block.tool_use_id, ""), text: blocksText(block.content) };
  }
  return null;
}

function anthropicChoice(v: unknown): NormChoice | null {
  if (!isObj(v)) return null;
  if (v.type === "auto") return { kind: "auto" };
  if (v.type === "any") return { kind: "any" };
  if (v.type === "none") return { kind: "none" };
  if (v.type === "tool" && typeof v.name === "string") return { kind: "one", name: v.name };
  return null;
}

function normalizeAnthropic(body: any): NormReq {
  const req = emptyReq();
  req.system = blocksText(body.system);
  for (const m of asArray(body.messages)) {
    if (!isObj(m)) continue;
    if (m.role !== "user" && m.role !== "assistant") continue;
    const parts: NormPart[] = [];
    if (typeof m.content === "string") {
      if (m.content !== "") parts.push({ kind: "text", text: m.content });
    } else {
      for (const b of asArray(m.content)) {
        const p = anthropicPart(b);
        if (p !== null) parts.push(p);
      }
    }
    req.messages.push({ role: m.role, toolId: "", parts });
  }
  for (const t of asArray(body.tools)) {
    if (!isObj(t) || typeof t.name !== "string") continue;
    req.tools.push({
      name: t.name,
      description: asText(t.description, ""),
      parameters: isObj(t.input_schema) ? t.input_schema : { type: "object", properties: {} },
    });
  }
  req.choice = anthropicChoice(body.tool_choice);
  req.maxTokens = typeof body.max_tokens === "number" ? body.max_tokens : null;
  req.temperature = typeof body.temperature === "number" ? body.temperature : null;
  req.topP = typeof body.top_p === "number" ? body.top_p : null;
  req.stops = stopList(body.stop_sequences);
  req.streaming = body.stream === true;
  return req;
}

// ---------------------------------------------------------------------------
// The neutral shape out to each upstream wire
// ---------------------------------------------------------------------------

function renderOpenAI(req: NormReq, model: string, needsJson: boolean): Record<string, any> {
  const msgs: any[] = [];
  if (req.system !== "") msgs.push({ role: "system", content: req.system });
  for (const m of req.messages) {
    if (m.role === "tool") {
      const text = textParts(m.parts);
      const result = m.parts.find((p) => p.kind === "result");
      msgs.push({ role: "tool", tool_call_id: m.toolId !== "" ? m.toolId : result !== undefined ? result.id : "", content: result !== undefined ? result.text : text });
      continue;
    }
    if (m.role === "user") {
      const parts: any[] = [];
      for (const p of m.parts) {
        if (p.kind === "text") parts.push({ type: "text", text: p.text });
        else if (p.kind === "image") {
          parts.push(
            p.url !== ""
              ? { type: "image_url", image_url: { url: p.url } }
              : { type: "image_url", image_url: { url: `data:${p.mime};base64,${p.data}` } },
          );
        } else if (p.kind === "result") {
          msgs.push({ role: "tool", tool_call_id: p.id, content: p.text });
        }
      }
      if (parts.length > 0) msgs.push({ role: "user", content: parts });
      continue;
    }
    const text = textParts(m.parts);
    const calls = m.parts.filter((p): p is Extract<NormPart, { kind: "call" }> => p.kind === "call");
    const msg: Record<string, any> = {
      role: "assistant",
      content: text !== "" ? text : calls.length > 0 ? null : "",
    };
    if (calls.length > 0) {
      msg.tool_calls = calls.map((c) => ({
        id: c.id,
        type: "function",
        function: { name: c.name, arguments: JSON.stringify(c.args) },
      }));
    }
    msgs.push(msg);
  }
  const out: Record<string, any> = { model, messages: msgs };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (req.choice !== null) {
    if (req.choice.kind === "auto") out.tool_choice = "auto";
    else if (req.choice.kind === "any") out.tool_choice = "required";
    else if (req.choice.kind === "none") out.tool_choice = "none";
    else out.tool_choice = { type: "function", function: { name: req.choice.name } };
  }
  if (needsJson) out.response_format = { type: "json_object" };
  if (req.streaming) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function renderAnthropic(req: NormReq, model: string): Record<string, any> {
  const out: Record<string, any> = { model, messages: [] };
  if (req.system !== "") out.system = req.system;
  for (const m of req.messages) {
    if (m.role === "tool") {
      const result = m.parts.find((p) => p.kind === "result");
      out.messages.push({
        role: "user",
        content: [{ type: "tool_result", tool_use_id: m.toolId, content: result !== undefined ? result.text : "" }],
      });
      continue;
    }
    const blocks: any[] = [];
    for (const p of m.parts) {
      if (p.kind === "text") blocks.push({ type: "text", text: p.text });
      else if (p.kind === "image") {
        blocks.push(
          p.url !== ""
            ? { type: "image", source: { type: "url", url: p.url } }
            : { type: "image", source: { type: "base64", media_type: p.mime, data: p.data } },
        );
      } else if (p.kind === "call") {
        blocks.push({ type: "tool_use", id: p.id, name: p.name, input: p.args });
      } else if (p.kind === "result") {
        blocks.push({ type: "tool_result", tool_use_id: p.id, content: p.text });
      }
    }
    if (blocks.length > 0) out.messages.push({ role: m.role, content: blocks });
  }
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.choice !== null) {
    if (req.choice.kind === "auto") out.tool_choice = { type: "auto" };
    else if (req.choice.kind === "any") out.tool_choice = { type: "any" };
    else if (req.choice.kind === "none") out.tool_choice = { type: "none" };
    else out.tool_choice = { type: "tool", name: req.choice.name };
  }
  if (req.streaming) out.stream = true;
  return out;
}

function renderGemini(req: NormReq, needsJson: boolean): Record<string, any> {
  const contents: any[] = [];
  for (const m of req.messages) {
    if (m.role === "tool") {
      const result = m.parts.find((p) => p.kind === "result");
      contents.push({
        role: "user",
        parts: [
          {
            functionResponse: {
              name: m.toolId !== "" ? m.toolId : "tool",
              response: { result: result !== undefined ? result.text : "" },
            },
          },
        ],
      });
      continue;
    }
    const parts: any[] = [];
    for (const p of m.parts) {
      if (p.kind === "text") parts.push({ text: p.text });
      else if (p.kind === "image") {
        if (p.data !== "") parts.push({ inlineData: { mimeType: p.mime, data: p.data } });
      } else if (p.kind === "call") {
        parts.push({ functionCall: { name: p.name, args: p.args } });
      } else if (p.kind === "result") {
        parts.push({ functionResponse: { name: p.id, response: { result: p.text } } });
      }
    }
    if (parts.length > 0) contents.push({ role: m.role === "assistant" ? "model" : "user", parts });
  }
  const out: Record<string, any> = { contents };
  if (req.system !== "") out.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  const gc: Record<string, any> = {};
  if (req.maxTokens !== null) gc.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) gc.temperature = req.temperature;
  if (req.topP !== null) gc.topP = req.topP;
  if (req.stops.length > 0) gc.stopSequences = req.stops;
  if (needsJson) gc.responseMimeType = "application/json";
  // The model id travels in the request URL (:generateContent), never in
  // the Gemini body, so this renderer takes no model argument at all.
  if (Object.keys(gc).length > 0) out.generationConfig = gc;
  return out;
}

// ---------------------------------------------------------------------------
// Auth headers
// ---------------------------------------------------------------------------

export function authHeaders(auth: Auth): Record<string, string> {
  if (auth === null || auth === undefined) return {};
  if (typeof auth.key !== "string" || auth.key === "") return {};
  if (auth.scheme === "bearer") return { Authorization: `Bearer ${auth.key}` };
  if (auth.scheme === "x-api-key") return { "x-api-key": auth.key };
  if (auth.scheme === "x-goog-api-key") return { "x-goog-api-key": auth.key };
  return {};
}

// ---------------------------------------------------------------------------
// Upstream request construction
// ---------------------------------------------------------------------------

export function buildUpstream(
  clientWire: ClientWire,
  providerWire: Wire,
  baseUrl: string,
  model: string,
  clientBody: any,
  needsJson: boolean,
  auth: Auth,
): { url: string; headers: Record<string, string>; body: string } {
  const base = baseUrl.replace(/\/+$/, "");
  const streaming = isObj(clientBody) && clientBody.stream === true;
  const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(auth) };
  let url = "";
  let payload: Record<string, any>;
  if (providerWire === "openai") {
    url = `${base}/v1/chat/completions`;
    if (clientWire === "openai") {
      // Same wire: pass the body through with only the routed model id,
      // the usage flag for streams, and JSON mode when the route needs it.
      payload = { ...clientBody, model };
      if (needsJson && !isObj(payload.response_format)) payload.response_format = { type: "json_object" };
      if (streaming) payload.stream_options = { include_usage: true };
    } else {
      payload = renderOpenAI(normalizeAnthropic(clientBody), model, needsJson);
    }
  } else if (providerWire === "anthropic") {
    url = `${base}/v1/messages`;
    headers["anthropic-version"] = "2023-06-01";
    if (clientWire === "anthropic") {
      payload = { ...clientBody, model };
    } else {
      payload = renderAnthropic(normalizeOpenAI(clientBody), model);
    }
  } else {
    url = streaming
      ? `${base}/v1beta/models/${model}:streamGenerateContent?alt=sse`
      : `${base}/v1beta/models/${model}:generateContent`;
    const req = clientWire === "openai" ? normalizeOpenAI(clientBody) : normalizeAnthropic(clientBody);
    payload = renderGemini(req, needsJson);
  }
  if (streaming) headers.accept = "text/event-stream";
  return { url, headers, body: JSON.stringify(payload) };
}

// ---------------------------------------------------------------------------
// Non-streaming answers: pull the answer apart, then rebuild it per wire
// ---------------------------------------------------------------------------

type AnswerBits = {
  text: string;
  calls: { id: string; name: string; args: Record<string, any> }[];
  input: number;
  output: number;
  finish: string;
};

// Canonical finish values: "stop" | "length" | "tool_calls" | "content_filter".
function canonicalFinish(raw: unknown, table: Record<string, string>): string {
  if (typeof raw === "string" && table[raw] !== undefined) return table[raw];
  return "stop";
}

function bitsFromOpenAI(doc: Record<string, any>): AnswerBits {
  const choice = isObj(asArray(doc.choices)[0]) ? asArray(doc.choices)[0] : {};
  const msg = isObj(choice.message) ? choice.message : {};
  const calls: AnswerBits["calls"] = [];
  for (const c of asArray(msg.tool_calls)) {
    if (!isObj(c) || !isObj(c.function)) continue;
    calls.push({ id: asText(c.id, ""), name: asText(c.function.name, ""), args: argsObject(c.function.arguments) });
  }
  const usage = isObj(doc.usage) ? doc.usage : {};
  return {
    text: blocksText(msg.content),
    calls,
    input: asCount(usage.prompt_tokens, 0),
    output: asCount(usage.completion_tokens, 0),
    finish: canonicalFinish(choice.finish_reason, {
      stop: "stop",
      length: "length",
      tool_calls: "tool_calls",
      function_call: "tool_calls",
      content_filter: "content_filter",
    }),
  };
}

function bitsFromAnthropic(doc: Record<string, any>): AnswerBits {
  const calls: AnswerBits["calls"] = [];
  const texts: string[] = [];
  for (const b of asArray(doc.content)) {
    if (!isObj(b)) continue;
    if (b.type === "text" && typeof b.text === "string") texts.push(b.text);
    else if (b.type === "tool_use") {
      calls.push({ id: asText(b.id, ""), name: asText(b.name, ""), args: isObj(b.input) ? b.input : {} });
    }
  }
  const usage = isObj(doc.usage) ? doc.usage : {};
  return {
    text: texts.join("\n"),
    calls,
    input: asCount(usage.input_tokens, 0),
    output: asCount(usage.output_tokens, 0),
    finish: canonicalFinish(doc.stop_reason, {
      end_turn: "stop",
      max_tokens: "length",
      tool_use: "tool_calls",
      stop_sequence: "stop",
      refusal: "content_filter",
    }),
  };
}

function bitsFromGemini(doc: Record<string, any>): AnswerBits {
  const cand = isObj(asArray(doc.candidates)[0]) ? asArray(doc.candidates)[0] : {};
  const content = isObj(cand.content) ? cand.content : {};
  const texts: string[] = [];
  for (const p of asArray(content.parts)) {
    if (isObj(p) && typeof p.text === "string") texts.push(p.text);
  }
  const usage = isObj(doc.usageMetadata) ? doc.usageMetadata : {};
  return {
    text: texts.join(""),
    calls: [],
    input: asCount(usage.promptTokenCount, 0),
    output: asCount(usage.candidatesTokenCount, 0),
    finish: canonicalFinish(cand.finishReason, {
      STOP: "stop",
      MAX_TOKENS: "length",
      SAFETY: "content_filter",
      RECITATION: "content_filter",
    }),
  };
}

function upstreamErrorText(doc: Record<string, any>, status: number): string {
  if (isObj(doc.error) && typeof doc.error.message === "string") return doc.error.message;
  if (typeof doc.message === "string") return doc.message;
  return `the provider answered HTTP ${status}`;
}

function renderOpenAIAnswer(bits: AnswerBits, modelId: string, doc: Record<string, any>): Record<string, any> {
  const message: Record<string, any> = {
    role: "assistant",
    content: bits.text !== "" ? bits.text : bits.calls.length > 0 ? null : "",
  };
  if (bits.calls.length > 0) {
    message.tool_calls = bits.calls.map((c) => ({
      id: c.id,
      type: "function",
      function: { name: c.name, arguments: JSON.stringify(c.args) },
    }));
  }
  return {
    id: asText(doc.id, "chatcmpl-aggregate"),
    object: "chat.completion",
    created: OPENAI_CREATED,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: bits.finish }],
    usage: {
      prompt_tokens: bits.input,
      completion_tokens: bits.output,
      total_tokens: bits.input + bits.output,
    },
  };
}

function renderAnthropicAnswer(bits: AnswerBits, modelId: string, doc: Record<string, any>): Record<string, any> {
  const content: any[] = [];
  if (bits.text !== "") content.push({ type: "text", text: bits.text });
  for (const c of bits.calls) content.push({ type: "tool_use", id: c.id, name: c.name, input: c.args });
  if (content.length === 0) content.push({ type: "text", text: "" });
  const stop =
    bits.finish === "tool_calls"
      ? "tool_use"
      : bits.finish === "length"
        ? "max_tokens"
        : bits.finish === "content_filter"
          ? "refusal"
          : "end_turn";
  const rawId = asText(doc.id, "aggregate");
  return {
    id: rawId.startsWith("msg_") ? rawId : `msg_${rawId}`,
    type: "message",
    role: "assistant",
    model: modelId,
    content,
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: bits.input, output_tokens: bits.output },
  };
}

export function translateAnswer(
  clientWire: ClientWire,
  providerWire: Wire,
  status: number,
  upstreamText: string,
  modelId: string,
): { status: number; contentType: string; body: string } {
  let doc: unknown = null;
  try {
    doc = JSON.parse(upstreamText);
  } catch {
    doc = null;
  }
  if (!isObj(doc)) {
    const err =
      clientWire === "anthropic"
        ? anthropicError("api_error", "the provider answered with unparsable JSON")
        : openaiError("the provider answered with unparsable JSON", "api_error");
    return { status: 502, contentType: "application/json", body: JSON.stringify(err) };
  }
  if (status >= 400) {
    const msg = upstreamErrorText(doc, status);
    const err = clientWire === "anthropic" ? anthropicError("api_error", msg) : openaiError(msg, "api_error");
    return { status, contentType: "application/json", body: JSON.stringify(err) };
  }
  const bits =
    providerWire === "openai" ? bitsFromOpenAI(doc) : providerWire === "anthropic" ? bitsFromAnthropic(doc) : bitsFromGemini(doc);
  const body = clientWire === "openai" ? renderOpenAIAnswer(bits, modelId, doc) : renderAnthropicAnswer(bits, modelId, doc);
  return { status: 200, contentType: "application/json", body: JSON.stringify(body) };
}

// ---------------------------------------------------------------------------
// Streaming answers: parse the provider's SSE fully, re-emit the client's
// ---------------------------------------------------------------------------

// Every `data:` payload in a raw SSE transcript, parsed; unparsable or
// terminal frames are skipped here and handled by the callers' defaults.
function ssePayloads(raw: string): any[] {
  const docs: any[] = [];
  for (const frame of raw.replace(/\r\n/g, "\n").split("\n\n")) {
    for (const line of frame.split("\n")) {
      const t = line.trim();
      if (!t.startsWith("data:")) continue;
      const payload = t.slice(5).trim();
      if (payload === "" || payload === "[DONE]") continue;
      try {
        docs.push(JSON.parse(payload));
      } catch {
        // A malformed frame carries no answer text: skip it.
      }
    }
  }
  return docs;
}

type StreamBits = { text: string; input: number; output: number; finish: string };

function bitsFromStream(providerWire: Wire, docs: any[]): StreamBits {
  let text = "";
  let input = 0;
  let output = 0;
  let finish = "stop";
  for (const d of docs) {
    if (!isObj(d)) continue;
    if (providerWire === "openai") {
      if (isObj(d.usage)) {
        input = asCount(d.usage.prompt_tokens, input);
        output = asCount(d.usage.completion_tokens, output);
      }
      const choice = isObj(asArray(d.choices)[0]) ? asArray(d.choices)[0] : {};
      const delta = isObj(choice.delta) ? choice.delta : {};
      if (typeof delta.content === "string") text += delta.content;
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        finish = canonicalFinish(choice.finish_reason, {
          stop: "stop",
          length: "length",
          tool_calls: "tool_calls",
          function_call: "tool_calls",
          content_filter: "content_filter",
        });
      }
    } else if (providerWire === "anthropic") {
      if (d.type === "message_start" && isObj(d.message) && isObj(d.message.usage)) {
        input = asCount(d.message.usage.input_tokens, input);
      }
      if (d.type === "content_block_delta" && isObj(d.delta) && d.delta.type === "text_delta" && typeof d.delta.text === "string") {
        text += d.delta.text;
      }
      if (d.type === "message_delta") {
        if (isObj(d.delta) && typeof d.delta.stop_reason === "string") {
          finish = canonicalFinish(d.delta.stop_reason, {
            end_turn: "stop",
            max_tokens: "length",
            tool_use: "tool_calls",
            stop_sequence: "stop",
            refusal: "content_filter",
          });
        }
        if (isObj(d.usage)) output = asCount(d.usage.output_tokens, output);
      }
    } else {
      const cand = isObj(asArray(d.candidates)[0]) ? asArray(d.candidates)[0] : {};
      const content = isObj(cand.content) ? cand.content : {};
      for (const p of asArray(content.parts)) {
        if (isObj(p) && typeof p.text === "string") text += p.text;
      }
      if (typeof cand.finishReason === "string") {
        finish = canonicalFinish(cand.finishReason, {
          STOP: "stop",
          MAX_TOKENS: "length",
          SAFETY: "content_filter",
          RECITATION: "content_filter",
        });
      }
      if (isObj(d.usageMetadata)) {
        input = asCount(d.usageMetadata.promptTokenCount, input);
        output = asCount(d.usageMetadata.candidatesTokenCount, output);
      }
    }
  }
  return { text, input, output, finish };
}

function sliceText(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += STREAM_SLICE) out.push(text.slice(i, i + STREAM_SLICE));
  return out;
}

function dataLine(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

function namedLine(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function emitOpenAIStream(bits: StreamBits, modelId: string): string {
  const cid = "chatcmpl-stream";
  let out = "";
  out += dataLine({
    id: cid,
    object: "chat.completion.chunk",
    created: OPENAI_CREATED,
    model: modelId,
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  });
  for (const slice of sliceText(bits.text)) {
    out += dataLine({
      id: cid,
      object: "chat.completion.chunk",
      created: OPENAI_CREATED,
      model: modelId,
      choices: [{ index: 0, delta: { content: slice }, finish_reason: null }],
    });
  }
  out += dataLine({
    id: cid,
    object: "chat.completion.chunk",
    created: OPENAI_CREATED,
    model: modelId,
    choices: [{ index: 0, delta: {}, finish_reason: bits.finish }],
  });
  out += dataLine({
    id: cid,
    object: "chat.completion.chunk",
    created: OPENAI_CREATED,
    model: modelId,
    choices: [],
    usage: { prompt_tokens: bits.input, completion_tokens: bits.output, total_tokens: bits.input + bits.output },
  });
  out += "data: [DONE]\n\n";
  return out;
}

function emitAnthropicStream(bits: StreamBits, modelId: string): string {
  const stop =
    bits.finish === "tool_calls"
      ? "tool_use"
      : bits.finish === "length"
        ? "max_tokens"
        : bits.finish === "content_filter"
          ? "refusal"
          : "end_turn";
  let out = "";
  out += namedLine("message_start", {
    type: "message_start",
    message: {
      id: "msg_stream",
      type: "message",
      role: "assistant",
      model: modelId,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: bits.input, output_tokens: 0 },
    },
  });
  out += namedLine("content_block_start", {
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  });
  for (const slice of sliceText(bits.text)) {
    out += namedLine("content_block_delta", {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: slice },
    });
  }
  out += namedLine("content_block_stop", { type: "content_block_stop", index: 0 });
  out += namedLine("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: bits.output },
  });
  out += namedLine("message_stop", { type: "message_stop" });
  return out;
}

export function translateStream(
  clientWire: ClientWire,
  providerWire: Wire,
  upstreamSseText: string,
  modelId: string,
): string {
  const bits = bitsFromStream(providerWire, ssePayloads(upstreamSseText));
  return clientWire === "openai" ? emitOpenAIStream(bits, modelId) : emitAnthropicStream(bits, modelId);
}

// ---------------------------------------------------------------------------
// Model listings and error envelopes
// ---------------------------------------------------------------------------

export function openaiModelList(models: Array<{ id: string; provider: string }>): any {
  return {
    object: "list",
    data: models.map((m) => ({ id: m.id, object: "model", created: OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: Array<{ id: string; provider: string }>): any {
  return {
    data: models.map((m) => ({ type: "model", id: m.id, display_name: m.id, created_at: ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function openaiError(message: string, type?: string, code?: string): any {
  return { error: { message, type: type || "api_error", code: code || null } };
}

export function anthropicError(type: string, message: string): any {
  return { type: "error", error: { type, message } };
}
