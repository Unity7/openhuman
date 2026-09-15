# Local Qwen Orchestration Roadmap

Status: active
Owner: OpenHuman host integration
Branch: `fix/local-qwen-history`
Baseline: `c49bfeecd5406d87c3727f744fcb988217c8dccd`
Current implementation head: `f6504d24248d85d837b13ae47c985022ab760257`

## Objective

Make `lmstudio:qwen38-openhuman` a reliable primary orchestrator without
replacing OpenHuman, TinyAgents, LM Studio, llama.cpp, the model, or the current
tool-call mode. Ordinary requests must expose only the capabilities they need,
permanent backend failures must stop immediately, and the context gauge must
show actual primary-model occupancy rather than cumulative turn traffic.

Qwen-Agent is a compatibility oracle and benchmark only. It is not a shipped
runtime dependency.

## Verified baseline

- LM Studio reports `qwen38-openhuman` as 27,320,697,856 parameters, Q5_K,
  `n_ctx=131072`, and `n_ctx_train=262144`.
- OpenHuman now resolves the alias to 131,072 before LM Studio's 8,192 fallback.
- The live local profile reports `supports_native_tools=false`; this work keeps
  the existing prompt-guided mode unchanged.
- A fresh orchestrator turn builds a 72,455-character system prompt.
- The first simple call reported 22,528 input tokens.
- The broad registry contains 238 tools; OpenHuman precomputation currently
  registers 28 for the orchestrator.
- TinyAgents emits a `ToolsFiltered` event showing the other 210 tools were
  withheld, but `OpenHumanToolExposureShadowMiddleware` does not enforce its
  own result on the live request.
- The image-generation example finished with 76,198 cumulative input tokens,
  71,654 cached-input tokens, and three parent iterations. These are turn
  totals, not proof that one request occupied 76,198 tokens.
- The UI's `lastTurnContextUsed` adds all primary model calls in a turn. It is
  therefore throughput, not current context occupancy, despite being presented
  as the latter.
- GMI image generation returned HTTP 400 `Insufficient balance`. This is a
  permanent account-level failure and cannot be repaired by retrying during the
  turn.

## Constraints

This feature must not change:

- model weights, GGUF, quantization, or chat template;
- llama.cpp or LM Studio launch configuration;
- the 131,072 runtime context;
- `http://127.0.0.1:1234/v1`;
- `lmstudio:qwen38-openhuman`;
- prompt-guided/native tool-mode selection;
- global LM Studio fallback values;
- approval, sandbox, path, credential, or autonomy policy;
- cloud behavior for unrelated providers.

No new orchestration framework is allowed unless the existing TinyAgents
surfaces demonstrably cannot meet an acceptance criterion. Python and
Qwen-Agent must not enter the shipped process.

## Architecture decision

OpenHuman remains the policy and execution host. TinyAgents remains the Rust
agent loop. LM Studio/llama.cpp remains the inference and Qwen-format adapter.

The production path becomes:

1. Resolve the agent's broad capability ceiling using existing security,
   profile, channel, MCP, and parent-agent restrictions.
2. Build a turn-stable exposure plan from the latest user request and current
   task state.
3. Intersect the exposure plan with the broad ceiling. Selection may only
   narrow; it can never grant a capability.
4. Register and advertise only the selected tools.
5. Enforce the same set in TinyAgents immediately before model and tool calls.
6. Load detailed skill instructions and packed tool schemas only after the
   model chooses `use_skill`.
7. Return bounded tool results and stop on terminal failures.

## Reference architecture sources

This roadmap borrows established behavior instead of introducing a new agent
framework:

- Qwen-Agent is the compatibility oracle for Qwen function schemas, canonical
  `{name, arguments}` calls, ordered tool results, and multi-step/parallel
  calls: <https://github.com/QwenLM/Qwen-Agent>.
- goose is the reference for a small provider-independent Rust execution loop
  that surfaces tool errors, revises context, and continues until a final model
  response: <https://github.com/aaif-goose/goose/blob/main/documentation/docs/goose-architecture/goose-architecture.md>.
- AnythingLLM is the reference for selecting a small relevant tool set before
  inference: <https://github.com/Mintplex-Labs/anything-llm-docs/blob/main/pages/agent/setup.mdx>.
- LangGraph is the reference for explicit state transitions, durable state,
  and idempotent re-execution: <https://langchain-ai.github.io/langgraph/reference/>.
- AutoGen is the reference for explicit success, failure, timeout, usage, and
  handoff termination conditions:
  <https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html>.

Only OpenHuman-specific policy is implemented locally: mapping an authorized
user intent and runtime state to OpenHuman tool capabilities, failure classes,
fallbacks, and completion conditions.

## Exposure policy

Selection is deterministic, local, explainable, and stable for the duration of
one parent turn so prompt-prefix caching remains effective.

### Always-visible recovery tools

- `ask_user_clarification`
- `use_skill`
- `tool_search` when that deferred-tool recovery surface is present

### State-required tools

- Goal tools only while the thread has an active goal.
- Plan/task-board tools only while a plan or task board is active or the user
  explicitly requests planning.
- Subagent lifecycle tools only while a child exists or the request is selected
  for delegation.
- Approval/recovery tools only when their corresponding state is active.

### Prompt-selected tools

Use the existing `tinyagents_harness::tool::rank_tools_by_prompt` implementation
against the already-authorized candidate set. Select at most eight ordinary
tools, in addition to state-required and recovery tools. Categories with a
strong explicit intent, such as web lookup, repository editing, memory, media,
or scheduling, receive their matching tools even when lexical ranking is thin.

A zero-match conversational request gets only recovery tools. A thin actionable
match gets recovery tools plus the best matching category, never the entire
catalogue. The previous "thin match means expose everything" behavior is not a
safe fallback for the primary orchestrator.

Every selected tool must be both advertised and callable. Every unselected tool
must be neither advertised nor callable. Unknown-tool recovery may explain what
is available but must not execute a hidden tool.

## Prompt policy

- Keep the current identity, security, workspace, and agent-role contracts.
- Record the character and estimated-token contribution of every prompt section
  before deleting or rewriting prose.
- Remove only duplicated instructions or instructions for capabilities absent
  from the turn's exposure plan.
- Do not render prose tool instructions when the same tool schema already
  carries the contract required by prompt-guided mode.
- Keep the initial no-tool request at or below 20,000 provider-reported input
  tokens. Keep an ordinary single-tool request at or below 24,000 on its first
  model call.
- Keep installed-skill descriptions sanitized and capped, and keep detailed
  packed-tool instructions behind the existing session-scoped `use_skill`
  lookup rather than injecting them into every model call.

## Context and usage telemetry

The system must expose two different concepts instead of conflating them:

- Turn traffic: the sum of input, cached input, output, and cost across all
  model calls and subagents. This remains the billing/activity breakdown.
- Context occupancy: input plus output tokens for the most recent primary-model
  call, or the maximum primary-model call occupancy in the turn if the provider
  cannot report the final call separately. Subagent tokens are excluded.

The context-window pill uses context occupancy. Its tooltip may show cumulative
turn/session traffic, but labels must say so. Occupancy must never be calculated
by summing repeated cached prefixes across model calls.

## Terminal failure policy

Classify account quota/balance exhaustion, invalid credentials, forbidden
provider configuration, and unsupported provider/model requests as terminal.
A terminal tool or delegated-inference failure:

1. records a stable failure class and bounded root cause;
2. halts that run after the first observed failure;
3. produces one actionable response;
4. does not suggest an automatic retry;
5. does not silently switch to another paid tool sharing the failed backend.

Timeouts, connection resets, rate limits with retry guidance, and 5xx responses
remain recoverable under the existing bounded retry policy. Independent tool
failures remain governed by the current no-progress ladder.

## Delivery phases and gates

### Phase 0: specification and baseline

- [x] Record constraints, architecture, acceptance criteria, and baseline.
- [x] Add reproducible prompt/tool fixtures and capture baseline logs.

Gate: baseline evidence distinguishes per-call occupancy from cumulative turn
traffic and records prompt sections and visible tool names.

### Phase 1: trustworthy telemetry

- [x] Add primary-call occupancy to the core turn-usage contract.
- [x] Preserve cumulative parent/subagent accounting separately.
- [x] Correct the context-window pill and labels.
- [x] Add Rust and frontend regression tests.

Gate: a three-call turn cannot make the gauge report the sum of all three
prompt prefixes; subagent usage cannot affect primary occupancy.

### Phase 2: authoritative exposure

- [x] Introduce a pure, unit-tested primary-agent exposure planner.
- [x] Reuse TinyAgents ranking and contextual selection.
- [x] Apply the plan before tool registration and on the live model request.
- [x] Install an execution-time allow guard for defense in depth.
- [x] Retire shadow-only comparison after parity tests pass.
- [x] Preserve stable tool order and prompt-cache layout within a turn.

Gate: all exposure/security tests pass, no hidden tool is callable, and the
four representative prompts expose only their expected capability families.

### Phase 3: prompt and skill payload

- [x] Add per-section prompt-size observability without logging prompt text.
- [x] Remove the duplicate frozen P-Format catalogue; TinyAgents remains the
  sole live prompt-guided/native tool-protocol owner.
- [x] Keep detailed skill/tool instructions behind the existing scoped
  `use_skill` lookup and retain the existing 240-character description cap.
- [x] Verify prompt-guided Qwen tool-call parsing remains green.

Gate: first-call token targets are met without changing tool mode or weakening
instructions.

### Phase 4: terminal failures

- [x] Reuse the existing terminal backend classifier rather than duplicate it.
- [x] Verify delegated insufficient-balance runs halt on the first occurrence.
- [x] Keep untrusted output from falsely triggering a terminal halt.
- [x] Verify 5xx remains recoverable under the bounded retry policy.

Gate: permanent failures make one attempt; recoverable failures retain bounded
headroom.

### Phase 5: evaluation and release

- [x] Run focused Rust and frontend tests.
- [x] Run formatting, clippy for changed targets, and repository checks.
- [ ] A/B the same fixtures through direct LM Studio and OpenHuman.
- [ ] Run live greeting, news, image-failure, and repository-tool scenarios.
- [x] Rebuild the Windows application, NSIS installer, and MSI with the MSVC
  toolchain.
- [x] Review the exact diff, commit, push, and record artifact hashes/paths.

Gate: all acceptance criteria pass and the source tree is clean.

### Phase 6: contract-driven Qwen orchestration (next build)

The Phase 5 live gate is reopened. Live testing after
`f6504d24248d85d837b13ae47c985022ab760257` found four contract violations:

- an image-retrieval request was routed to image generation instead of search;
- `spawn_async_subagent` instructed the model to call `wait_subagent`, although
  that tool was not callable by the parent;
- a later parent turn instructed Qwen to call `memory_recall`, although the
  live allowlist omitted it;
- Qwen emitted `<tool_call>{"arguments":{"url":...}}</tool_call>` without a
  tool name, which was rendered to the user instead of being executed or
  corrected.

#### Single orchestration contract

Add one immutable per-turn contract derived from the already-authorized tool
ceiling. It must be the sole input to:

1. prompt capability instructions;
2. advertised model tool schemas;
3. execution-time allowlist enforcement;
4. retry/fallback behavior; and
5. completion validation.

The contract contains:

- `intent_family`: conversation, web/news, image retrieval, image generation,
  repository, memory, scheduling, or delegation;
- `allowed_tools`: stable ordered names intersected with the security ceiling;
- `required_state_tools`: tools justified by live goal, memory, approval, or
  child-agent state;
- `fallbacks`: ordered alternatives that do not change modality or paid/local
  boundaries without user authorization;
- `terminal_failures`: typed failures that cannot improve during this run;
- `completion`: final text, explicit handoff, or one actionable terminal error;
- `limits`: maximum calls, identical retries, result bytes, and correction
  attempts.

Prompt text must be generated from this contract. A tool absent from
`allowed_tools` must not be named as an instruction, advertised to the model,
or accepted for execution. This invariant is checked immediately before every
model call and every tool call.

#### Intent and capability rules

| Intent | Allowed capability | Forbidden implicit fallback | Completion |
| --- | --- | --- | --- |
| `find/show/get an image from the internet` | web/image search and retrieval | image generation | retrieved result or clear retrieval failure |
| `generate/create/draw an image` | image generation | paid web/media service not already authorized | generated artifact or one actionable failure |
| current news/web lookup | search then fetch | workspace, memory, media, delegation | bounded sourced answer |
| memory request | memory only when module health and access allow it | pretending recall occurred | recalled result or clear unavailable state |
| repository work | workspace/read/edit/shell under existing policy | web/media unless explicitly requested | verified change or blocker |
| delegated work | spawn/continue and automatic result delivery | unavailable polling tool | delivered result or terminal child failure |

Follow-up turns such as `try a different site` inherit the active intent family
but not stale tool results, failed provider choices, or hidden capabilities.

#### Qwen protocol boundary

Prefer native structured provider `tool_calls`. For prompt-guided text calls,
apply Qwen-Agent-compatible parsing only after the complete stream is assembled.

- Accept canonical `{ "name": string, "arguments": object }` calls.
- Retain the current narrow bare-name repair for `qwen38-openhuman`.
- A missing name may be inferred only when exactly one allowed tool matches the
  intent family and validates the argument object against its schema.
- Otherwise return one bounded validation correction to the model. A second
  invalid call ends with one clear error.
- Never execute an unadvertised tool, invent missing arguments, or infer a
  destructive tool.
- Never persist or render raw `<tool_call>` markup as assistant text.
- Preserve call IDs, call order, reasoning metadata required by the provider,
  and one result for every accepted call.

Tool-call dialect parsing remains owned by TinyAgents. OpenHuman supplies the
turn contract, authorized schemas, and model-specific compatibility decision;
it must not grow a second general parser or agent loop.

#### Lifecycle and failure rules

- `spawn_async_subagent` must not instruct the parent to call a tool outside
  its contract. Use the existing automatic result-delivery path; expose a wait
  tool only if it becomes an explicit, state-required capability.
- Memory instructions are emitted only when `memory_recall` is callable. A
  failed or unloaded memory module produces one typed unavailable result.
- Terminal billing, quota, authentication, configuration, and unsupported
  capability failures propagate across parent and child boundaries and stop
  that failing route after one attempt.
- Retry only typed transient failures. At most one retry is allowed unless the
  tool supplies explicit retry guidance.
- A successful non-terminal tool result returns to the model for a final
  response. A turn may not finish with only a URL, raw call, or internal status.
- Web/RSS/HTML results receive a deterministic byte/item cap before optional
  model summarization. Summarizer failure falls back to bounded parsed content,
  never the complete raw document.
- Side-effecting calls carry a stable idempotency key for the turn and call so
  stream replay or resume cannot duplicate the action.

#### Ownership scope

Expected OpenHuman touch points are limited to:

- `crates/openhuman-core/src/agent/harness/primary_tool_exposure.rs` for intent
  families and authorized tool selection;
- `crates/openhuman-core/src/agent/harness/session/turn/graph.rs` for live
  state-required capabilities and the per-turn contract;
- `crates/openhuman-core/src/agent/tinyagents/middleware/tool_exposure.rs` for
  prompt/advertisement/execution parity;
- `crates/openhuman-core/src/agent/orchestration/tools/spawn_async_subagent.rs`
  for truthful async lifecycle instructions;
- `crates/openhuman-core/src/agent/tinyagents/middleware/loop_guards.rs` and
  `repeated_failure.rs` for typed parent/child failure propagation;
- the existing TinyAgents tool-call adapter seam for bounded Qwen correction;
- existing tool-result artifact/context code for deterministic result limits.

No frontend policy, second tool registry, Python sidecar, new agent framework,
or provider-specific copy of OpenHuman's security rules is in scope.

#### Delivery order and gates

1. **Contract parity:** create the per-turn contract and prove prompt,
   advertisement, and execution contain the same tool names.
2. **Intent routing:** separate image retrieval from generation and make
   follow-up intent inheritance explicit.
3. **Protocol recovery:** handle canonical, safely inferable, ambiguous,
   unadvertised, malformed, streamed, and parallel Qwen calls.
4. **Lifecycle:** remove unavailable async instructions, gate memory on health,
   propagate terminal child failures, and enforce final-response completion.
5. **Result control:** bound web/media results and provide a non-LLM fallback
   when summarization fails.
6. **Release:** run focused suites, live scenarios, MSVC packaging, exact diff
   review, clean-tree verification, and publish artifact paths/hashes.

Each gate must pass before the next behavior is integrated. Existing context,
Windows loading, approval, sandbox, and security regressions remain required.

#### Phase 6 acceptance scenarios

- `hey`: one final response, no ordinary task tool, no raw protocol markup.
- `show me the top three Google News headlines`: web-only route, at most three
  primary calls, bounded results, three sourced headlines, and final text.
- `get me a pic of a blonde Asian woman from the internet; don't generate`:
  retrieval tools only, no image agent or generation call, and a returned
  result rather than a bare URL/tool call.
- `try a different site`: preserves image-retrieval intent and excludes the
  previously failed source without switching modalities.
- `generate a portrait ...` with `Insufficient balance`: exactly one media
  attempt and one actionable final response.
- memory enabled and healthy: `memory_recall` is both advertised and callable;
  memory unavailable: it is neither instructed nor advertised and no
  `not on the allowlist` failure occurs.
- async delegation: no unavailable `wait_subagent` instruction or call; child
  completion is delivered once.
- missing-name Qwen call: execute only when schema/intent matching yields one
  safe candidate; otherwise perform one correction and finish clearly.
- malformed, partial, or unclosed streamed call: no execution and no raw tag
  appears in persisted history or the UI.

#### Phase 6 measurable release criteria

- zero prompt/advertisement/allowlist name mismatches in unit and live logs;
- zero raw `<tool_call>` leaks across the fixture suite;
- zero retries after a terminal failure;
- no more than one correction attempt for an invalid model call;
- one execution per accepted call ID, including resume/replay tests;
- greeting: one primary call; news and image retrieval: at most three primary
  calls; media quota failure: one media call;
- no user-message trim in the representative scenarios;
- the existing 20,000/24,000 first-call input targets remain satisfied;
- Windows executable, NSIS, and MSI build through the repository's MSVC path;
- live logs show model context `131072`, selected intent/tool names, typed stop
  reason, final-response completion, and no false 8K budgeting.

Gate: every Phase 6 scenario passes against deterministic fixtures and the live
`lmstudio:qwen38-openhuman` route, all relevant existing tests remain green,
the exact diff is reviewed, and the source tree is clean.

## Representative acceptance scenarios

### Conversation

Prompt: `hey`

- No task tool is called.
- At most the recovery tools are exposed.
- One model call completes the turn.
- First-call input is at most 20,000 tokens.

### News

Prompt: `show me the top three Google News headlines`

- Only web/recovery tools are exposed.
- No workspace, goal, subagent-management, media, or repository tool appears.
- The task completes in at most three primary model calls and one successful
  fetch/search path.
- No user message is trimmed.

### Media quota failure

Prompt: `generate a portrait of a blonde K-pop-inspired woman`

- Only media/recovery/delegation capabilities needed for the route are exposed.
- HTTP 400 `Insufficient balance` results in one generation attempt and one
  actionable final response.
- The run does not retry GMI or switch to another paid TinyHumans backend.

### Repository edit

Prompt: `change the README heading and run its focused test`

- Repository read/edit/shell tools are exposed.
- Media, web, memory-maintenance, scheduling, and unrelated integrations are
  absent.
- Existing approval and action-directory restrictions remain effective.

## Required automated coverage

- Exposure planning: zero-match chat, explicit category match, ambiguous match,
  state-required tools, stable ordering, top-K, parent ceiling, denylist, empty
  allowlist, and hidden-tool execution rejection.
- Prompt construction: unavailable capability prose omitted; required security
  and recovery contracts retained; deterministic section-size snapshot.
- Usage: last primary call versus cumulative calls, cached tokens, subagent
  subtraction, unknown context window, and resumed thread.
- Failures: terminal balance/quota/auth/config, recoverable timeout/rate-limit/
  5xx, parallel results, and delegated terminal inference.
- Existing context-resolution regression for `qwen38-openhuman` remains green.

## Rollout and rollback

Land telemetry with the behavior change. The planner only narrows the existing
authorized ceiling and logs counts and names, never prompt text or tool
arguments.

Rollback is reverting authoritative relevance narrowing while retaining the
correct 131,072 context resolver and Windows loader fix; the security ceiling
continues to apply independently.

## Explicit non-goals

- Replacing TinyAgents with Qwen-Agent, LangGraph, PydanticAI, or Smolagents.
- Changing the model, endpoint, context length, chat template, or tool mode.
- Adding local image generation.
- Fixing unrelated memory persistence, cloud billing, UI, or module issues.
- Claiming that prompt caching reduces context occupancy; it reduces repeated
  computation/cost, not the number of tokens the model attends to.

## Completion evidence

Completion requires the final report to include:

- exact source diff and commits;
- baseline versus final prompt characters, visible tools, per-call occupancy,
  cumulative turn traffic, model calls, and wall time for each scenario;
- focused and relevant suite results;
- MSVC build output, executable/installer paths, sizes, and SHA-256 hashes;
- live log lines proving 131,072 context, selected tool counts, no false trim,
  and terminal-failure behavior;
- remote branch verification and a clean worktree.
