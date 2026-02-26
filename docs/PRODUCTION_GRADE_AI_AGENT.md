# Designing a Production-Grade AI Agent: What Actually Changes

> This article is a companion to [Building a ReAct + HITL Agent with MCP, DSPy, and LangGraph](./MEDIUM_ARTICLE.md). That article walks through the architecture and code. This one asks: what happens when you move that prototype to production?

The architecture in the companion article is clean, working, and testable. But when you move it to production — real traffic, real data, real consequences — a handful of components need to be swapped or hardened. Here's the honest map of what changes and why.

---

## The Prototype Stack (Quick Recap)

The working prototype uses:

- **stdio MCP** — the tool server runs as a subprocess the agent forks at startup
- **SQLite checkpointing** — conversation state persists to a local file via `AsyncSqliteSaver`
- **Inline DSPy signatures** — guardrail and eligibility prompts live as Python docstrings
- **stdout logs** — observability is `print()` statements
- **No auth** — a single `AgentSession` is shared across all users
- **No reliability layer** — no retries, no timeouts, no stale session cleanup
- **Single process** — the agent, the tool server, and the API all run together

Everything below is about replacing each of these with something that holds under real load.

---

## Tools Layer: stdio MCP → FastMCP over HTTP

In the prototype, the MCP server runs as a subprocess over stdio. That's perfect for local development: one command starts everything. In production, you need the tool server to be a standalone, independently deployable service.

The switch is from stdio transport to HTTP/SSE transport. The tool server becomes a **FastMCP application** that registers all its tools as HTTP endpoints and exposes them over Server-Sent Events. The agent connects over a URL instead of spawning a subprocess.

This gives you everything that comes naturally with a proper HTTP service:

- **Independent deployment** — the tool server and the agent server deploy and scale separately. A refund spike doesn't affect order lookup latency.
- **Versioning** — tool schemas are versioned endpoints. You can run `v1` and `v2` simultaneously during a migration.
- **Authentication** — the MCP transport layer supports bearer tokens. Tools are gated, not open to any process that can fork the binary.
- **Health checks and load balancing** — the tool server sits behind a standard reverse proxy. Any orchestration layer (Kubernetes, ECS, Railway) treats it like any other service.
- **Independent testing** — the tool server is a standalone FastAPI application. Integration tests hit its HTTP endpoints directly, with no LangGraph, no graph, no agent involved.

The agent side sees no change — it connects to a URL instead of a command, and the tool objects it receives are identical LangChain `Tool` instances.

---

## Checkpointing: SQLite → PostgreSQL

SQLite is a single-file database. It works perfectly when you have one agent process. The moment you scale horizontally — two API pods behind a load balancer, a background worker resuming HITL confirmations — SQLite's file-locking model breaks.

The production swap is **`AsyncPostgresSaver`**, which uses the same interface as `AsyncSqliteSaver`. One import, one connection string, the rest of the code unchanged. PostgreSQL gives you:

- **Multi-instance state sharing** — any pod can resume any session. A HITL confirmation that arrives on pod B can resume a conversation started on pod A.
- **Crash recovery** — PostgreSQL's WAL means a pod restart doesn't lose in-flight state. The interrupted graph state is durable before `ainvoke()` even returns.
- **Concurrent reads** — multiple processes can read checkpoint state simultaneously. Your analytics pipeline can query conversation history without blocking the agent.
- **Schema migrations** — `AsyncPostgresSaver.setup()` creates and migrates tables idempotently. Add it to your startup sequence; it's safe to run on every deploy.
- **Connection pooling** — pair it with `asyncpg` and `pgbouncer` for high-throughput deployments. The checkpointer interface abstracts the pool away.

For distributed HITL specifically: when a user responds to a confirmation 10 minutes after requesting it, the response may arrive on a different pod than the one that saved the interrupted state. PostgreSQL makes that invisible.

---

## DSPy Prompts: Inline Strings → YAML Configs

In the prototype, the DSPy signature docstrings are Python strings embedded in the class definition. They define the domain boundary, the injection patterns, the eligibility policy. They work. But in production, three things make inline strings painful:

**Non-technical stakeholders can't edit them.** A product manager who wants to add a new in-scope topic — "product defects" — has to open a Python file, edit a docstring, and trigger a deployment. That's a friction-to-change problem that accumulates.

**Version control is noisy.** A policy change shows up as a diff in a Python file alongside unrelated code changes. Auditors reviewing "what changed in the refund eligibility policy last quarter" have to read git diffs through Python syntax.

**A/B testing is impossible.** You can't easily serve different prompt variants to different traffic segments when the prompt is baked into the class definition.

The production pattern is to move all DSPy signatures and their associated field descriptions, few-shot examples, and policy text into **YAML configuration files** that are loaded at startup. The signature classes become thin shells that read their content from config. This means:

- Policy updates are YAML file changes — reviewable, diffable, deployable independently of code.
- Non-engineers can propose prompt changes via a standard config PR without touching Python.
- You can load different YAML configs per environment (stricter guardrails in production, looser in staging for testing).
- DSPy's optimization artifacts (compiled `.json` files from `dspy.compile()`) are stored alongside the YAML, making the full prompt engineering history auditable.
- Prompt versions are tagged in your config management system, not your code repository.

The operational model is: **code changes deploy monthly, prompt changes deploy daily**.

---

## Observability: Print Statements → LangSmith or Langfuse

In production you need to see inside the agent — not just that it responded, but what it reasoned, which tools it called, how long each step took, and what it cost. Two frameworks dominate this space: **LangSmith** and **Langfuse**. They solve overlapping problems in different ways.

---

### LangSmith — Deep LangGraph Integration

LangSmith is built by the LangChain team, which means it understands LangGraph's internals natively. Zero instrumentation code is needed — set two environment variables and every `ainvoke()` call automatically produces a full trace.

Every graph execution becomes a hierarchical trace in LangSmith's UI. You see the full run decomposed into spans: `guardrail` node (DSPy Predict call + decision), `agent` node (LLM call with full prompt and response), each `tools` iteration (tool name, input args, output, latency), the `human_review` interrupt (timestamp of pause, timestamp of resume, user's yes/no response). For a 4-step refund workflow you get 4 nested spans, each with token counts, latency, and the exact content the LLM saw.

This is particularly valuable for this architecture because:

- **HITL gaps are visible.** The trace records the exact moment the graph paused and the exact moment it resumed. You can see that a user took 8 minutes to confirm a refund, and what the full pending state looked like during that gap.
- **ReAct iteration count is tracked automatically.** If your `tools → agent` loop runs 6 times instead of the expected 3, that shows up as 6 nested spans in a single trace. No custom metrics needed.
- **DSPy calls are captured.** The guardrail's `dspy.Predict` call — the message it received, the decision it returned, and the reasoning — appears as a span inside the `guardrail` node span.
- **Prompt versions are stored.** Every trace captures the exact system prompt, tool descriptions, and few-shot examples that were active at that moment. You can diff traces from before and after a prompt change.

LangSmith also surfaces aggregate metrics: p50/p95 latency by node, token cost by workflow type, error rate by tool. "What percentage of refund requests reach `human_review` vs. getting rejected at the guardrail?" is a dashboard query.

**The tradeoff:** LangSmith is a closed SaaS product. Your traces — including the full content of every LLM call — are sent to LangChain's infrastructure. For most B2C applications this is fine. For healthcare, finance, or any domain with strict data residency requirements, it may not be.

---

### Langfuse — Open Source, Self-Hostable

Langfuse solves the data residency problem. It is open source, actively maintained, and can be self-hosted on your own infrastructure — your traces never leave your network. It also has a managed cloud offering if you want the SaaS convenience without building the pipeline.

Langfuse does not integrate with LangGraph automatically. You instrument it manually by adding trace and span calls at the points that matter. The effort is higher, but the control is complete.

**What manual instrumentation gives you that automatic tracing doesn't:**

You choose exactly what constitutes a "trace" for your business. In this architecture, a natural business trace is the full conversation thread — from the user's first message through guardrail, through all ReAct iterations, through the HITL pause, through the confirmation, through the final refund. LangSmith's automatic tracing creates one trace per `ainvoke()` call. That means a 3-step refund is 3 separate traces linked by `thread_id`. Langfuse lets you wrap all of them in a single session-level trace with a shared session ID, which maps directly to how your team thinks about the conversation.

**Langfuse-specific features relevant to this architecture:**

- **Prompt management.** Langfuse has a first-class prompt registry where you version, store, and pull prompts at runtime. This is the infrastructure the YAML-based DSPy prompt approach fits into — your prompts live in Langfuse, the agent pulls the current version on startup, and every trace records which prompt version was active.
- **LLM cost tracking by model and session.** Langfuse aggregates token usage across all LLM calls in a session, including DSPy calls. You see the total cost of one refund conversation end to end.
- **Human annotation queues.** Relevant for HITL-heavy workflows: Langfuse can surface flagged traces to a human review queue where your team annotates agent decisions as correct or incorrect. These annotations feed back into DSPy optimization.
- **Evals.** Langfuse has an evaluation framework where you define test cases and run them against traces. "Did the guardrail correctly block this injection attempt?" can be a scored eval that runs on every deploy.

**The tradeoff:** Langfuse requires more setup. Self-hosting means you own the infrastructure — a PostgreSQL database, a web server, a background worker. Manual instrumentation means more code. The payoff is full data ownership and tighter integration with your product's domain model.

---

### Which One?

| Choose LangSmith if... | Choose Langfuse if... |
|---|---|
| You want observability in an hour, not a week | You operate in healthcare, finance, or legal |
| Data residency is not a constraint | Your traces cannot leave your infrastructure |
| You want automatic LangGraph tracing | You need session-level traces across multiple `ainvoke()` calls |
| You need HITL pause/resume visible immediately | You want annotation queues and evals built in from day one |

**What you should not do:** skip observability entirely and treat it as "something to add later." The hardest bugs in production agents are not errors — they are silent misbehaviors: the guardrail that passes an injection attempt because the wording was slightly different, the ReAct loop that runs 8 iterations on one query type and burns 4× the expected tokens, the HITL gap that averages 4 minutes but occasionally hits 40. None of these appear in your error logs. They only appear in traces.

**The three signals you must instrument, regardless of framework:**

| Signal | What it tells you |
|---|---|
| Node-level latency (guardrail, agent, tools) | Where time is actually being spent; which tool calls are slow |
| ReAct iteration count per session | Prompt efficiency; whether the LLM is reasoning in circles |
| HITL pause duration + resolution | User behaviour; whether confirmations are being abandoned |

---

## Authentication and Multi-Tenancy

The prototype has a single `AgentSession` shared across all users. Production needs isolation:

**Session pool keyed by tenant.** Each tenant (or user, depending on your isolation model) gets its own `thread_id` namespace in the checkpoint database. The `configurable.thread_id` that LangGraph uses for state isolation is your primary key — it must encode both the user and the conversation to prevent cross-tenant state leakage.

**API authentication.** The `/chat` endpoint needs a bearer token or session cookie. The `session_id` returned by `/session` must be cryptographically opaque — not a UUID that an attacker can enumerate. Map it to the authenticated user server-side.

**Tool-level authorization.** Not every user should be able to call `process_refund`. The MCP tool server should receive the caller's identity (passed through the agent as a context header) and enforce authorization at the tool level. A Gold-tier customer can request a full refund; a Standard-tier customer gets store credit. These are policy decisions that belong in the tool server, not the agent prompt.

---

## Reliability: Retries, Timeouts, and Circuit Breakers

LLM APIs fail. Tool servers have latency spikes. The production agent needs to handle both without surfacing errors to users:

**LLM retries with exponential backoff** — provider outages are short and infrequent. A retry strategy with jitter handles them transparently. LangChain's retry mechanism wraps the LLM call; configure max attempts and backoff parameters per environment.

**Tool call timeouts** — a slow database query inside an MCP tool should not hang the agent indefinitely. Set hard timeouts on tool node execution. If a tool times out, inject a `ToolMessage` with an error and let the agent reason about it — "I couldn't retrieve your order. Please try again in a moment."

**Circuit breakers for downstream services** — if the payment service is down, `process_refund` will fail repeatedly. A circuit breaker stops calling the failing service and returns a fast failure instead, giving the service time to recover without hammering it.

**Dead letter queue for HITL** — when a user never responds to a confirmation prompt, that interrupted graph state sits in the checkpoint database indefinitely. A background job should scan for stale interrupted sessions (older than N hours), inject a timeout cancellation message, and resume the graph cleanly.

---

## Deployment Architecture

The production topology separates concerns that are co-located in the prototype:

```
                     ┌─────────────────────────────────────┐
                     │       API Gateway / Load Balancer    │
                     └────────────┬───────────┬────────────┘
                                  │           │
                     ┌────────────▼──┐   ┌────▼────────────┐
                     │  Agent API    │   │   Agent API     │
                     │  (Pod 1)      │   │   (Pod 2)       │
                     └────────┬──────┘   └──────┬──────────┘
                              │                 │
                     ┌────────▼─────────────────▼──────────┐
                     │       PostgreSQL (checkpoints)        │
                     └──────────────────────────────────────┘
                              │
                     ┌────────▼──────────────────────────┐
                     │   FastMCP Tool Server              │
                     │   (independently scaled)           │
                     │   GET  /tools                      │
                     │   POST /tools/get_order            │
                     │   POST /tools/process_refund       │
                     └────────┬──────────────────────────┘
                              │
                     ┌────────▼──────────────────────────┐
                     │   Backend Services                 │
                     │   Order DB │ Payment API │ CRM     │
                     └───────────────────────────────────┘
```

Each component has a clear boundary and can be scaled, deployed, and monitored independently. The agent API pods are stateless — all state lives in PostgreSQL. The MCP tool server is the only component that talks to backend business systems. If you need to add a new integration, you add a tool to the MCP server, and every agent that connects to it gets that tool on the next startup.

---

## The Summary Table

| Component | Prototype | Production |
|---|---|---|
| MCP transport | stdio subprocess | FastMCP over HTTP/SSE |
| Checkpointing | SQLite file | PostgreSQL with connection pool |
| DSPy prompts | Inline docstrings | YAML config files, version-controlled |
| Observability | stdout logs | LangSmith or Langfuse + structured JSON logs + token cost metrics |
| Authentication | None | Bearer tokens, tenant-scoped thread IDs, tool-level authz |
| Reliability | None | Retries, timeouts, circuit breakers, stale session cleanup |
| Deployment | Single process | Stateless API pods + independent tool server + shared DB |
| Prompt updates | Code deploy | Config-only deploy, independently versioned |

The prototype and production implementations share the same graph topology, the same routing logic, the same HITL mechanism, and the same DSPy module structure. The production version is the prototype with each external dependency replaced by something that can handle real traffic.

That's what "production-ready architecture" actually means: not a different design, but the same design with every layer's infrastructure assumptions made explicit and replaced with something that holds under load.

---

*The code this article references is at [github.com/your-username/react-hitl-mcp-agent](https://github.com/your-username/react-hitl-mcp-agent).*

*The architecture was built as part of the [TalkToYourData](https://github.com/your-username/talktoyourdata) project, a production semantic layer management agent. This article generalizes the core patterns to a domain-neutral use case.*

---

**Tags**: `AI Agents` `LangGraph` `MCP` `DSPy` `Production` `LLM` `FastMCP` `PostgreSQL` `Observability` `LangSmith` `Langfuse`

---

### About the Author

*Building production AI agents for data and analytics workflows. Opinions are my own.*

*Found this useful? Follow for more production agent patterns.*
