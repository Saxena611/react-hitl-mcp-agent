"""
Agent Session
=============
High-level interface for one multi-turn conversation.

Responsibilities:
  - Start/stop the MCP server subprocess (stdio transport)
  - Manage the checkpointer lifecycle via AsyncExitStack
  - Build and hold the compiled LangGraph graph
  - Route incoming messages to the right graph invocation:
      · Interrupted state → handle yes/no confirmation
      · Normal message    → graph.ainvoke with new HumanMessage

Checkpointer modes:
  SQLite (default, durable)
    AgentSession(db_path="agent_checkpoints.db")
    Conversations survive process restarts. HITL pauses survive server crashes.

  In-memory (ephemeral)
    AgentSession(in_memory=True)
    Use for CLI demos and unit tests that don't need persistence.

The session is async because MCP tool calls are async — the graph must be
invoked with ainvoke() / aget_state() rather than the sync equivalents.
"""
import logging
import sys
from contextlib import AsyncExitStack

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from .checkpointing import get_db_path, memory_checkpointer, sqlite_checkpointer
from .graph import build_graph

logger = logging.getLogger(__name__)


# ── Confirmation detection ──────────────────────────────────────────────────

_YES_WORDS = frozenset({"yes", "confirm", "proceed", "ok", "sure", "go", "yep", "do it"})
_NO_WORDS  = frozenset({"no", "cancel", "stop", "abort", "nope", "don't"})


def is_confirmation(message: str) -> bool | None:
    """
    Parse a yes/no reply from the user.

    Returns:
        True  — confirmed
        False — rejected
        None  — ambiguous (caller decides how to handle)
    """
    lower = message.lower().strip()
    if any(w in lower for w in _YES_WORDS):
        return True
    if any(w in lower for w in _NO_WORDS):
        return False
    return None


# ── Cancellation helpers ────────────────────────────────────────────────────

def build_cancel_messages(state) -> list:
    """
    Build a ToolMessage cancellation for every pending tool_call in state.

    The OpenAI (and compatible) API requires a ToolMessage for every
    AIMessage tool_call — even cancelled ones. Without this, the next
    LLM invocation fails with an API error.
    """
    cancel_msgs = []
    for msg in reversed(list(state.values.get("messages", []))):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                cancel_msgs.append(ToolMessage(
                    content="Action cancelled by user.",
                    tool_call_id=tc["id"],
                ))
            break
    return cancel_msgs


async def handle_cancellation(graph, config: dict, state) -> dict:
    """
    Cancel a pending HITL action by satisfying its tool call requirements,
    then resume the graph so the agent can acknowledge the cancellation.
    """
    cancel_msgs = build_cancel_messages(state)
    if cancel_msgs:
        return await graph.ainvoke({"messages": cancel_msgs}, config=config)
    return await graph.ainvoke(None, config=config)


# ── Pending action description ──────────────────────────────────────────────

def describe_pending(pending: dict) -> str:
    """Format a human-readable confirmation prompt for a pending tool call."""
    tool = pending["tool"]
    args = pending.get("args", {})

    if tool == "process_refund":
        amount     = args.get("amount")
        amount_str = f"${float(amount):.2f}" if amount is not None else "an amount"
        order_id   = args.get("order_id", "unknown")
        return (
            f"I'm ready to process a refund of **{amount_str}** "
            f"for order {order_id}.\n\n"
            "This is a financial transaction that cannot be undone.\n\n"
            "Shall I proceed? (yes / no)"
        )

    if tool == "cancel_order":
        order_id = args.get("order_id", "unknown")
        return (
            f"I'm ready to cancel order **{order_id}**.\n\n"
            "Once cancelled, this cannot be undone.\n\n"
            "Shall I proceed? (yes / no)"
        )

    return f"I'm ready to perform: `{tool}`. Shall I proceed? (yes / no)"


# ── AgentSession ────────────────────────────────────────────────────────────

class AgentSession:
    """
    Manages one agent session lifecycle.

    Args:
        db_path:   Path to the SQLite checkpoint database file.
                   Defaults to CHECKPOINT_DB_PATH env var or "agent_checkpoints.db".
        in_memory: If True, use MemorySaver instead of SQLite.
                   Conversations are lost when the process exits.
                   Suitable for tests and single-run CLI demos.

    Usage:
        session = AgentSession()                      # SQLite, durable
        session = AgentSession(in_memory=True)        # MemorySaver, ephemeral
        session = AgentSession(db_path="/tmp/test.db")# custom path

        await session.start()
        response = await session.chat(session_id, "Refund order ORD-001")
        await session.stop()
    """

    def __init__(self, db_path: str | None = None, in_memory: bool = False):
        self._db_path   = db_path
        self._in_memory = in_memory
        self._client: MultiServerMCPClient | None = None
        self._graph     = None
        # AsyncExitStack ties the checkpointer's connection lifetime to the session.
        # Entering the sqlite_checkpointer context on start() and exiting on stop()
        # ensures the aiosqlite connection is properly opened and closed.
        self._exit_stack = AsyncExitStack()

    async def start(self) -> None:
        """
        Connect to the MCP server, open the checkpointer, and build the graph.

        SQLite path: opens the aiosqlite connection and runs setup() (creates
        tables if needed). The connection stays open until stop() is called.
        """
        # ── Checkpointer ──────────────────────────────────────────────────
        if self._in_memory:
            checkpointer = memory_checkpointer()
            logger.info("[session] Using in-memory checkpointer (ephemeral)")
        else:
            path = self._db_path or get_db_path()
            checkpointer = await self._exit_stack.enter_async_context(
                sqlite_checkpointer(path)
            )
            logger.info("[session] Using SQLite checkpointer at: %s", path)

        # ── MCP client ────────────────────────────────────────────────────
        self._client = MultiServerMCPClient({
            "shopease": {
                "command": sys.executable,
                "args":    ["mcp_server.py"],
                "transport": "stdio",
            }
        })
        tools = await self._client.get_tools()

        # ── Graph ─────────────────────────────────────────────────────────
        self._graph = build_graph(tools, checkpointer=checkpointer)

        logger.info(
            "[session] Ready. %d tools: %s",
            len(tools), [t.name for t in tools],
        )
        print(f"[Agent] Connected. {len(tools)} tools: {[t.name for t in tools]}")

    async def stop(self) -> None:
        """
        Shut down MCP subprocess and close the checkpointer connection.

        AsyncExitStack.aclose() exits all contexts in reverse order — the
        SQLite connection is closed cleanly even if an exception occurred.
        """
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        await self._exit_stack.aclose()

    # ── State helpers ───────────────────────────────────────────────────────

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    async def get_state(self, session_id: str):
        return await self._graph.aget_state(self._config(session_id))

    async def is_interrupted(self, session_id: str) -> bool:
        return bool((await self.get_state(session_id)).next)

    async def get_pending_action(self, session_id: str) -> dict | None:
        state = await self.get_state(session_id)
        for msg in reversed(list(state.values.get("messages", []))):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                tc = msg.tool_calls[0]
                return {"tool": tc["name"], "args": tc.get("args", {})}
        return None

    async def get_history(self, session_id: str) -> list[dict]:
        """
        Return the conversation history for a session as a list of
        { role: "user"|"assistant", content: str } dicts.

        Tool messages (internal to the ReAct loop) are excluded — they are
        implementation details, not part of the user-facing conversation.
        """
        state = await self.get_state(session_id)
        history = []
        for msg in state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif (
                isinstance(msg, AIMessage)
                and msg.content
                and not getattr(msg, "tool_calls", None)
            ):
                history.append({"role": "assistant", "content": msg.content})
        return history

    # ── Message extraction ──────────────────────────────────────────────────

    @staticmethod
    def _last_ai_text(messages) -> str:
        """Return the last AIMessage that has text content (not a tool call)."""
        for msg in reversed(list(messages)):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                return msg.content
        return "..."

    # ── Main chat interface ─────────────────────────────────────────────────

    async def chat(self, session_id: str, message: str) -> dict:
        """
        Send a message to the agent and return the response.

        Returns a dict with:
            content        — the agent's reply text
            interrupted    — True if the graph paused for HITL confirmation
            pending_action — { tool, args } if interrupted, else None

        Graph resume semantics:
            - Confirmed: ainvoke(None, config) resumes from saved checkpoint
            - Rejected:  ainvoke({messages: cancel_msgs}, config) satisfies the
                         pending tool calls, then resumes so the agent can
                         acknowledge the cancellation politely
        """
        config        = self._config(session_id)
        current_state = await self.get_state(session_id)

        # Handle interrupted state (graph is waiting for yes/no)
        if current_state.next:
            confirmed = is_confirmation(message)

            if confirmed is True:
                result = await self._graph.ainvoke(None, config=config)
                return {"content": self._last_ai_text(result["messages"]), "interrupted": False}

            if confirmed is False:
                result = await handle_cancellation(self._graph, config, current_state)
                return {"content": self._last_ai_text(result["messages"]), "interrupted": False}

            # Ambiguous reply — fall through to normal message flow

        # Normal message
        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=message)], "session_id": session_id},
            config=config,
        )

        # Check whether a new interrupt fired
        new_state = await self.get_state(session_id)
        if new_state.next:
            pending = await self.get_pending_action(session_id)
            desc    = describe_pending(pending) if pending else "Would you like me to proceed?"
            return {"content": desc, "interrupted": True, "pending_action": pending}

        return {"content": self._last_ai_text(result["messages"]), "interrupted": False}
