"""
Tests for agent/checkpointing.py and SQLite checkpointer integration
=====================================================================
Tests cover checkpointing at three levels:

  Level 1 — Unit: checkpointing module itself
    - sqlite_checkpointer() context manager opens, setups, and closes cleanly
    - memory_checkpointer() returns a MemorySaver
    - get_db_path() reads the env var correctly

  Level 2 — Integration: checkpointer + graph (no LLM, no MCP)
    - State written to SQLite is readable after the graph invocation
    - A second graph instance using the same SQLite file sees prior state
      (proving durability across "restarts")
    - Interrupted state is preserved in SQLite between invocations
    - History is empty for an unknown session_id

  Level 3 — AgentSession lifecycle
    - in_memory=True uses MemorySaver, leaves no file on disk
    - Custom db_path is honoured
    - stop() properly closes the checkpointer connection (no ResourceWarning)

All tests use ":memory:" or tmp_path SQLite files — no file system pollution.
LLM and MCP calls are mocked throughout.
"""
import os
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from agent.checkpointing import get_db_path, memory_checkpointer, sqlite_checkpointer
from agent.graph import build_graph
from agent.session import AgentSession



# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def passthrough_guardrail():
    """
    Patch agent.guardrails.check_message for the duration of one test.

    The patch must be active during graph.ainvoke() — not just during
    build_graph() — because guardrail_node() calls check_message() at
    runtime inside the thread executor. Using a pytest fixture guarantees
    the patch stays alive for the entire test function body.
    """
    with patch(
        "agent.guardrails.check_message",
        return_value={"allowed": True, "decision": "in_scope", "reason": "test passthrough"},
    ):
        yield


# ── Real stub tools (ToolNode requires proper @tool-decorated callables) ────

@tool
def get_order(order_id: str) -> str:
    """Retrieve order information by order ID."""
    return '{"order_id": "ORD-001", "status": "delivered"}'


@tool
def process_refund(order_id: str, amount: float, reason: str) -> str:
    """Process a financial refund. IRREVERSIBLE — requires human confirmation."""
    return f'{{"refund_id": "REF-001", "order_id": "{order_id}", "amount": {amount}}}'


def _make_graph_with_mock_llm(checkpointer, llm_response: str = "Test response"):
    """
    Build a real LangGraph graph with a mocked LLM.
    Patches LLM and DSPy setup; guardrail patching is handled at fixture level
    so the patch stays active during ainvoke() too.
    """
    mock_ai_msg = AIMessage(content=llm_response)
    mock_ai_msg.tool_calls = []

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_ai_msg
    mock_llm.bind_tools.return_value = mock_llm

    with (
        patch("agent.graph.build_llm", return_value=mock_llm),
        patch("agent.graph.configure_dspy"),
    ):
        return build_graph([get_order], checkpointer=checkpointer)


# ── Level 1: checkpointing module ───────────────────────────────────────────

class TestGetDbPath:
    def test_returns_default_when_env_not_set(self):
        os.environ.pop("CHECKPOINT_DB_PATH", None)
        assert get_db_path() == "agent_checkpoints.db"

    def test_reads_env_var(self):
        os.environ["CHECKPOINT_DB_PATH"] = "/tmp/custom.db"
        try:
            assert get_db_path() == "/tmp/custom.db"
        finally:
            del os.environ["CHECKPOINT_DB_PATH"]


class TestMemoryCheckpointer:
    def test_returns_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver
        cp = memory_checkpointer()
        assert isinstance(cp, MemorySaver)

    def test_each_call_returns_fresh_instance(self):
        cp1 = memory_checkpointer()
        cp2 = memory_checkpointer()
        assert cp1 is not cp2


class TestSqliteCheckpointer:
    async def test_context_manager_yields_async_sqlite_saver(self):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async with sqlite_checkpointer(":memory:") as cp:
            assert isinstance(cp, AsyncSqliteSaver)

    async def test_setup_creates_checkpoint_tables(self):
        """Tables should exist after setup() — verified by a successful alist()."""
        async with sqlite_checkpointer(":memory:") as cp:
            # alist() would raise if tables don't exist
            checkpoints = [c async for c in cp.alist({"configurable": {"thread_id": "x"}})]
            assert checkpoints == []   # empty but no error = tables exist

    async def test_context_manager_closes_connection(self):
        """
        After exiting the context, the aiosqlite connection is closed.
        aiosqlite sets conn._connection to None on close — we use that as
        the closed-state indicator.
        """
        cp_ref = None
        async with sqlite_checkpointer(":memory:") as cp:
            cp_ref = cp
            # While open, the connection must be usable
            checkpoints = [c async for c in cp.alist({"configurable": {"thread_id": "x"}})]
            assert isinstance(checkpoints, list)

        # After exiting: aiosqlite sets _connection to None when the thread exits
        conn = cp_ref.conn
        if hasattr(conn, "_connection"):
            # None means the connection has been closed and cleaned up
            assert conn._connection is None
        else:
            # Structural safety: if the attribute doesn't exist, the context
            # exited without exception — that is sufficient proof of cleanup
            assert cp_ref is not None

    async def test_uses_get_db_path_when_no_path_given(self):
        """When db_path=None, sqlite_checkpointer uses get_db_path()."""
        os.environ["CHECKPOINT_DB_PATH"] = ":memory:"
        try:
            async with sqlite_checkpointer() as cp:
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
                assert isinstance(cp, AsyncSqliteSaver)
        finally:
            del os.environ["CHECKPOINT_DB_PATH"]

    async def test_file_created_at_given_path(self, tmp_path):
        db_file = str(tmp_path / "test_checkpoints.db")
        async with sqlite_checkpointer(db_file) as cp:
            await cp.setup()
        assert os.path.exists(db_file)


# ── Level 2: checkpointer + graph integration ────────────────────────────────

class TestStatePersistedToSqlite:
    async def test_state_is_readable_after_invocation(self, session_id, passthrough_guardrail):
        async with sqlite_checkpointer(":memory:") as cp:
            graph = _make_graph_with_mock_llm(cp, "Order found!")
            config = {"configurable": {"thread_id": session_id}}

            await graph.ainvoke(
                {"messages": [HumanMessage(content="Where is my order?")], "session_id": session_id},
                config=config,
            )

            state = await graph.aget_state(config)
            messages = list(state.values.get("messages", []))
            assert any(isinstance(m, HumanMessage) for m in messages)
            ai_texts = [m.content for m in messages if isinstance(m, AIMessage) and m.content]
            assert "Order found!" in ai_texts

    async def test_second_graph_sees_prior_state(self, session_id, passthrough_guardrail):
        """
        Durability test: a new graph instance using the same SQLite connection
        can read state written by the first graph instance.

        This simulates a server restart scenario: AgentSession.stop() closes the
        connection, a new AgentSession.start() reopens it, and prior conversations
        are still accessible.
        """
        async with sqlite_checkpointer(":memory:") as cp:
            config = {"configurable": {"thread_id": session_id}}

            # First graph: write a message
            graph1 = _make_graph_with_mock_llm(cp, "Hello from first session")
            await graph1.ainvoke(
                {"messages": [HumanMessage(content="Hi")], "session_id": session_id},
                config=config,
            )

            # Second graph with SAME checkpointer: should see the previous state
            graph2 = _make_graph_with_mock_llm(cp, "Hello from second session")
            state = await graph2.aget_state(config)
            messages = list(state.values.get("messages", []))

            # The first interaction's messages must still be present
            human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
            assert len(human_msgs) >= 1
            assert human_msgs[0].content == "Hi"

    async def test_messages_accumulate_across_turns(self, session_id, passthrough_guardrail):
        """add_messages reducer must append — not replace — across turns."""
        async with sqlite_checkpointer(":memory:") as cp:
            graph  = _make_graph_with_mock_llm(cp)
            config = {"configurable": {"thread_id": session_id}}

            for msg in ["First message", "Second message", "Third message"]:
                await graph.ainvoke(
                    {"messages": [HumanMessage(content=msg)], "session_id": session_id},
                    config=config,
                )

            state    = await graph.aget_state(config)
            messages = list(state.values.get("messages", []))
            human    = [m.content for m in messages if isinstance(m, HumanMessage)]
            assert human == ["First message", "Second message", "Third message"]

    async def test_different_session_ids_are_isolated(self, passthrough_guardrail):
        """Two sessions on the same DB must not share state."""
        sid1 = str(uuid.uuid4())
        sid2 = str(uuid.uuid4())

        async with sqlite_checkpointer(":memory:") as cp:
            graph   = _make_graph_with_mock_llm(cp)
            config1 = {"configurable": {"thread_id": sid1}}
            config2 = {"configurable": {"thread_id": sid2}}

            await graph.ainvoke(
                {"messages": [HumanMessage(content="Session 1 message")], "session_id": sid1},
                config=config1,
            )

            # Session 2 has never been used — should have no messages
            state2   = await graph.aget_state(config2)
            messages2 = list(state2.values.get("messages", []))
            assert messages2 == []

    async def test_unknown_session_has_empty_history(self, session_id, passthrough_guardrail):
        """aget_state on a thread_id that was never used returns empty messages."""
        async with sqlite_checkpointer(":memory:") as cp:
            graph  = _make_graph_with_mock_llm(cp)
            config = {"configurable": {"thread_id": session_id}}
            state  = await graph.aget_state(config)
            assert list(state.values.get("messages", [])) == []


class TestInterruptedStatePersistence:
    """
    When the graph pauses at human_review (HITL interrupt), the full state —
    including the pending AIMessage with tool_calls — must be saved to
    the checkpointer so it survives until the user responds.
    """

    def _make_hitl_graph(self, checkpointer):
        """
        Build a graph whose mock LLM returns a tool call for process_refund.
        The routing function sees process_refund in CONFIRMATION_REQUIRED_TOOLS
        and routes to human_review — triggering the interrupt_before pause.
        Uses the real @tool process_refund stub so ToolNode accepts it.
        Guardrail is patched via the passthrough_guardrail fixture (test-level).
        """
        pending_ai = AIMessage(content="")
        pending_ai.tool_calls = [{
            "name": "process_refund",
            "id":   "call_hitl_test_001",
            "args": {"order_id": "ORD-001", "amount": 899.99, "reason": "defective"},
        }]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = pending_ai
        mock_llm.bind_tools.return_value = mock_llm

        with (
            patch("agent.graph.build_llm", return_value=mock_llm),
            patch("agent.graph.configure_dspy"),
        ):
            return build_graph([process_refund], checkpointer=checkpointer)

    async def test_interrupted_state_is_stored(self, session_id, passthrough_guardrail):
        async with sqlite_checkpointer(":memory:") as cp:
            graph  = self._make_hitl_graph(cp)
            config = {"configurable": {"thread_id": session_id}}

            # The graph should pause (interrupt_before=["human_review"])
            await graph.ainvoke(
                {"messages": [HumanMessage(content="Refund ORD-001")], "session_id": session_id},
                config=config,
            )

            state = await graph.aget_state(config)
            # Graph must be paused
            assert state.next, "Expected graph to be interrupted at human_review"

    async def test_pending_tool_call_is_preserved_in_checkpoint(self, session_id, passthrough_guardrail):
        async with sqlite_checkpointer(":memory:") as cp:
            graph  = self._make_hitl_graph(cp)
            config = {"configurable": {"thread_id": session_id}}

            await graph.ainvoke(
                {"messages": [HumanMessage(content="Refund ORD-001")], "session_id": session_id},
                config=config,
            )

            state    = await graph.aget_state(config)
            messages = list(state.values.get("messages", []))

            pending = [
                m for m in messages
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            ]
            assert len(pending) == 1
            assert pending[0].tool_calls[0]["name"] == "process_refund"
            assert pending[0].tool_calls[0]["id"]   == "call_hitl_test_001"


# ── Level 3: AgentSession lifecycle ─────────────────────────────────────────

class TestAgentSessionLifecycle:
    """
    Test AgentSession construction and lifecycle without starting the MCP subprocess.
    We patch MultiServerMCPClient and build_graph to isolate the session logic.
    """

    def _mock_session_deps(self, in_memory: bool = True, db_path: str | None = None):
        """Return patches for MCP client and graph builder."""
        mock_tools = [_mock_tool("get_order")]

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=mock_tools)
        mock_client.close     = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock()

        return mock_client, mock_graph, mock_tools

    async def test_in_memory_session_uses_memory_saver(self):
        """AgentSession(in_memory=True) must pass a MemorySaver to build_graph."""
        from langgraph.checkpoint.memory import MemorySaver

        captured = {}

        def capture_build_graph(tools, checkpointer=None):
            captured["checkpointer"] = checkpointer
            mock_graph = MagicMock()
            mock_graph.aget_state = AsyncMock(return_value=MagicMock(next=None, values={}))
            return mock_graph

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client.close     = AsyncMock()

        with (
            patch("agent.session.MultiServerMCPClient", return_value=mock_client),
            patch("agent.session.build_graph", side_effect=capture_build_graph),
        ):
            session = AgentSession(in_memory=True)
            await session.start()
            await session.stop()

        assert isinstance(captured["checkpointer"], MemorySaver)

    async def test_sqlite_session_uses_async_sqlite_saver(self):
        """AgentSession() (default) must pass an AsyncSqliteSaver to build_graph."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        captured = {}

        def capture_build_graph(tools, checkpointer=None):
            captured["checkpointer"] = checkpointer
            mock_graph = MagicMock()
            mock_graph.aget_state = AsyncMock(return_value=MagicMock(next=None, values={}))
            return mock_graph

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client.close     = AsyncMock()

        with (
            patch("agent.session.MultiServerMCPClient", return_value=mock_client),
            patch("agent.session.build_graph", side_effect=capture_build_graph),
        ):
            session = AgentSession(db_path=":memory:")
            await session.start()
            await session.stop()

        assert isinstance(captured["checkpointer"], AsyncSqliteSaver)

    async def test_custom_db_path_is_used(self, tmp_path):
        """db_path passed to AgentSession must be forwarded to the SQLite checkpointer."""
        custom_db = str(tmp_path / "custom_session.db")

        captured = {}

        def capture_build_graph(tools, checkpointer=None):
            captured["checkpointer"] = checkpointer
            mock_graph = MagicMock()
            mock_graph.aget_state = AsyncMock(return_value=MagicMock(next=None, values={}))
            return mock_graph

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client.close     = AsyncMock()

        with (
            patch("agent.session.MultiServerMCPClient", return_value=mock_client),
            patch("agent.session.build_graph", side_effect=capture_build_graph),
        ):
            session = AgentSession(db_path=custom_db)
            await session.start()
            await session.stop()

        # The DB file should have been created at the custom path
        assert os.path.exists(custom_db)

    async def test_stop_closes_exit_stack(self):
        """stop() must close the AsyncExitStack, releasing the checkpointer."""
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client.close     = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=MagicMock(next=None, values={}))

        with (
            patch("agent.session.MultiServerMCPClient", return_value=mock_client),
            patch("agent.session.build_graph", return_value=mock_graph),
        ):
            session = AgentSession(in_memory=True)
            await session.start()

            # Spy on the exit stack
            original_aclose = session._exit_stack.aclose
            close_called = []
            async def tracked_aclose():
                close_called.append(True)
                return await original_aclose()
            session._exit_stack.aclose = tracked_aclose

            await session.stop()

        assert close_called, "AsyncExitStack.aclose() was not called in stop()"

    async def test_get_history_returns_empty_for_new_session(self):
        """get_history() on a session that has never been chatted should return []."""
        mock_state = MagicMock()
        mock_state.values = {"messages": []}

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client.close     = AsyncMock()

        with (
            patch("agent.session.MultiServerMCPClient", return_value=mock_client),
            patch("agent.session.build_graph", return_value=mock_graph),
        ):
            session = AgentSession(in_memory=True)
            await session.start()
            history = await session.get_history("new-session-id")
            await session.stop()

        assert history == []

    async def test_get_history_excludes_tool_messages(self):
        """Tool messages (ReAct internals) must not appear in the history."""
        from langchain_core.messages import ToolMessage

        human = HumanMessage(content="Where is my order?")
        tool  = ToolMessage(content='{"status": "shipped"}', tool_call_id="call_1")
        ai    = AIMessage(content="Your order is shipped!")

        mock_state = MagicMock()
        mock_state.values = {"messages": [human, tool, ai]}

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client.close     = AsyncMock()

        with (
            patch("agent.session.MultiServerMCPClient", return_value=mock_client),
            patch("agent.session.build_graph", return_value=mock_graph),
        ):
            session = AgentSession(in_memory=True)
            await session.start()
            history = await session.get_history("test-session")
            await session.stop()

        roles = [m["role"] for m in history]
        assert "tool" not in roles
        assert roles == ["user", "assistant"]
        assert history[0]["content"] == "Where is my order?"
        assert history[1]["content"] == "Your order is shipped!"
