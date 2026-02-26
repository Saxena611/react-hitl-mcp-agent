"""
Checkpointing
=============
Manages the LangGraph checkpoint backend that makes HITL and multi-turn
conversations durable.

Two backends:

  SQLite (default)
  ─────────────────
  Uses AsyncSqliteSaver from langgraph.checkpoint.sqlite.aio (bundled with
  LangGraph — no extra install needed). Conversations survive process restarts.
  The DB file path is controlled by the CHECKPOINT_DB_PATH env var, defaulting
  to "agent_checkpoints.db" in the current working directory.

  Memory (in-process only)
  ────────────────────────
  Uses MemorySaver. Lost on process exit. Appropriate for tests and one-shot
  CLI sessions where durability is not required.

Usage pattern — SQLite:

    async with sqlite_checkpointer() as cp:
        graph = build_graph(tools, checkpointer=cp)
        # cp is kept alive for the duration of the context

Usage pattern — memory (tests / dev):

    cp = memory_checkpointer()
    graph = build_graph(tools, checkpointer=cp)

Design note:
  AsyncSqliteSaver.from_conn_string() is an async context manager that owns
  the aiosqlite connection. Entering it opens the connection; exiting closes it.
  AgentSession uses AsyncExitStack to enter this context on start() and exit it
  on stop(), tying the connection lifetime to the session lifetime.
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "agent_checkpoints.db"


def get_db_path() -> str:
    """
    Return the SQLite database file path.

    Resolution order:
      1. CHECKPOINT_DB_PATH environment variable
      2. DEFAULT_DB_PATH ("agent_checkpoints.db" in the cwd)
    """
    return os.getenv("CHECKPOINT_DB_PATH", DEFAULT_DB_PATH)


@asynccontextmanager
async def sqlite_checkpointer(
    db_path: str | None = None,
) -> AsyncIterator[AsyncSqliteSaver]:
    """
    Async context manager that opens an AsyncSqliteSaver and runs setup().

    setup() is idempotent — it creates the checkpoint tables if they don't
    exist yet, and is safe to call on every startup.

    Args:
        db_path: Path to the SQLite file. Defaults to get_db_path().
                 Pass ":memory:" for a fully in-process SQLite (useful in tests
                 when you want SQL semantics but no file on disk).

    Yields:
        A ready-to-use AsyncSqliteSaver instance.
    """
    path = db_path if db_path is not None else get_db_path()
    logger.info("[checkpointing] Opening SQLite checkpointer at: %s", path)

    async with AsyncSqliteSaver.from_conn_string(path) as checkpointer:
        await checkpointer.setup()
        logger.info("[checkpointing] SQLite checkpointer ready")
        yield checkpointer


def memory_checkpointer() -> MemorySaver:
    """
    Return an in-memory MemorySaver.

    State is lost when the process exits. Use for:
      - Unit tests that don't need persistence
      - CLI demos where session durability doesn't matter
    """
    return MemorySaver()
