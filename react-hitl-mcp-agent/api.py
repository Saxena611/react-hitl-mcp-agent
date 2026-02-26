"""
FastAPI HTTP Interface
======================
Exposes the ReAct + HITL + MCP agent over HTTP.

Endpoints:
  POST /session              → create a new session, returns session_id
  POST /chat                 → send a message, returns response + HITL state
  GET  /history/{session_id} → conversation history for a session
  GET  /health               → liveness check

Run:
    uvicorn api:app --reload --port 8000

Example cURL flow:

    # 1. Create session
    curl -X POST http://localhost:8000/session

    # 2. Chat (triggers multi-step ReAct + HITL pause)
    curl -X POST http://localhost:8000/chat \\
         -H "Content-Type: application/json" \\
         -d '{"session_id": "<id>", "message": "Refund order ORD-001"}'

    # 3. Confirm pending action
    curl -X POST http://localhost:8000/chat \\
         -H "Content-Type: application/json" \\
         -d '{"session_id": "<id>", "message": "yes"}'
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from agent import AgentSession

_session: AgentSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the MCP server subprocess and open the SQLite checkpointer on startup.
    Both are cleanly closed on shutdown via AgentSession.stop().

    SQLite path is read from the CHECKPOINT_DB_PATH env var (default:
    "agent_checkpoints.db"). Conversations survive server restarts — HITL
    pauses can be resumed even after a crash.
    """
    global _session
    _session = AgentSession()   # SQLite by default; pass in_memory=True for ephemeral
    await _session.start()
    yield
    await _session.stop()


app = FastAPI(
    title="ReAct + HITL + MCP Agent",
    description="Customer support agent with Human-in-the-Loop confirmation.",
    lifespan=lifespan,
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


class PendingAction(BaseModel):
    tool: str
    args: dict


class ChatResponse(BaseModel):
    session_id: str
    content: str
    interrupted: bool = False
    pending_action: PendingAction | None = None


class SessionResponse(BaseModel):
    session_id: str


class HistoryMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/session", response_model=SessionResponse)
async def create_session():
    """
    Create a new conversation session.
    Returns a session_id that must be passed with every /chat request.
    """
    return SessionResponse(session_id=str(uuid.uuid4()))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the agent.

    Normal flow:
      Response: { content: "...", interrupted: false }

    HITL flow (agent wants to call a dangerous tool):
      Response: { content: "I'm ready to refund $X. Proceed?",
                  interrupted: true,
                  pending_action: { tool: "process_refund", args: {...} } }
      → Send "yes" or "no" as the next message to confirm or cancel.
    """
    if not _session:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    result = await _session.chat(request.session_id, request.message)

    pending = None
    if result.get("pending_action"):
        pending = PendingAction(**result["pending_action"])

    return ChatResponse(
        session_id=request.session_id,
        content=result["content"],
        interrupted=result.get("interrupted", False),
        pending_action=pending,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """
    Retrieve conversation history for a session.
    Tool messages are excluded (internal implementation detail).
    """
    if not _session:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    state = await _session.get_state(session_id)
    messages = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append(HistoryMessage(role="user", content=msg.content))
        elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            messages.append(HistoryMessage(role="assistant", content=msg.content))

    return HistoryResponse(session_id=session_id, messages=messages)


@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": _session is not None}
