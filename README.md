# ReAct + HITL + MCP Agent

A production-pattern AI agent for e-commerce customer support, demonstrating four patterns that matter in real deployments:

| Pattern | What it does |
|---|---|
| **Guardrails** | Blocks off-topic and prompt-injection attempts before the agent sees them |
| **ReAct Loop** | Agent reasons → calls tools → observes results → repeats until done |
| **Human-in-the-Loop** | Pauses before irreversible actions (refunds, cancellations) and waits for confirmation |
| **MCP Tools** | Tools run in a separate MCP server, consumed via `langchain-mcp-adapters` |

Built with **LangGraph**, **DSPy**, **FastMCP**, and **FastAPI**.

## Quickstart

```bash
cd react-hitl-mcp-agent
pip install -r requirements.txt

# Add your API key
cp .env.example .env

# Run the interactive CLI demo
python demo.py
```

## Project Structure

```
react-hitl-mcp-agent/
├── agent/           # LangGraph graph, nodes, routing, state, guardrails
├── mcp_server.py    # MCP tool server (get_order, process_refund, etc.)
├── api.py           # FastAPI HTTP wrapper
├── demo.py          # Interactive CLI demo
└── tests/           # Pytest test suite
```

## Tech Stack

- [LangGraph](https://langchain-ai.github.io/langgraph/) — stateful agent graph with checkpointing
- [DSPy](https://dspy.ai) — structured LLM reasoning for guardrails & eligibility
- [FastMCP](https://github.com/jlowin/fastmcp) — MCP tool server
- [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) — MCP → LangChain tool bridge
- [FastAPI](https://fastapi.tiangolo.com) — HTTP API layer
