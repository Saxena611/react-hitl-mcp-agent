"""
pytest configuration for the react-hitl-mcp-agent test suite.

Sets PYTHONPATH so tests can import from the project root.
Prevents DSPy and LangChain from making real LLM calls during unit tests.

asyncio_mode = "auto" (set in pytest.ini equivalent below) means all async
test functions are automatically collected as asyncio tests — no @pytest.mark.asyncio
needed on individual tests. This keeps the test code clean.
"""
import os
import sys

import pytest

# Ensure the project root is on sys.path so `import agent` and `import mcp_server` work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Stub out .env loading — tests use mocked values and should not need real keys
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("LLM_PROVIDER", "groq")
