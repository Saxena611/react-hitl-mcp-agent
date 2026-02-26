"""
LLM Providers
=============
Builds the LangChain LLM and configures DSPy from environment variables.

Provider auto-detection priority: Groq → Azure OpenAI → OpenAI
Override with LLM_PROVIDER=groq|azure|openai to force a specific provider.

Keeping provider selection in one place means swapping LLMs never touches
graph, node, or session code.
"""
import logging
import os

import dspy

logger = logging.getLogger(__name__)


def detect_provider() -> str:
    """
    Return which LLM provider to use.

    Checks LLM_PROVIDER env var first (explicit override), then falls back
    to whichever API key is present in the environment.
    """
    forced = os.getenv("LLM_PROVIDER", "").lower()
    if forced in ("groq", "azure", "openai"):
        return forced
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return "azure"
    return "openai"


def build_llm():
    """
    Return a LangChain chat model for the detected provider.

    Groq   → ChatGroq  (llama-3.3-70b-versatile by default)
    Azure  → AzureChatOpenAI (temperature omitted — o-series rejects it)
    OpenAI → ChatOpenAI (gpt-4o-mini by default)
    """
    provider = detect_provider()
    logger.info("[LLM] Provider: %s", provider)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


def configure_dspy() -> None:
    """
    Point DSPy at the same provider as the main LLM.

    DSPy is used for structured sub-reasoning (guardrails, eligibility).
    Both DSPy and LangChain must use the same provider to avoid double billing.
    """
    provider = detect_provider()

    if provider == "groq":
        lm = dspy.LM(
            "groq/" + os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "azure":
        lm = dspy.LM(
            "azure/" + os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    else:
        lm = dspy.LM(
            "openai/" + os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    dspy.configure(lm=lm)
