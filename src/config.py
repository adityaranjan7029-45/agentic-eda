import os
from contextvars import ContextVar

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


_session_api_key: ContextVar[str | None] = ContextVar("_session_api_key", default=None)


def set_session_api_key(key: str | None) -> None:
    """Scope an API key to the CURRENT session/thread only.

    Call this once per Streamlit script run, before the pipeline starts.
    Never writes to os.environ -- see the note above for why that leaks
    keys between concurrent visitors."""
    _session_api_key.set((key or "").strip() or None)


def _resolve_key(*env_names: str) -> str | None:
    """Session override first, then the environment (.env / host secrets)."""
    session_key = _session_api_key.get()
    if session_key:
        return session_key
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    return None


def get_llm(model: str = None):
    """
    Returns a configured LLM instance for any node that needs one.

    model: optional override for the model name, so different agents can use
    different models through the same provider/API key (e.g. a bigger model
    for the Insight/Critic agents, a smaller/faster one for Synthesis).
    If omitted, falls back to the provider's default env var, same as before.

    Which backend gets used is controlled by the LLM_PROVIDER env var:

      LLM_PROVIDER=groq          (default) -- open-source models (Llama, Gemma,
                                    etc.) served by Groq's free API tier.
                                    Reliable with_structured_output support.
                                    Needs: GROQ_API_KEY

      LLM_PROVIDER=huggingface   -- open-source models routed through Hugging
                                    Face's OpenAI-compatible endpoint
                                    (router.huggingface.co). Uses ChatOpenAI
                                    under the hood instead of ChatHuggingFace,
                                    because ChatHuggingFace's
                                    with_structured_output is currently broken
                                    (langchain-ai/langchain#29569).
                                    Needs: HF_TOKEN
                                    Note: free HF accounts only get ~$0.10/mo
                                    in inference credit.

      LLM_PROVIDER=openai        -- original OpenAI path, kept for reference.
                                    Needs: OPENAI_API_KEY
    """

    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    temperature = float(os.getenv("TEMPERATURE", 0.2))

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = _resolve_key("GROQ_API_KEY")
        resolved_model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env (get a free key at console.groq.com)")

        return ChatGroq(
            api_key=api_key,
            model=resolved_model,
            temperature=temperature,
        )

    if provider == "huggingface":
        from langchain_openai import ChatOpenAI

        api_key = _resolve_key("HF_TOKEN")
        resolved_model = model or os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

        if not api_key:
            raise ValueError("HF_TOKEN not found in .env (get a free token at huggingface.co/settings/tokens)")

        # Hugging Face's OpenAI-compatible router. Using ChatOpenAI here (instead
        # of ChatHuggingFace) is what makes with_structured_output work reliably.
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://router.huggingface.co/v1",
            model=resolved_model,
            temperature=temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = _resolve_key("OPENAI_API_KEY")
        resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")

        return ChatOpenAI(
            api_key=api_key,
            model=resolved_model,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'groq', 'huggingface', or 'openai'."
    )