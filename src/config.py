import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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

        api_key = os.getenv("GROQ_API_KEY")
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

        api_key = os.getenv("HF_TOKEN")
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

        api_key = os.getenv("OPENAI_API_KEY")
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