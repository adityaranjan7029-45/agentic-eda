import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def get_llm():
    """
    Returns configured LLM instance
    """

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", 0.2))

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature
    )