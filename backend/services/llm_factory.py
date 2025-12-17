# services/llm_factory.py

from typing import Any
from core import get_logger
from config import get_settings
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq  # if you already use Groq

logger = get_logger(__name__)


def get_llm() -> Any:
    """
    Return the chat LLM used across the app.
    Mirror the same provider/model config as your RAG pipeline.
    """
    settings = get_settings()

    # Example: prefer Groq, fallback to local Ollama.
    if settings.llm_provider == "groq":
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.1,
        )

    # Default: Ollama
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )
