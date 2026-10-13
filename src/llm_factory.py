from langchain_groq import ChatGroq
from src.config import (
    GROQ_API_KEY,
    MODEL_PRIMARY,
    MODEL_SECONDARY,
    MODEL_TERTIARY,
)


def get_llm_with_fallbacks(temperature: float = 0.2):
    """
    Returns an LLM with a 3-model fallback chain.
    Order: GPT-OSS-120B -> Qwen3.8-27B -> Llama-3.3-70B
    """
    primary = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_PRIMARY,
        temperature=temperature,
    )
    secondary = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_SECONDARY,
        temperature=temperature,
    )
    tertiary = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_TERTIARY,
        temperature=temperature,
    )
    return primary.with_fallbacks([secondary, tertiary])