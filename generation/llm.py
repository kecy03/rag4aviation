from langchain_openai import ChatOpenAI

from config.settings import get_settings


def get_chat_model(temperature: float = 0) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )
