from langchain_openai import OpenAIEmbeddings

from config.settings import get_settings


def get_embedding_function() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        chunk_size=settings.EMBEDDING_BATCH_SIZE,
        check_embedding_ctx_length=False,
    )
