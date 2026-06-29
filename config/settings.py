import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DASHSCOPE_BASE_URL: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "deepseek-v4-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "5"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    COLLECTIONS: list[str] = ("aircraft_structure", "flight_ops", "regulations")

    def _validate_api_key(self, name: str, value: str) -> str:
        if not value or value.startswith("your-"):
            raise RuntimeError(f"Please set {name} in .env file.")
        return value

    def __post_init__(self):
        object.__setattr__(
            self,
            "DEEPSEEK_API_KEY",
            self._validate_api_key("DEEPSEEK_API_KEY", self.DEEPSEEK_API_KEY),
        )
        object.__setattr__(
            self,
            "DASHSCOPE_API_KEY",
            self._validate_api_key("DASHSCOPE_API_KEY", self.DASHSCOPE_API_KEY),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
