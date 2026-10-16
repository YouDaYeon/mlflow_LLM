from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    # .env가 있으면 덮어씀, 없으면 fallback
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    mlflow_tracking_uri: str = "http://127.0.0.1:2522"
    # mlflow mcp subprocess 기동·initialize/list_tools가 5초를 넘기기 쉬움
    mcp_client_session_timeout_seconds: float = 60.0

    # chat
    chat_base_url: str = "http://127.0.0.1:9983"
    chat_api_key: str = "llama-local"
    chat_model: str = "Qwen3.6-35B-A3B"
    chat_temperature: float = 0.2
    chat_history_limit: int = 6

    # database
    # 로컬 uvicorn: compose postgres 호스트 포트(기본 5435). 컨테이너 backend: postgres:5432
    database_url: str = (
        "postgresql+psycopg://mlflow_llm:mlflow_llm@127.0.0.1:5435/mlflow_llm"
    )

    # report
    llama_base_url: str = "http://127.0.0.1:9980"
    llama_api_key: str = "llama-local"
    llama_model: str = "gemma-4-26B-A4B-it-UD-Q8_K_XL"
    llama_temperature: float = 0.2


settings = Settings()
