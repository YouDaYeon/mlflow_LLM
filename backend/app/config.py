from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://127.0.0.1:2522"
    # uv run으로 mlflow mcp 기동 시 initialize/list_tools가 5초를 넘기기 쉬움
    mcp_client_session_timeout_seconds: float = 60.0

    # chat
    chat_base_url: str = "http://127.0.0.1:9978"
    chat_api_key: str = "llama-local"
    chat_model: str = "Qwen3.6-35B-A3B"
    chat_temperature: float = 0.2

    # report
    llama_base_url: str = "http://127.0.0.1:9976/v1"
    llama_api_key: str = "llama-local"
    llama_model: str = "gemma-4-31B-it-UD-Q8_K_XL"
    llama_temperature: float = 0.2


settings = Settings()
