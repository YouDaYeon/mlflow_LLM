from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://127.0.0.1:2522"

    llama_base_url: str = "http://127.0.0.1:9976/v1"
    llama_api_key: str = "llama-local"
    llama_model: str = "gemma-4-31B-it-UD-Q8_K_XL"
    llama_temperature: float = 0.2


settings = Settings()
