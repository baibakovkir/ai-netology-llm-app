import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_SYSTEM_PROMPT = "Ты — полезный ассистент. Отвечай кратко, точно и по существу."


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    timeout_seconds: float = 10.0
    max_attempts: int = 3
    retry_base_delay_seconds: float = 0.5
    cache_ttl_seconds: float = 600.0
    max_message_length: int = 1000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    log_level: str = "INFO"
    log_file: str = "logs/service.log"
    host: str = "0.0.0.0"
    port: int = 8000

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.retry_base_delay_seconds < 0:
            raise ValueError("retry_base_delay_seconds must not be negative")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        if self.max_message_length < 1:
            raise ValueError("max_message_length must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
            temperature=_float_env("LLM_TEMPERATURE", 0.2),
            timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 10.0),
            max_attempts=_int_env("LLM_MAX_ATTEMPTS", 3),
            retry_base_delay_seconds=_float_env("LLM_RETRY_BASE_DELAY_SECONDS", 0.5),
            cache_ttl_seconds=_float_env("CACHE_TTL_SECONDS", 600.0),
            max_message_length=_int_env("MAX_MESSAGE_LENGTH", 1000),
            system_prompt=os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_file=os.getenv("LOG_FILE", "logs/service.log"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=_int_env("PORT", 8000),
        )

    @property
    def log_path(self) -> Path:
        return Path(self.log_file)
