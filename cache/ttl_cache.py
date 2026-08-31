from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Callable, Dict, Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Small process-local TTL cache safe for concurrent threads."""

    def __init__(
        self,
        ttl_seconds: float,
        time_func: Callable[[], float] = monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._time = time_func
        self._entries: Dict[str, _Entry[T]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[T]:
        now = self._time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                return None
            return entry.value

    def set(self, key: str, value: T) -> None:
        now = self._time()
        with self._lock:
            self._remove_expired(now)
            self._entries[key] = _Entry(value=value, expires_at=now + self._ttl_seconds)

    def _remove_expired(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            del self._entries[key]

