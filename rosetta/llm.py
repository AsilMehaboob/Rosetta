"""LLM access: JSON-schema output at temperature 0, every response cached.

The cache key covers the model, prompt version, prompt text and schema, so an
unchanged input never costs a second call and a rerun gives identical output.
The evaluation harness turns cache reads off, so each of its runs asks the model afresh.
"""

import hashlib
import json
import logging
import re
import time
from typing import Protocol

from rosetta.storage import Storage

LLM_CACHE = "llm_cache"
log = logging.getLogger(__name__)


class QuotaExhausted(RuntimeError):
    """The provider's daily limit is spent; further calls today would only fail."""


class JsonLLM(Protocol):
    model: str
    calls: int  # requests actually sent
    cached: int  # requests answered from the cache

    def generate_json(self, *, system: str, prompt: str, schema: dict, prompt_version: str) -> dict: ...


def cache_key(model: str, prompt_version: str, system: str, prompt: str, schema: dict) -> str:
    payload = json.dumps([model, prompt_version, system, prompt, schema], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class CachedLLM:
    """Shared caching around any backend's single `_send` call."""

    def __init__(self, model: str, storage: Storage, use_cache: bool = True):
        self.model = model
        self._storage = storage
        self._use_cache = use_cache  # False: always send, still store the newest response
        self.calls = 0
        self.cached = 0

    def _send(self, system: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        raise NotImplementedError

    def generate_json(self, *, system: str, prompt: str, schema: dict, prompt_version: str) -> dict:
        key = cache_key(self.model, prompt_version, system, prompt, schema)
        hit = self._storage.read(LLM_CACHE, key) if self._use_cache else None
        if hit is not None:
            self.cached += 1
            return hit["response"]
        response, usage = self._send(system, prompt, schema)
        self.calls += 1
        self._storage.write(
            LLM_CACHE,
            key,
            {"model": self.model, "prompt_version": prompt_version, "usage": usage, "response": response},
        )
        return response


class GeminiLLM(CachedLLM):
    def __init__(
        self,
        api_key: str,
        model: str,
        storage: Storage,
        max_retries: int = 4,
        retry_base_seconds: float = 10.0,
        request_timeout_seconds: int = 300,
        requests_per_minute: int = 15,
        use_cache: bool = True,
    ):
        if not api_key or not model:
            raise RuntimeError("Missing GEMINI_API_KEY or GEMINI_MODEL; see .env.example")
        from google import genai
        from google.genai import types

        super().__init__(model, storage, use_cache)
        # without a timeout a stalled request waits forever
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=request_timeout_seconds * 1000)
        )
        self._max_retries = max_retries
        self._retry_base = retry_base_seconds
        # stay under the per-minute limit instead of hitting it and paying a retry
        self._min_interval = 60.0 / requests_per_minute * 1.1
        self._last_sent = 0.0

    def _pace(self) -> None:
        wait = self._last_sent + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_sent = time.monotonic()

    def _send(self, system: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        import httpx
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            seed=0,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        for attempt in range(self._max_retries + 1):
            delay = self._retry_base * 2**attempt
            self._pace()
            try:
                response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
                data = json.loads(response.text)
            except errors.APIError as exc:
                text = str(exc)
                if exc.code == 429 and "PerDay" in text:
                    # every retry would fail and still count against the quota
                    raise QuotaExhausted(
                        f"Daily request limit reached for {self.model}; it resets at midnight Pacific time"
                    ) from exc
                # per-minute limits and overloads pass; anything else is a real error
                if exc.code not in (429, 500, 503) or attempt == self._max_retries:
                    raise
                suggested = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", text)
                if suggested:
                    delay = max(delay, float(suggested.group(1)) + 1)
                log.warning("%s: %s, retrying in %.0fs", self.model, exc.code, delay)
            except (json.JSONDecodeError, httpx.TimeoutException) as exc:
                if attempt == self._max_retries:
                    raise
                log.warning("%s: %s, retrying in %.0fs", self.model, type(exc).__name__, delay)
            else:
                meta = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(meta, "prompt_token_count", None),
                    "output_tokens": getattr(meta, "candidates_token_count", None),
                }
                return data, usage
            time.sleep(delay)
        raise RuntimeError("unreachable")
