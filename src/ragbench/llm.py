"""Thin OpenAI-compatible client for DeepInfra with an on-disk cache, retries and a hard call cap.

The key is read from the environment (`DI_TOKEN`, then `DEEPINFRA_API_KEY`) and is never written anywhere.
A finished call is appended to a JSON Lines cache keyed by the full request, so re-running a job after a crash
(or re-running a finished one) costs nothing and returns the same text.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Optional

BASE_URL = "https://api.deepinfra.com/v1/openai"
KEY_ENV_NAMES = ("DI_TOKEN", "DEEPINFRA_API_KEY")

# USD per 1M tokens (input, output). DeepInfra pages, November 2026; the DeepSeek rate is a promotional one
# (list price 0.20 / 0.60). Used only for the running cost estimate.
PRICES = {
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": (0.098, 0.294),
    "zai-org/GLM-5.3-Flash": (0.15, 0.50),
}

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class BudgetExceeded(RuntimeError):
    """Raised before a live call that would exceed `max_calls`."""


def find_api_key() -> Optional[str]:
    for name in KEY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def request_key(model: str, messages: list, temperature: float, max_tokens: int, response_format: Optional[dict]) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
         "response_format": response_format},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in RETRY_STATUS
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError"}


def _openai_transport(api_key: str, base_url: str, timeout: float) -> Callable[..., dict]:
    from openai import OpenAI  # imported lazily so the package imports without the SDK

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    def call(model, messages, temperature, max_tokens, response_format):
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if response_format:
            kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        return {
            "text": resp.choices[0].message.content or "",
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "finish_reason": resp.choices[0].finish_reason,
        }

    return call


class LLMClient:
    def __init__(self, cache_path: Optional[str | Path] = None, max_calls: Optional[int] = None, max_workers: int = 16,
                 api_key: Optional[str] = None, base_url: str = BASE_URL, timeout: float = 120.0,
                 transport: Optional[Callable[..., dict]] = None, max_retries: int = 5,
                 sleep: Callable[[float], None] = time.sleep):
        self.max_calls, self.max_workers, self.max_retries, self._sleep = max_calls, max_workers, max_retries, sleep
        self._lock = threading.Lock()
        self.live_calls = 0
        self.cached_hits = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.cost_usd = 0.0
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        self._cache[row["key"]] = row
        if transport is None:
            key = api_key or find_api_key()
            if not key:
                raise RuntimeError("No API key: set DI_TOKEN (or DEEPINFRA_API_KEY) in the environment")
            transport = _openai_transport(key, base_url, timeout)
        self._transport = transport

    def complete(self, model: str, messages: list, temperature: float = 0.0, max_tokens: int = 512,
                 response_format: Optional[dict] = None) -> dict:
        key = request_key(model, messages, temperature, max_tokens, response_format)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self.cached_hits += 1
                return {**hit, "cached": True}
            if self.max_calls is not None and self.live_calls >= self.max_calls:
                raise BudgetExceeded(f"call cap of {self.max_calls} reached")
            self.live_calls += 1
        attempt = 0
        while True:
            try:
                out = self._transport(model, messages, temperature, max_tokens, response_format)
                break
            except Exception as exc:  # noqa: BLE001 - classified below
                attempt += 1
                if attempt > self.max_retries or not _is_retryable(exc):
                    raise
                self._sleep(min(30.0, 2.0 ** attempt))
        row = {"key": key, "model": model, **out}
        with self._lock:
            self._cache[key] = row
            self.usage["prompt_tokens"] += out.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += out.get("completion_tokens", 0)
            price = PRICES.get(model)
            if price:
                self.cost_usd += (out.get("prompt_tokens", 0) * price[0] + out.get("completion_tokens", 0) * price[1]) / 1e6
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {**row, "cached": False}

    def complete_many(self, model: str, batches: Iterable[list], **kwargs) -> list[dict]:
        """One completion per message list, in order, `max_workers` at a time."""
        batches = list(batches)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda m: self.complete(model, m, **kwargs), batches))

    def summary(self) -> str:
        return (f"{self.live_calls} live calls, {self.cached_hits} cache hits, "
                f"{self.usage['prompt_tokens']} in / {self.usage['completion_tokens']} out tokens, ~${self.cost_usd:.3f}")
