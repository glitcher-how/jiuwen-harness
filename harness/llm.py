"""LLM access through openJiuwen's Model client, with strict token accounting.

Cost unit is the competition one: output tokens + input tokens / 10.
Reasoning tokens are already inside completion_tokens for OpenAI-compatible
APIs; if a provider reports them separately we add them on top.
"""
from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .log import log, traj, quiet_framework_logs

MODELS = {
    "fast": os.environ.get("FAST_MODEL", "qwen3.8-27b-noreason"),
    "think": os.environ.get("THINK_MODEL", "qwen3.8-27b"),
    "moe_fast": "qwen3.6-35b-a3b-noreason",
    "moe_think": "qwen3.6-35b-a3b",
}


class BudgetExhausted(Exception):
    pass


class ProxyDown(Exception):
    pass


@dataclass
class Ledger:
    """Shared spend counter for all workers."""
    budget: Optional[float] = None
    spent: float = 0.0
    calls: int = 0
    in_tok: int = 0
    out_tok: int = 0
    dead: bool = False  # proxy stopped answering
    by_task: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, inp: int, out: int, tag: str = "") -> float:
        cost = out + inp / 10.0
        with self.lock:
            key = tag.split("#")[0]
            self.by_task[key] = self.by_task.get(key, 0.0) + cost
            self.spent += cost
            self.calls += 1
            self.in_tok += inp
            self.out_tok += out
        return cost

    def remaining(self) -> float:
        if self.budget is None:
            return float("inf")
        return self.budget - self.spent


def _approx_tokens(text: str) -> int:
    # Qwen tokenizer: ~3.6 chars/token on English+code; err on the high side.
    return int(len(text) / 3.3) + 4


class LLM:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("LLM_API_KEY", "")
        self._models = {}
        self._fail_streak = 0
        self._v1_state = "untried"  # untried -> trying -> done
        self._base0 = self.base
        from openjiuwen.core.foundation.llm.model import Model
        from openjiuwen.core.foundation.llm.schema.config import (
            ModelClientConfig, ModelRequestConfig)
        self._Model = Model
        self._MCC = ModelClientConfig
        self._MRC = ModelRequestConfig

    def _model(self, name: str):
        m = self._models.get(name)
        if m is None:
            cc = self._MCC(client_provider="OpenAI", api_key=self.key or "none",
                           api_base=self.base, timeout=180.0,
                           stream_first_chunk_timeout=170.0, verify_ssl=True)
            rc = self._MRC(model=name, temperature=0.0)
            m = self._Model(cc, rc)
            self._models[name] = m
            quiet_framework_logs()
        return m

    def estimate(self, messages: List[dict], max_out: int) -> float:
        inp = sum(_approx_tokens(m["content"]) for m in messages)
        return inp / 10.0 + max_out

    async def chat(self, messages: List[dict], model: str = MODELS["fast"],
                   max_tokens: int = 512, tag: str = "", **extra) -> str:
        """One completion. Returns assistant text (without reasoning)."""
        if self.ledger.dead:
            raise ProxyDown()
        worst = self.estimate(messages, max_tokens)
        if self.ledger.remaining() < worst * 0.5:
            raise BudgetExhausted()
        kwargs = {}
        if model == MODELS["think"]:
            kwargs["reasoning_effort"] = extra.pop("reasoning_effort", "low")
        last_err = None
        for attempt in range(6):
            t0 = time.time()
            try:
                resp = await self._model(model).invoke(
                    messages, max_tokens=max_tokens, temperature=0.0, **kwargs)
            except Exception as e:  # network / proxy / rate limit
                last_err = e
                msg = str(e)
                log(f"[llm] {tag} attempt {attempt} error: {msg[:300]}")
                if re.search(r"\b404\b", msg):
                    if self._v1_state == "untried" and not self.base.rstrip("/").endswith("/v1"):
                        # LLM_BASE_URL may be given without /v1: try once with it
                        self._v1_state, self._base0 = "trying", self.base
                        self.base = self.base.rstrip("/") + "/v1"
                        self._models.clear()
                        log(f"[llm] 404, trying base URL {self.base}")
                        continue
                    if self._v1_state == "trying":
                        # still 404, so the address was not the problem: put it back
                        self._v1_state, self.base = "done", self._base0
                        self._models.clear()
                        log(f"[llm] still 404, base URL back to {self.base}")
                rate = re.search(r"\b429\b|rate.?limit|too many requests|inflight|concurren", msg, re.I)
                spent = re.search(r"\b(401|402|403)\b|budget|exhaust|insufficient|balance", msg, re.I)
                if spent and not rate:
                    self._fail_streak += 1
                    if self._fail_streak >= 3:  # three different calls refused: budget is gone
                        self.ledger.dead = True
                    raise BudgetExhausted(msg)
                await asyncio.sleep(min(30, (5 if rate else 2) * (attempt + 1)))
                continue
            self._fail_streak = 0
            if self._v1_state == "trying":
                self._v1_state = "done"  # the /v1 address works, keep it
            quiet_framework_logs()  # openJiuwen creates its loggers lazily on first use
            u = resp.usage_metadata
            inp = getattr(u, "input_tokens", 0) or 0
            out = getattr(u, "output_tokens", 0) or 0
            reasoning = resp.reasoning_content or ""
            if not inp:
                inp = sum(_approx_tokens(m["content"]) for m in messages)
            if not out:
                out = _approx_tokens((resp.content or "") + reasoning)
            cost = self.ledger.add(inp, out, tag)
            text = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            text = re.sub(r"(?s)<think>.*?</think>", "", text).strip()
            traj({"tag": tag, "model": model, "messages": messages, "reply": text,
                  "reasoning_chars": len(reasoning), "in": inp, "out": out,
                  "cost": cost, "sec": round(time.time() - t0, 2),
                  "finish": resp.finish_reason})
            return text
        # one call failing six times may be a glitch; three calls in a row means the
        # proxy has stopped answering (budget gone) and the run should wind down
        self._fail_streak += 1
        if self._fail_streak >= 3:
            self.ledger.dead = True
        raise ProxyDown(str(last_err))
