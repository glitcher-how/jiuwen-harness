"""The model side of the harness: openJiuwen ReAct agents, their tools, and token accounting.

Every model call in the harness is made by an openJiuwen ``ReActAgent``:

* the agent gets a system prompt, the task context prepared by the harness, and at most
  one tool (``run_python``, ``apply_fix`` or ``bash``) registered in openJiuwen's
  resource manager;
* the model answers with a tool call, openJiuwen executes the tool, and the harness reads
  the outcome from the tool itself. Most tasks need a single model call, so the agent
  runs with ``max_iterations=1``; the free-form agent for unknown task types runs a
  real multi-step loop;
* the agent's model is a ``MeteredModel`` — openJiuwen's ``Model`` with accounting
  (output + input/10), retries on rate limits, and a stop when the budget is gone.

Cost unit is the competition one: output tokens + input tokens / 10. Reasoning tokens
are already inside completion_tokens for OpenAI-compatible APIs.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from openjiuwen.core.foundation.llm.model import Model
from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.foundation.tool import tool
from openjiuwen.core.runner import Runner
from openjiuwen.core.single_agent.agents.react_agent import ReActAgent, ReActAgentConfig
from openjiuwen.core.single_agent.schema.agent_card import AgentCard

from .log import log, quiet_framework_logs, traj
from .sandbox import extract_code

MODELS = {
    "fast": os.environ.get("FAST_MODEL", "qwen3.8-27b-noreason"),
    "think": os.environ.get("THINK_MODEL", "qwen3.8-27b"),
    "moe_fast": "qwen3.6-35b-a3b-noreason",
    "moe_think": "qwen3.6-35b-a3b",
}


# How a single-step agent hands its program or patch to its tool (env AGENT_TOOLS):
#   0 (default) - as a fenced code block; the harness passes it to the openJiuwen tool.
#   1 - as a native tool call. Same accuracy on our sets, but the tool schema and the
#       model's tool preamble are sent with every request: +63% tokens (8571 vs 5269
#       on 40 independent tasks).
# The multi-step agent for unknown task types always uses native tool calls.
NATIVE_TOOLS = os.environ.get("AGENT_TOOLS", "0") == "1"


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


def _text(m) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
    return c if isinstance(c, str) else str(c or "")


def _role(m) -> str:
    return (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) or ""


# What the current worker is doing: read by the metered model and by the tools, which
# openJiuwen calls from inside the agent loop. One dict per ``AgentSession.ask``.
_CALL: contextvars.ContextVar = contextvars.ContextVar("harness_call", default=None)


# ------------------------------------------------------------------ tools (openJiuwen LocalFunction)

def _dispatch(arg: str) -> str:
    call = _CALL.get()
    if call is None or call.get("handler") is None:
        return "tool is not available here"
    call["called"] = True
    try:
        return str(call["handler"](arg))
    except Exception as e:  # a tool must answer, not break the agent loop
        call["tool_error"] = e
        return f"{type(e).__name__}: {e}"


@tool(name="run_python", description="Run a Python 3 program in the task directory; returns its output.")
async def run_python_tool(code: str) -> str:
    return _dispatch(code)


@tool(name="apply_fix", description="Replace functions of the buggy module with the given corrected "
                                    "definitions and run the tests; returns the test result.")
async def apply_fix_tool(code: str) -> str:
    return _dispatch(code)


@tool(name="bash", description="Run a shell command in the task directory; returns its output.")
async def bash_tool(command: str) -> str:
    return _dispatch(command)


TOOLS = {"run_python": run_python_tool, "apply_fix": apply_fix_tool, "bash": bash_tool}
_tools_registered = False


def _register_tools() -> None:
    global _tools_registered
    if not _tools_registered:
        for t in TOOLS.values():
            Runner.resource_mgr.add_tool(t)
        _tools_registered = True


# ------------------------------------------------------------------ metered model

class MeteredModel(Model):
    """openJiuwen Model that counts what every call costs and survives a flaky proxy."""

    def __init__(self, owner: "LLM", name: str):
        self._owner, self._name = owner, name
        cc = ModelClientConfig(client_provider="OpenAI", api_key=owner.key or "none",
                               api_base=owner.base, timeout=180.0,
                               stream_first_chunk_timeout=170.0, verify_ssl=True)
        super().__init__(cc, ModelRequestConfig(model=name, temperature=0.0))

    async def invoke(self, messages, **kw):
        call = _CALL.get() or {}
        tag = call.get("tag", "")
        if call.get("err") is not None:
            raise call["err"]
        try:
            return await self._invoke(messages, call, tag, kw)
        except (BudgetExhausted, ProxyDown) as e:
            call["err"] = e  # the agent loop may swallow it; AgentSession.ask re-raises
            raise

    async def _invoke(self, messages, call, tag, kw):
        owner, ledger = self._owner, self._owner.ledger
        if ledger.dead:
            raise ProxyDown()
        max_tokens = int(call.get("max_tokens") or kw.get("max_tokens") or 512)
        msgs = messages if isinstance(messages, list) else [{"role": "user", "content": str(messages)}]
        est_in = sum(_approx_tokens(_text(m)) for m in msgs)
        if ledger.remaining() < (est_in / 10.0 + max_tokens) * 0.5:
            raise BudgetExhausted()
        kw = dict(kw, max_tokens=max_tokens, temperature=0.0)
        kw.pop("model", None)
        if self._name == MODELS["think"]:
            kw.setdefault("reasoning_effort", "low")
        last_err = None
        for attempt in range(6):
            t0 = time.time()
            try:
                resp = await super().invoke(messages, **kw)
            except Exception as e:  # network / proxy / rate limit
                last_err = e
                msg = str(e)
                log(f"[llm] {tag} attempt {attempt} error: {msg[:300]}")
                if re.search(r"\b404\b", msg) and owner.v1_state != "done":
                    # LLM_BASE_URL may be given without /v1: try once with it, and put
                    # the address back if that was not the problem
                    if owner.v1_state == "untried" and not owner.base.endswith("/v1"):
                        owner.v1_state, owner.base = "trying", owner.base + "/v1"
                    else:
                        owner.v1_state, owner.base = "done", owner.base0
                    log(f"[llm] 404, base URL is now {owner.base}")
                    owner.rebuild_models()
                    continue
                rate = re.search(r"\b429\b|rate.?limit|too many requests|inflight|concurren", msg, re.I)
                spent = re.search(r"\b(401|402|403)\b|budget|exhaust|insufficient|balance", msg, re.I)
                if spent and not rate:
                    owner.fail_streak += 1
                    if owner.fail_streak >= 3:  # three different calls refused: budget is gone
                        ledger.dead = True
                    raise BudgetExhausted(msg)
                await asyncio.sleep(min(30, (5 if rate else 2) * (attempt + 1)))
                continue
            owner.fail_streak = 0
            if owner.v1_state == "trying":
                owner.v1_state = "done"  # the /v1 address works, keep it
            quiet_framework_logs()  # openJiuwen creates its loggers lazily on first use
            u = resp.usage_metadata
            inp = getattr(u, "input_tokens", 0) or est_in
            reasoning = resp.reasoning_content or ""
            calls = [(t.name, t.arguments) for t in (resp.tool_calls or [])]
            out = getattr(u, "output_tokens", 0) or _approx_tokens(
                _text(resp) + reasoning + "".join(str(a) for _, a in calls))
            cost = ledger.add(inp, out, tag)
            call["last"] = resp
            traj({"tag": tag, "model": self._name,
                  "messages": [{"role": _role(m), "content": _text(m)} for m in msgs],
                  "tools": [getattr(t, "name", None) or (t.get("function", {}).get("name") if isinstance(t, dict)
                                                         else None) for t in (kw.get("tools") or [])],
                  "reply": _text(resp), "tool_calls": calls,
                  "reasoning_chars": len(reasoning), "in": inp, "out": out, "cost": cost,
                  "sec": round(time.time() - t0, 2), "finish": resp.finish_reason})
            return resp
        # one call failing six times may be a glitch; three calls in a row means the
        # proxy has stopped answering (budget gone) and the run should wind down
        owner.fail_streak += 1
        if owner.fail_streak >= 3:
            ledger.dead = True
        raise ProxyDown(str(last_err))


# ------------------------------------------------------------------ agents

@dataclass
class Reply:
    content: str   # the model's text (empty when it only called the tool)
    called: bool   # the tool ran (called by the model, or recovered from a code block)


class AgentSession:
    """One openJiuwen ReActAgent working on one task. ``ask`` may be repeated: the agent
    keeps the conversation, so a retry only needs a short follow-up."""

    def __init__(self, llm: "LLM", tag: str, system: str, tool_name: str = "",
                 handler: Optional[Callable[[str], str]] = None, max_iterations: int = 1):
        self.llm, self.tag, self.handler, self.tool_name = llm, tag, handler, tool_name
        self.sid = f"{tag}-{uuid.uuid4().hex[:8]}"
        cfg = ReActAgentConfig()
        cfg.configure_prompt_template([{"role": "system", "content": system}])
        cfg.configure_max_iterations(max_iterations)
        self.agent = ReActAgent(AgentCard(id=self.sid, name="solver",
                                          description="solves one task of the benchmark"))
        self.agent.configure(cfg)
        if tool_name and (NATIVE_TOOLS or max_iterations > 1):
            _register_tools()
            self.agent.ability_manager.add(TOOLS[tool_name].card)

    async def ask(self, text: str, model: str = MODELS["fast"], max_tokens: int = 512) -> Reply:
        self.llm.n_calls[self.tag] = self.llm.n_calls.get(self.tag, 0) + 1
        call = {"tag": f"{self.tag}#{self.llm.n_calls[self.tag] - 1}", "max_tokens": max_tokens,
                "handler": self.handler, "called": False, "last": None, "err": None}
        self.agent.set_llm(self.llm.model(model))
        token = _CALL.set(call)
        try:
            try:
                await self.agent.invoke({"query": text, "conversation_id": self.sid})
            except (BudgetExhausted, ProxyDown):
                raise
            except Exception as e:
                if call["err"] is None and call["last"] is None:
                    raise
                log(f"[agent] {call['tag']}: {type(e).__name__}: {str(e)[:200]}")
            content = ""
            if call["err"] is None and call["last"] is not None:
                content = re.sub(r"(?s)<think>.*?</think>", "", _text(call["last"])).strip()
                if self.tool_name and not call["called"] and "```" in content:
                    # The model handed its code over as a fenced block (the default
                    # protocol, see NATIVE_TOOLS): run the same openJiuwen tool with it.
                    arg = next(iter(TOOLS[self.tool_name].card.input_params["properties"]))
                    await TOOLS[self.tool_name].invoke({arg: extract_code(content)})
        finally:
            _CALL.reset(token)
            quiet_framework_logs()
        if call["err"] is not None:
            raise call["err"]
        if call["last"] is None:
            raise ProxyDown("the agent made no model call")
        return Reply(content, call["called"])


class LLM:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("LLM_API_KEY", "")
        self.base0 = self.base
        self.v1_state = "untried"  # untried -> trying -> done
        self.fail_streak = 0
        self.n_calls: dict = {}
        self._models: dict = {}

    def rebuild_models(self) -> None:
        """Point the existing model objects at the current base URL."""
        for name, m in self._models.items():
            MeteredModel.__init__(m, self, name)

    def model(self, name: str) -> MeteredModel:
        m = self._models.get(name)
        if m is None:
            m = self._models[name] = MeteredModel(self, name)
            quiet_framework_logs()
        return m

    def session(self, tag: str, system: str, tool_name: str = "",
                handler: Optional[Callable[[str], str]] = None, max_iterations: int = 1) -> AgentSession:
        return AgentSession(self, tag, system, tool_name, handler, max_iterations)
