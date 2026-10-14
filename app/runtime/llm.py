"""模型网关（M2）：OpenAI 兼容 Provider + 重试/退避 + 熔断 + 多模型降级链。

- OpenAICompatibleProvider：httpx 调用 /chat/completions；
  可重试错误（超时 / 429 / 5xx）指数退避 + 抖动重试，耗尽后抛 LLMError。
- CircuitBreaker：连续失败 N 次熔断 M 秒，冷却后放行探测（half-open）。
- ModelGateway：按顺序尝试 provider 链（主 → 备 → 兜底），
  熔断中的直接跳过；全部失败抛 GatewayError。

HTTP client 与 sleep/clock 均可注入 → 全链路可离线测试。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import httpx


class LLMError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class GatewayError(Exception):
    pass


@dataclass
class CallMeta:
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    error: str | None = None

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatibleProvider:
    """任何 OpenAI 兼容端点（OpenAI / Azure / vLLM / Ollama / DeepSeek ...）。"""

    def __init__(self, name: str, base_url: str, api_key: str, model: str,
                 timeout: float = 180.0, max_retries: int = 3,
                 backoff_base: float = 0.5,
                 http_client: httpx.Client | None = None,
                 sleep_fn=time.sleep):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep_fn
        self._http = http_client or httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str,
                 timeout: float | None = None) -> tuple[str, CallMeta]:
        """timeout：本次调用的读超时基础窗口（覆盖实例默认 self.timeout）；
        重试翻倍逻辑不变（基础 ×2，上限 2x）。供 WRITE 等长生成阶段放宽。"""
        base = timeout if timeout is not None else self.timeout
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err = "unknown"
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.monotonic()
                # 读超时窗口：首次用基础值；读超时后的重试翻倍（上限 2x）。
                # 慢模型（如 GLM-5.x 长生成 30-50s/次）在基础窗口内会超时，
                # 而生成不可续传——重试只能靠更长的窗口救回来。
                resp = self._http.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers,
                    timeout=base * (2 if attempt > 0 else 1))
                latency_ms = round((time.monotonic() - t0) * 1000, 1)

                if resp.status_code in _RETRYABLE_STATUS:
                    last_err = f"HTTP {resp.status_code}"
                    if attempt < self.max_retries:
                        self._sleep(self.backoff_base * (2 ** attempt)
                                    + random.uniform(0, 0.1))
                        continue
                    raise LLMError(
                        f"{self.name}: 重试 {self.max_retries} 次后仍为 "
                        f"HTTP {resp.status_code}", retryable=True)
                if resp.status_code != 200:
                    raise LLMError(
                        f"{self.name}: HTTP {resp.status_code}: {resp.text[:200]}",
                        retryable=False)
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                return text, CallMeta(
                    provider=self.name, model=self.model,
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    latency_ms=latency_ms, retries=attempt)
            except httpx.TimeoutException as e:
                if attempt >= self.max_retries:
                    # 读超时单独提示：绝大多数情况是“模型生成太慢”而非网络故障，
                    # 直接给出调参出口，避免用户对着“网络错误”排查
                    raise LLMError(
                        f"{self.name}: 读超时重试耗尽（{type(e).__name__}；模型生成可能"
                        f"超过 {int(base * 2)}s 窗口——调大 RE_EMP_LLM_TIMEOUT "
                        f"或换更快的模型/降级模型）", retryable=True) from e
                last_err = str(e)
                self._sleep(self.backoff_base * (2 ** attempt)
                            + random.uniform(0, 0.1))
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise LLMError(f"{self.name}: 网络错误重试耗尽: {e}",
                                   retryable=True) from e
                last_err = str(e)
                self._sleep(self.backoff_base * (2 ** attempt)
                            + random.uniform(0, 0.1))
        raise LLMError(f"{self.name}: 重试 {self.max_retries} 次后仍失败: {last_err}",
                       retryable=True)


class CircuitBreaker:
    """连续失败 threshold 次 → open；cooldown 秒后 half-open 放行一次探测。"""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0,
                 clock=time.monotonic):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self.failures = 0
        self.opened_at: float | None = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if self._clock() - self.opened_at >= self.cooldown_seconds:
            self.opened_at = None      # half-open：放行探测
            self.failures = 0
        return self.opened_at is None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        return self._clock() - self.opened_at < self.cooldown_seconds

    def record_ok(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold and self.opened_at is None:
            self.opened_at = self._clock()


class ModelGateway:
    """多 provider 降级链。实现 ModelProvider 协议（complete -> str）。"""

    name = "model-gateway"

    def __init__(self, providers: list[OpenAICompatibleProvider],
                 failure_threshold: int = 3, cooldown_seconds: float = 60.0,
                 clock=time.monotonic):
        if not providers:
            raise ValueError("至少需要一个 provider")
        self.providers = providers
        self.breakers = {p.name: CircuitBreaker(failure_threshold, cooldown_seconds,
                                                clock)
                         for p in providers}
        self.last_meta: CallMeta | None = None
        self.last_errors: list[str] = []

    def complete(self, system: str, user: str,
                 timeout: float | None = None) -> str:
        """timeout：本次调用的读超时基础窗口（透传给 provider）；
        None 时各 provider 用自身默认。仅真实 provider 支持该参数，
        mock/测试 fake 保持两参签名不受影响。"""
        self.last_errors = []
        for p in self.providers:
            breaker = self.breakers[p.name]
            if not breaker.allow():
                self.last_errors.append(f"{p.name}: 熔断中（circuit open）")
                continue
            try:
                if timeout is not None:
                    text, meta = p.complete(system, user, timeout=timeout)
                else:
                    text, meta = p.complete(system, user)
                breaker.record_ok()
                self.last_meta = meta
                return text
            except LLMError as e:
                breaker.record_failure()
                self.last_errors.append(str(e))
        raise GatewayError("所有 provider 均失败: " + " | ".join(self.last_errors))
