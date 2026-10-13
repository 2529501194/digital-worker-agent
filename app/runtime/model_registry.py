"""模型配置注册表（Web 前端「模型配置」面板 + CLI 共用，零新增依赖）。

- ModelRegistry：可变 LLM 降级链（list[LLMProviderSpec]）。
  增删改 / 重排序后立刻重建 ModelGateway，并持久化到 storage/models.json
  （llm 段）；启动时若 models.json 存在，覆盖环境变量配置的降级链。
  已运行任务持有旧 gateway 引用不受影响；新任务 / 恢复任务用新链。
- WebBackendRegistry：管理搜索 API key（langsearch / brave / serpapi）与
  you-mcp 开关；变更后按 build_web 逻辑重建搜索后端，持久化到同一文件
  （web 段）。

两个注册表共享一个 models.json 文件，保存时按段合并（读-改-写，
file_lock 串行化文件级竞争）。key 明文存于该文件——本地单机工具，
勿把 storage 目录提交到仓库。
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path

from ..config import AppConfig, LLMProviderSpec, build_web
from ..runtime.llm import ModelGateway, OpenAICompatibleProvider
from ..runtime.model_gateway import MockResearchLLM

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_WEB_KEY_NAMES = ("langsearch", "brave", "serpapi")  # build_web 优先级顺序


def _load_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_section(path: Path | None, section: str, value,
                  file_lock: threading.Lock) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock:
        data = _load_file(path)
        data[section] = value
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


class ModelRegistry:
    """LLM 降级链管理：查看 / 增删改 / 重排序 + 持久化 + gateway 重建。"""

    def __init__(self, specs: list[LLMProviderSpec],
                 persist_path: str | Path | None = None,
                 file_lock: threading.Lock | None = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self._lock = threading.Lock()
        self._file_lock = file_lock or threading.Lock()
        self.specs: list[LLMProviderSpec] = list(specs)
        if self.persist_path is not None and self.persist_path.exists():
            saved = _load_file(self.persist_path).get("llm")
            if isinstance(saved, list) and saved \
                    and all(isinstance(e, dict) for e in saved):
                try:
                    self.specs = [LLMProviderSpec(**e) for e in saved]
                except (TypeError, ValueError):
                    pass  # 持久化内容损坏 → 回退环境变量配置
        self._gateway = None
        self._rebuild()

    # ---- 重建 / 查看 ----
    def _rebuild(self) -> None:
        if self.specs:
            self._gateway = ModelGateway([
                OpenAICompatibleProvider(s.name, s.base_url, s.api_key, s.model,
                                         s.timeout, s.max_retries)
                for s in self.specs])
        else:
            self._gateway = MockResearchLLM()

    @property
    def gateway(self):
        """当前生效的 ModelProvider（ModelGateway 或 MockResearchLLM）。"""
        return self._gateway

    def chain(self) -> list[dict]:
        """有序视图（前端用；不回传 key 明文，只有 has_key）。"""
        return [{"position": i, "name": s.name, "model": s.model,
                 "base_url": s.base_url, "has_key": bool(s.api_key)}
                for i, s in enumerate(self.specs)]

    def description(self) -> str:
        if not self.specs:
            return "mock"
        return " → ".join(s.name for s in self.specs)

    # ---- 变更（每次变更：重建 + 持久化） ----
    @staticmethod
    def _view(s: LLMProviderSpec) -> dict:
        return {"name": s.name, "model": s.model, "base_url": s.base_url,
                "has_key": bool(s.api_key)}

    def _find(self, name: str) -> LLMProviderSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise ValueError(f"模型 {name} 不存在")

    def add(self, name: str, model: str, key: str = "",
            base_url: str = "") -> dict:
        with self._lock:
            name = str(name or "").strip()
            model = str(model or "").strip()
            if not name:
                raise ValueError("模型名称（name）不能为空")
            if not model:
                raise ValueError("模型名（model）不能为空")
            if any(s.name == name for s in self.specs):
                raise ValueError(f"模型 {name} 已存在")
            spec = LLMProviderSpec(
                name=name,
                base_url=str(base_url or "").strip() or DEFAULT_BASE_URL,
                api_key=str(key or "").strip(), model=model)
            self.specs.append(spec)
            self._rebuild()
            _save_section(self.persist_path, "llm",
                         [asdict(s) for s in self.specs], self._file_lock)
            return {"position": len(self.specs) - 1, **self._view(spec)}

    def update(self, name: str, model: str | None = None,
               key: str | None = None, base_url: str | None = None) -> dict:
        with self._lock:
            s = self._find(name)
            if model is not None and not str(model).strip():
                raise ValueError("model 不能置空")
            if base_url is not None and not str(base_url).strip():
                raise ValueError("base_url 不能置空")
            if model is not None:
                s.model = str(model).strip()
            if key is not None:
                s.api_key = str(key).strip()  # 传空串 = 清除 key
            if base_url is not None:
                s.base_url = str(base_url).strip()
            self._rebuild()
            _save_section(self.persist_path, "llm",
                         [asdict(s) for s in self.specs], self._file_lock)
            return self._view(s)

    def remove(self, name: str) -> dict:
        with self._lock:
            s = self._find(name)
            self.specs.remove(s)
            self._rebuild()
            _save_section(self.persist_path, "llm",
                         [asdict(s) for s in self.specs], self._file_lock)
            return {"removed": name}

    def reorder(self, order: list[str]) -> list[dict]:
        """order 必须是现有 name 的排列；首位即主模型。"""
        with self._lock:
            names = [s.name for s in self.specs]
            if sorted(order) != sorted(names):
                raise ValueError("order 必须是现有模型列表的重排")
            self.specs = [next(s for s in self.specs if s.name == n)
                          for n in order]
            self._rebuild()
            _save_section(self.persist_path, "llm",
                         [asdict(s) for s in self.specs], self._file_lock)
            return self.chain()


class WebBackendRegistry:
    """Web 搜索 key 管理：langsearch / brave / serpapi 的增删改 + you-mcp 开关。"""

    def __init__(self, cfg: AppConfig, storage_root: str | Path,
                 persist_path: str | Path | None = None,
                 file_lock: threading.Lock | None = None):
        self.storage_root = Path(storage_root)
        self.web_mode = cfg.web_mode
        self.persist_path = Path(persist_path) if persist_path else None
        self._lock = threading.Lock()
        self._file_lock = file_lock or threading.Lock()
        self.keys = {"langsearch": cfg.langsearch_key or "",
                     "brave": cfg.brave_key or "",
                     "serpapi": cfg.serp_key or ""}
        self.you_mcp_enabled = bool(cfg.you_mcp_enabled)
        self._version = 0
        self._backend = None
        if self.persist_path is not None and self.persist_path.exists():
            saved = _load_file(self.persist_path).get("web")
            if isinstance(saved, dict):
                sk = saved.get("keys")
                if isinstance(sk, dict):
                    for k in self.keys:
                        v = sk.get(k)
                        if isinstance(v, str) and v:
                            self.keys[k] = v
                if isinstance(saved.get("you_mcp"), bool):
                    self.you_mcp_enabled = saved["you_mcp"]
        self._rebuild()

    def _rebuild(self) -> None:
        cfg = AppConfig(brave_key=self.keys["brave"] or None,
                        serp_key=self.keys["serpapi"] or None,
                        langsearch_key=self.keys["langsearch"] or None,
                        you_mcp_enabled=self.you_mcp_enabled,
                        web_mode=self.web_mode)
        self._backend = build_web(cfg, self.storage_root, verbose=False)
        self._version += 1

    def _persist(self) -> None:
        _save_section(self.persist_path, "web",
                      {"keys": self.keys, "you_mcp": self.you_mcp_enabled},
                      self._file_lock)

    @property
    def backend(self):
        """当前生效的搜索后端（RealWebBackend / YouMCPWebBackend / Mock）。"""
        return self._backend

    @property
    def version(self) -> int:
        """每次重建自增；TaskManager 据此判断是否需要重建 ToolRouter。"""
        return self._version

    def available(self) -> dict:
        """前端视图：当前后端 + 各后端可用性/激活状态。"""
        active = getattr(self._backend, "name", type(self._backend).__name__)
        backends = [
            {"name": n, "kind": "http", "has_key": bool(self.keys[n]),
             "available": bool(self.keys[n]),
             "active": active.startswith(n + "+")}
            for n in _WEB_KEY_NAMES
        ]
        backends.append({"name": "you-mcp", "kind": "mcp", "has_key": None,
                         "available": bool(self.you_mcp_enabled
                                           and not any(self.keys.values())),
                         "active": active == "you-mcp"})
        backends.append({"name": "mock", "kind": "mock", "has_key": None,
                         "available": True, "active": active == "mock"})
        return {"active": active, "backends": backends,
                "you_mcp_enabled": self.you_mcp_enabled}

    def set_key(self, backend: str, key: str) -> dict:
        """设置/清除某搜索后端的 key（空串 = 清除）。"""
        with self._lock:
            if backend not in _WEB_KEY_NAMES:
                raise ValueError(
                    f"未知搜索后端 {backend}（可选: {' / '.join(_WEB_KEY_NAMES)}）")
            self.keys[backend] = str(key or "").strip()
            self._rebuild()
            self._persist()
            return self.available()

    def set_you_mcp(self, enabled: bool) -> dict:
        with self._lock:
            self.you_mcp_enabled = bool(enabled)
            self._rebuild()
            self._persist()
            return self.available()
