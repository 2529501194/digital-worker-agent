"""ModelRegistry / WebBackendRegistry：LLM 降级链 + Web 搜索 key 的增删改/重排、
持久化（models.json 分节合并）、gateway/后端重建。"""
from __future__ import annotations

import json
import threading

import pytest

from app.config import AppConfig, LLMProviderSpec
from app.runtime.llm import ModelGateway
from app.runtime.model_gateway import MockResearchLLM
from app.runtime.model_registry import ModelRegistry, WebBackendRegistry


def spec(name="primary", model="gpt-4o-mini", key="sk-1"):
    return LLMProviderSpec(name=name, base_url="https://x/v1", api_key=key,
                           model=model)


class TestModelRegistry:

    def test_chain_view_and_gateway(self):
        reg = ModelRegistry([spec("a"), spec("b", key="")])
        assert reg.chain() == [
            {"position": 0, "name": "a", "model": "gpt-4o-mini",
             "base_url": "https://x/v1", "has_key": True},
            {"position": 1, "name": "b", "model": "gpt-4o-mini",
             "base_url": "https://x/v1", "has_key": False}]
        assert isinstance(reg.gateway, ModelGateway)
        assert [p.name for p in reg.gateway.providers] == ["a", "b"]
        assert reg.description() == "a → b"

    def test_empty_specs_fall_back_to_mock(self):
        reg = ModelRegistry([])
        assert isinstance(reg.gateway, MockResearchLLM)
        assert reg.description() == "mock"

    def test_add_update_remove(self):
        reg = ModelRegistry([spec("a")])
        view = reg.add("b", "gpt-4o", key="sk-2")
        assert view["position"] == 1 and view["has_key"]
        assert [p.name for p in reg.gateway.providers] == ["a", "b"]
        with pytest.raises(ValueError, match="已存在"):
            reg.add("b", "m")
        with pytest.raises(ValueError, match="不能为空"):
            reg.add("  ", "m")
        with pytest.raises(ValueError, match="不能为空"):
            reg.add("c", "   ")
        v = reg.update("b", model="gpt-5", key="sk-3")
        assert v["model"] == "gpt-5" and v["has_key"]
        reg.update("b", key="")  # 传空串 = 清除 key
        assert not reg.chain()[1]["has_key"]
        with pytest.raises(ValueError, match="不存在"):
            reg.update("zzz")
        reg.remove("b")
        assert [s.name for s in reg.specs] == ["a"]
        with pytest.raises(ValueError, match="不存在"):
            reg.remove("zzz")

    def test_reorder_changes_primary(self):
        reg = ModelRegistry([spec("a"), spec("b")])
        chain = reg.reorder(["b", "a"])
        assert [p.name for p in reg.gateway.providers] == ["b", "a"]
        assert [c["name"] for c in chain] == ["b", "a"]
        with pytest.raises(ValueError, match="重排"):
            reg.reorder(["a"])
        with pytest.raises(ValueError, match="重排"):
            reg.reorder(["a", "b", "c"])

    def test_persistence_roundtrip_and_section_isolation(self, tmp_path):
        f = tmp_path / "models.json"
        lock = threading.Lock()
        # web 段由 WebBackendRegistry 写入（变更时落盘）
        web = WebBackendRegistry(AppConfig(brave_key="bk"), tmp_path,
                                persist_path=f, file_lock=lock)
        web.set_key("brave", "bk")
        reg = ModelRegistry([spec("env-seed")], persist_path=f, file_lock=lock)
        reg.add("user-added", "glm-5.3", key="sk-9")
        data = json.loads(f.read_text("utf-8"))
        # llm 段保存不能冲掉 web 段
        assert [e["name"] for e in data["llm"]] == ["env-seed", "user-added"]
        assert data["web"]["keys"]["brave"] == "bk"
        # 重启：env 只有 env-seed，文件内容优先
        reg2 = ModelRegistry([spec("env-seed")], persist_path=f, file_lock=lock)
        assert [s.name for s in reg2.specs] == ["env-seed", "user-added"]
        assert reg2.gateway is not reg.gateway  # 重启后重建

    def test_corrupted_file_falls_back_to_env(self, tmp_path):
        f = tmp_path / "models.json"
        f.write_text("{not json", "utf-8")
        reg = ModelRegistry([spec("env-seed")], persist_path=f)
        assert [s.name for s in reg.specs] == ["env-seed"]


class TestWebBackendRegistry:

    @staticmethod
    def _cfg(**kw):
        base = dict(you_mcp_enabled=True, web_mode="auto")
        base.update(kw)
        return AppConfig(**base)

    def test_priority_langsearch_over_brave(self, tmp_path):
        reg = WebBackendRegistry(
            self._cfg(langsearch_key="lk", brave_key="bk"), tmp_path)
        assert reg.backend.name == "langsearch+http"
        by = {b["name"]: b for b in reg.available()["backends"]}
        assert by["langsearch"]["active"] and by["brave"]["available"]
        assert by["you-mcp"]["available"] is False  # 有 key 时 http 优先

    def test_set_key_switches_backend(self, tmp_path):
        reg = WebBackendRegistry(self._cfg(brave_key="bk"), tmp_path)
        assert reg.backend.name == "brave+http"
        view = reg.set_key("langsearch", "lk")
        assert view["active"] == "langsearch+http"
        v0 = reg.version
        view = reg.set_key("langsearch", "")  # 清除 → 回落到 brave
        assert view["active"] == "brave+http"
        assert reg.version > v0  # 版本自增（ToolRouter 重建信号）

    def test_no_keys_you_mcp_or_mock(self, tmp_path):
        reg = WebBackendRegistry(self._cfg(), tmp_path)
        assert reg.backend.name == "you-mcp"
        reg.set_you_mcp(False)
        assert reg.backend.name == "mock"
        view = reg.set_key("serpapi", "sk")
        assert view["active"] == "serpapi+http"

    def test_unknown_backend_rejected(self, tmp_path):
        reg = WebBackendRegistry(self._cfg(), tmp_path)
        with pytest.raises(ValueError, match="未知搜索后端"):
            reg.set_key("google", "x")

    def test_persistence_roundtrip(self, tmp_path):
        f = tmp_path / "models.json"
        lock = threading.Lock()
        WebBackendRegistry(self._cfg(), tmp_path,
                          persist_path=f, file_lock=lock).set_key("brave", "bk2")
        reg2 = WebBackendRegistry(self._cfg(), tmp_path,
                                 persist_path=f, file_lock=lock)
        assert reg2.keys["brave"] == "bk2"
        assert reg2.backend.name == "brave+http"
