"""AppConfig：Agent 运行时参数（默认值 / env 覆盖 / 下限保护）。

来源：t-eb3dbd0f 时间线复盘——max_steps=12 全部撞顶、WRITE 证据截断
400 字符、WRITE 156s 贴 180s 超时线，三项参数从硬编码收敛到配置。
"""
from __future__ import annotations

from app.config import AppConfig


def test_agent_runtime_defaults():
    cfg = AppConfig()
    assert cfg.agent_max_steps == 20        # 旧硬编码 12 → 正常研究模式撞顶
    assert cfg.write_evidence_chars == 4000  # 旧硬编码 400 → 报告卡在摘要级
    assert cfg.llm_write_timeout == 270.0    # 基础 180s × 1.5


def test_agent_runtime_env_overrides(monkeypatch):
    monkeypatch.setenv("RE_EMP_AGENT_MAX_STEPS", "30")
    monkeypatch.setenv("RE_EMP_WRITE_EVIDENCE_CHARS", "8000")
    monkeypatch.setenv("RE_EMP_LLM_WRITE_TIMEOUT", "300")
    cfg = AppConfig.from_env()
    assert cfg.agent_max_steps == 30
    assert cfg.write_evidence_chars == 8000
    assert cfg.llm_write_timeout == 300.0


def test_agent_runtime_bounds(monkeypatch):
    """非法值保护：步数至少 1；证据字符数不低于 200（防误配成 0 把报告打瞎）。"""
    monkeypatch.setenv("RE_EMP_AGENT_MAX_STEPS", "0")
    monkeypatch.setenv("RE_EMP_WRITE_EVIDENCE_CHARS", "50")
    cfg = AppConfig.from_env()
    assert cfg.agent_max_steps == 1
    assert cfg.write_evidence_chars == 200
