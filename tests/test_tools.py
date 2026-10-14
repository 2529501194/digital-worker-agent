"""ToolRouter：read_artifact 空 id / 参数别名 / 错误时列出可用工件。

回归背景：GLM-5.3 时间线中模型三次调用 read_artifact 不带 id
（工具参数表未写入 SYSTEM_PROMPT，模型盲猜参数名），错误信息
"artifact 不存在: "（空 id）无法指导模型自纠，浪费 5/36 个执行步数。
"""
from __future__ import annotations

from app.tools.artifact_store import ArtifactStore
from app.tools.registry import ToolRouter
from app.tools.web import MockWebBackend


def _router(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    return ToolRouter(MockWebBackend(), store), store


def test_read_artifact_missing_id_says_so(tmp_path):
    """空 args → 明确报"缺少 id"而非误导性的"artifact 不存在"。"""
    router, store = _router(tmp_path)
    store.write("t1", "evidence", "A", "内容 A", ["https://a"])
    r = router.run("t1", "read_artifact", {})
    assert not r.ok
    assert "缺少 id" in r.error
    assert "list_artifacts" in r.scratch


def test_read_artifact_accepts_common_aliases(tmp_path):
    """模型常见误传参数名（path/artifact_id）也能命中。"""
    router, store = _router(tmp_path)
    art = store.write("t1", "evidence", "A", "内容 A", [])
    for alias in ("id", "artifact_id", "path"):
        r = router.run("t1", "read_artifact", {alias: art.id})
        assert r.ok, f"别名 {alias} 未生效"


def test_read_artifact_unknown_id_lists_available(tmp_path):
    """id 错误时列出当前可用工件，模型一步即可自纠。"""
    router, store = _router(tmp_path)
    a = store.write("t1", "evidence", "A", "内容 A", [])
    r = router.run("t1", "read_artifact", {"id": "E-999"})
    assert not r.ok
    assert "E-999" in r.scratch
    assert a.id in r.scratch


def test_web_fetch_missing_url_still_reported(tmp_path):
    router, _ = _router(tmp_path)
    r = router.run("t1", "web_fetch", {})
    assert not r.ok and "缺少 url" in r.error
