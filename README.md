# research-employee（M1 + M2）

研究数字员工：事件溯源 + 检查点 + Agent 循环 + 模型网关 + 真实搜索/抓取。

对应设计文档：`../设计方案.md` 的 M1（骨架）与 M2（真接入）里程碑。

## 快速开始

```bash
cd research-employee
# 离线演示（无任何 key，自动降级 mock 并告警）
python -m app.cli run --prompt "调研主流 Coding Agent 架构" --task-id t-001

# 真实模型（OpenAI 兼容端点均可：OpenAI/Azure/vLLM/Ollama/DeepSeek…）
set RE_EMP_LLM_API_KEY=sk-...
set RE_EMP_LLM_BASE_URL=https://api.openai.com/v1   :: 可选
set RE_EMP_LLM_MODEL=gpt-4o-mini                    :: 可选
:: 可选降级模型（主模型失败/熔断后自动切换）：
set RE_EMP_LLM_FALLBACK_API_KEY=...
set RE_EMP_LLM_FALLBACK_BASE_URL=...
set RE_EMP_LLM_FALLBACK_MODEL=...

# 真实搜索（三种方式，按优先级自动选择）：
# 1) You.com 免费 MCP（默认，无需 key，按 IP 限流；RE_EMP_YOU_MCP=off 禁用）
# 2) Brave 搜索 API（set BRAVE_API_KEY=***
# 3) SerpApi（set SERPAPI_API_KEY=***

python -m app.cli run --prompt "…" --task-id t-001 --limit-subtasks 3
python -m app.cli timeline --task-id t-001      :: 事件时间线（含 token/延迟/重试）
python -m app.cli resume   --task-id t-001      :: 崩溃后断点续跑
python -m app.cli serve  --port 8000      :: 启动 Web 前端（浏览器打开 http://127.0.0.1:8000）
```

## M2 新增能力

| 能力 | 实现 | 说明 |
|---|---|---|
| 真实模型接入 | `runtime/llm.py` `OpenAICompatibleProvider` | httpx 调 /chat/completions；可重试错误（超时/429/5xx）指数退避+抖动重试 |
| 多模型降级链 | `runtime/llm.py` `ModelGateway` | 主→备顺序尝试；`CircuitBreaker` 连续失败熔断、冷却后 half-open 探测 |
| token 预算 | `runtime/budget.py` | `max_total_tokens` 软限 → 收尾模式（部分报告 + paused） |
| 调用元数据观测 | `llm.called` 事件 | provider/model/tokens_in/out/latency_ms/retries 全入事件流，timeline 直接可读 |
| 真实搜索 | `tools/search_providers.py` | Brave / SerpApi 两个 Provider，统一 `search()` 接口；失败作为 observation 交回 LLM |
| 免费搜索/抓取 | `tools/mcp_you.py` `YouMCPWebBackend` | You.com MCP（`api.you.com/mcp?profile=free`，**无需 key**，IP 限流）；`you-search` 一次调用带回结果 + 整页 Markdown，fetch 复用免二次限流调用；无 Brave/SerpApi key 时 auto 模式自动启用，`RE_EMP_WEB=mcp` 可强制 |
| Web 前端 | `web/server.py` + `web/index.html` | 零新增依赖（stdlib http.server + 原生 JS 单页）；JSON API（建任务/恢复/事件增量/报告）+ 任务列表/实时时间线/报告渲染/一键恢复；`python -m app.cli serve` |
| 主题自适应报告 | `runtime/orchestrator.py` | PLAN 阶段产出 `compare_dimensions`（对比维度由研究主题推导，如手机市场→市场份额/价格带/目标人群），WRITE 阶段动态注入提示词，不再写死固定模板 |
| 真实抓取 | `tools/fetch_backends.py` `HttpxFetchBackend` | httpx 抓取 + 标准库 HTML→Markdown + **磁盘缓存（TTL）**→ 恢复重放不重复抓 |
| 缩小版任务 | `--limit-subtasks N`（默认 3） | PLAN 阶段注入上限，真实模型不遵守时硬截断兜底 |
| 配置装配 | `app/config.py` | 全 env 驱动；无 key 自动降级 mock 并打印告警 | 兼容任意 OpenAI 端点（已验证：OpenAI / 智谱 GLM `https://open.bigmodel.cn/api/paas/v4`） |
| 引用校验 | `orchestrator._verify_citations` | 报告中的 URL 必须能在任务证据/抓取记录中验证；编造链接被标注并写入 `citation.checked` 事件；WRITE 上下文注入 ALLOWED_URLS 白名单 |

## 架构对应关系

```
app/runtime/orchestrator.py   状态机 + 断点恢复 + 预算收尾
app/runtime/agent_loop.py     ReAct 循环（llm.called 事件带 token/延迟/重试）
app/runtime/llm.py            M2：OpenAI 兼容 Provider + 重试 + 熔断 + 降级链
app/runtime/model_gateway.py  Mock LLM（离线演示/测试）+ ScriptedLLM
app/runtime/{events,budget,context}.py  事件溯源 / 预算 / 上下文组装
app/tools/search_providers.py M2：Brave / SerpApi / Mock 搜索
app/tools/fetch_backends.py   M2：httpx 抓取 + HTML→MD + 磁盘缓存
app/tools/{registry,web,artifact_store}.py  工具路由 / 后端门面 / 工件库
app/config.py                 env 配置 → 组件装配（无 key 降级 mock）
app/cli.py                    run / resume / timeline
```

## 测试（29 项，全离线可跑）

```bash
pytest tests/ -q
```

- `test_agent_loop.py` — 解析失败恢复、工具错误观察、步数/预算护栏
- `test_orchestrator.py` — 端到端研究任务、幂等重跑、预算收尾
- `test_recovery.py` — kill 模拟断点续跑、外部调用不重复
- `test_llm_gateway.py` — 重试/退避、401 不重试、降级链、熔断生命周期、token 预算、plan_limit
- `test_web_real.py` — 本地 HTTP server 验证真实抓取链路（HTML→MD、404、磁盘缓存防重复抓取、断网优雅降级）、Brave/SerpApi 响应解析

HTTP client / sleep / clock 全部可注入，测试不发真实网络请求。

## 运行逻辑

1. **PLAN**：LLM 生成子任务计划（受 `--limit-subtasks` 约束）→ 工件 + 事件 + checkpoint
2. **EXECUTE**：每子任务一个 Agent 循环（搜索 → 抓取 → 证据工件 → finish），完成落 checkpoint
3. **WRITE**：汇总计划 + 全部证据 → 带引用 Markdown 报告 → DONE
4. **异常路径**：模型 429/5xx 退避重试 → 熔断 → 降级链；搜索 API 挂 → LLM 换路/记 gap；
   预算耗尽 → 部分报告 + paused；进程崩溃 → `resume` 从 checkpoint 续跑

## 已知边界（留给 M3/M4）

- VERIFY 阶段（证据充分性检查 + 计划修订回环）未实现，当前 PLAN 后直线执行；
  已实现最小 SELF-CHECK：报告引用校验（`citation.checked` 事件 + 编造 URL 标注）
- 无心跳看门狗 / 多任务并发 worker / OTel 导出（M3/M4）
- 金额预算、抓取 robots 检查、向量检索未接
- 上下文压缩仍是 M1 最小版（长任务 M3 加滚动摘要）
