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
:: 慢模型（如 GLM-5.x，单次生成 30-50s）需留足读超时（默认 180s；
   读超时后重试窗口自动翻倍；仍失败会提示调大该值）：
set RE_EMP_LLM_TIMEOUT=300
:: Agent 运行时参数（时间线复盘 t-eb3dbd0f 后从硬编码收敛到 env，均有下限保护）：
set RE_EMP_AGENT_MAX_STEPS=20        :: 子任务执行步数上限（默认 20；旧硬编码 12 时正常研究模式撞顶）
set RE_EMP_WRITE_EVIDENCE_CHARS=4000 :: WRITE 阶段注入的证据字符数/篇（默认 4000；旧硬编码 400 把报告质量卡在摘要级）
set RE_EMP_LLM_WRITE_TIMEOUT=270     :: WRITE 阶段读超时（默认 270 = 基础 180s × 1.5；长报告生成贴线）

# 真实搜索（四种方式，有 key 时按优先级自动选择：LangSearch > Brave > SerpApi）：
# 1) LangSearch 免费 Web Search API（set LANGSEARCH_API_KEY=*** key 在
#    https://langsearch.com/dashboard 申请，免费每日 token 额度，POST /v1/web-search）
# 2) Brave 搜索 API（set BRAVE_API_KEY=***
# 3) SerpApi（set SERPAPI_API_KEY=***
# 4) You.com 免费 MCP（无任何 key 时的默认，无需 key，按 IP 限流；RE_EMP_YOU_MCP=off 禁用）

python -m app.cli run --prompt "…" --task-id t-001 --limit-subtasks 3
python -m app.cli timeline --task-id t-001      :: 事件时间线（含 token/延迟/重试）
python -m app.cli resume   --task-id t-001      :: 崩溃后恢复（有检查点断点续跑；无检查点整体重跑）
python -m app.cli serve  --port 8000      :: 启动 Web 前端（浏览器打开 http://127.0.0.1:8000）
```

Web 前端「模型配置」面板可直接增删改 LLM 降级链（含选主模型/重排序）
与各搜索后端的 API key，持久化到 `storage/models.json`（重启仍生效，
CLI 与 Web 服务共用；无此文件时用环境变量配置）。变更对新任务/恢复
任务生效，运行中任务沿用原配置。

## M2 新增能力

| 能力 | 实现 | 说明 |
|---|---|---|
| 真实模型接入 | `runtime/llm.py` `OpenAICompatibleProvider` | httpx 调 /chat/completions；可重试错误（超时/429/5xx）指数退避+抖动重试 |
| 多模型降级链 | `runtime/llm.py` `ModelGateway` | 主→备顺序尝试；`CircuitBreaker` 连续失败熔断、冷却后 half-open 探测 |
| token 预算 | `runtime/budget.py` | `max_total_tokens` 软限 → 收尾模式（部分报告 + paused） |
| 调用元数据观测 | `llm.called` 事件 | provider/model/tokens_in/out/latency_ms/retries 全入事件流，timeline 直接可读 |
| 真实搜索 | `tools/search_providers.py` | LangSearch / Brave / SerpApi 三个 Provider，统一 `search()` 接口（LangSearch：langsearch.com 免费 API，Bearer key，snippet 模式）；失败作为 observation 交回 LLM |
| 免费搜索/抓取 | `tools/mcp_you.py` `YouMCPWebBackend` | You.com MCP（`api.you.com/mcp?profile=free`，**无需 key**，IP 限流）；`you-search` 一次调用带回结果 + 整页 Markdown，fetch 复用免二次限流调用；无 Brave/SerpApi key 时 auto 模式自动启用，`RE_EMP_WEB=mcp` 可强制 |
| Web 前端 | `web/server.py` + `web/index.html` | 零新增依赖（stdlib http.server + 原生 JS 单页）；JSON API（建任务/恢复/事件增量/报告/模型配置）+ 任务列表/实时时间线/报告渲染/一键恢复失败任务（有检查点→断点续跑；无检查点（如 PLAN 阶段崩溃）→重新运行）；`python -m app.cli serve` |
| 模型配置（Web） | `runtime/model_registry.py` + `/api/models*` | 前端「模型配置」面板：LLM 降级链查看/增删改/重排序（首位为主模型）+ 搜索 key（LangSearch/Brave/SerpApi）增删改；变更后立即重建 ModelGateway/搜索后端并持久化 `storage/models.json`（llm/web 分节，重启仍生效，CLI 共用）；测试注入 llm/web 时 API 只读（409） |
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
app/tools/search_providers.py M2：LangSearch / Brave / SerpApi / Mock 搜索
app/tools/fetch_backends.py   M2：httpx 抓取 + HTML→MD + 磁盘缓存
app/tools/{registry,web,artifact_store}.py  工具路由 / 后端门面 / 工件库
app/config.py                 env 配置 → 组件装配（无 key 降级 mock）
app/cli.py                    run / resume / timeline
```

## 测试（112 项 Python + 16 项前端 DOM，全离线可跑）

```bash
pytest tests/ -q
```

- `test_agent_loop.py` — 解析失败恢复、工具错误观察、步数/预算护栏、SYSTEM_PROMPT 工具参数表完整性；顶层参数合并端到端、REPEAT_FAIL 退化循环护栏（同参失败 2 次即强提示换路）、max_steps 零证据明示、llm.called 带 raw_head 原文片段（t-a58c3f76 回归）
- `test_tools.py` — read_artifact 空 id 明确报错 / 参数别名（path、artifact_id）/ 未知 id 时列出可用工件；web_search 别名（q、keyword）/ web_fetch 别名（link、href）；缺参报错带正确调用示例（t-eb3dbd0f / t-a58c3f76 回归）
- `test_config.py` — Agent 运行时参数默认值 / env 覆盖 / 下限保护（agent_max_steps、write_evidence_chars、llm_write_timeout）
- `test_actions.py` — JSON 提取健壮性：括号平衡取第一个完整对象（多 JSON / 后随杂文 / 字符串内花括号 / 转义），回归贪婪正则 `\\{.*\\}` 跨对象过度匹配缺陷（PLAN 提取共用同一实现）；顶层参数容忍（模型把 query 写顶层时合并进 args，t-a58c3f76 回归）
- `test_orchestrator.py` — 端到端研究任务、幂等重跑、预算收尾、max_steps 透传、WRITE 证据截断字符数、WRITE 读超时透传至 provider（旧式两参 LLM 回退兼容）、零证据 WRITE 降级提示（禁编造链接）
- `test_recovery.py` — kill 模拟断点续跑、外部调用不重复
- `test_llm_gateway.py` — 重试/退避、401 不重试、降级链、熔断生命周期、读超时窗口翻倍救慢模型、`timeout=` 覆盖窗口并跟随重试翻倍、token 预算、plan_limit
- `test_web_real.py` — 本地 HTTP server 验证真实抓取链路（HTML→MD、404、磁盘缓存防重复抓取、断网优雅降级）、LangSearch/Brave/SerpApi 响应解析与 build_web 优先级
- `test_model_registry.py` — LLM 降级链/搜索 key 增删改/重排序、models.json 分节持久化、损坏回退
- `test_web_server.py` — `/api/models*` 全周期（增删改/重排/web key/持久化重载）+ 注入模式只读 409
- `frontend_dom_test.js` — 含模型配置面板（链渲染/重排/添加）回归

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
