/**
 * 前端 DOM 回归测试（node tests/frontend_dom_test.js，node 可选依赖）。
 *
 * 与浏览器一致的关键语义：对元素赋 innerHTML 会**销毁其子元素**，
 * 之后 querySelector 找不到被销毁的 id（返回 null）。
 * 场景：连续切换两个已完成历史任务（done→done），时间线必须都能显示。
 */
const fs = require('fs');
const path = require('path');
const htmlPath = process.argv[2] || path.join(__dirname, '..', 'app', 'web', 'index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];

// ---- 静态解析 <script> 之前的 HTML：parent id -> 嵌套的 id 集合 ----
function nestedIdMap(htmlPart) {
  const VOID = new Set(['input', 'br', 'img', 'hr', 'meta', 'link', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr']);
  const map = {}; const stack = [];
  const re = /<\/?([a-zA-Z][a-zA-Z0-9-]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)(\/?)>/g;
  let m;
  while ((m = re.exec(htmlPart))) {
    const name = m[1], attrs = m[2] || '', selfClose = !!m[3];
    const closing = m[0][1] === '/';
    const id = (/id="([^"]+)"/.exec(attrs) || [])[1];
    if (closing) {
      for (let i = stack.length - 1; i >= 0; i--)
        if (stack[i].name === name) { stack.length = i; break; }
    } else {
      if (id) for (const a of stack) if (a.id) (map[a.id] || (map[a.id] = new Set())).add(id);
      if (!VOID.has(name.toLowerCase()) && !selfClose) stack.push({ name, id });
    }
  }
  return map;
}
const nested = nestedIdMap(html.split('<script>')[0]);

// ---- DOM 桩（含销毁语义） ----
const destroyed = new Set();
function makeEl(id) {
  const el = {
    id, textContent: '', value: '3', style: {}, dataset: {}, children: [],
    scrollTop: 0, scrollHeight: 0,
    classList: { toggle() {}, add() {}, remove() {} },
    appendChild(c) { this.children.push(c); },
    addEventListener(ev, fn) { if (ev === 'click') this.__click = fn; },
    querySelectorAll() { return []; },
  };
  Object.defineProperty(el, 'innerHTML', {
    get: () => el.__html || '',
    set: (v) => {
      el.__html = v;
      if (v === '') el.children.length = 0;
      // 销毁语义：新 innerHTML 不包含某嵌套 id → 该元素从 DOM 消失
      if (id && nested[id]) for (const nid of nested[id])
        if (!v.includes(`id="${nid}"`)) destroyed.add(nid);
    },
  });
  return el;
}
const els = {};
function $(sel) {
  const id = sel.replace('#', '').split(' ')[0];
  if (destroyed.has(id)) return null;   // 与浏览器一致：被销毁 → null
  if (!els[id]) els[id] = makeEl(id);
  if (id === 'limit') els[id].value = '3';
  return els[id];
}
const tabTL = { dataset: { tab: 'timeline' }, classList: { toggle(){} } };
const tabRP = { dataset: { tab: 'report' }, classList: { toggle(){} } };
$('tabs').querySelectorAll = (s) => (s === '.tab' ? [tabTL, tabRP] : []);
const document = { querySelector: (sel) => $(sel), createElement: (t) => makeEl('<' + t + '>') };

// ---- fetch 桩：两个 done 历史任务 + 一个 failed（无检查点）+ 一个 interrupted（有检查点）----
let resumeCalls = 0;
function fetch(path, opt) {
  const key = (opt && opt.method === 'POST' ? 'POST ' : 'GET ') + path;
  const J = (o) => Promise.resolve({ ok: true, headers: { get: () => 'application/json' }, json: () => Promise.resolve(o) });
  const T = (t) => Promise.resolve({ ok: true, headers: { get: () => 'text/markdown' }, text: () => Promise.resolve(t) });
  if (key.startsWith('GET /api/status')) return J({ llm: 'mock', web: 'mock' });
  if (key.startsWith('GET /api/tasks/A/timeline'))
    return J({ events: [[1, 'task.created', {}], [2, 'llm.called', {}], [3, 'task.completed', {}]], status: 'done' });
  if (key.startsWith('GET /api/tasks/B/timeline'))
    return J({ events: [[1, 'task.created', {}], [2, 'tool.called', { tool: 'web_search', ok: true }], [3, 'task.completed', {}]], status: 'done' });
  // 失败且无检查点（PLAN 阶段崩溃）→ 可“重新运行”
  if (key.startsWith('GET /api/tasks/C/timeline'))
    return J({ events: [[1, 'task.created', {}], [2, 'llm.called', {}],
                        [3, 'task.failed', { error: 'LLMError: 500' }]], status: 'failed' });
  // 中断且有检查点（进程被杀，跑到子任务 1）→ 可“恢复运行”
  if (key.startsWith('GET /api/tasks/D/timeline'))
    return J({ events: [[1, 'task.created', {}], [2, 'plan.created', {}],
                        [3, 'checkpoint.saved', { stage: 'EXECUTE', subtask_cursor: 1 }]], status: 'interrupted' });
  if (key.startsWith('GET /api/tasks/A/report')) return T('# 报告 A\n\n内容A');
  if (key.startsWith('GET /api/tasks/B/report')) return T('# 报告 B\n\n内容B');
  if (key === 'POST /api/tasks/C/resume') { resumeCalls++; return J({ task_id: 'C', status: 'running' }); }
  if (key.startsWith('GET /api/tasks')) return J({ tasks: [
    { task_id: 'A', status: 'done', resumable: false, has_checkpoint: true, subtasks_done: 3, subtasks_total: 3, has_report: true },
    { task_id: 'B', status: 'done', resumable: false, has_checkpoint: true, subtasks_done: 3, subtasks_total: 3, has_report: true },
    { task_id: 'C', status: 'failed', resumable: true, has_checkpoint: false, subtasks_done: 0, subtasks_total: 3, has_report: false },
    { task_id: 'D', status: 'interrupted', resumable: true, has_checkpoint: true, subtasks_done: 1, subtasks_total: 3, has_report: false }] });
  return Promise.resolve({ ok: false, code: 404, headers: { get: () => 'application/json' }, json: () => Promise.resolve({ error: 'nf' }) });
}
function alert(m) { console.log('ALERT:', m); }
// 页面的 setInterval 不能拖住 Node 进程：unref 掉
const realSetInterval = global.setInterval;
function setInterval(fn, ms) { const h = realSetInterval(fn, ms); if (h.unref) h.unref(); return h; }

const errors = [];
process.on('unhandledRejection', (e) => errors.push('unhandled: ' + (e && e.message)));
const fn = new Function('document', 'fetch', 'setTimeout', 'setInterval', 'alert', 'encodeURIComponent', script);
fn(document, fetch, setTimeout, setInterval, alert, encodeURIComponent);
const tick = (ms) => new Promise(r => setTimeout(r, ms));
// 模拟浏览器点击：handler 抛错不上抛（浏览器行为）
function click(el) { try { el.onclick && el.onclick(); } catch (e) { errors.push('onclick throw: ' + e.message); } }

let fail = 0;
function check(name, cond) { console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}`); if (!cond) fail++; }

(async () => {
  let stepErr = null;
  try {
    await tick(20);
    // 选历史任务 A
    click(els['task-list'].children[0]);
    await tick(30);
    check('任务A：时间线页可见且 3 条事件',
      els['view-timeline'].style.display !== 'none' && els['events'].children.length === 3);
    for (let i = 0; i < 4; i++) await tick(10);   // 4 个轮询周期
    check('任务A：停留轮询周期后仍在时间线页', els['view-timeline'].style.display !== 'none');
    // 切换到历史任务 B（核心回归点：#ph 若被销毁，selectTask 会在 pollEvents 前抛错）
    click(els['task-list'].children[1]);
    await tick(30);
    check('任务B：时间线页可见且 3 条事件',
      els['view-timeline'].style.display !== 'none' && els['events'].children.length === 3);
    for (let i = 0; i < 4; i++) await tick(10);
    check('任务B：停留轮询周期后仍在时间线页', els['view-timeline'].style.display !== 'none');
    // 报告页
    try { $('tabs').__click({ target: tabRP }); } catch (e) { errors.push('tab throw: ' + e.message); }
    await tick(10);
    check('任务B：手动点报告页可渲染', els['report'].innerHTML.includes('报告 B'));

    // ---- 任务C：失败且无检查点 → 显示“重新运行”按钮，点击后恢复轮询 ----
    click(els['task-list'].children[2]);
    await tick(30);
    check('任务C（failed 无检查点）：按钮可见且文案为“重新运行”',
      els['btn-resume'].style.display !== 'none' && els['btn-resume'].textContent === '重新运行');
    check('任务C：无检查点提示语可见',
      els['resume-hint'].style.display !== 'none' && els['resume-hint'].textContent.includes('重新生成计划'));
    click(els['btn-resume']);
    await tick(30);
    check('任务C：点击“重新运行”后时间线重拉（3 条事件）且无报错',
      els['events'].children.length === 3 && errors.length === 0);
    check('任务C：POST /api/tasks/C/resume 被调用', resumeCalls === 1);

    // ---- 任务D：中断且有检查点 → 显示“恢复运行”按钮 ----
    click(els['task-list'].children[3]);
    await tick(30);
    check('任务D（interrupted 有检查点）：按钮文案为“恢复运行”',
      els['btn-resume'].style.display !== 'none' && els['btn-resume'].textContent === '恢复运行');
    check('任务D：检查点提示语可见', els['resume-hint'].textContent.includes('检查点'));
  } catch (e) { stepErr = e; }
  check('无 JS 错误（含未捕获 promise rejection / 点击抛错）',
    errors.length === 0 && !stepErr);
  if (errors.length) console.log('JS ERRORS:\n  ' + errors.join('\n  '));
  if (stepErr) console.log('STEP ERROR:', stepErr.stack);
  console.log(fail === 0 ? '\nALL PASS' : `\n${fail} FAILED`);
  process.exit(fail ? 1 : 0);
})();
