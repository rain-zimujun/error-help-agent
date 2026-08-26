# error-help-agent

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的故障自愈系统。用 4 个 LLM Agent（Monitor → RCA → Heal → Change）串联起一次故障从检测到审批的完整处理流程，每个 Agent 通过 ReAct 循环自主调用工具完成分析，而不是硬编码的 if/else 流程。

## 📖 这个项目是什么？

你是否遇到过这些问题：

- 运维团队每天被 **200+ 条告警**淹没，大部分是误报
- 服务出了问题，排查根因要花 **40 分钟**
- 凌晨 3 点被电话叫醒，半睡半醒地排查故障

本项目用 **4 个 AI Agent 协作**，自动完成从「告警检测」到「故障修复」的全流程，把 MTTR（平均修复时间）从 40 分钟降到 5 分钟。

```
告警来了
  ↓  Agent 1：这是真的异常吗？（时序分析，过滤误报）
  ↓  Agent 2：根因在哪里？（知识图谱推理）
  ↓  Agent 3：怎么修复？能自动执行吗？（安全护栏 + 分级策略）
  ↓  Agent 4：这个操作风险有多大？需要审批吗？（风险评分）
  ✅ 5 分钟内修复完成，全程自动
```

> 

## 架构

```
START
  │
  ▼
[Monitor Agent] 异常检测 + 告警去重 + 分级
  │  条件边：is_anomaly=True 且 is_duplicate=False 才继续
  ▼
[RCA Agent] 根因分析（贝叶斯推理）
  │  条件边：confidence >= CONFIDENCE_THRESHOLD 才继续
  ▼
[Heal Agent] 匹配修复方案 + dry-run 验证
  │  条件边：status 是 SUCCESS 或 PENDING_APPROVAL 才继续
  ▼
[Change Agent] 风险评分 + 审批决策
  │
  ▼
 END
```

每个 Agent 都是一个独立的 ReAct Agent（`langchain.agents.create_agent`），自己决定调用哪些工具、调用顺序，最后通过 `response_format` 强制输出结构化结果（`AlertEvent`/`RCAResult`/`HealAction`/`ChangeDecision`），写回 `IncidentState`。State 上的 `status` 字段既是给人看的事件状态，也直接驱动条件边的路由判断。

图用 `InMemorySaver` 做 checkpoint，按 `thread_id` 存取每次事件处理的完整状态，支持事后查询和断点恢复。

### 文件结构

```
.
├── state.py              # IncidentState / MetricData 的 TypedDict 定义
├── knowledge_graph.py     # 服务依赖知识图谱（内存版）
├── tools/
│   ├── monitor_tools.py   # 异常检测、告警去重、分级
│   ├── rca_tools.py       # 依赖查询、影响链追踪、根因候选
│   ├── heal_tools.py      # Playbook 匹配、熔断器、dry-run
│   └── change_tools.py    # 风险评分、审批策略、oncall 通知
├── agents.py              # 4 个 Agent 的构造（system prompt + response_format schema）
├── graph.py               # StateGraph 组装：节点、条件边、checkpointer
├── main.py                # Demo 运行入口
└── tests/                 # pytest 单元测试 + 图路由集成测试
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 ICE_API_KEY / ICE_BASE_URL（或替换成你自己的 OpenAI 兼容 LLM 服务）

# 3. 跑一次 demo
python main.py

# 4. 跑测试（不需要真实 LLM 调用，纯逻辑测试用 mock 跑）
pytest tests/ -v
```

`main.py` 会用一份写死的示例数据（`order-service` 的 CPU 95% 异常）跑一遍完整流程，并打印每个 Agent 的结构化输出和最终审批结果。

## 配置项（`.env`）

| 变量 | 说明 |
|---|---|
| `ICE_API_KEY` / `ICE_BASE_URL` | OpenAI 兼容接口的 API key 和 base url |
| `CONFIDENCE_THRESHOLD` | RCA 置信度阈值（0-1，默认 0.3），低于这个值不会进入 Heal 阶段 |

## 事件状态（`status`）一览

| 阶段 | 继续下一步 | 停在这里 |
|---|---|---|
| Monitor | `investigating` | `duplicate_alert` / `no_anomaly` / `monitor_failed` |
| RCA | `analyzing_cause` | `low_confidence` / `rca_failed` |
| Heal | `SUCCESS` / `PENDING_APPROVAL`（进入 Change） | `FAILED` |
| Change（终点） | — | `resolved` / `pending_approval` / `rejected` / `change_failed` |
