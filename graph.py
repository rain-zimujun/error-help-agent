from langgraph.graph import StateGraph, END, START
from state import IncidentState
from agents import create_monitor_agent, create_rca_agent, create_heal_agent, create_change_agent
import logging
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

logger = logging.getLogger(__name__)

# 从环境变量读取置信度阈值，默认值为 0.3
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

# 创建StateGraph传入全局状态类型
builder = StateGraph(IncidentState)

# 初始化四个agent
monitor_agent = create_monitor_agent()
rca_agent = create_rca_agent()
heal_agent = create_heal_agent()
change_agent = create_change_agent()

# 为四个agent定义结点
def monitor_node(state: IncidentState) -> IncidentState:
    print("[Monitor Agent] 开始分析告警...")
    result = monitor_agent.invoke({
        "messages": [HumanMessage(content=f"分析告警数据：{state['metric_data']}")]
    })
    print("[Monitor Agent] 完成")

    agent_output = result.get("structured_response")

    # 根据 alert_event 的判定结果动态设置 incident 状态：
    # 是否异常、是否重复，决定了会不会继续进入 rca
    if agent_output and isinstance(agent_output, dict):
        if agent_output.get("is_duplicate"):
            incident_status = "duplicate_alert"
        elif not agent_output.get("is_anomaly"):
            incident_status = "no_anomaly"
        else:
            incident_status = "investigating"
    else:
        incident_status = "monitor_failed"

    return {
        "alert_event": agent_output,
        "status": incident_status,
        "messages": state["messages"] + [HumanMessage(content="Monitor Agent 分析完成")],
    }


def rca_node(state: IncidentState) -> IncidentState:
    print("[RCA Agent] 开始根因分析...")
    result = rca_agent.invoke({
        "messages": [HumanMessage(content=f"根因分析。告警：{state['alert_event']}")]
    })
    print("[RCA Agent] 完成")

    agent_output = result.get("structured_response")

    # 置信度是否达标决定了会不会继续进入 heal
    if agent_output and isinstance(agent_output, dict):
        if agent_output.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
            incident_status = "analyzing_cause"
        else:
            incident_status = "low_confidence"
    else:
        incident_status = "rca_failed"

    return {
        "rca_result": agent_output,
        "status": incident_status,
        "messages": state["messages"] + [HumanMessage(content="RCA Agent 分析完成")],
    }


def heal_node(state: IncidentState) -> IncidentState:
    print("[Heal Agent] 开始执行自愈...")
    result = heal_agent.invoke({
        "messages": [HumanMessage(content=f"执行自愈。根因：{state['rca_result']}")]
    })
    print("[Heal Agent] 完成")

    agent_output = result.get("structured_response")

    # 根据 heal_action.status 动态设置 incident 状态，
    # 避免 FAILED/没有返回值时 status 还停留在 "healing"，事后看着像卡住了而不是已经失败结束
    if agent_output and isinstance(agent_output, dict):
        heal_status = agent_output.get("status")
    else:
        heal_status = "FAILED"

    return {
        "heal_action": agent_output,
        "status": heal_status,
        "messages": state["messages"] + [HumanMessage(content="Heal Agent 执行完成")],
    }


def change_node(state: IncidentState) -> IncidentState:
    print("[Change Agent] 开始批准决策...")
    result = change_agent.invoke({
        "messages": [HumanMessage(content=f"批准决策。修复：{state['heal_action']}")]
    })
    print("[Change Agent] 完成")

    agent_output = result.get("structured_response")

    # change 是终点，approval 结果直接映射成最终的 incident 状态
    if agent_output and isinstance(agent_output, dict):
        approval = agent_output.get("approval")
        if approval == "AUTO_APPROVE":
            incident_status = "resolved"
        elif approval == "NEED_ONCALL_APPROVAL":
            incident_status = "pending_approval"
        elif approval == "REJECT":
            incident_status = "rejected"
        else:
            incident_status = "change_failed"
    else:
        incident_status = "change_failed"

    return {
        "change_decision": agent_output,
        "status": incident_status,
        "messages": state["messages"] + [HumanMessage(content="Change Agent 决策完成")],
    }

# 添加结点
builder.add_node("monitor", monitor_node)
builder.add_node("rca", rca_node)
builder.add_node("heal", heal_node)
builder.add_node("change", change_node)

# 添加边
builder.add_edge(START, "monitor")
# builder.add_edge("monitor", "rca")
# builder.add_edge("rca", "heal")
# builder.add_edge("heal", "change")
# builder.add_edge("change", END)
# 条件边, 没有必要的时候不用调用后面的agent
# 每个 node 已经把"继不继续"的判断结果编码进 status 里了，条件边直接读 status 即可
builder.add_conditional_edges(
    "monitor",
    lambda state: "rca" if state.get("status") == "investigating" else END
)
builder.add_conditional_edges(
    "rca",
    lambda state: "heal" if state.get("status") == "analyzing_cause" else END
)
builder.add_conditional_edges(
    "heal",
    lambda state: "change" if state.get("status") in ("SUCCESS", "PENDING_APPROVAL") else END
)
builder.add_edge("change", END)

checkpoint = InMemorySaver()

graph = builder.compile(checkpointer=checkpoint)

if __name__ == "__main__":
    import sys
    from state import create_initial_state

    # 修复 Windows 终端下中文输出乱码
    sys.stdout.reconfigure(encoding="utf-8")

    test_state = create_initial_state(
        incident_id="test_001",
        metric_data={
            "metric_history": [20, 21, 19, 22, 20, 21, 20, 22, 21, 20],
            "current_value": 95,
            "alert_name": "high_cpu_usage",
            "target_service": "order-service",
            "labels": {"pod": "order-service-xyz", "namespace": "production"}
        }
    )

    print("=" * 60)
    print("完整 Graph 流程测试")
    print("=" * 60)

    thread_config = {"configurable": {"thread_id": "test_001"}}

    try:
        result = graph.invoke(test_state, config=thread_config)
    except Exception as e:
        print(f"\n[graph.invoke 失败] {type(e).__name__}: {e}")
        raise

    print(f"\n最终状态: {result['status']}")
    print(f"\nAlert Event:\n{result.get('alert_event', 'N/A')}")
    print(f"\nRCA Result:\n{result.get('rca_result', 'N/A')}")
    print(f"\nHeal Action:\n{result.get('heal_action', 'N/A')}")
    print(f"\nChange Decision:\n{result.get('change_decision', 'N/A')}")

    # 验证 checkpointer 真的把这次运行的状态存下来了
    saved = graph.get_state(thread_config)
    print("\n" + "=" * 60)
    print("Checkpoint 验证")
    print("=" * 60)
    print(f"是否存在已保存的状态: {saved is not None and bool(saved.values)}")
    print(f"下一步待执行节点（空说明已经跑到终点）: {saved.next}")