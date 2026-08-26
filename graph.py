from langgraph.graph import StateGraph, END, START
from state import IncidentState
from agents import create_monitor_agent, create_rca_agent, create_heal_agent, create_change_agent
import logging
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

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

    agent_output = None
    if "messages" in result and result["messages"]:
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "content"):
            agent_output = last_msg.content

    return {
        "alert_event": agent_output,
        "status": "investigating",
        "messages": state["messages"] + [HumanMessage(content="Monitor Agent 分析完成")],
    }


def rca_node(state: IncidentState) -> IncidentState:
    print("[RCA Agent] 开始根因分析...")
    result = rca_agent.invoke({
        "messages": [HumanMessage(content=f"根因分析。告警：{state['alert_event']}")]
    })
    print("[RCA Agent] 完成")

    agent_output = None
    if "messages" in result and result["messages"]:
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "content"):
            agent_output = last_msg.content

    return {
        "rca_result": agent_output,
        "status": "analyzing_cause",
        "messages": state["messages"] + [HumanMessage(content="RCA Agent 分析完成")],
    }


def heal_node(state: IncidentState) -> IncidentState:
    print("[Heal Agent] 开始执行自愈...")
    result = heal_agent.invoke({
        "messages": [HumanMessage(content=f"执行自愈。根因：{state['rca_result']}")]
    })
    print("[Heal Agent] 完成")

    agent_output = None
    if "messages" in result and result["messages"]:
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "content"):
            agent_output = last_msg.content

    return {
        "heal_action": agent_output,
        "status": "healing",
        "messages": state["messages"] + [HumanMessage(content="Heal Agent 执行完成")],
    }


def change_node(state: IncidentState) -> IncidentState:
    print("[Change Agent] 开始批准决策...")
    result = change_agent.invoke({
        "messages": [HumanMessage(content=f"批准决策。修复：{state['heal_action']}")]
    })
    print("[Change Agent] 完成")

    agent_output = None
    if "messages" in result and result["messages"]:
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "content"):
            agent_output = last_msg.content

    return {
        "change_decision": agent_output,
        "status": "resolved",
        "messages": state["messages"] + [HumanMessage(content="Change Agent 决策完成")],
    }

# 添加结点
builder.add_node("monitor", monitor_node)
builder.add_node("rca", rca_node)
builder.add_node("heal", heal_node)
builder.add_node("change", change_node)

# 添加边
builder.add_edge(START, "monitor")
builder.add_edge("monitor", "rca")
builder.add_edge("rca", "heal")
builder.add_edge("heal", "change")
builder.add_edge("change", END)

graph = builder.compile()

if __name__ == "__main__":
    from state import create_initial_state

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

    result = graph.invoke(test_state)

    print(f"\n最终状态: {result['status']}")
    print(f"\nAlert Event:\n{result.get('alert_event', 'N/A')[:500]}...")
    print(f"\nRCA Result:\n{result.get('rca_result', 'N/A')[:500]}...")
    print(f"\nHeal Action:\n{result.get('heal_action', 'N/A')[:500]}...")
    print(f"\nChange Decision:\n{result.get('change_decision', 'N/A')[:500]}...")