from langgraph.graph import StateGraph, END
from state import IncidentState
from agents import create_monitor_agent, create_rca_agent, create_heal_agent, create_change_agent
import logging
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

# 创建StateGraph传入全局状态类型
graph = StateGraph(IncidentState)

# 初始化四个agent
monitor_agent = create_monitor_agent()
rca_agent = create_rca_agent()
heal_agent = create_heal_agent()
change_agent = create_change_agent()

# 为四个agent定义结点
def monitor_node(state: IncidentState) -> IncidentState:
    # 调用agent
    result = monitor_agent.invoke(HumanMessage(content=f"分析这个监控告警数据：{state['metric_data']}"))

    return {
        "alert_event": result.get("output"),
        "status": "investigating",
        "messages": state["messages"] + [HumanMessage(content="Monitor Agent 分析完成")],
    }


def rca_node(state: IncidentState) -> IncidentState:
    # 调用agent
    result = rca_agent.invoke(HumanMessage(content=f"根因分析。告警数据：{state['alert_event']}"))

    return {
        "rca_result": result.get("output"),
        "status": "analyzing_cause",
        "messages": state["messages"] + [HumanMessage(content="RCA Agent 分析完成")],
    }


def heal_node(state: IncidentState) -> IncidentState:
    # 调用agent
    result = heal_agent.invoke(HumanMessage(content=f"执行自愈。根因分析结果：{state['rca_result']}"))

    return {
        "heal_action": result.get("output"),
        "status": "healing",
        "messages": state["messages"] + [HumanMessage(content="Heal Agent 执行完成")],
    }


def change_node(state: IncidentState) -> IncidentState:
    # 调用agent
    result = change_agent.invoke(HumanMessage(content=f"评估变更风险和批准。修复动作：{state['heal_action']}"))

    return {
        "change_decision": result.get("output"),
        "status": "resolved",
        "messages": state["messages"] + [HumanMessage(content="Change Agent 决策完成")],
    }
