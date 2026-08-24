from typing import TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage, add_messages
from datetime import datetime

class IncidentState(TypedDict):
    incident_id: str # 事件ID(唯一标识)
    status: str #状态: open/investigating/healing/resolved/pending_approval
    created_at: datetime # 创建时间
    updated_at: datetime # 更新时间

    # 输入数据
    metric_data: dict # 监控指标数据(Monitor Agent的输入)

    # 各 Agent 的输出结果
    alert_event: Optional[dict]  # Monitor Agent 的输出
    rca_result: Optional[dict]  # RCA Agent 的输出
    heal_action: Optional[dict]  # Heal Agent 的输出
    change_decision: Optional[dict]  # Change Agent 的输出

    # LLM 消息链(用于追踪ReAct过程)
    messages: Annotated[list[BaseMessage], add_messages]


def create_initial_state(incident_id: str, metric_data: dict) -> IncidentState:
    """创建初始状态"""
    now = datetime.now()
    return {
        "incident_id": incident_id,
        "status": "open",
        "created_at": now,
        "updated_at": now,
        "metric_data": metric_data,
        "alert_event": None,
        "rca_result": None,
        "heal_action": None,
        "change_decision": None,
        "messages": [],
    }
