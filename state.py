from typing_extensions import TypedDict
from typing import Optional
from langchain_core.messages import BaseMessage
from datetime import datetime


class MetricData(TypedDict):
    """Monitor Agent 的输入数据结构"""
    metric_history: list[float]  # 历史指标值列表，用于异常检测算法建立基线
    current_value: float         # 当前采集到的指标值
    alert_name: str              # 告警类型，需与 rca_tools.FAULT_PATTERNS 里定义的 key 一致（如 "high_cpu_usage"）
    target_service: str          # 故障服务名称
    labels: dict[str, str]       # 附加标签（如 pod、namespace），用于告警去重指纹


class IncidentState(TypedDict):
    incident_id: str # 事件ID(唯一标识)
    status: str #状态: open/investigating/healing/resolved/pending_approval
    created_at: datetime # 创建时间
    updated_at: datetime # 更新时间

    # 输入数据
    metric_data: MetricData # 监控指标数据(Monitor Agent的输入)

    # 各 Agent 的输出结果
    alert_event: Optional[dict]  # Monitor Agent 的输出
    rca_result: Optional[dict]  # RCA Agent 的输出
    heal_action: Optional[dict]  # Heal Agent 的输出
    change_decision: Optional[dict]  # Change Agent 的输出

    # LLM 消息链(用于追踪ReAct过程)
    messages: list[BaseMessage]



def create_initial_state(incident_id: str, metric_data: MetricData) -> IncidentState:
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
