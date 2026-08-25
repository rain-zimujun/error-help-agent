from enum import Enum
from datetime import datetime
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)

class RiskScorer:
    """风险评分模型

    综合多维度因素计算变更风险分数 (0-1)：
    - 爆炸半径权重: 0.30
    - 操作类型权重: 0.25
    - 时间窗口权重: 0.20
    - 历史成功率权重: 0.15
    - 服务关键度权重: 0.10
    """

    OPERATION_RISK = {
        "restart_pod": 0.2,
        "scale_up": 0.15,
        "rate_limit": 0.1,
        "circuit_breaker": 0.1,
        "rollback_config": 0.5,
        "rollback": 0.6,
        "heap_dump": 0.05,
    }

    CRITICAL_SERVICES = {
        "payment-service": 1.0,
        "order-service": 0.9,
        "user-service": 0.8,
        "inventory-service": 0.7,
    }

    def compute(
        self,
        action_type: str,
        blast_radius: float,
        target_service: str,
        hour: int,
    ) -> float:
        op_risk = self.OPERATION_RISK.get(action_type, 0.5)
        svc_criticality = self.CRITICAL_SERVICES.get(target_service, 0.5)

        # 非工作时间风险加成（凌晨/周末操作风险更高，因为响应人力少）
        time_risk = 0.3 if (hour < 8 or hour > 22) else 0.1

        # 模拟历史成功率（实际应从数据库查询）
        history_risk = 0.1

        score = (
            0.30 * blast_radius
            + 0.25 * op_risk
            + 0.20 * time_risk
            + 0.15 * history_risk
            + 0.10 * svc_criticality
        )
        return round(min(score, 1.0), 3)


class ApprovalLevel(str, Enum):
    """批准级别"""
    AUTO_APPROVE = "AUTO_APPROVE"
    NEED_ONCALL_APPROVAL = "NEED_ONCALL_APPROVAL"
    REJECT = "REJECT"

# 批准阈值
APPROVAL_THRESHOLDS = {
    "auto_approve_max": 3.0,
    # 风险 < 3.0 → 自动批准
    "oncall_approval_max": 6.5,
    # 风险 3.0-6.5 → 需要 oncall 审批
    "reject_min": 6.5,
    # 风险 >= 6.5 → 拒绝
}

score = RiskScorer()
@tool(parse_docstring=True)
def calculate_risk_score(
      action_type: str,
      blast_radius: float,
      target_service: str,
      estimated_duration_sec: int = 30
) -> dict:
    """
    计算修复方案的综合风险分数（0-10）。

    综合考虑：操作类型、爆炸半径、服务关键度、时间窗口等因素。

    Args:
        action_type: 修复动作（restart_pod / rollback / scale_up 等）
        blast_radius: 爆炸半径（0-1）
        target_service: 目标服务名称
        estimated_duration_sec: 预计耗时（秒）

    Returns:
        风险分数（0-10）及详细分析
    """
    # 获取当前小时(用于判断是否非工作时间)
    current_hour = datetime.now().hour
    # 调用RiskScorer.compute()获得 0-1 范围的分数
    normalized_score = score.compute(action_type=action_type,blast_radius=blast_radius,target_service=target_service,hour=current_hour)
    # 转换为 0-10 范围
    risk_score = normalized_score * 10.0

    return {
        "risk_score": round(risk_score, 2),
        "action_type": action_type,
        "blast_radius": blast_radius,
        "target_service": target_service,
        "estimated_duration_sec": estimated_duration_sec,
        "current_hour": current_hour,
        "time_window": "非工作时间" if (current_hour < 8 or current_hour > 22) else "工作时间",
        "explanation": f"综合风险分数={round(risk_score, 2)}/10.0 (action={action_type}, service={target_service}, blast_radius={blast_radius})"
    }

@tool(parse_docstring=True)
def apply_approval_policy(
    risk_score: float,
    heal_level: str,
    incident_severity: str
) -> dict:
    """
    根据风险分数和修复级别，决定是否批准修复。

    规则：
    - 低风险（score < 3.0）：自动批准
    - 中风险（3.0 <= score < 6.5）：需要 oncall 审批（除非 L0_AUTO + CRITICAL）
    - 高风险（score >= 6.5）：拒绝

    特殊情况：L0_AUTO 级别 + CRITICAL 告警 → 紧急自动批准

    Args:
        risk_score: 风险分数（0-10）
        heal_level: 修复级别（L0_AUTO / L1_SEMI / L2_MANUAL）
        incident_severity: 告警严重程度（CRITICAL / HIGH / MEDIUM / LOW）

    Returns:
        批准决策、原因说明和建议行动
    """
    # 根据风险分数判断基础审批级别
    if risk_score < APPROVAL_THRESHOLDS["auto_approve_max"]:
        base_approval = ApprovalLevel.AUTO_APPROVE
    elif risk_score < APPROVAL_THRESHOLDS["oncall_approval_max"]:
        base_approval = ApprovalLevel.NEED_ONCALL_APPROVAL
    else:
        base_approval = ApprovalLevel.REJECT

    # 检查特殊情况, 如果是 L0_AUTO + CRITICAL -> 紧急自动批准
    final_approval = base_approval
    if(base_approval == ApprovalLevel.NEED_ONCALL_APPROVAL and heal_level == "L0_AUTO" and incident_severity == "CRITICAL"):
        final_approval = ApprovalLevel.AUTO_APPROVE

    # 根据最终审批级别来决定建议行动
    if final_approval == ApprovalLevel.AUTO_APPROVE:
        recommendation = "立即执行修复"
    elif final_approval == ApprovalLevel.NEED_ONCALL_APPROVAL:
        recommendation = "通知 oncall 值班人员进行审批"
    else:
        recommendation = "拒绝执行, 建议人工分析"

    return {
        "approval": final_approval.value,
        "approved": final_approval != ApprovalLevel.REJECT,
        "risk_score": risk_score,
        "heal_level": heal_level,
        "incident_severity": incident_severity,
        "recommendation": recommendation,
        "explanation": f"风险分数={risk_score}/10，修复级别={heal_level}，告警级别={incident_severity} → {final_approval.value}"
    }

@tool(parse_docstring=True)
def notify_oncall(
    incident_id: str,
    service_name: str,
    risk_score: float,
    proposed_action: str,
    approval_timeout_sec: int = 1800
) -> dict:
    """
    通知 oncall 值班人员，请求审批修复方案。

    在实际部署时，这会调用 PagerDuty / Slack / 钉钉 等通知系统。

    Args:
        incident_id: 告警事件 ID
        service_name: 故障服务名称
        risk_score: 风险分数
        proposed_action: 建议的修复动作
        approval_timeout_sec: 审批超时时间（秒，默认 30 分钟）

    Returns:
        通知状态和追踪 ID
    """
    # 生成追踪 ID（用于追踪这次通知）
    from datetime import datetime as dt
    timestamp = dt.now().timestamp()
    notification_id = f"notify_{incident_id}_{int(timestamp)}"

    # ② 构造通知消息
    message = f"""                                                                                                                                                                                                               
    🚨 异常自愈系统需要您的审批                                                                                                                                                                                                    

    📊 告警信息：                                                                                                                                                                                                                    
    - 事件 ID: {incident_id}                                                                                                                                                                                                         
    - 故障服务: {service_name}                                                                                                                                                                                                       
    - 风险分数: {risk_score:.2f}/10.0                                                                                                                                                                                                

    💊 建议修复:                                                                                                                                                                                                                     
    - 动作: {proposed_action}                                                                                                                                                                                                        

    ⏱️ 请在 {approval_timeout_sec // 60} 分钟内审批（否则自动拒绝）                                                                                                                                                                  
        """

    return {
        "status": "NOTIFIED",
        "notification_id": notification_id,
        "incident_id": incident_id,
        "service_name": service_name,
        "proposed_action": proposed_action,
        "message_preview": message.strip(),
        "timeout_seconds": approval_timeout_sec,
        "channel": "pagerduty/slack",
        "auto_reject_on_timeout": True,
    }