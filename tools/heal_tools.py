import logging
from typing import Optional
from datetime import datetime, timedelta
from enum import Enum
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class HealLevel(str, Enum):
    """修复动作的风险级别"""
    L0_AUTO = "L0_AUTO"       # 全自动，无需审批
    L1_SEMI = "L1_SEMI"       # 半自动，需要 oncall 审批
    L2_MANUAL = "L2_MANUAL"   # 纯人工，不能自动执行


# 修复方案库（生产环境中可以从数据库或 RAG 检索）
REMEDIATION_PLAYBOOKS = {
    "restart_pod": {
        "level": HealLevel.L0_AUTO,
        "description": "重启故障 Pod",
        "command_template": "kubectl rollout restart deployment/{service} -n {namespace}",
        "estimated_duration_sec": 30,
        "blast_radius_factor": 0.05,
        "rollback_command": None,
    },
    "scale_up": {
        "level": HealLevel.L0_AUTO,
        "description": "水平扩容 Pod 副本数",
        "command_template": "kubectl scale deployment/{service} --replicas={replicas} -n {namespace}",
        "estimated_duration_sec": 60,
        "blast_radius_factor": 0.02,
        "rollback_command": "kubectl scale deployment/{service} --replicas={original_replicas} -n {namespace}",
    },
    "rollback": {
        "level": HealLevel.L1_SEMI,
        "description": "回滚到上一个稳定版本",
        "command_template": "kubectl rollout undo deployment/{service} -n {namespace}",
        "estimated_duration_sec": 120,
        "blast_radius_factor": 0.15,
        "rollback_command": None,
    },
    "rate_limit": {
        "level": HealLevel.L0_AUTO,
        "description": "启用限流保护",
        "command_template": "kubectl annotate svc/{service} rate-limit={rate} -n {namespace} --overwrite",
        "estimated_duration_sec": 10,
        "blast_radius_factor": 0.01,
        "rollback_command": "kubectl annotate svc/{service} rate-limit- -n {namespace}",
    },
    "circuit_breaker": {
        "level": HealLevel.L0_AUTO,
        "description": "开启熔断器",
        "command_template": "kubectl annotate svc/{service} circuit-breaker=open -n {namespace} --overwrite",
        "estimated_duration_sec": 5,
        "blast_radius_factor": 0.08,
        "rollback_command": "kubectl annotate svc/{service} circuit-breaker=closed -n {namespace} --overwrite",
    },
    "rollback_config": {
        "level": HealLevel.L1_SEMI,
        "description": "回滚配置变更",
        "command_template": "kubectl rollout undo configmap/{service}-config -n {namespace}",
        "estimated_duration_sec": 30,
        "blast_radius_factor": 0.10,
        "rollback_command": None,
    },
    "heap_dump": {
        "level": HealLevel.L2_MANUAL,
        "description": "执行堆转储分析",
        "command_template": "kubectl exec {pod} -- jmap -dump:live,format=b,file=/tmp/heap.hprof 1",
        "estimated_duration_sec": 300,
        "blast_radius_factor": 0.0,
        "rollback_command": None,
    },
}


class CircuitBreaker:
    """熔断器模式

    面试要点：
    - CLOSED → 正常状态，失败计数
    - OPEN → 熔断状态，拒绝所有请求
    - HALF_OPEN → 试探状态，允许少量请求通过
    """

    def __init__(self, threshold: int = 5, timeout_sec: int = 60):
        self._threshold = threshold
        self._timeout = timedelta(seconds=timeout_sec)
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "CLOSED"

    def allow_request(self) -> bool:
        if self._state == "CLOSED":
            return True
        if self._state == "OPEN":
            if datetime.utcnow() - self._last_failure_time >= self._timeout:
                self._state = "HALF_OPEN"
                return True
            return False
        return True  # HALF_OPEN

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        if self._failure_count >= self._threshold:
            self._state = "OPEN"
            logger.warning("[CircuitBreaker] Circuit OPEN — auto-heal suspended")

    @property
    def state(self) -> str:
        return self._state


# 全局熔断器实例
circuit_breaker = CircuitBreaker()

@tool(parse_docstring=True)
def check_circuit_breaker_status() -> dict:
    """
    检查熔断器状态，判断是否允许继续自愈。

    Returns:
        当前熔断器状态和是否允许继续
    """
    # 查看当前是否允许
    allowed = circuit_breaker.allow_request()

    return {
        "state": circuit_breaker.state,
        "allowed": allowed,
        "explanation": "CLOSED=正常可执行 / OPEN=熔断已禁止 / HALF_OPEN=试探阶段"
    }


@tool(parse_docstring=True)
def match_remediation_playbook(suggested_actions: list[str]) -> dict:
    """
    根据建议动作匹配最合适的 Playbook（优先选择低风险的 L0 动作）。

    Args:
        suggested_actions: RCA 建议的动作列表（如 ["rollback", "restart_pod"]）

    Returns:
        匹配到的具体修复方案
    """
    # 从 REMEDIATION_PLAYBOOKS 中筛选出存在的动作
    matched = [
        (action, REMEDIATION_PLAYBOOKS[action])
        for action in suggested_actions
        if action in REMEDIATION_PLAYBOOKS
    ]

    # 如果没找到任何匹配，返回失败
    if not matched:
        return {
            "action": None,
            "message": f"没有找到匹配的 Playbook，建议动作：{suggested_actions}"
        }

    # 优先选择风险最低的(0->1->2)
    level_priority = {"L0_AUTO": 0, "L1_SEMI": 1, "L2_MANUAL": 2}
    matched.sort(key=lambda x: level_priority.get(x[1]["level"].value, 999))
    action, playbook = matched[0]

    # 返回选中的方案
    return {
        "action": action,
        "level": playbook["level"].value,
        "description": playbook["description"],
        "blast_radius_factor": playbook["blast_radius_factor"],
        "command_template": playbook["command_template"],
        "estimated_duration_sec": playbook["estimated_duration_sec"],
    }


@tool(parse_docstring=True)
def simulate_dry_run(action_type: str, service_name: str) -> dict:
    """
    模拟执行修复命令，不实际修改系统。

    Args:
        action_type: 修复动作类型（如 "rollback"、"restart_pod"）
        service_name: 目标服务名称

    Returns:
        模拟执行的结果（SUCCESS/FAILED）和输出信息
    """
    # 查找 Playbook
    playbook = REMEDIATION_PLAYBOOKS.get(action_type)
    if not playbook:
        return {
            "status": "FAILED",
            "output": f"未知的修复动作：{action_type}"
        }

    # 替换命令模板中的变量
    command = playbook["command_template"].format(
        service=service_name,
        namespace="production",
        replicas=3,
        original_replicas=2,
        rate="100req/s",
        pod="pod-0"
    )
    # 模拟执行（这里简化为总是成功）
    return {
        "status": "SUCCESS",
        "output": f"DRY-RUN OK: 模拟执行命令 `{command}` 成功，耗时约 {playbook['estimated_duration_sec']}s，预计影响范围 {playbook['blast_radius_factor'] * 100:.1f}%"
    }


@tool(parse_docstring=True)
def record_heal_result(success: bool) -> dict:
    """
    记录自愈执行结果，更新熔断器状态。

    Args:
        success: 本次修复是否成功

    Returns:
        更新后的熔断器状态
    """
    # 根据结果更新熔断器
    if success:
        circuit_breaker.record_success()
    else:
        circuit_breaker.record_failure()

    # 返回更新后的状态
    return {
        "status": "recorded",
        "circuit_breaker_state": circuit_breaker.state,
        "failure_count": circuit_breaker._failure_count,
    }