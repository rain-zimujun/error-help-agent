"""
测试 graph.py 里的条件边路由逻辑。

思路：把 graph.py 里的 4 个 agent 对象（monitor_agent/rca_agent/heal_agent/change_agent）
用 mock 替换掉，让它们的 .invoke() 直接返回指定的 structured_response，
这样真实的 StateGraph（节点函数、条件边、checkpointer）会完整跑一遍，
但不会真的发起 LLM 调用，跑得快、不烧 token、结果确定。
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import graph as graph_module
from state import create_initial_state


SAMPLE_METRIC_DATA = {
    "metric_history": [20, 21, 19, 22, 20, 21, 20, 22, 21, 20],
    "current_value": 95,
    "alert_name": "high_cpu_usage",
    "target_service": "order-service",
    "labels": {"pod": "order-service-xyz", "namespace": "production"},
}


def mock_agent(structured_response: dict) -> MagicMock:
    """构造一个假 agent，.invoke() 不管传什么，都直接返回指定的 structured_response"""
    agent = MagicMock()
    agent.invoke.return_value = {"structured_response": structured_response}
    return agent


@contextmanager
def patched_agents(**agents):
    """
    批量替换 graph_module 里的 agent 对象。
    用法：with patched_agents(monitor_agent=mock_agent({...}), rca_agent=mock_agent({...})):
    没传的 agent 保持原样（如果流程真的调用到它，会真实调 LLM，所以测试里该 mock 的都要 mock 全）
    """
    with ExitStack() as stack:
        for name, agent in agents.items():
            stack.enter_context(patch.object(graph_module, name, agent))
        yield


def run_graph(thread_id: str):
    """用固定的示例数据跑一次 graph"""
    state = create_initial_state(incident_id=thread_id, metric_data=SAMPLE_METRIC_DATA)
    return graph_module.graph.invoke(state, config={"configurable": {"thread_id": thread_id}})


# 几个可复用的"正常通过"假数据
PASSING_ALERT_EVENT = {
    "is_anomaly": True, "is_duplicate": False, "severity": "CRITICAL",
    "anomaly_score": 10.0, "service": "order-service", "alert_name": "high_cpu_usage",
}
PASSING_RCA_RESULT = {
    "root_cause": "测试用根因", "confidence": 0.9,
    "affected_services": ["order-service"], "suggested_actions": ["restart_pod"],
}


class TestMonitorRouting:
    """monitor -> rca 的条件边：is_anomaly=True 且 is_duplicate=False 才继续"""

    def test_stops_when_duplicate(self):
        with patched_agents(monitor_agent=mock_agent({
            **PASSING_ALERT_EVENT, "is_duplicate": True,
        })):
            result = run_graph("test-monitor-dup")

        assert result["status"] == "duplicate_alert"
        assert result.get("rca_result") is None

    def test_stops_when_not_anomaly(self):
        with patched_agents(monitor_agent=mock_agent({
            **PASSING_ALERT_EVENT, "is_anomaly": False,
        })):
            result = run_graph("test-monitor-noanomaly")

        assert result["status"] == "no_anomaly"
        assert result.get("rca_result") is None

    def test_stops_when_monitor_returns_nothing(self):
        """LLM 没能生成 structured_response 的兜底情况"""
        with patched_agents(monitor_agent=mock_agent(None)):
            result = run_graph("test-monitor-empty")

        assert result["status"] == "monitor_failed"
        assert result.get("rca_result") is None

    def test_continues_to_rca_when_anomaly_and_not_duplicate(self):
        # 故意让 rca 返回低置信度，这样流程会在 rca 停下，不需要继续 mock heal/change
        with patched_agents(
            monitor_agent=mock_agent(PASSING_ALERT_EVENT),
            rca_agent=mock_agent({**PASSING_RCA_RESULT, "confidence": 0.0}),
        ):
            result = run_graph("test-monitor-continue")

        assert result["status"] == "low_confidence"
        assert result["rca_result"] is not None


class TestRcaRouting:
    """rca -> heal 的条件边：confidence >= CONFIDENCE_THRESHOLD 才继续"""

    def test_stops_when_confidence_below_threshold(self):
        with patched_agents(
            monitor_agent=mock_agent(PASSING_ALERT_EVENT),
            rca_agent=mock_agent({
                **PASSING_RCA_RESULT,
                "confidence": graph_module.CONFIDENCE_THRESHOLD - 0.01,
            }),
        ):
            result = run_graph("test-rca-lowconf")

        assert result["status"] == "low_confidence"
        assert result.get("heal_action") is None

    def test_continues_to_heal_when_confidence_meets_threshold(self):
        # 故意让 heal 返回 FAILED，这样流程会在 heal 停下，不需要继续 mock change
        with patched_agents(
            monitor_agent=mock_agent(PASSING_ALERT_EVENT),
            rca_agent=mock_agent({
                **PASSING_RCA_RESULT,
                "confidence": graph_module.CONFIDENCE_THRESHOLD,
            }),
            heal_agent=mock_agent({
                "action": "restart_pod", "level": "L0_AUTO", "blast_radius": 0.05,
                "status": "FAILED", "dry_run_output": "模拟失败", "circuit_breaker_state": "CLOSED",
            }),
        ):
            result = run_graph("test-rca-continue")

        assert result["status"] == "FAILED"
        assert result["heal_action"] is not None


class TestHealRouting:
    """heal -> change 的条件边：status 是 SUCCESS 或 PENDING_APPROVAL 才继续"""

    def test_stops_when_heal_failed(self):
        with patched_agents(
            monitor_agent=mock_agent(PASSING_ALERT_EVENT),
            rca_agent=mock_agent(PASSING_RCA_RESULT),
            heal_agent=mock_agent({
                "action": "restart_pod", "level": "L0_AUTO", "blast_radius": 0.05,
                "status": "FAILED", "dry_run_output": "模拟失败", "circuit_breaker_state": "CLOSED",
            }),
        ):
            result = run_graph("test-heal-failed")

        assert result["status"] == "FAILED"
        assert result.get("change_decision") is None

    def test_stops_when_heal_returns_nothing(self):
        with patched_agents(
            monitor_agent=mock_agent(PASSING_ALERT_EVENT),
            rca_agent=mock_agent(PASSING_RCA_RESULT),
            heal_agent=mock_agent(None),
        ):
            result = run_graph("test-heal-empty")

        # heal_node 的兜底状态用的是 LLM 词汇表的大写 "FAILED"，不是别的节点那种 snake_case
        assert result["status"] == "FAILED"
        assert result.get("change_decision") is None

    def test_continues_to_change_when_success(self):
        with patched_agents(
            monitor_agent=mock_agent(PASSING_ALERT_EVENT),
            rca_agent=mock_agent(PASSING_RCA_RESULT),
            heal_agent=mock_agent({
                "action": "scale_up", "level": "L0_AUTO", "blast_radius": 0.02,
                "status": "SUCCESS", "dry_run_output": "模拟成功", "circuit_breaker_state": "CLOSED",
            }),
            change_agent=mock_agent({
                "approval": "AUTO_APPROVE", "risk_score": 1.5, "recommendation": "低风险，自动批准",
            }),
        ):
            result = run_graph("test-heal-success")

        assert result["status"] == "resolved"
        assert result["change_decision"] is not None

    def test_continues_to_change_when_pending_approval(self):
        with patched_agents(
            monitor_agent=mock_agent(PASSING_ALERT_EVENT),
            rca_agent=mock_agent(PASSING_RCA_RESULT),
            heal_agent=mock_agent({
                "action": "rollback", "level": "L1_SEMI", "blast_radius": 0.15,
                "status": "PENDING_APPROVAL", "dry_run_output": "模拟成功", "circuit_breaker_state": "CLOSED",
            }),
            change_agent=mock_agent({
                "approval": "NEED_ONCALL_APPROVAL", "risk_score": 4.0,
                "notification_id": "notify_test_123", "recommendation": "中风险，需要审批",
            }),
        ):
            result = run_graph("test-heal-pending")

        assert result["status"] == "pending_approval"


class TestChangeStatusMapping:
    """change_node 内部：approval 值 -> 最终 incident status 的映射"""

    def _run_with_change_response(self, thread_id: str, change_response: dict):
        with patched_agents(
            monitor_agent=mock_agent(PASSING_ALERT_EVENT),
            rca_agent=mock_agent(PASSING_RCA_RESULT),
            heal_agent=mock_agent({
                "action": "scale_up", "level": "L0_AUTO", "blast_radius": 0.02,
                "status": "SUCCESS", "dry_run_output": "模拟成功", "circuit_breaker_state": "CLOSED",
            }),
            change_agent=mock_agent(change_response),
        ):
            return run_graph(thread_id)

    def test_auto_approve_maps_to_resolved(self):
        result = self._run_with_change_response("test-change-auto", {
            "approval": "AUTO_APPROVE", "risk_score": 1.0, "recommendation": "go",
        })
        assert result["status"] == "resolved"

    def test_need_oncall_maps_to_pending_approval(self):
        result = self._run_with_change_response("test-change-oncall", {
            "approval": "NEED_ONCALL_APPROVAL", "risk_score": 4.0,
            "notification_id": "notify_123", "recommendation": "review",
        })
        assert result["status"] == "pending_approval"

    def test_reject_maps_to_rejected(self):
        result = self._run_with_change_response("test-change-reject", {
            "approval": "REJECT", "risk_score": 8.0, "recommendation": "no",
        })
        assert result["status"] == "rejected"

    def test_stops_when_change_returns_nothing(self):
        result = self._run_with_change_response("test-change-empty", None)
        assert result["status"] == "change_failed"


class TestCheckpointer:
    """验证 checkpointer 真的把状态存下来了，并且能按 thread_id 查到"""

    def test_state_is_saved_and_reachable_after_run(self):
        thread_id = "test-checkpoint-verify"
        with patched_agents(monitor_agent=mock_agent({
            **PASSING_ALERT_EVENT, "is_anomaly": False,
        })):
            run_graph(thread_id)

        saved = graph_module.graph.get_state({"configurable": {"thread_id": thread_id}})

        assert saved.values["status"] == "no_anomaly"
        assert saved.next == ()  # 空元组说明已经跑到终点，没有待执行节点

    def test_different_thread_ids_do_not_interfere(self):
        with patched_agents(monitor_agent=mock_agent({
            **PASSING_ALERT_EVENT, "is_duplicate": True,
        })):
            run_graph("test-checkpoint-a")

        with patched_agents(monitor_agent=mock_agent({
            **PASSING_ALERT_EVENT, "is_anomaly": False,
        })):
            run_graph("test-checkpoint-b")

        state_a = graph_module.graph.get_state({"configurable": {"thread_id": "test-checkpoint-a"}})
        state_b = graph_module.graph.get_state({"configurable": {"thread_id": "test-checkpoint-b"}})

        assert state_a.values["status"] == "duplicate_alert"
        assert state_b.values["status"] == "no_anomaly"
