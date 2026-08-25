import pytest
from tools.monitor_tools import (
    run_anomaly_detection,
    check_alert_duplicate,
    classify_alert_severity,
    fingerprint_tracker
)


class TestMonitorTools:
    """Monitor 工具测试"""

    def setup_method(self):
        """每个测试前清理 fingerprint_tracker"""
        fingerprint_tracker._recent.clear()

    def test_run_anomaly_detection_normal_data(self):
        """测试正常数据 - 不应该检测到异常"""
        metric_history = [20, 21, 19, 22, 20, 21, 20, 22]
        result = run_anomaly_detection.invoke({
            "metric_history": metric_history,
            "current_value": 21
        })

        assert result["is_anomaly"] == False
        assert result["score"] < 3.0  # 正常分数应该较低
        assert result["votes"] < 2  # 投票数应该少于 2

    def test_run_anomaly_detection_anomaly_data(self):
        """测试异常数据 - 应该检测到异常"""
        # 需要至少 20 个数据点（Isolation Forest 的要求）
        metric_history = [20, 21, 19, 22, 20, 21, 20, 22, 21, 20] * 2  # 20 个点
        result = run_anomaly_detection.invoke({
            "metric_history": metric_history,
            "current_value": 95  # 突然飙升
        })

        assert result["is_anomaly"] == True
        assert result["votes"] >= 2  # 至少 2 个算法投票
        assert "three_sigma" in result["algorithms"]
        assert "ewma" in result["algorithms"]
        assert "isolation_forest" in result["algorithms"]

    def test_run_anomaly_detection_insufficient_data(self):
        """测试数据不足 - 无法判断异常"""
        metric_history = [20, 21]  # 数据太少
        result = run_anomaly_detection.invoke({
            "metric_history": metric_history,
            "current_value": 95
        })

        # 数据不足时，各算法都返回 False
        assert result["is_anomaly"] == False
        assert result["votes"] == 0

    def test_check_alert_duplicate_first_occurrence(self):
        """测试首次告警 - 不应该是重复"""
        result = check_alert_duplicate.invoke({
            "alert_name": "high_cpu_usage",
            "target_service": "order-service",
            "labels": {"pod": "order-service-xyz", "namespace": "production"}
        })

        assert result["is_duplicate"] == False
        assert "首次" in result["reason"]

    def test_check_alert_duplicate_second_occurrence(self):
        """测试重复告警 - 应该检测到重复（5分钟内）"""
        alert_params = {
            "alert_name": "high_cpu_usage",
            "target_service": "order-service",
            "labels": {"pod": "order-service-xyz", "namespace": "production"}
        }

        # 第一次调用
        result1 = check_alert_duplicate.invoke(alert_params)
        assert result1["is_duplicate"] == False

        # 第二次立即调用（应该是重复）
        result2 = check_alert_duplicate.invoke(alert_params)
        assert result2["is_duplicate"] == True
        assert "5分钟内" in result2["reason"]

    def test_check_alert_duplicate_different_service(self):
        """测试不同服务的告警 - 不应该是重复"""
        params1 = {
            "alert_name": "high_cpu_usage",
            "target_service": "order-service",
            "labels": {"pod": "order-xyz"}
        }
        params2 = {
            "alert_name": "high_cpu_usage",
            "target_service": "payment-service",  # 不同服务
            "labels": {"pod": "payment-xyz"}
        }

        result1 = check_alert_duplicate.invoke(params1)
        result2 = check_alert_duplicate.invoke(params2)

        assert result1["is_duplicate"] == False
        assert result2["is_duplicate"] == False  # 不同服务，不是重复

    def test_classify_alert_severity_critical(self):
        """测试严重程度分类 - CRITICAL"""
        result = classify_alert_severity.invoke({"anomaly_score": 8.0})
        assert result["severity"] == "CRITICAL"

    def test_classify_alert_severity_high(self):
        """测试严重程度分类 - HIGH"""
        result = classify_alert_severity.invoke({"anomaly_score": 5.5})
        assert result["severity"] == "HIGH"

    def test_classify_alert_severity_medium(self):
        """测试严重程度分类 - MEDIUM"""
        result = classify_alert_severity.invoke({"anomaly_score": 3.5})
        assert result["severity"] == "MEDIUM"

    def test_classify_alert_severity_low(self):
        """测试严重程度分类 - LOW"""
        result = classify_alert_severity.invoke({"anomaly_score": 1.0})
        assert result["severity"] == "LOW"

    def test_classify_alert_severity_boundary(self):
        """测试边界值"""
        assert classify_alert_severity.invoke({"anomaly_score": 7.0})["severity"] == "CRITICAL"
        assert classify_alert_severity.invoke({"anomaly_score": 6.99})["severity"] == "HIGH"
        assert classify_alert_severity.invoke({"anomaly_score": 5.0})["severity"] == "HIGH"
        assert classify_alert_severity.invoke({"anomaly_score": 4.99})["severity"] == "MEDIUM"
        assert classify_alert_severity.invoke({"anomaly_score": 3.0})["severity"] == "MEDIUM"
        assert classify_alert_severity.invoke({"anomaly_score": 2.99})["severity"] == "LOW"


class TestRCATools:
    """RCA 工具测试"""

    def test_query_service_dependencies(self):
        """测试查询服务依赖"""
        from tools.rca_tools import query_service_dependencies

        result = query_service_dependencies.invoke({
            "service_name": "order-service"
        })

        assert result["service"] == "order-service"
        assert "payment-service" in result["direct_dependencies"]
        assert "inventory-service" in result["direct_dependencies"]
        assert result["count"] > 0

    def test_query_service_dependencies_no_deps(self):
        """测试查询无依赖的服务"""
        from tools.rca_tools import query_service_dependencies

        result = query_service_dependencies.invoke({
            "service_name": "mysql-primary"  # 数据库没有依赖其他服务
        })

        assert result["service"] == "mysql-primary"
        assert result["count"] == 0

    def test_trace_impact_chain(self):
        """测试追踪故障影响链"""
        from tools.rca_tools import trace_impact_chain

        result = trace_impact_chain.invoke({
            "service_name": "order-service",
            "max_depth": 5
        })

        assert result["root_service"] == "order-service"
        assert len(result["dependency_paths"]) > 0
        assert len(result["affected_services"]) > 0
        # order-service 的影响范围应该包括 payment-service 和 mysql
        assert "payment-service" in result["affected_services"]
        assert "mysql-primary" in result["affected_services"]

    def test_trace_impact_chain_leaf_node(self):
        """测试追踪叶子节点（无下游依赖）的影响"""
        from tools.rca_tools import trace_impact_chain

        result = trace_impact_chain.invoke({
            "service_name": "mysql-primary",
            "max_depth": 5
        })

        assert result["root_service"] == "mysql-primary"
        # mysql-primary 没有依赖其他服务，所以 affected_services 应该为空
        assert len(result["affected_services"]) == 0

    def test_find_recent_changes_in_service(self):
        """测试查询近期变更"""
        from tools.rca_tools import find_recent_changes_in_service

        result = find_recent_changes_in_service.invoke({
            "service_name": "order-service",
            "hours": 24
        })

        assert result["service"] == "order-service"
        assert result["time_window_hours"] == 24
        assert isinstance(result["changes"], list)
        assert result["change_count"] >= 0

    def test_find_recent_changes_with_different_timewindow(self):
        """测试不同时间窗口的变更查询"""
        from tools.rca_tools import find_recent_changes_in_service

        result_24h = find_recent_changes_in_service.invoke({
            "service_name": "order-service",
            "hours": 24
        })

        result_1h = find_recent_changes_in_service.invoke({
            "service_name": "order-service",
            "hours": 1
        })

        # 1小时的变更数应该 <= 24小时的
        assert result_1h["change_count"] <= result_24h["change_count"]

    def test_list_fault_candidates_high_cpu(self):
        """测试列出 CPU 告警的根因候选"""
        from tools.rca_tools import list_fault_candidates

        result = list_fault_candidates.invoke({
            "alert_type": "high_cpu_usage"
        })

        assert result["alert_type"] == "high_cpu_usage"
        assert result["candidate_count"] > 0
        assert len(result["candidates"]) > 0

        # 检查候选的结构
        for candidate in result["candidates"]:
            assert "root_cause" in candidate
            assert "evidence_pattern" in candidate
            assert "base_probability" in candidate
            assert "suggested_actions" in candidate
            assert 0 <= candidate["base_probability"] <= 1

    def test_list_fault_candidates_high_memory(self):
        """测试列出内存告警的根因候选"""
        from tools.rca_tools import list_fault_candidates

        result = list_fault_candidates.invoke({
            "alert_type": "high_memory_usage"
        })

        assert result["alert_type"] == "high_memory_usage"
        assert result["candidate_count"] > 0

    def test_list_fault_candidates_high_error_rate(self):
        """测试列出错误率告警的根因候选"""
        from tools.rca_tools import list_fault_candidates

        result = list_fault_candidates.invoke({
            "alert_type": "high_error_rate"
        })

        assert result["alert_type"] == "high_error_rate"
        assert result["candidate_count"] > 0

    def test_list_fault_candidates_unknown_type(self):
        """测试未知告警类型"""
        from tools.rca_tools import list_fault_candidates

        result = list_fault_candidates.invoke({
            "alert_type": "unknown_alert_type"
        })

        assert result["alert_type"] == "unknown_alert_type"
        assert result["candidate_count"] == 0
        assert len(result["candidates"]) == 0
        assert "未知" in result["message"]


class TestHealTools:
    """Heal 工具测试"""

    def setup_method(self):
        """每个测试前重置熔断器状态"""
        from tools.heal_tools import circuit_breaker
        circuit_breaker._failure_count = 0
        circuit_breaker._state = "CLOSED"

    def test_check_circuit_breaker_closed(self):
        """测试熔断器正常状态"""
        from tools.heal_tools import check_circuit_breaker_status

        result = check_circuit_breaker_status.invoke({})

        assert result["state"] == "CLOSED"
        assert result["allowed"] == True

    def test_match_remediation_playbook_found(self):
        """测试匹配到 Playbook"""
        from tools.heal_tools import match_remediation_playbook

        result = match_remediation_playbook.invoke({
            "suggested_actions": ["rollback", "restart_pod"]
        })

        # 应该选择风险更低的 restart_pod（L0）而不是 rollback（L1）
        assert result["action"] == "restart_pod"
        assert result["level"] == "L0_AUTO"
        assert "blast_radius_factor" in result

    def test_match_remediation_playbook_not_found(self):
        """测试没有匹配的 Playbook"""
        from tools.heal_tools import match_remediation_playbook

        result = match_remediation_playbook.invoke({
            "suggested_actions": ["unknown_action", "another_unknown"]
        })

        assert result["action"] is None
        assert "没有找到" in result["message"]

    def test_match_remediation_playbook_priority(self):
        """测试优先级排序（L0 优先于 L1）"""
        from tools.heal_tools import match_remediation_playbook

        result = match_remediation_playbook.invoke({
            "suggested_actions": ["rollback", "scale_up"]  # L1 和 L0
        })

        # 应该选 L0（scale_up）
        assert result["action"] == "scale_up"
        assert result["level"] == "L0_AUTO"

    def test_simulate_dry_run_success(self):
        """测试模拟执行成功"""
        from tools.heal_tools import simulate_dry_run

        result = simulate_dry_run.invoke({
            "action_type": "restart_pod",
            "service_name": "order-service"
        })

        assert result["status"] == "SUCCESS"
        assert "kubectl" in result["output"]
        assert "order-service" in result["output"]

    def test_simulate_dry_run_unknown_action(self):
        """测试未知动作"""
        from tools.heal_tools import simulate_dry_run

        result = simulate_dry_run.invoke({
            "action_type": "unknown_action",
            "service_name": "order-service"
        })

        assert result["status"] == "FAILED"
        assert "未知" in result["output"]

    def test_record_heal_result_success(self):
        """测试记录修复成功"""
        from tools.heal_tools import record_heal_result, circuit_breaker

        # 先触发一次失败
        record_heal_result.invoke({"success": False})
        assert circuit_breaker._failure_count == 1

        # 再记录成功，应该重置计数
        result = record_heal_result.invoke({"success": True})

        assert result["circuit_breaker_state"] == "CLOSED"
        assert result["failure_count"] == 0

    def test_record_heal_result_failure_accumulates(self):
        """测试多次失败会增加计数"""
        from tools.heal_tools import record_heal_result, circuit_breaker

        # 连续记录 3 次失败
        for _ in range(3):
            record_heal_result.invoke({"success": False})

        assert circuit_breaker._failure_count == 3

    def test_circuit_breaker_opens_on_threshold(self):
        """测试熔断器在达到阈值后打开"""
        from tools.heal_tools import record_heal_result, check_circuit_breaker_status, circuit_breaker

        # 连续失败 5 次（达到阈值）
        for _ in range(5):
            record_heal_result.invoke({"success": False})

        # 熔断器应该打开
        assert circuit_breaker.state == "OPEN"

        result = check_circuit_breaker_status.invoke({})
        assert result["state"] == "OPEN"
        assert result["allowed"] == False
