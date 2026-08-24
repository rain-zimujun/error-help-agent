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
