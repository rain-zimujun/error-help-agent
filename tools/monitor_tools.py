import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from langchain_core.tools import tool
import numpy as np

logger = logging.getLogger(__name__)


class AlertFingerprint:
    """告警指纹，用于告警收敛去重"""

    def __init__(self, window_seconds: int = 300):
        self._window = timedelta(seconds=window_seconds)
        self._recent: dict[str, datetime] = {}

    def compute(self, alert_name: str, target_service: str, labels: dict) -> str:
        raw = f"{alert_name}:{target_service}:{sorted(labels.items())}"
        return hashlib.md5(raw.encode()).hexdigest()

    def is_duplicate(self, fingerprint: str) -> bool:
        now = datetime.utcnow()
        if fingerprint in self._recent:
            if now - self._recent[fingerprint] < self._window:
                return True
        self._recent[fingerprint] = now
        self._cleanup(now)
        return False

    def _cleanup(self, now: datetime) -> None:
        expired = [k for k, v in self._recent.items() if now - v >= self._window]
        for k in expired:
            del self._recent[k]


class AnomalyDetector:
    """多算法异常检测器

    面试要点：
    - 3-sigma: 假设正态分布，超过 μ±3σ 即异常，简单高效但不适合非正态
    - EWMA: 指数加权移动平均，对近期数据更敏感，适合趋势变化检测
    - Isolation Forest: 基于随机森林的无监督异常检测，适合多维场景
    """

    @staticmethod
    def three_sigma(values: list[float], current: float) -> tuple[bool, float]:
        """3-sigma 异常检测"""
        if len(values) < 10:
            return False, 0.0
        arr = np.array(values)
        mean, std = arr.mean(), arr.std()
        if std == 0:
            return False, 0.0
        z_score = abs(current - mean) / std
        return z_score > 3.0, z_score

    @staticmethod
    def ewma(
        values: list[float], current: float, alpha: float = 0.3, threshold: float = 3.0
    ) -> tuple[bool, float]:
        """EWMA 指数加权移动平均异常检测"""
        if len(values) < 5:
            return False, 0.0
        ewma_val = values[0]
        ewma_var = 0.0
        for v in values[1:]:
            ewma_val = alpha * v + (1 - alpha) * ewma_val
            ewma_var = alpha * (v - ewma_val) ** 2 + (1 - alpha) * ewma_var

        ewma_std = np.sqrt(ewma_var)
        if ewma_std == 0:
            return False, 0.0
        deviation = abs(current - ewma_val) / ewma_std
        return deviation > threshold, deviation

    @staticmethod
    def isolation_forest_score(values: list[float], current: float) -> tuple[bool, float]:
        """Isolation Forest 异常检测（简化版，完整版见 models/time_series.py）"""
        if len(values) < 20:
            return False, 0.0
        from sklearn.ensemble import IsolationForest

        data = np.array(values + [current]).reshape(-1, 1)
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(data[:-1])
        score = clf.decision_function(data[-1:])
        is_anomaly = clf.predict(data[-1:])[0] == -1
        return is_anomaly, float(-score[0])

fingerprint_tracker = AlertFingerprint()
detector = AnomalyDetector()

@tool(parse_docstring=True)
def run_anomaly_detection(
    metric_history: list[float],
    current_value: float
) -> dict:
    """
    检测时序异常，使用三种算法投票。

    Args:
        metric_history: 历史指标数据列表
        current_value: 当前指标值

    Returns:
        包含三个算法的检测结果、投票数、综合异常分数
    """
    # 调用三种算法
    three_sigma_result, three_sigma_score = detector.three_sigma(metric_history, current_value)
    ewma_result, ewma_score = detector.ewma(metric_history, current_value)
    isolation_result, isolation_score = detector.isolation_forest_score(metric_history, current_value)

    # 投票: 至少两个算法觉得有"异常"就认为出现了异常
    votes = sum([three_sigma_result, ewma_result, isolation_result])
    is_anomaly = votes >= 2

    # 综合异常分数(0 - 10)
    score = (three_sigma_score + ewma_score + isolation_score) / 3.0
    score = min(score, 10.0)

    return {
        "algorithms": {
            "three_sigma": {"is_anomaly": three_sigma_result, "score": float(three_sigma_score)},
            "ewma": {"is_anomaly": ewma_result, "score": float(ewma_score)},
            "isolation_forest": {"is_anomaly": isolation_result, "score": float(isolation_score)}
        },
        "votes": int(votes),
        "is_anomaly": bool(is_anomaly),
        "score": float(score)
    }

@tool(parse_docstring=True)
def check_alert_duplicate(
      alert_name: str,
      target_service: str,
      labels: dict
  ) -> dict:
    """
    检查告警是否重复（5分钟滑动窗口）。

    Args:
        alert_name: 告警名称
        target_service: 目标服务
        labels: 告警标签（如 pod、namespace）

    Returns:
        是否重复，以及原因说明
    """
    # ① 生成指纹
    fingerprint = fingerprint_tracker.compute(alert_name, target_service, labels)

    # ② 检查是否重复（自动更新时间戳）
    is_dup = fingerprint_tracker.is_duplicate(fingerprint)

    reason = "5分钟内已出现过此告警" if is_dup else "首次告警"

    return {
        "is_duplicate": is_dup,
        "reason": reason
    }


@tool(parse_docstring=True)
def classify_alert_severity(anomaly_score: float) -> dict:
    """
    根据异常分数分级告警严重程度。

    Args:
        anomaly_score: 异常分数（0-10）

    Returns:
        告警严重程度等级
    """
    if anomaly_score >= 7.0:
        severity = "CRITICAL"
    elif anomaly_score >= 5.0:
        severity = "HIGH"
    elif anomaly_score >= 3.0:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {"severity": severity}