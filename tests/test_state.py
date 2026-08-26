from datetime import datetime

from state import create_initial_state


SAMPLE_METRIC_DATA = {
    "metric_history": [20, 21, 19, 22, 20],
    "current_value": 95,
    "alert_name": "high_cpu_usage",
    "target_service": "order-service",
    "labels": {"pod": "order-service-xyz", "namespace": "production"},
}


class TestCreateInitialState:
    """测试 create_initial_state 工厂函数"""

    def test_basic_fields(self):
        state = create_initial_state("incident-001", SAMPLE_METRIC_DATA)

        assert state["incident_id"] == "incident-001"
        assert state["status"] == "open"
        assert state["metric_data"] == SAMPLE_METRIC_DATA
        assert isinstance(state["created_at"], datetime)
        assert isinstance(state["updated_at"], datetime)

    def test_agent_output_fields_default_to_none(self):
        state = create_initial_state("incident-002", SAMPLE_METRIC_DATA)

        assert state["alert_event"] is None
        assert state["rca_result"] is None
        assert state["heal_action"] is None
        assert state["change_decision"] is None

    def test_messages_starts_empty(self):
        state = create_initial_state("incident-003", SAMPLE_METRIC_DATA)

        assert state["messages"] == []

    def test_metric_data_is_preserved_as_is(self):
        """metric_data 是 TypedDict，运行时不做校验，原样存进 state"""
        custom_metric_data = {**SAMPLE_METRIC_DATA, "current_value": 50}
        state = create_initial_state("incident-004", custom_metric_data)

        assert state["metric_data"]["current_value"] == 50
