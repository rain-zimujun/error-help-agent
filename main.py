import uuid
from datetime import datetime

from graph import graph
from state import create_initial_state, MetricData

# 生成唯一id的函数
def generate_incident_id() -> str:
    """生成唯一的 incident_id，时间戳+短uuid，方便按时间排序和去重"""
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"incident-{ts}-{short_uuid}"

# 打印结果的函数
def print_result(result: dict) -> None:
    print("=" * 60)
    print("  故障处理结果")
    print("=" * 60)
    print(f"  事件 ID:    {result['incident_id']}")
    print(f"  状态:       {result['status']}")

    alert = result.get("alert_event")
    if alert:
        print("\n  [告警]")
        print(f"    名称:     {alert.get('alert_name')}")
        print(f"    严重度:   {alert.get('severity')}")
        print(f"    服务:     {alert.get('service')}")
        print(f"    异常分数: {alert.get('anomaly_score')}")

    rca = result.get("rca_result")
    if rca:
        print("\n  [根因分析]")
        print(f"    根因:     {rca.get('root_cause')}")
        print(f"    置信度:   {rca.get('confidence')}")
        print(f"    影响服务: {', '.join(rca.get('affected_services', []))}")
        print(f"    建议动作: {rca.get('suggested_actions')}")

    heal = result.get("heal_action")
    if heal:
        print("\n  [自愈]")
        print(f"    操作:     {heal.get('action')}")
        print(f"    级别:     {heal.get('level')}")
        print(f"    爆炸半径: {heal.get('blast_radius')}")
        print(f"    Dry-run:  {heal.get('dry_run_output')}")

    change = result.get("change_decision")
    if change:
        print("\n  [审批]")
        print(f"    状态:     {change.get('approval')}")
        print(f"    风险分:   {change.get('risk_score')}")
        print(f"    原因:     {change.get('recommendation')}")

    print("=" * 60)


# main入口
def main():
    metric_data: MetricData = {
        "metric_history": [20, 21, 19, 22, 20, 21, 20, 22, 21, 20],
        "current_value": 95,
        "alert_name": "high_cpu_usage",
        "target_service": "order-service",
        "labels": {"pod": "order-service-xyz", "namespace": "production"}
    }

    incident_id = generate_incident_id()
    initial_state = create_initial_state(incident_id=incident_id, metric_data=metric_data)

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": incident_id}}
    )

    print_result(result)

if __name__ == "__main__":
    main()