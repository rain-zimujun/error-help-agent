from langchain_core.tools import tool
from knowledge_graph import create_demo_knowledge_graph

# 全局知识图谱实例
kg = create_demo_knowledge_graph()


@tool(parse_docstring=True)
def query_service_dependencies(service_name: str) -> dict:
    """
    查询某服务的直接依赖。

    Args:
        service_name: 服务名称（如 "order-service"）

    Returns:
        该服务的所有直接依赖列表
    """
    # 调用知识图谱查询
    dependencies = kg.get_dependencies(service_name)

    return {
        "service": service_name,
        "direct_dependencies": dependencies,
        "count": len(dependencies)
    }

@tool(parse_docstring=True)
def trace_impact_chain(service_name: str, max_depth: int = 5) -> dict:
    """
     追踪故障影响链 - 从某个服务出发，找出它的所有下游依赖。

     这个工具用来评估：如果这个服务故障了，会波及多少个下游服务？

     Args:
         service_name: 故障服务的名称
         max_depth: 最多追踪几层依赖（默认 5 层）

     Returns:
         所有可能的依赖路径和受影响的服务列表
     """
    # 用 BFS 遍历所有依赖路径
    paths = kg.bfs_trace(service_name, rel_type="DEPENDS_ON", max_depth=max_depth)

    # 提取出所有出现过的服务并去重
    affected_services = set()
    for path in paths:
        affected_services.update(path)
    affected_services.discard(service_name) # 过滤掉服务本身

    return {
        "root_service": service_name,
        "dependency_paths": paths,
        # 所有可能的路径
        "affected_services": sorted(list(affected_services)),
        # 所有受影响的服务
        "impact_depth": max_depth
    }


@tool(parse_docstring=True)
def find_recent_changes_in_service(service_name: str, hours: int = 24) -> dict:
    """
    查询某服务及其直接依赖的近期变更。

    这是根因分析的关键！因为最近改过的服务最可能是根因。

    Args:
        service_name: 服务名称
        hours: 查询最近多少小时的变更（默认 24 小时）

    Returns:
        最近的变更记录列表
    """
    # 调用知识图谱查询最近变更
    changes = kg.find_recent_changes(service_name, within_hours=hours)

    # 按时间排序（最新的在前）
    changes_sorted = sorted(changes, key=lambda x: x.get("time", ""), reverse=True)

    return {
        "service": service_name,
        "time_window_hours": hours,
        "changes": changes_sorted,
        "change_count": len(changes_sorted)
    }

# 常见故障模式与根因的映射
FAULT_PATTERNS = {
    "high_cpu_usage": [
        {
            "root_cause": "上游服务请求量激增导致 CPU 过载",
            "evidence_pattern": "request_rate_increase",
            "base_probability": 0.35,
            "suggested_actions": ["scale_up", "rate_limit"],
        },
        {
            "root_cause": "内存泄漏导致 GC 频繁触发",
            "evidence_pattern": "memory_leak",
            "base_probability": 0.25,
            "suggested_actions": ["restart_pod", "heap_dump"],
        },
        {
            "root_cause": "近期代码部署引入性能退化",
            "evidence_pattern": "recent_deploy",
            "base_probability": 0.30,
            "suggested_actions": ["rollback", "profiling"],
        },
        {
            "root_cause": "下游服务响应慢导致线程池耗尽",
            "evidence_pattern": "downstream_slow",
            "base_probability": 0.10,
            "suggested_actions": ["circuit_breaker", "timeout_adjust"],
        },
    ],
    "high_memory_usage": [
        {
            "root_cause": "内存泄漏（未释放的对象引用）",
            "evidence_pattern": "memory_leak",
            "base_probability": 0.50,
            "suggested_actions": ["restart_pod", "heap_dump"],
        },
        {
            "root_cause": "缓存未设置过期策略",
            "evidence_pattern": "cache_overflow",
            "base_probability": 0.30,
            "suggested_actions": ["clear_cache", "set_ttl"],
        },
    ],
    "high_error_rate": [
        {
            "root_cause": "下游依赖服务不可用",
            "evidence_pattern": "dependency_down",
            "base_probability": 0.40,
            "suggested_actions": ["check_dependency", "circuit_breaker"],
        },
        {
            "root_cause": "近期配置变更导致异常",
            "evidence_pattern": "recent_config_change",
            "base_probability": 0.35,
            "suggested_actions": ["rollback_config", "review_change"],
        },
    ],
}


@tool(parse_docstring=True)
def list_fault_candidates(alert_type: str) -> dict:
    """
    列出某个告警类型的所有可能根因。

    这个工具告诉 LLM：对于这个告警，有哪些可能的根因，
    以及每个根因的先验概率是多少。

    LLM 会结合这些候选和实际的证据（变更记录等），
    用贝叶斯推理计算后验概率。

    Args:
        alert_type: 告警类型（如 "high_cpu_usage"）

    Returns:
        该告警类型的所有根因候选及其概率
    """
    candidates = FAULT_PATTERNS.get(alert_type, [])

    if not candidates:
        return {
            "alert_type": alert_type,
            "candidates": [],
            "candidate_count": 0,
            "message": f"未知的告警类型：{alert_type}"
        }

    return {
        "alert_type": alert_type,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "explanation": "这些是该告警类型的常见根因及其概率，结合实际证据（变更、依赖等）进行贝叶斯推理"
    }