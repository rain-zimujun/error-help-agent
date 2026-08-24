import pytest
from knowledge_graph import InMemoryKnowledgeGraph, create_demo_knowledge_graph

def test_add_node():
    """测试添加节点"""
    kg = InMemoryKnowledgeGraph()
    kg.add_node("order-service", "microservice", {"tier": "backend"})

    node = kg.get_node("order-service")
    assert node is not None
    assert node["name"] == "order-service"
    assert node["type"] == "microservice"


def test_get_dependencies():
    """测试获取依赖"""
    kg = InMemoryKnowledgeGraph()
    kg.add_node("order-service", "microservice")
    kg.add_node("payment-service", "microservice")
    kg.add_relationship("order-service", "payment-service", "DEPENDS_ON")

    deps = kg.get_dependencies("order-service")
    assert "payment-service" in deps
    assert len(deps) == 1

def test_bfs_trace():
    """测试 BFS 遍历依赖链"""
    kg = InMemoryKnowledgeGraph()
    kg.add_node("a", "service")
    kg.add_node("b", "service")
    kg.add_node("c", "service")
    kg.add_relationship("a", "b", "DEPENDS_ON")
    kg.add_relationship("b", "c", "DEPENDS_ON")

    paths = kg.bfs_trace("a", max_depth=5)
    assert any(["a", "b", "c"] == path for path in paths)


def test_reverse_bfs_trace():
    """测试反向 BFS 找根因候选"""
    kg = InMemoryKnowledgeGraph()
    kg.add_node("a", "service")
    kg.add_node("b", "service")
    kg.add_node("c","service")  # 叶子节点
    kg.add_relationship("a", "b", "DEPENDS_ON")
    kg.add_relationship("b", "c", "DEPENDS_ON")

    roots = kg.reverse_bfs_trace("a", max_depth=5)
    assert "c" in roots  # c 是叶子节点，应该是根因候选


def test_create_demo_knowledge_graph():
    """测试演示知识图谱"""
    kg = create_demo_knowledge_graph()

    # 检查节点数
    topo = kg.get_topology_summary()
    assert topo["total_nodes"] == 14
    assert topo["total_edges"] == 15 + 6

    # 检查特定依赖
    order_deps = kg.get_dependencies("order-service")
    assert "payment-service" in order_deps
    assert "inventory-service" in order_deps