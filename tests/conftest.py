"""
Pytest configuration and fixtures for Verdict.
"""

import pytest
from verdict.graph_store import GraphStore
from verdict.agent import VerdictAgent


@pytest.fixture
def test_graph():
    """Provides a fresh isolated FalkorDB graph for each test."""
    store = GraphStore(host="localhost", port=6379, graph_name="verdict_pytest")
    store.delete_graph()
    store.init_schema()
    yield store
    store.delete_graph()


@pytest.fixture
def agent():
    """Provides a fresh VerdictAgent instance using a dedicated test graph."""
    agt = VerdictAgent(host="localhost", port=6379, graph_name="verdict_agent_test")
    yield agt
    agt.graph_store.delete_graph()
