"""
Test 4: PageRank Sanity on a Toy Citation Graph.
"""

import pytest
from verdict.trust import TrustEngine


def test_pagerank_on_toy_citation_network(test_graph):
    # Setup toy citation network:
    # Source A cites Source B
    # Source C cites Source B
    # Source D cites Source A
    # Expected: Source B has the highest in-degree/authority and therefore highest PageRank.
    test_graph.add_source("source_A", "Source A", "wikipedia")
    test_graph.add_source("source_B", "Source B", "primary")
    test_graph.add_source("source_C", "Source C", "news")
    test_graph.add_source("source_D", "Source D", "blog")

    test_graph.add_citation("source_A", "source_B")
    test_graph.add_citation("source_C", "source_B")
    test_graph.add_citation("source_D", "source_A")

    engine = TrustEngine(test_graph)
    scores = engine.compute_source_trust()

    assert "source_B" in scores
    assert "source_A" in scores
    # Source B receives endorsements from both A and C -> must rank highest
    assert scores["source_B"] > scores["source_A"]
    assert scores["source_B"] > scores["source_C"]
    assert scores["source_B"] > scores["source_D"]

    # Verify score is persisted in FalkorDB node property
    res = test_graph.run_query("MATCH (s:Source {url: 'source_B'}) RETURN s.pagerank")
    assert abs(res.result_set[0][0] - scores["source_B"]) < 1e-4
