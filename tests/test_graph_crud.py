"""
Test 3: FalkorDB Graph CRUD Round-Trip and Pitfall Guards.
"""

import pytest
from verdict.graph_store import GraphStore, format_vector, flatten_properties
from verdict.claim_decomposer import SubClaim
from verdict.retriever import EvidenceItem


def test_vector_formatting_pitfall():
    # Wrap embedding vectors: vecf32([...]) in Cypher. Never pass raw lists.
    vec = [0.1, 0.25, 0.99]
    formatted = format_vector(vec)
    assert formatted.startswith("vecf32([")
    assert formatted.endswith("])")
    assert "0.250000" in formatted


def test_flatten_properties_pitfall():
    # No nested maps as node/relationship properties - flatten first.
    nested = {
        "title": "Claim 1",
        "author": {"name": "Gustave", "details": {"origin": "France"}},
        "tags": ["history", "architecture"]
    }
    flat = flatten_properties(nested)
    assert flat["title"] == "Claim 1"
    assert flat["author_name"] == "Gustave"
    assert flat["author_details_origin"] == "France"
    assert isinstance(flat["tags"], list)


def test_graph_crud_and_empty_delete_guard(test_graph):
    # Guard empty delete: deleting twice ensures the second attempt hits an empty key
    test_graph.delete_graph()
    assert test_graph.delete_graph() is False

    test_graph.init_schema()
    test_graph.add_root_claim("claim_1", "Test Claim Text")
    sub = SubClaim(id="sub_1", text="Test Subclaim Text")
    test_graph.add_subclaim("claim_1", sub)

    ev = EvidenceItem(
        id="ev_1",
        subclaim_id="sub_1",
        text="Test evidence text supporting the claim.",
        source_name="Test Source",
        source_url="https://example.com/source1",
        source_type="wikipedia",
        stance="SUPPORTS",
        weight=0.9
    )
    test_graph.add_evidence(ev, "sub_1")

    counts = test_graph.get_counts()
    assert counts["claims"] == 2
    assert counts["evidence"] == 1
    assert counts["sources"] >= 1
    assert counts["total_edges"] >= 2

    # Query back using Cypher
    res = test_graph.run_query("MATCH (c:Claim {id: 'claim_1'}) RETURN c.text")
    assert res.result_set[0][0] == "Test Claim Text"
