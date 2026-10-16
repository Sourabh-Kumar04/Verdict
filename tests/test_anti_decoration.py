"""
Test 8: Anti-Decoration Test.
Asserts that FalkorDB is strictly load-bearing and not decorative:
- Verdict queries directly execute Cypher on FalkorDB.
- No direct LLM bypass exists.
- If the graph is empty or nodes are deleted, verdict resolution halts with load-bearing failure.
"""

import pytest
from verdict.graph_store import GraphStore
from verdict.verdict import VerdictEngine


def test_anti_decoration_empty_graph_fails_resolution(test_graph):
    # Completely empty graph - no nodes created
    engine = VerdictEngine(test_graph)
    result = engine.evaluate_claim("non_existent_claim")

    # Must fail to verify and report load-bearing failure
    assert result.verdict == "UNVERIFIED"
    assert result.confidence == 0.0
    assert "Load-bearing failure" in result.reason
    assert "FalkorDB" in result.reason


def test_anti_decoration_verdict_persisted_to_graph_node(test_graph):
    # When evaluated, the verdict must be committed into the FalkorDB node itself
    test_graph.add_root_claim("claim_anti_dec", "Factual statement")

    from verdict.claim_decomposer import SubClaim
    from verdict.retriever import EvidenceItem

    sub = SubClaim(id="sub_anti", text="Factual statement")
    test_graph.add_subclaim("claim_anti_dec", sub)

    ev = EvidenceItem(
        id="ev_anti",
        subclaim_id="sub_anti",
        text="Corroborating statement.",
        source_name="Source A",
        source_url="https://source-a.org",
        source_type="wikipedia",
        stance="SUPPORTS",
        weight=0.9
    )
    test_graph.add_evidence(ev, "sub_anti")

    engine = VerdictEngine(test_graph)
    engine.evaluate_claim("claim_anti_dec")

    # Assert that FalkorDB node contains the verdict properties from Cypher
    res = test_graph.run_query("""
        MATCH (c:Claim {id: 'claim_anti_dec'})
        RETURN c.verdict, c.confidence, c.resolved_at
    """)
    assert len(res.result_set) == 1
    verdict, confidence, resolved_at = res.result_set[0]
    assert verdict == "SUPPORTED"
    assert confidence > 0.0
    assert resolved_at > 0


def test_anti_decoration_invalid_connection_throws():
    # If FalkorDB is severed or unreachable, no store or evaluation can proceed
    with pytest.raises(Exception):
        dead_store = GraphStore(host="localhost", port=9999, graph_name="ghost")
        engine = VerdictEngine(dead_store)
        engine.evaluate_claim("any_claim")
