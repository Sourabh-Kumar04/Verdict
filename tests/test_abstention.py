"""
Test 7: Honest Abstention on Insufficient or Missing Evidence.
"""

import pytest
from verdict.claim_decomposer import SubClaim
from verdict.verdict import VerdictEngine


def test_abstention_on_zero_evidence(test_graph):
    test_graph.add_root_claim("claim_abs", "The Loch Ness monster was photographed by Isaac Newton in 1675")
    sub = SubClaim(id="sub_abs_1", text="The Loch Ness monster was photographed by Isaac Newton in 1675.")
    test_graph.add_subclaim("claim_abs", sub)

    # Note: No evidence nodes added to graph
    engine = VerdictEngine(test_graph)
    result = engine.evaluate_claim("claim_abs")

    assert result.verdict == "UNVERIFIED"
    assert result.confidence == 0.0
    assert result.total_evidence_nodes == 0
    assert "Abstained" in result.reason
    assert result.is_abstention() is True


def test_abstention_on_partial_subclaim_coverage(test_graph):
    # Compound claim with 2 subclaims: only 1 has evidence, the other has no evidence
    test_graph.add_root_claim("claim_compound", "Entity A did X and Entity B did Y")
    sub1 = SubClaim(id="sub_c1", text="Entity A did X")
    sub2 = SubClaim(id="sub_c2", text="Entity B did Y")
    test_graph.add_subclaim("claim_compound", sub1)
    test_graph.add_subclaim("claim_compound", sub2)

    from verdict.retriever import EvidenceItem
    ev1 = EvidenceItem(
        id="ev_part_1",
        subclaim_id="sub_c1",
        text="Verified that Entity A did X.",
        source_name="Source 1",
        source_url="https://example.com/s1",
        source_type="wikipedia",
        stance="SUPPORTS",
        weight=0.9
    )
    test_graph.add_evidence(ev1, "sub_c1")
    # sub2 has NO evidence in graph

    engine = VerdictEngine(test_graph)
    result = engine.evaluate_claim("claim_compound")

    # Because sub2 is unverified, the compound claim must honestly abstain!
    assert result.verdict == "UNVERIFIED"
    assert "Abstained" in result.reason
