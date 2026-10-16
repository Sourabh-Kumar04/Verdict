"""
Test 6: Graph-Driven Verdict - REFUTES.
"""

import pytest
from verdict.claim_decomposer import SubClaim
from verdict.retriever import EvidenceItem
from verdict.trust import TrustEngine
from verdict.verdict import VerdictEngine
from verdict.proof_path import ProofPathEngine


def test_verdict_refutes(test_graph):
    test_graph.add_root_claim("claim_ref", "Apollo 11 landed on Mars in 1969")
    sub = SubClaim(id="sub_ref_1", text="Apollo 11 landed on Mars in 1969.")
    test_graph.add_subclaim("claim_ref", sub)

    # Add refuting evidence
    ev = EvidenceItem(
        id="ev_ref_1",
        subclaim_id="sub_ref_1",
        text="Apollo 11 was the spaceflight that landed the first humans on the Moon, not Mars.",
        source_name="Wikipedia: Apollo 11",
        source_url="https://en.wikipedia.org/wiki/Apollo_11",
        source_type="wikipedia",
        stance="REFUTES",
        weight=0.95
    )
    test_graph.add_evidence(ev, "sub_ref_1")

    trust = TrustEngine(test_graph)
    trust.compute_source_trust()

    engine = VerdictEngine(test_graph)
    result = engine.evaluate_claim("claim_ref")

    assert result.verdict == "REFUTED"
    assert result.confidence > 0.0
    assert "Refuted" in result.reason

    proof_engine = ProofPathEngine(test_graph)
    proof_steps = proof_engine.get_ranked_proof_path("claim_ref")
    assert len(proof_steps) >= 1
    assert proof_steps[0].stance == "REFUTES"
