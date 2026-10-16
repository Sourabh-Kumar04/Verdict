"""
Test 5: Graph-Driven Verdict - SUPPORTS.
"""

import pytest
from verdict.claim_decomposer import SubClaim
from verdict.retriever import EvidenceItem
from verdict.trust import TrustEngine
from verdict.verdict import VerdictEngine
from verdict.proof_path import ProofPathEngine


def test_verdict_supports(test_graph):
    test_graph.add_root_claim("claim_sup", "The Eiffel Tower was completed in 1889")
    sub = SubClaim(id="sub_sup_1", text="The Eiffel Tower was completed in 1889.")
    test_graph.add_subclaim("claim_sup", sub)

    # Add supporting evidence
    ev = EvidenceItem(
        id="ev_sup_1",
        subclaim_id="sub_sup_1",
        text="Gustave Eiffel designed and built the tower from 1887 to 1889.",
        source_name="Wikipedia: Eiffel Tower",
        source_url="https://en.wikipedia.org/wiki/Eiffel_Tower",
        source_type="wikipedia",
        stance="SUPPORTS",
        weight=0.95
    )
    test_graph.add_evidence(ev, "sub_sup_1")

    # Add citation and compute trust
    test_graph.add_source("https://www.wikidata.org/wiki/Q243", "Wikidata: Q243", "wikidata")
    test_graph.add_citation("https://en.wikipedia.org/wiki/Eiffel_Tower", "https://www.wikidata.org/wiki/Q243")

    trust = TrustEngine(test_graph)
    trust.compute_source_trust()

    engine = VerdictEngine(test_graph)
    result = engine.evaluate_claim("claim_sup")

    assert result.verdict == "SUPPORTED"
    assert result.confidence > 0.0
    assert "Supported" in result.reason
    assert result.total_evidence_nodes >= 1

    proof_engine = ProofPathEngine(test_graph)
    proof_steps = proof_engine.get_ranked_proof_path("claim_sup")
    assert len(proof_steps) >= 1
    assert proof_steps[0].stance == "SUPPORTS"
