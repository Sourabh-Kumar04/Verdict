"""
Test 9: End-to-End Real Claim Investigation Tests.
Connects live to Wikipedia/Wikidata public APIs, constructs FalkorDB graph,
and checks verdicts on real factual claims.
"""

import pytest
from verdict.agent import VerdictAgent


@pytest.mark.e2e
def test_e2e_real_claim_supported(agent):
    claim = "The Eiffel Tower was completed in 1889"
    result = agent.investigate(claim)

    assert result["verdict"] == "SUPPORTED"
    assert result["confidence"] > 0.0
    assert result["graph_counts"]["evidence"] > 0
    assert len(result["proof_steps"]) > 0
    assert "1889" in result["one_line_proof"] or "Eiffel" in result["one_line_proof"]


@pytest.mark.e2e
def test_e2e_real_claim_refuted(agent):
    claim = "Apollo 11 landed on Mars in 1969"
    result = agent.investigate(claim)

    assert result["verdict"] == "REFUTED"
    assert result["confidence"] > 0.0
    assert result["graph_counts"]["evidence"] > 0
    assert len(result["proof_steps"]) > 0
    assert any(step.stance == "REFUTES" for step in result["proof_steps"])


@pytest.mark.e2e
def test_e2e_real_claim_abstention(agent):
    claim = "The Loch Ness monster was photographed by Isaac Newton in 1675"
    result = agent.investigate(claim)

    assert result["verdict"] == "UNVERIFIED"
    assert "Abstained" in result["reason"]
