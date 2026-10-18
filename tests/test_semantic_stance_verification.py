"""
Tests for Semantic Stance Verification Layer:
- Verifies that false positive lexical matches (e.g., 'openai is indian company') are REFUTED.
- Verifies that true positive factual claims (e.g., 'The Eiffel Tower was completed in 1889') remain SUPPORTED.
- Verifies that debunked claims (e.g., 'vaccines cause autism') remain REFUTED.
- Direct unit tests for SemanticStanceVerifier and nationality contradiction detection.
"""

import pytest
from verdict.agent import VerdictAgent
from verdict.semantic_verifier import (
    SemanticStanceVerifier,
    detect_nationality_contradiction
)
from verdict.claim_decomposer import SubClaim
from verdict.retriever import EvidenceItem


def test_openai_is_indian_company_returns_refuted(agent):
    """
    Validates that the false-positive lexical matching bug is fixed:
    'openai is indian company' must return REFUTED with high confidence,
    not SUPPORTED.
    """
    res = agent.investigate("openai is indian company")

    assert res["verdict"] == "REFUTED"
    assert res["confidence"] >= 0.85
    assert not any(step.stance == "SUPPORTS" for step in res["proof_steps"])
    assert any(step.stance == "REFUTES" for step in res["proof_steps"])


def test_eiffel_tower_returns_supported(agent):
    """
    Validates that true positives are not broken by the semantic verifier:
    'The Eiffel Tower was completed in 1889' must still return SUPPORTED.
    """
    res = agent.investigate("The Eiffel Tower was completed in 1889")

    assert res["verdict"] == "SUPPORTED"
    assert res["confidence"] >= 0.70
    assert any(step.stance == "SUPPORTS" for step in res["proof_steps"])


def test_vaccines_cause_autism_returns_refuted(agent):
    """
    Validates that debunked claims remain REFUTED:
    'vaccines cause autism' must return REFUTED with confidence >= 0.90.
    """
    res = agent.investigate("vaccines cause autism")

    assert res["verdict"] == "REFUTED"
    assert res["confidence"] >= 0.90
    assert any(step.stance == "REFUTES" for step in res["proof_steps"])


def test_nationality_contradiction_detection():
    """Unit test for country/nationality contradiction detection."""
    # OpenAI American vs Indian -> Contradiction
    assert detect_nationality_contradiction(
        "openai is indian company",
        "OpenAI: American artificial intelligence research organization"
    ) is True

    # Indian American in evidence while claim is Indian -> Compound ethnicity, not pure contradiction
    assert detect_nationality_contradiction(
        "openai is indian company",
        "OpenAI names Indian American Uday Ruddarraju to lead infrastructure"
    ) is False

    # Eiffel tower in France -> No contradiction
    assert detect_nationality_contradiction(
        "The Eiffel Tower was completed in 1889",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    ) is False


def test_semantic_stance_verifier_direct():
    """Direct unit tests for SemanticStanceVerifier calling local Ollama LLM."""
    verifier = SemanticStanceVerifier()

    # False positive test: hiring an Indian-American person does NOT support being an Indian company
    is_sup1, stance1, reason1 = verifier.verify_single(
        "openai is indian company",
        "OpenAI names Indian American Uday Ruddarraju to lead infrastructure and computing operations - The American Bazaar."
    )
    assert is_sup1 is False
    assert stance1 in ("QUALIFIES", "REFUTES")

    # True positive test: Eiffel Tower completed in 1889
    is_sup2, stance2, reason2 = verifier.verify_single(
        "The Eiffel Tower was completed in 1889",
        "As one of the most iconic and recognizable structures in the world, the Eiffel Tower, completed in 1889, has been the inspiration for the creation of over 50 similar towers around the world."
    )
    assert is_sup2 is True
    assert stance2 == "SUPPORTS"
