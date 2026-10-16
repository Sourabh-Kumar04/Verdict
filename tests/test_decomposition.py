"""
Test 1: Claim Decomposition Unit Tests.
"""

import pytest
from verdict.claim_decomposer import ClaimDecomposer


def test_atomic_claim_decomposition():
    decomposer = ClaimDecomposer()
    claim = "The Eiffel Tower was completed in 1889"
    subclaims = decomposer.decompose(claim)

    assert len(subclaims) == 1
    assert "The Eiffel Tower" in subclaims[0].entities
    assert subclaims[0].time_expression == "1889"
    assert "1889" in subclaims[0].keywords


def test_compound_claim_decomposition():
    decomposer = ClaimDecomposer()
    claim = "Marie Curie was born in Warsaw and won two Nobel Prizes"
    subclaims = decomposer.decompose(claim)

    assert len(subclaims) == 2
    assert "Marie Curie was born in Warsaw." == subclaims[0].text
    # Check subject ellipsis resolution carried over "Marie Curie"
    assert "Marie Curie won two Nobel Prizes." == subclaims[1].text
    assert "Warsaw" in subclaims[0].entities
    assert "Marie Curie" in subclaims[1].entities


def test_apollo_claim_decomposition():
    decomposer = ClaimDecomposer()
    claim = "Apollo 11 landed on Mars in 1969"
    subclaims = decomposer.decompose(claim)

    assert len(subclaims) == 1
    assert "Apollo 11" in subclaims[0].entities
    assert "Mars" in subclaims[0].entities
    assert subclaims[0].time_expression == "1969"
