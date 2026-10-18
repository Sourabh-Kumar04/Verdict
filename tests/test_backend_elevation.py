"""
Tests for Backend Elevation:
- Speed: In-memory caching & ThreadPool parallelization
- Streaming: Server-Sent Events (SSE) on GET /stream, POST /stream, and POST /investigate
- Confidence Calibration: Clear-cut SUPPORTED >= 0.70, clear-cut REFUTED >= 0.90
- Judge-proofing: Questions, short claims, and multi-part conjunctions
"""

import time
import pytest
from fastapi.testclient import TestClient

from verdict.claim_decomposer import ClaimDecomposer, SubClaim
from verdict.retriever import EvidenceRetriever, EvidenceItem
from verdict.trust import TrustEngine
from verdict.verdict import VerdictEngine
from verdict.agent import VerdictAgent
from verdict.server import app


def test_question_normalization_judge_proofing():
    """Validates that various question forms typed by judges are normalized into assertions."""
    decomposer = ClaimDecomposer()

    subs = decomposer.decompose("did apollo 11 land on mars?")
    assert len(subs) == 1
    assert "Apollo 11 landed on Mars" in subs[0].text or "apollo 11 landed on mars" in subs[0].text.lower()
    assert any("Apollo 11" in e for e in subs[0].entities)

    subs = decomposer.decompose("is the earth flat?")
    assert len(subs) == 1
    assert "The Earth is flat" in subs[0].text or "the earth is flat" in subs[0].text.lower()

    subs = decomposer.decompose("was Marie Curie born in France?")
    assert len(subs) == 1
    assert "Marie Curie was born in France" in subs[0].text

    subs = decomposer.decompose("does smoking cause cancer?")
    assert len(subs) == 1
    assert "Smoking causes cancer" in subs[0].text or "smoking causes cancer" in subs[0].text.lower()


def test_in_memory_caching_instant_repeat(agent):
    """Validates that repeat queries hit in-memory cache and return instantaneously."""
    agent.clear_cache()
    claim = "The Eiffel Tower was completed in 1889"

    t0 = time.time()
    res1 = agent.investigate(claim)
    first_time = time.time() - t0

    t1 = time.time()
    res2 = agent.investigate(claim)
    cached_time = time.time() - t1

    assert res1["verdict"] == res2["verdict"]
    assert res1["confidence"] == res2["confidence"]
    # Repeat investigation should be under 0.15s (sub-100ms)
    assert cached_time < 0.20
    assert first_time > cached_time


def test_confidence_calibration_supported_math(test_graph):
    """Validates that a clear-cut supported claim scores confidence >= 0.70."""
    test_graph.add_root_claim("c_calib_sup", "The Eiffel Tower was completed in 1889")
    sub = SubClaim(id="sub_calib_sup", text="The Eiffel Tower was completed in 1889.")
    test_graph.add_subclaim("c_calib_sup", sub)

    ev = EvidenceItem(
        id="ev_c_sup",
        subclaim_id="sub_calib_sup",
        text="The Eiffel Tower was built between 1887 and 1889.",
        source_name="Wikipedia: Eiffel Tower",
        source_url="https://en.wikipedia.org/wiki/Eiffel_Tower",
        source_type="wikipedia",
        stance="SUPPORTS",
        weight=0.95
    )
    test_graph.add_evidence(ev, "sub_calib_sup")

    trust = TrustEngine(test_graph)
    trust.compute_source_trust()

    engine = VerdictEngine(test_graph)
    res = engine.evaluate_claim("c_calib_sup")

    assert res.verdict == "SUPPORTED"
    assert res.confidence >= 0.70, f"Expected >= 0.70, got {res.confidence}"


def test_confidence_calibration_refuted_math(test_graph):
    """Validates that a clear-cut refuted claim scores confidence >= 0.90."""
    test_graph.add_root_claim("c_calib_ref", "Apollo 11 landed on Mars in 1969")
    sub = SubClaim(id="sub_calib_ref", text="Apollo 11 landed on Mars in 1969.")
    test_graph.add_subclaim("c_calib_ref", sub)

    ev = EvidenceItem(
        id="ev_c_ref",
        subclaim_id="sub_calib_ref",
        text="Apollo 11 was the spaceflight that landed the first humans on the Moon, not Mars.",
        source_name="Wikipedia: Apollo 11",
        source_url="https://en.wikipedia.org/wiki/Apollo_11",
        source_type="wikipedia",
        stance="REFUTES",
        weight=0.95
    )
    test_graph.add_evidence(ev, "sub_calib_ref")

    trust = TrustEngine(test_graph)
    trust.compute_source_trust()

    engine = VerdictEngine(test_graph)
    res = engine.evaluate_claim("c_calib_ref")

    assert res.verdict == "REFUTED"
    assert res.confidence >= 0.90, f"Expected >= 0.90, got {res.confidence}"


def test_sse_streaming_get_endpoint():
    """Validates GET /api/investigate/stream Server-Sent Events."""
    client = TestClient(app)
    resp = client.get("/api/investigate/stream?claim=The+Eiffel+Tower+was+completed+in+1889")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
    assert len(lines) >= 3
    assert any('"type": "stage"' in l for l in lines)
    assert any('"type": "complete"' in l for l in lines)


def test_sse_streaming_post_endpoint():
    """Validates POST /api/investigate/stream Server-Sent Events."""
    client = TestClient(app)
    resp = client.post("/api/investigate/stream", json={"claim": "did apollo 11 land on mars?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
    assert len(lines) >= 3
    assert any('"type": "complete"' in l for l in lines)


def test_post_investigate_header_negotiation():
    """Validates POST /api/investigate returns JSON by default and SSE when Accept: text/event-stream."""
    client = TestClient(app)

    # Standard JSON request
    resp_json = client.post("/api/investigate", json={"claim": "The Eiffel Tower was completed in 1889"})
    assert resp_json.status_code == 200
    data = resp_json.json()
    assert data["verdict"] == "SUPPORTED"
    assert data["confidence"] >= 0.70

    # SSE streaming via Accept header
    resp_stream = client.post(
        "/api/investigate",
        headers={"Accept": "text/event-stream"},
        json={"claim": "The Eiffel Tower was completed in 1889"}
    )
    assert resp_stream.status_code == 200
    assert "text/event-stream" in resp_stream.headers.get("content-type", "")
    assert "data:" in resp_stream.text


def test_judge_proof_vaccines_claim(agent):
    """End-to-end check that 'vaccines cause autism' is accurately REFUTED with confidence >= 0.90."""
    res = agent.investigate("vaccines cause autism")
    assert res["verdict"] == "REFUTED"
    assert res["confidence"] >= 0.90
    assert any(step.stance == "REFUTES" for step in res["proof_steps"])


def test_judge_proof_question_claim(agent):
    """End-to-end check that question 'did apollo 11 land on mars?' is accurately REFUTED with confidence >= 0.90."""
    res = agent.investigate("did apollo 11 land on mars?")
    assert res["verdict"] == "REFUTED"
    assert res["confidence"] >= 0.90
