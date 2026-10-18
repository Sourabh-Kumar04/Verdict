"""
Tests for Robustness and Fact-Checker Evidence Integration:
- Fact-checker sources (Snopes, PolitiFact, FactCheck.org) site-scoped retrieval & parsing
- Higher trust weight and PageRank authority for fact-checkers in FalkorDB
- Input sanitization (null bytes, control chars, overly long strings, empty strings)
- Graceful handling when FalkorDB is unreachable (clear error message)
- Graceful abstention when all evidence sources return empty
"""

from unittest.mock import MagicMock
import pytest
import requests

from verdict.claim_decomposer import SubClaim
from verdict.sanitizer import sanitize_claim
from verdict.graph_store import GraphStore, FalkorDBConnectionError
from verdict.trust import TrustEngine
from verdict.news_retriever import LiveNewsRetriever
from verdict.retriever import EvidenceItem, EvidenceRetriever
from verdict.agent import VerdictAgent


def test_input_sanitizer_null_bytes_and_control_chars():
    # Null bytes and unprintable control characters are stripped
    raw = "Special \x00 null \t chars \r\n and \x08\x0b\x0c\x1f\x7f symbols <>?!"
    cleaned = sanitize_claim(raw)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "Special null chars and symbols <>?!" in cleaned


def test_input_sanitizer_length_capping():
    # Very long claims are truncated on word boundary
    long_claim = "word " * 500  # 2500 characters
    cleaned = sanitize_claim(long_claim, max_length=100)
    assert len(cleaned) <= 100
    assert not cleaned.endswith(" ")
    assert cleaned.startswith("word")


def test_input_sanitizer_empty_and_none():
    assert sanitize_claim("") == ""
    assert sanitize_claim("   ") == ""
    assert sanitize_claim(None) == ""


def test_agent_investigate_empty_claim_returns_clean_unverified(agent):
    res = agent.investigate("   \x00   \t\n  ")
    assert res["verdict"] == "UNVERIFIED"
    assert res["confidence"] == 0.0
    assert "Abstained" in res["reason"]
    assert res["graph_counts"]["total_nodes"] == 0


def test_falkordb_unreachable_raises_clear_error():
    with pytest.raises(FalkorDBConnectionError) as exc_info:
        GraphStore(host="localhost", port=59999, graph_name="ghost")
    err = str(exc_info.value)
    assert "FalkorDB is unreachable" in err
    assert "docker run" in err


def test_fact_checker_site_scoped_retrieval_mocked():
    mock_session = MagicMock(spec=requests.Session)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Fact Check: False claim that Apollo 11 landed on Mars - Snopes</title>
          <link>https://www.snopes.com/fact-check/apollo-11-mars/</link>
          <pubDate>Thu, 15 Oct 2026 10:00:00 GMT</pubDate>
          <description>Snopes verified that Apollo 11 landed on the Moon, not Mars.</description>
          <source url="https://www.snopes.com">Snopes</source>
        </item>
      </channel>
    </rss>"""
    mock_session.get.return_value = mock_resp
    mock_session.headers = {}

    retriever = LiveNewsRetriever(session=mock_session)
    sub = SubClaim(
        id="sub_fc_test",
        text="Apollo 11 landed on Mars in 1969.",
        keywords=["Apollo", "11", "landed", "Mars", "1969"],
        entities=["Apollo 11"],
        time_expression="1969"
    )

    ev_items, citations = retriever.retrieve_fact_checks_for_subclaim(sub, top_k_per_site=1)

    assert len(ev_items) >= 1
    ev = ev_items[0]
    assert ev.source_type == "fact_checker"
    assert "FactCheck: Snopes" in ev.source_name
    assert ev.stance == "REFUTES"
    assert ev.weight >= 0.90
    assert any(c.source_to_url == "https://www.snopes.com" for c in citations)


def test_fact_checker_higher_trust_weight_in_falkordb(test_graph):
    # Add regular news source and fact-checker source
    test_graph.add_root_claim("claim_fc_trust", "Test Claim")
    sub = SubClaim(id="sub_fc_t", text="Test Claim.")
    test_graph.add_subclaim("claim_fc_trust", sub)

    ev_news = EvidenceItem(
        id="ev_std_news",
        subclaim_id="sub_fc_t",
        text="Standard blog report.",
        source_name="News: Regular Outlet",
        source_url="https://standard-outlet.com/article1",
        source_type="news",
        stance="SUPPORTS",
        weight=0.70
    )
    ev_fc = EvidenceItem(
        id="ev_snopes_fc",
        subclaim_id="sub_fc_t",
        text="Snopes official investigation.",
        source_name="FactCheck: Snopes",
        source_url="https://www.snopes.com/fact-check/test-claim",
        source_type="fact_checker",
        stance="REFUTES",
        weight=0.98
    )

    test_graph.add_evidence(ev_news, "sub_fc_t")
    test_graph.add_evidence(ev_fc, "sub_fc_t")

    # Add citations
    test_graph.add_citation("https://standard-outlet.com/article1", "https://standard-outlet.com")
    test_graph.add_citation("https://www.snopes.com/fact-check/test-claim", "https://www.snopes.com")

    # Run trust engine
    trust = TrustEngine(test_graph)
    scores = trust.compute_source_trust()

    # Fact-checker source should have higher trust score in FalkorDB
    fc_url = "https://www.snopes.com/fact-check/test-claim"
    news_url = "https://standard-outlet.com/article1"
    assert scores[fc_url] >= 0.85
    assert scores[fc_url] > scores[news_url]
