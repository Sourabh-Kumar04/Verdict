"""
Tests for Live Web/News Evidence Retriever:
- SerpApi Google News mocked retrieval & parsing
- Google News RSS mocked retrieval & XML parsing
- Publication date parsing & recency weighting calculations
- FalkorDB graph integration (published_date, snippet, source_url)
- Primary/Secondary pipeline integration & gap fallback
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import pytest
import requests

from verdict.claim_decomposer import SubClaim
from verdict.retriever import EvidenceItem, EvidenceRetriever
from verdict.news_retriever import (
    LiveNewsRetriever,
    parse_publication_date,
    compute_recency_multiplier,
    clean_html_snippet
)
from verdict.trust import TrustEngine
from verdict.proof_path import ProofPathEngine
from verdict.verdict import VerdictEngine


def test_publication_date_parsing():
    # RFC 2822
    dt, s = parse_publication_date("Wed, 14 Oct 2026 11:46:00 GMT")
    assert dt is not None
    assert dt.year == 2026 and dt.month == 10 and dt.day == 14

    # ISO 8601
    dt_iso, s_iso = parse_publication_date("2026-10-14T11:46:00Z")
    assert dt_iso is not None
    assert dt_iso.year == 2026 and dt_iso.day == 14

    # Relative time
    dt_rel, s_rel = parse_publication_date("2 days ago")
    assert dt_rel is not None
    now = datetime.now(timezone.utc)
    assert abs((now - dt_rel).total_seconds() - 2 * 86400) < 300

    # Common format
    dt_fmt, s_fmt = parse_publication_date("Oct 14, 2026")
    assert dt_fmt is not None
    assert dt_fmt.month == 10 and dt_fmt.day == 14


def test_recency_weighting_calculation():
    now = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)

    # 3 days ago (breaking boost)
    d_fresh = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert compute_recency_multiplier(d_fresh, ref_time=now) == 1.05

    # 20 days ago (current month)
    d_month = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
    assert compute_recency_multiplier(d_month, ref_time=now) == 1.00

    # 60 days ago (recent quarter)
    d_quarter = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)
    assert compute_recency_multiplier(d_quarter, ref_time=now) == 0.90

    # 200 days ago (past year)
    d_year = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert compute_recency_multiplier(d_year, ref_time=now) == 0.80

    # 500 days ago (older)
    d_old = datetime(2025, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert compute_recency_multiplier(d_old, ref_time=now) == 0.65

    # None / unknown
    assert compute_recency_multiplier(None, ref_time=now) == 0.75


def test_clean_html_snippet():
    raw = '<a href="https://example.com" target="_blank">Headline Title</a>&nbsp;&nbsp;<font color="#6f6f6f">Reuters</font>'
    cleaned = clean_html_snippet(raw)
    assert "<" not in cleaned
    assert "Headline Title" in cleaned
    assert "Reuters" in cleaned
    assert "&nbsp;" not in cleaned


def test_serpapi_news_retriever_mocked():
    mock_session = MagicMock(spec=requests.Session)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "news_results": [
            {
                "title": "Tesla Cybercab Robotaxi Unveiled in Los Angeles",
                "link": "https://reuters.com/technology/tesla-cybercab-unveiled-2024",
                "snippet": "Tesla and Elon Musk officially unveiled the Cybercab autonomous robotaxi in October 2024.",
                "date": "2 days ago",
                "source": {
                    "name": "Reuters",
                    "link": "https://reuters.com"
                }
            }
        ]
    }
    mock_session.get.return_value = mock_resp
    mock_session.headers = {}

    retriever = LiveNewsRetriever(session=mock_session, serpapi_key="mock_serpapi_key_123")

    subclaim = SubClaim(
        id="sub_test_news",
        text="Tesla unveiled the Cybercab robotaxi in October 2024.",
        keywords=["Tesla", "unveiled", "Cybercab", "robotaxi", "October", "2024"],
        entities=["Tesla", "Cybercab"],
        time_expression="2024"
    )

    ev_items, citations = retriever.retrieve_news_for_subclaim(subclaim, top_k=2)

    assert len(ev_items) == 1
    ev = ev_items[0]
    assert ev.source_type == "news"
    assert ev.stance == "SUPPORTS"
    assert "reuters.com" in ev.source_url
    assert "Cybercab" in ev.snippet
    assert ev.published_date != ""
    assert ev.weight > 0.6

    assert len(citations) >= 1
    assert citations[0].source_to_url == "https://reuters.com"
    assert citations[0].relationship == "CITES"


def test_google_news_rss_retriever_mocked():
    mock_session = MagicMock(spec=requests.Session)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Google News</title>
        <item>
          <title>Tesla Cybercab Robotaxi Unveiled - Bloomberg</title>
          <link>https://news.google.com/rss/articles/mock123?oc=5</link>
          <pubDate>Mon, 12 Oct 2026 14:00:00 GMT</pubDate>
          <description>&lt;a href="https://example.com"&gt;Tesla Cybercab Robotaxi Unveiled&lt;/a&gt;</description>
          <source url="https://www.bloomberg.com">Bloomberg</source>
        </item>
      </channel>
    </rss>
    """
    mock_session.get.return_value = mock_resp
    mock_session.headers = {}

    retriever = LiveNewsRetriever(session=mock_session, serpapi_key=None)

    subclaim = SubClaim(
        id="sub_test_rss",
        text="Tesla unveiled the Cybercab robotaxi in October 2026.",
        keywords=["Tesla", "unveiled", "Cybercab", "robotaxi", "October", "2026"],
        entities=["Tesla", "Cybercab"],
        time_expression="2026"
    )

    ev_items, citations = retriever.retrieve_news_for_subclaim(subclaim, top_k=2)

    assert len(ev_items) == 1
    ev = ev_items[0]
    assert ev.source_type == "news"
    assert ev.stance == "SUPPORTS"
    assert "Bloomberg" in ev.source_name
    assert "2026-10-12" in ev.published_date
    assert len(citations) >= 1
    assert citations[0].source_to_url == "https://www.bloomberg.com"


def test_news_evidence_falkordb_graph_integration(test_graph):
    # Tests adding news evidence with publication date, snippet, and citation to FalkorDB
    test_graph.add_root_claim("claim_news_integ", "Tesla unveiled the Cybercab in 2024")
    sub = SubClaim(id="sub_news_1", text="Tesla unveiled the Cybercab in 2024.")
    test_graph.add_subclaim("claim_news_integ", sub)

    ev = EvidenceItem(
        id="ev_news_test_1",
        subclaim_id="sub_news_1",
        text="Tesla officially unveiled its autonomous Cybercab vehicle in 2024.",
        source_name="News: Reuters",
        source_url="https://reuters.com/article/cybercab-2024",
        source_type="news",
        stance="SUPPORTS",
        weight=0.92,
        snippet="Elon Musk presented the two-seater Cybercab robotaxi.",
        published_date="2024-10-11 18:00:00 UTC"
    )
    test_graph.add_evidence(ev, "sub_news_1")

    # Add citation and run PageRank
    test_graph.add_citation("https://reuters.com/article/cybercab-2024", "https://reuters.com")
    trust = TrustEngine(test_graph)
    scores = trust.compute_source_trust()
    assert len(scores) >= 2
    assert "https://reuters.com" in scores

    # Query FalkorDB node properties directly via Cypher
    res = test_graph.run_query("""
        MATCH (e:Evidence {id: 'ev_news_test_1'})
        RETURN e.published_date, e.snippet, e.weight
    """)
    assert len(res.result_set) == 1
    pub_date, snippet, weight = res.result_set[0]
    assert pub_date == "2024-10-11 18:00:00 UTC"
    assert "Cybercab" in snippet
    assert float(weight) == 0.92

    # Verify proof path includes published_date and snippet
    proof_engine = ProofPathEngine(test_graph)
    proof_steps = proof_engine.get_ranked_proof_path("claim_news_integ")
    assert len(proof_steps) >= 1
    step = proof_steps[0]
    assert step.published_date == "2024-10-11 18:00:00 UTC"
    assert "published: 2024-10-11 18:00:00 UTC" in step.to_line()

    # Evaluate verdict via graph queries
    v_engine = VerdictEngine(test_graph)
    v_res = v_engine.evaluate_claim("claim_news_integ")
    assert v_res.verdict == "SUPPORTED"
    assert v_res.confidence > 0.0


def test_evidence_pipeline_gap_triggers_secondary_news():
    # When Wikipedia/Wikidata returns no results (gap), the secondary news retriever is called
    mock_session = MagicMock(spec=requests.Session)

    def mock_get(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        if "wikipedia.org" in url or "wikidata.org" in url:
            # Empty Wikipedia/Wikidata response
            mock_resp.json.return_value = {"query": {"search": []}, "search": []}
            return mock_resp
        elif "news.google.com" in url:
            # News RSS returns evidence
            mock_resp.content = b"""<?xml version="1.0" encoding="UTF-8"?>
            <rss version="2.0">
              <channel>
                <item>
                  <title>Tesla Cybercab Robotaxi Unveiled in Texas - AP News</title>
                  <link>https://apnews.com/article/tesla-cybercab-2024</link>
                  <pubDate>Fri, 11 Oct 2024 16:00:00 GMT</pubDate>
                  <description>Tesla revealed the Cybercab robotaxi.</description>
                  <source url="https://apnews.com">AP News</source>
                </item>
              </channel>
            </rss>"""
            return mock_resp
        return mock_resp

    mock_session.get.side_effect = mock_get
    mock_session.headers = {}

    retriever = EvidenceRetriever(session=mock_session, news_mode="on_gap")

    subclaim = SubClaim(
        id="sub_gap_test",
        text="Tesla unveiled the Cybercab robotaxi in October 2024.",
        keywords=["Tesla", "unveiled", "Cybercab", "robotaxi", "October", "2024"],
        entities=["Tesla", "Cybercab"],
        time_expression="2024"
    )

    ev_items, citations = retriever.retrieve_evidence_for_subclaim(subclaim, top_k=2)

    # Wikipedia had 0 results, so news filled the gap
    assert len(ev_items) >= 1
    assert any(e.source_type == "news" for e in ev_items)
    assert any(e.stance == "SUPPORTS" for e in ev_items)
