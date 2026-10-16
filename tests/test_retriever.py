"""
Test 2: Evidence Retriever Tests with Mocked HTTP Responses.
"""

from unittest.mock import MagicMock
import pytest
import requests

from verdict.claim_decomposer import SubClaim
from verdict.retriever import EvidenceRetriever


def test_mocked_retriever_wikipedia_and_wikidata():
    mock_session = MagicMock(spec=requests.Session)

    def mock_get(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        if "w/api.php" in url and kwargs.get("params", {}).get("action") == "query":
            mock_resp.json.return_value = {
                "query": {
                    "search": [
                        {"title": "Eiffel Tower", "snippet": "Built between 1887 and 1889."}
                    ]
                }
            }
        elif "page/summary" in url:
            mock_resp.json.return_value = {
                "extract": "The Eiffel Tower was built from 1887 to 1889. It was inaugurated in 1889.",
                "content_urls": {
                    "desktop": {"page": "https://en.wikipedia.org/wiki/Eiffel_Tower"}
                }
            }
        elif "w/api.php" in url and kwargs.get("params", {}).get("action") == "wbsearchentities":
            mock_resp.json.return_value = {
                "search": [
                    {
                        "id": "Q243",
                        "label": "Eiffel Tower",
                        "description": "tower located on Champ de Mars in Paris"
                    }
                ]
            }
        else:
            mock_resp.json.return_value = {}

        return mock_resp

    mock_session.get.side_effect = mock_get
    mock_session.headers = {}
    retriever = EvidenceRetriever(session=mock_session)

    subclaim = SubClaim(
        id="sub_test",
        text="The Eiffel Tower was completed in 1889.",
        keywords=["Eiffel", "Tower", "1889"],
        entities=["The Eiffel Tower"],
        time_expression="1889"
    )

    ev_items, citations = retriever.retrieve_evidence_for_subclaim(subclaim, top_k=2)

    assert len(ev_items) > 0
    assert any(e.stance == "SUPPORTS" for e in ev_items)
    assert any("wikipedia.org" in e.source_url for e in ev_items)
    assert any(c.relationship == "CITES" for c in citations)
