"""
Retriever: Fetches public evidence from Wikipedia and Wikidata APIs.
Extracts factual snippets, source metadata, citation links, and evaluates evidence stance.
"""

from __future__ import annotations
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import requests

from verdict.claim_decomposer import SubClaim


@dataclass
class EvidenceItem:
    id: str
    subclaim_id: str
    text: str
    source_name: str
    source_url: str
    source_type: str  # "wikipedia" | "wikidata" | "news"
    stance: str       # "SUPPORTS" | "REFUTES" | "QUALIFIES"
    weight: float     # 0.0 to 1.0
    snippet: str = ""
    target_attribute: Optional[str] = None
    published_date: Optional[str] = None


@dataclass
class CitationLink:
    source_from_url: str
    source_to_url: str
    relationship: str = "CITES"


from verdict.news_retriever import LiveNewsRetriever


class EvidenceRetriever:
    """
    Evidence Retriever: Fetches public evidence from Wikipedia/Wikidata (PRIMARY)
    and live web/news APIs (SECONDARY).
    Wikipedia/Wikidata serves as the primary encyclopedic knowledge base.
    The secondary live news source is seamlessly queried when Wikipedia has gaps
    or when verifying recent events where static encyclopedias lag.
    """

    USER_AGENT = "VerdictAgent/1.0 (Hackathon Track 1; connected-data-agent; contact@example.com)"
    WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"
    WIKI_REST_API = "https://en.wikipedia.org/api/rest_v1/page/summary"
    WIKIDATA_API = "https://www.wikidata.org/w/api.php"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        timeout: float = 6.0,
        serpapi_key: Optional[str] = None,
        news_mode: str = "on_gap"
    ):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        self.timeout = timeout
        self.news_mode = news_mode  # "on_gap" | "always" | "disabled"
        self.news_retriever = LiveNewsRetriever(
            session=self.session,
            timeout=self.timeout,
            serpapi_key=serpapi_key
        )

    def retrieve_evidence_for_subclaim(
        self,
        subclaim: SubClaim,
        top_k: int = 3
    ) -> Tuple[List[EvidenceItem], List[CitationLink]]:
        """
        Retrieves top-k evidence items and cross-source citations for a subclaim.
        """
        evidence_list: List[EvidenceItem] = []
        citations: List[CitationLink] = []

        # 1. Search Wikipedia
        search_query = self._build_search_query(subclaim)
        wiki_pages = self._search_wikipedia(search_query, limit=top_k)

        ev_idx = 1
        for page in wiki_pages:
            title = page.get("title", "")
            if not title:
                continue

            summary, page_url = self._get_wikipedia_summary(title)
            if not summary:
                continue

            # Extract best candidate sentences from summary
            sentences = self._split_sentences(summary)
            for sent in sentences:
                stance, weight = self._evaluate_stance(subclaim, sent)
                if weight > 0.35:
                    ev_item = EvidenceItem(
                        id=f"ev_{subclaim.id}_{ev_idx}",
                        subclaim_id=subclaim.id,
                        text=sent.strip(),
                        source_name=f"Wikipedia: {title}",
                        source_url=page_url,
                        source_type="wikipedia",
                        stance=stance,
                        weight=round(weight, 3),
                        snippet=summary[:200]
                    )
                    evidence_list.append(ev_item)
                    ev_idx += 1

        # 2. Search Wikidata for structured entity facts using cleaned entity names
        wikidata_query = self._build_wikidata_query(subclaim)
        wikidata_entities = self._search_wikidata(wikidata_query, limit=2)
        for ent in wikidata_entities:
            ent_id = ent.get("id")
            ent_label = ent.get("label", "")
            ent_desc = ent.get("description", "")
            ent_url = f"https://www.wikidata.org/wiki/{ent_id}"

            if ent_desc:
                fact_text = f"{ent_label}: {ent_desc}"
                stance, weight = self._evaluate_stance(subclaim, fact_text)
                if weight > 0.30:
                    ev_item = EvidenceItem(
                        id=f"ev_{subclaim.id}_{ev_idx}",
                        subclaim_id=subclaim.id,
                        text=fact_text,
                        source_name=f"Wikidata: {ent_label} ({ent_id})",
                        source_url=ent_url,
                        source_type="wikidata",
                        stance=stance,
                        weight=round(weight, 3),
                        snippet=ent_desc
                    )
                    evidence_list.append(ev_item)
                    ev_idx += 1

            # Cross-source citation links for PageRank:
            # Wikipedia articles cite Wikidata items and primary sources
            for ev in evidence_list:
                if ev.source_type == "wikipedia" and ev.source_url != ent_url:
                    citations.append(CitationLink(
                        source_from_url=ev.source_url,
                        source_to_url=ent_url,
                        relationship="CITES"
                    ))

        # 3. SECONDARY: Live Web/News Evidence
        # Triggered when Wikipedia/Wikidata has gaps, or when claim refers to recent events.
        if self.news_mode != "disabled":
            should_fetch_news = False
            if self.news_mode == "always":
                should_fetch_news = True
            else:
                has_zero_evidence = (len(evidence_list) == 0)
                has_only_weak = (max((e.weight for e in evidence_list), default=0.0) < 0.50)
                has_no_definitive_stance = all(e.stance == "QUALIFIES" for e in evidence_list)
                has_recency_cue = self._has_recency_cue(subclaim)

                should_fetch_news = (
                    has_zero_evidence
                    or has_only_weak
                    or has_no_definitive_stance
                    or has_recency_cue
                )

            if should_fetch_news:
                news_items, news_citations = self.news_retriever.retrieve_news_for_subclaim(
                    subclaim=subclaim,
                    top_k=top_k,
                    start_idx=ev_idx
                )
                evidence_list.extend(news_items)
                citations.extend(news_citations)

        # Deduplicate citations
        unique_citations = []
        seen = set()
        for c in citations:
            key = (c.source_from_url, c.source_to_url, c.relationship)
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        return evidence_list[:top_k * 3], unique_citations

    def _has_recency_cue(self, subclaim: SubClaim) -> bool:
        """Checks if subclaim references recent years, breaking events, or temporal phrases."""
        text = subclaim.text.lower()
        time_expr = (subclaim.time_expression or "").lower()
        recent_years = {"2024", "2025", "2026", "2027"}
        if any(yr in time_expr or yr in text for yr in recent_years):
            return True
        recency_words = {
            "recent", "recently", "latest", "breaking", "unveiled", "announced",
            "this week", "this month", "this year", "yesterday", "today", "newly",
            "robotaxi", "cybercab", "starship", "flight 5", "flight 6"
        }
        return any(w in text for w in recency_words)

    def _build_search_query(self, subclaim: SubClaim) -> str:
        """Constructs an effective search query from subclaim entities and keywords."""
        if subclaim.entities:
            query = " ".join(subclaim.entities)
            if subclaim.time_expression and subclaim.time_expression not in query:
                query += f" {subclaim.time_expression}"
            return query
        # Fallback to keywords
        return " ".join(subclaim.keywords[:5])

    def _build_wikidata_query(self, subclaim: SubClaim) -> str:
        """Constructs a clean entity search query for Wikidata."""
        if subclaim.entities:
            # Pick first entity and strip leading 'The ', 'A '
            raw = subclaim.entities[0]
            cleaned = re.sub(r'^(The|A|An)\s+', '', raw, flags=re.IGNORECASE)
            return cleaned.strip()
        return " ".join(subclaim.keywords[:2])

    def _search_wikipedia(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        try:
            resp = self.session.get(
                self.WIKI_SEARCH_API,
                params={"action": "query", "list": "search", "srsearch": query, "format": "json"},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("query", {}).get("search", [])[:limit]
        except Exception:
            return []

    def _get_wikipedia_summary(self, title: str) -> Tuple[str, str]:
        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        url = f"{self.WIKI_REST_API}/{encoded_title}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            extract = data.get("extract", "")
            desktop_url = data.get("content_urls", {}).get("desktop", {}).get(
                "page", f"https://en.wikipedia.org/wiki/{encoded_title}"
            )
            return extract, desktop_url
        except Exception:
            return "", f"https://en.wikipedia.org/wiki/{encoded_title}"

    def _search_wikidata(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        try:
            resp = self.session.get(
                self.WIKIDATA_API,
                params={"action": "wbsearchentities", "search": query, "language": "en", "format": "json"},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("search", [])[:limit]
        except Exception:
            return []

    def _split_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter on punctuation followed by space
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]

    def _evaluate_stance(self, subclaim: SubClaim, evidence_text: str) -> Tuple[str, float]:
        """
        Determines if evidence SUPPORTS, REFUTES, or QUALIFIES the subclaim.
        Computes stance weight based on semantic overlap and contradiction cues.
        """
        claim_text = subclaim.text.lower()
        ev_lower = evidence_text.lower()

        # Keyword overlap
        claim_keywords = set(w.lower() for w in subclaim.keywords if len(w) > 2)
        if not claim_keywords:
            return "QUALIFIES", 0.3

        ev_words = set(re.findall(r'\b[a-z0-9_-]+\b', ev_lower))
        overlap = claim_keywords.intersection(ev_words)
        overlap_ratio = len(overlap) / max(1, len(claim_keywords))

        # Check year/date contradiction
        claim_year = subclaim.time_expression
        years_in_ev = re.findall(r'\b(1[6-9][0-9]{2}|20[0-3][0-9])\b', ev_lower)

        # Contradiction detection:
        # e.g. Claim: Apollo 11 landed on Mars. Ev: Apollo 11 landed humans on the Moon.
        # Key locations/targets:
        known_celestial = {"moon", "mars", "venus", "jupiter", "saturn", "mercury"}
        claim_celestial = {w for w in claim_keywords if w in known_celestial}
        ev_celestial = {w for w in ev_words if w in known_celestial}

        if claim_celestial and ev_celestial:
            if claim_celestial != ev_celestial:
                # Direct contradiction in key target entity
                return "REFUTES", 0.92

        # Check temporal contradiction if year is specified
        if claim_year and years_in_ev:
            if claim_year in years_in_ev:
                # Exact year match + high overlap
                if overlap_ratio >= 0.4:
                    return "SUPPORTS", min(0.95, 0.6 + overlap_ratio * 0.4)
            else:
                # If subject is explicitly matched and event year is different
                if overlap_ratio >= 0.5:
                    return "REFUTES", 0.85

        # Check explicit debunking or negated predicate phrases
        debunk_match = re.search(
            r'\b(was not|were not|is not|did not|never|debunked|refuted|fictional|myth|false|disproved|untrue|contrary to)\b',
            ev_lower
        )
        if debunk_match and overlap_ratio >= 0.45:
            return "REFUTES", 0.88

        # Standard support / qualifier evaluation
        if overlap_ratio >= 0.6:
            return "SUPPORTS", min(0.95, overlap_ratio)
        elif overlap_ratio >= 0.35:
            return "SUPPORTS" if any(ent.lower() in ev_lower for ent in subclaim.entities) else "QUALIFIES", 0.65
        elif overlap_ratio >= 0.2:
            return "QUALIFIES", 0.40
        else:
            return "QUALIFIES", 0.20
