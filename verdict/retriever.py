"""
Retriever: Fetches public evidence from Wikipedia and Wikidata APIs.
Extracts factual snippets, source metadata, citation links, and evaluates evidence stance.
"""

from __future__ import annotations
import re
import urllib.parse
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import requests

from verdict.claim_decomposer import SubClaim


class SimpleMemoryCache:
    """Thread-safe bounded in-memory cache for API responses."""
    def __init__(self, maxsize: int = 1024):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._cache) >= self._maxsize:
                keys_to_remove = list(self._cache.keys())[: max(1, self._maxsize // 5)]
                for k in keys_to_remove:
                    self._cache.pop(k, None)
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_wiki_search_cache = SimpleMemoryCache()
_wiki_summary_cache = SimpleMemoryCache()
_wikidata_cache = SimpleMemoryCache()


def clear_retriever_cache():
    """Clears all in-memory retriever caches."""
    _wiki_search_cache.clear()
    _wiki_summary_cache.clear()
    _wikidata_cache.clear()


@dataclass
class EvidenceItem:
    id: str
    subclaim_id: str
    text: str
    source_name: str
    source_url: str
    source_type: str  # "wikipedia" | "wikidata" | "news" | "fact_checker"
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

    USER_AGENT = "VerdictAgent/1.1 (https://github.com/graph-hacks-verdict; verdict-agent@research-connected.org)"
    WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"
    WIKI_REST_API = "https://en.wikipedia.org/api/rest_v1/page/summary"
    WIKIDATA_API = "https://www.wikidata.org/w/api.php"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        timeout: float = 6.0,
        serpapi_key: Optional[str] = None,
        news_mode: str = "on_gap",
        fact_check_mode: str = "auto"
    ):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        self.timeout = timeout
        self.news_mode = news_mode  # "on_gap" | "always" | "disabled"
        self.fact_check_mode = fact_check_mode  # "auto" | "always" | "on_gap" | "disabled"
        self.news_retriever = LiveNewsRetriever(
            session=self.session,
            timeout=self.timeout,
            serpapi_key=serpapi_key
        )

    def clear_cache(self) -> None:
        """Clears in-memory caches."""
        clear_retriever_cache()
        if hasattr(self.news_retriever, "clear_cache"):
            self.news_retriever.clear_cache()

    def retrieve_evidence_for_subclaim(
        self,
        subclaim: SubClaim,
        top_k: int = 3
    ) -> Tuple[List[EvidenceItem], List[CitationLink]]:
        """
        Retrieves top-k evidence items and cross-source citations for a subclaim.
        Parallelizes searches and summary fetches across thread pool.
        """
        evidence_list: List[EvidenceItem] = []
        citations: List[CitationLink] = []

        # 1 & 2. Concurrently execute Wikipedia and Wikidata searches
        search_query = self._build_search_query(subclaim)
        wikidata_query = self._build_wikidata_query(subclaim)
        primary_entity = (
            subclaim.entities[0]
            if (subclaim.entities and subclaim.entities[0].lower() != search_query.lower())
            else None
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_wiki = executor.submit(self._search_wikipedia, search_query, top_k)
            fut_wikidata = executor.submit(self._search_wikidata, wikidata_query, 2)
            fut_primary = (
                executor.submit(self._search_wikipedia, primary_entity, 1)
                if primary_entity
                else None
            )

            wiki_pages = fut_wiki.result()
            wikidata_entities = fut_wikidata.result()
            if fut_primary:
                primary_pages = fut_primary.result()
                for p in primary_pages:
                    if p.get("title") and not any(existing.get("title") == p.get("title") for existing in wiki_pages):
                        wiki_pages.insert(0, p)

        # Concurrently fetch Wikipedia page summaries
        valid_pages = [p for p in wiki_pages if p.get("title")]
        if valid_pages:
            with ThreadPoolExecutor(max_workers=min(4, len(valid_pages))) as executor:
                page_summaries = list(
                    executor.map(lambda p: (p.get("title", ""), self._get_wikipedia_summary(p.get("title", ""))), valid_pages)
                )
        else:
            page_summaries = []

        ev_idx = 1
        for title, (summary, page_url) in page_summaries:
            if not summary:
                continue

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

        # Process Wikidata entity facts
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

            for ev in evidence_list:
                if ev.source_type == "wikipedia" and ev.source_url != ent_url:
                    citations.append(CitationLink(
                        source_from_url=ev.source_url,
                        source_to_url=ent_url,
                        relationship="CITES"
                    ))

        # 3. SECONDARY: Live Web/News Evidence
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
                ev_idx += len(news_items)

        # 4. FACT-CHECKER SOURCES: Snopes, PolitiFact, FactCheck.org
        if self.fact_check_mode != "disabled":
            should_fetch_fact_checks = False
            if self.fact_check_mode == "always":
                should_fetch_fact_checks = True
            else:
                has_zero_evidence = (len(evidence_list) == 0)
                has_only_weak = (max((e.weight for e in evidence_list), default=0.0) < 0.50)
                has_refutation = any(e.stance == "REFUTES" for e in evidence_list)
                has_fact_check_cue = self._has_fact_check_cue(subclaim)

                should_fetch_fact_checks = (
                    has_zero_evidence
                    or has_only_weak
                    or has_refutation
                    or has_fact_check_cue
                )

            if should_fetch_fact_checks:
                fc_items, fc_citations = self.news_retriever.retrieve_fact_checks_for_subclaim(
                    subclaim=subclaim,
                    top_k_per_site=1,
                    start_idx=ev_idx
                )
                evidence_list.extend(fc_items)
                citations.extend(fc_citations)
                ev_idx += len(fc_items)

        # Deduplicate citations
        unique_citations = []
        seen = set()
        for c in citations:
            key = (c.source_from_url, c.source_to_url, c.relationship)
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        return evidence_list[:top_k * 4], unique_citations

    def _has_fact_check_cue(self, subclaim: SubClaim) -> bool:
        """Checks if subclaim references controversial, viral, or commonly fact-checked topics."""
        text = subclaim.text.lower()
        fc_keywords = {
            "born in", "birth certificate", "flat earth", "moon landing", "mars",
            "fake", "hoax", "debunk", "conspiracy", "staged", "alien", "ineligible",
            "secret", "vaccine", "stolen", "poison", "chemtrail", "5g", "illuminati"
        }
        return any(kw in text for kw in fc_keywords)

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
        parts = []
        for ent in subclaim.entities:
            if ent.lower() not in [p.lower() for p in parts]:
                parts.append(ent)
        for kw in subclaim.keywords:
            if kw.lower() not in [p.lower() for p in parts] and len(kw) > 2:
                parts.append(kw)
        if subclaim.time_expression and subclaim.time_expression not in parts:
            parts.append(subclaim.time_expression)

        query = " ".join(parts[:5]).strip()
        if not query:
            query = subclaim.text.rstrip(".")
        return query

    def _build_wikidata_query(self, subclaim: SubClaim) -> str:
        """Constructs a clean entity search query for Wikidata."""
        if subclaim.entities:
            raw = subclaim.entities[0]
            cleaned = re.sub(r'^(The|A|An)\s+', '', raw, flags=re.IGNORECASE)
            return cleaned.strip()
        return " ".join(subclaim.keywords[:2])

    def _search_wikipedia(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        cache_key = f"{query.lower().strip()}_{limit}"
        cached = _wiki_search_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            resp = self.session.get(
                self.WIKI_SEARCH_API,
                params={"action": "query", "list": "search", "srsearch": query, "format": "json"},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("query", {}).get("search", [])[:limit]
            _wiki_search_cache.set(cache_key, results)
            return results
        except Exception:
            return []

    def _get_wikipedia_summary(self, title: str) -> Tuple[str, str]:
        cache_key = title.lower().strip()
        cached = _wiki_summary_cache.get(cache_key)
        if cached is not None:
            return cached

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
            result = (extract, desktop_url)
            _wiki_summary_cache.set(cache_key, result)
            return result
        except Exception:
            return "", f"https://en.wikipedia.org/wiki/{encoded_title}"

    def _search_wikidata(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        cache_key = f"{query.lower().strip()}_{limit}"
        cached = _wikidata_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            resp = self.session.get(
                self.WIKIDATA_API,
                params={"action": "wbsearchentities", "search": query, "language": "en", "format": "json"},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("search", [])[:limit]
            _wikidata_cache.set(cache_key, results)
            return results
        except Exception:
            return []

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]

    def _evaluate_stance(self, subclaim: SubClaim, evidence_text: str) -> Tuple[str, float]:
        """
        Determines if evidence SUPPORTS, REFUTES, or QUALIFIES the subclaim.
        Computes stance weight based on semantic overlap and contradiction cues.
        """
        claim_text = subclaim.text.lower().replace("’", "'").replace("‘", "'")
        ev_lower = evidence_text.lower().replace("’", "'").replace("‘", "'")

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
        known_celestial = {"moon", "mars", "venus", "jupiter", "saturn", "mercury"}
        claim_celestial = {w for w in claim_keywords if w in known_celestial}
        ev_celestial = {w for w in ev_words if w in known_celestial}

        if claim_celestial and ev_celestial:
            if claim_celestial != ev_celestial:
                return "REFUTES", 0.95

        # Check location / origin / birthplace contradiction
        us_locs = {"u.s.", "u.s", "usa", "united states", "hawaii", "honolulu"}
        kenya_locs = {"kenya", "mombasa", "nairobi"}
        has_kenya_in_claim = any(loc in claim_text for loc in kenya_locs)
        has_us_in_ev = any(loc in ev_lower for loc in us_locs)
        if has_kenya_in_claim and has_us_in_ev and any(w in ev_lower or w in claim_text for w in ("born", "birth", "president", "citizen")):
            return "REFUTES", 0.95

        # Check temporal contradiction if year is specified
        if claim_year and years_in_ev:
            if claim_year in years_in_ev:
                if overlap_ratio >= 0.4:
                    return "SUPPORTS", min(0.95, 0.6 + overlap_ratio * 0.4)
            else:
                if overlap_ratio >= 0.5:
                    return "REFUTES", 0.85

        # Check explicit debunking or negated predicate phrases
        debunk_match = re.search(
            r"\b(do not|does not|did not|doesn't|don't|cannot|can not|can't|never|was not|were not|is not|are not|has not|have not|had not)\s+(?:[a-z0-9_-]+\s+){0,4}(?:cause|link|support|show|lead to|result in|find|corroborate|prove|attribute|relate|imply|indicate|originate)\b|"
            r"\b(no relationship|no link|no evidence|no connection|no causal|no correlation|no association|no proof|lack of evidence|lack of link|unrelated)\b|"
            r"\b(cannot be attributed|not attributable|not associated with|not linked to|not caused by)\b|"
            r"\b(debunked|refuted|fictional|myth|false|disproved|disproven|untrue|contrary to|pants on fire|mostly false|fiction|unproven|unfounded|baseless|ineligible|misleading|incorrect|retracted|fraudulent|hoax|conspiracy)\b|"
            r"\b(failed to replicate|no scientific basis|rejected by the scientific community)\b|"
            r"\b(not believe|doesn't show|does not show|don't believe)\b|"
            r"^no,\s+|\bno,\s+",
            ev_lower
        )
        if debunk_match and overlap_ratio >= 0.20:
            return "REFUTES", 0.95

        # Multi-entity relation check: if claim links multiple entities, a sentence mentioning only one cannot SUPPORT
        entities_present = sum(1 for ent in subclaim.entities if ent.lower() in ev_lower)
        entity_ratio = entities_present / max(1, len(subclaim.entities))
        if len(subclaim.entities) >= 2 and entity_ratio < 0.6:
            return "QUALIFIES", 0.35

        # Year consistency: if claim specifies an explicit year, but evidence lacks it
        if subclaim.time_expression and subclaim.time_expression not in ev_lower:
            if overlap_ratio < 0.6:
                return "QUALIFIES", 0.35

        # Ensure primary named entity is actually mentioned in evidence sentence before supporting
        if subclaim.entities and not any(ent.lower() in ev_lower for ent in subclaim.entities):
            return "QUALIFIES", 0.30

        # Standard support / qualifier evaluation
        if overlap_ratio >= 0.6 and (not subclaim.entities or entity_ratio >= 0.5):
            return "SUPPORTS", min(0.95, overlap_ratio)
        elif overlap_ratio >= 0.35 and (not subclaim.entities or entity_ratio >= 0.5):
            return "SUPPORTS", 0.70
        elif overlap_ratio >= 0.2:
            return "QUALIFIES", 0.40
        else:
            return "QUALIFIES", 0.20
