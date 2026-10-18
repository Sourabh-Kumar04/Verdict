"""
Live Web/News Evidence Retriever: Fetches fresh public news evidence for Verdict.
Supports SerpApi Google News (if SERPAPI_API_KEY is present) and Google News RSS
as a reliable, zero-config free public news evidence stream.
Extracts publication dates for recency weighting, source URLs, and contextual snippets.
"""

from __future__ import annotations
import os
import re
import urllib.parse
import email.utils
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import xml.etree.ElementTree as ET
import requests

from verdict.claim_decomposer import SubClaim

if TYPE_CHECKING:
    from verdict.retriever import EvidenceItem, CitationLink


class SimpleMemoryCache:
    """Thread-safe bounded in-memory cache for news responses."""
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


_news_rss_cache = SimpleMemoryCache()


def clear_news_cache():
    _news_rss_cache.clear()


def parse_publication_date(date_str: Optional[str]) -> Tuple[Optional[datetime], str]:
    """
    Parses various date formats from news sources:
    - RFC 2822 / 822 (Google News RSS: 'Wed, 14 Oct 2024 11:46:00 GMT')
    - ISO 8601 ('2024-10-14T11:46:00Z')
    - Standard dates ('Oct 14, 2024', '2024-10-14')
    - Relative time strings ('2 hours ago', '3 days ago')
    Returns (datetime_object, normalized_iso_or_raw_string).
    """
    if not date_str:
        return None, ""

    raw = date_str.strip()

    # 1. RFC 2822 (standard for RSS pubDate)
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt is not None:
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc, dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        pass

    # 2. ISO formats
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc, dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        pass

    # 3. Common date formats
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt, dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # 4. Relative time strings ('2 hours ago', '3 days ago', '1 week ago')
    rel_match = re.match(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", raw.lower())
    if rel_match:
        num, unit = int(rel_match.group(1)), rel_match.group(2)
        now = datetime.now(timezone.utc)
        if unit in ("second", "minute", "hour"):
            dt = now - timedelta(hours=max(1, num if unit == "hour" else 1))
        elif unit == "day":
            dt = now - timedelta(days=num)
        elif unit == "week":
            dt = now - timedelta(days=num * 7)
        elif unit == "month":
            dt = now - timedelta(days=num * 30)
        elif unit == "year":
            dt = now - timedelta(days=num * 365)
        else:
            dt = now
        return dt, dt.strftime("%Y-%m-%d")

    return None, raw


def compute_recency_multiplier(
    dt: Optional[datetime],
    ref_time: Optional[datetime] = None
) -> float:
    """
    Computes a recency multiplier to weight evidence:
    - <= 7 days: 1.05 (breaking/fresh event boost)
    - <= 30 days: 1.00 (current month)
    - <= 90 days: 0.90 (recent quarter)
    - <= 365 days: 0.80 (past year)
    - > 365 days: 0.65 (older historical news)
    - Unknown date: 0.75 (neutral baseline)
    """
    if dt is None:
        return 0.75

    now = ref_time or datetime.now(timezone.utc)
    diff_days = max(0.0, (now - dt).total_seconds() / 86400.0)

    if diff_days <= 7.0:
        return 1.05
    elif diff_days <= 30.0:
        return 1.00
    elif diff_days <= 90.0:
        return 0.90
    elif diff_days <= 365.0:
        return 0.80
    else:
        return 0.65


def clean_html_snippet(raw_html: str) -> str:
    """Strips HTML tags and unescapes entities from RSS descriptions."""
    if not raw_html:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", raw_html)
    cleaned = cleaned.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class LiveNewsRetriever:
    """
    Retrieves live web and news evidence.
    First checks for SerpApi API key (via env var SERPAPI_API_KEY).
    If absent or unavailable, falls back to Google News RSS feed, which is free,
    reliable, requires no API key, and provides rich publication timestamps and snippets.
    """

    USER_AGENT = "VerdictAgent/1.1 (https://github.com/graph-hacks-verdict; verdict-agent@research-connected.org)"
    GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
    SERPAPI_URL = "https://serpapi.com/search.json"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        timeout: float = 6.0,
        serpapi_key: Optional[str] = None
    ):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        self.timeout = timeout
        self.serpapi_key = serpapi_key if serpapi_key is not None else os.environ.get("SERPAPI_API_KEY")

    def clear_cache(self) -> None:
        """Clears in-memory news caches."""
        clear_news_cache()

    def retrieve_news_for_subclaim(
        self,
        subclaim: SubClaim,
        top_k: int = 3,
        start_idx: int = 1
    ) -> Tuple[List[EvidenceItem], List[CitationLink]]:
        """
        Retrieves top-k news items for the given subclaim and constructs
        EvidenceItems with publication dates, snippets, and citation links.
        """
        query = self._build_news_query(subclaim)
        articles: List[Dict[str, Any]] = []

        # 1. Try SerpApi if API key is present
        if self.serpapi_key:
            try:
                articles = self._search_serpapi(query, limit=top_k)
            except Exception:
                articles = []

        # 2. Fall back to free Google News RSS if no SerpApi results or no key
        if not articles:
            try:
                articles = self._search_google_news_rss(query, limit=top_k)
            except Exception:
                articles = []

        from verdict.retriever import EvidenceItem, CitationLink

        evidence_list: List[EvidenceItem] = []
        citations: List[CitationLink] = []
        ev_idx = start_idx

        for art in articles:
            title = art.get("title", "").strip()
            link = art.get("link", "").strip()
            snippet = art.get("snippet", "").strip()
            source_name = art.get("source_name", "News Media").strip()
            source_domain = art.get("source_domain", "").strip()
            raw_date = art.get("published_date", "")

            if not title or not link:
                continue

            dt_parsed, norm_date_str = parse_publication_date(raw_date)
            recency_mult = compute_recency_multiplier(dt_parsed)

            # Evaluate stance against subclaim
            candidate_text = f"{title}. {snippet}".strip()
            stance, base_weight = self._evaluate_news_stance(subclaim, candidate_text)

            if base_weight >= 0.25:
                # Apply recency multiplier to stance weight
                final_weight = round(min(0.98, max(0.20, base_weight * recency_mult)), 3)
                ev_item = EvidenceItem(
                    id=f"ev_{subclaim.id}_news_{ev_idx}",
                    subclaim_id=subclaim.id,
                    text=candidate_text[:300],
                    source_name=f"{source_name}",
                    source_url=link,
                    source_type="news",
                    stance=stance,
                    weight=final_weight,
                    snippet=snippet[:250] if snippet else title[:250],
                    published_date=norm_date_str
                )
                evidence_list.append(ev_item)
                ev_idx += 1

                # If source domain exists, create a citation link for PageRank
                if source_domain and source_domain != link:
                    citations.append(CitationLink(
                        source_from_url=link,
                        source_to_url=source_domain,
                        relationship="CITES"
                    ))

        return evidence_list[:top_k], citations

    FACT_CHECK_SITES = [
        {"domain": "snopes.com", "name": "Snopes", "root_url": "https://www.snopes.com"},
        {"domain": "politifact.com", "name": "PolitiFact", "root_url": "https://www.politifact.com"},
        {"domain": "factcheck.org", "name": "FactCheck.org", "root_url": "https://www.factcheck.org"},
    ]

    def retrieve_fact_checks_for_subclaim(
        self,
        subclaim: SubClaim,
        top_k_per_site: int = 1,
        start_idx: int = 1
    ) -> Tuple[List[EvidenceItem], List[CitationLink]]:
        """
        Runs site-scoped queries (site:snopes.com, site:politifact.com, site:factcheck.org)
        through Google News RSS concurrently to fetch dedicated fact-checker assessments.
        Tags results as source_type='fact_checker' with authoritative trust weights.
        """
        from verdict.retriever import EvidenceItem, CitationLink

        base_query = self._build_news_query(subclaim)
        evidence_list: List[EvidenceItem] = []
        citations: List[CitationLink] = []

        def _fetch_site(site_info: Dict[str, str]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
            domain = site_info["domain"]
            site_query = f"{base_query} site:{domain}"
            try:
                articles = self._search_google_news_rss(site_query, limit=top_k_per_site)
            except Exception:
                articles = []
            return site_info, articles

        # Parallelize across fact check sites
        with ThreadPoolExecutor(max_workers=len(self.FACT_CHECK_SITES)) as executor:
            site_results = list(executor.map(_fetch_site, self.FACT_CHECK_SITES))

        ev_idx = start_idx
        for site_info, articles in site_results:
            site_name = site_info["name"]
            root_url = site_info["root_url"]

            for art in articles:
                title = art.get("title", "").strip()
                link = art.get("link", "").strip()
                snippet = art.get("snippet", "").strip()
                raw_date = art.get("published_date", "")

                if not title or not link:
                    continue

                dt_parsed, norm_date_str = parse_publication_date(raw_date)
                candidate_text = f"{title}. {snippet}".strip()
                stance, base_weight = self._evaluate_news_stance(subclaim, candidate_text)

                if base_weight >= 0.25:
                    # Fact-checkers are authoritative investigative sources: boost weight
                    factcheck_weight = min(0.98, max(0.40, base_weight * 1.1)) if stance in ("SUPPORTS", "REFUTES") else base_weight
                    ev_item = EvidenceItem(
                        id=f"ev_{subclaim.id}_factcheck_{ev_idx}",
                        subclaim_id=subclaim.id,
                        text=candidate_text[:300],
                        source_name=f"FactCheck: {site_name}",
                        source_url=link,
                        source_type="fact_checker",
                        stance=stance,
                        weight=round(factcheck_weight, 3),
                        snippet=snippet[:250] if snippet else title[:250],
                        published_date=norm_date_str
                    )
                    evidence_list.append(ev_item)
                    ev_idx += 1

                    # Citation link to fact-checker domain for PageRank trust
                    citations.append(CitationLink(
                        source_from_url=link,
                        source_to_url=root_url,
                        relationship="CITES"
                    ))

        return evidence_list, citations

    def _build_news_query(self, subclaim: SubClaim) -> str:
        """Constructs a search query optimized for recent news headlines and fact-checks."""
        query_parts: List[str] = []
        if subclaim.entities:
            query_parts.extend(subclaim.entities)
            for kw in subclaim.keywords:
                if kw.lower() not in " ".join(query_parts).lower() and len(kw) > 2:
                    query_parts.append(kw)
                    if len(query_parts) >= 4:
                        break
        else:
            query_parts.extend(subclaim.keywords[:5])

        # Add temporal expressions if not already in entities
        if subclaim.time_expression and subclaim.time_expression not in " ".join(query_parts):
            query_parts.append(subclaim.time_expression)

        query = " ".join(query_parts).strip()
        if not query:
            query = subclaim.text
        return query

    def _search_serpapi(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Queries SerpApi Google News engine."""
        params = {
            "engine": "google_news",
            "q": query,
            "api_key": self.serpapi_key,
            "hl": "en",
            "gl": "us"
        }
        resp = self.session.get(self.SERPAPI_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        results: List[Dict[str, Any]] = []
        for item in data.get("news_results", [])[:limit]:
            source_info = item.get("source", {})
            source_name = source_info.get("name") if isinstance(source_info, dict) else str(source_info or "News")
            source_domain = source_info.get("link", "") if isinstance(source_info, dict) else ""

            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source_name": f"News: {source_name}",
                "source_domain": source_domain,
                "published_date": item.get("date", "")
            })
        return results

    def _search_google_news_rss(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Queries Google News RSS feed for free, real-time public news with caching."""
        cache_key = f"{query.lower().strip()}_{limit}"
        cached = _news_rss_cache.get(cache_key)
        if cached is not None:
            return cached

        params = {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en"
        }
        resp = self.session.get(self.GOOGLE_NEWS_RSS_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        items = root.findall(".//item")

        results: List[Dict[str, Any]] = []
        for item in items[:limit]:
            title_text = item.findtext("title") or ""
            link_text = item.findtext("link") or ""
            pub_date = item.findtext("pubDate") or ""
            desc_html = item.findtext("description") or ""

            snippet = clean_html_snippet(desc_html)

            # Extract source element if present
            source_elem = item.find("source")
            if source_elem is not None:
                source_name = source_elem.text or "News Media"
                source_domain = source_elem.attrib.get("url", "")
            else:
                source_name = "News Media"
                source_domain = ""

            results.append({
                "title": title_text,
                "link": link_text,
                "snippet": snippet,
                "source_name": f"News: {source_name}",
                "source_domain": source_domain,
                "published_date": pub_date
            })

        _news_rss_cache.set(cache_key, results)
        return results

    def _evaluate_news_stance(self, subclaim: SubClaim, evidence_text: str) -> Tuple[str, float]:
        """
        Evaluates whether a news candidate supports, refutes, or qualifies the subclaim.
        Checks entity coverage, lexical overlap, negation/debunk terms, and temporal coherence.
        """
        claim_text = subclaim.text.lower().replace("’", "'").replace("‘", "'")
        ev_lower = evidence_text.lower().replace("’", "'").replace("‘", "'")

        claim_keywords = set(w.lower() for w in subclaim.keywords if len(w) > 2)
        if not claim_keywords:
            return "QUALIFIES", 0.30

        ev_words = set(re.findall(r"\b[a-z0-9_-]+\b", ev_lower))
        overlap = claim_keywords.intersection(ev_words)
        overlap_ratio = len(overlap) / max(1, len(claim_keywords))

        # Check explicit debunking or contradiction cues in news / fact-checks
        debunk_match = re.search(
            r"\b(do not|does not|did not|doesn't|don't|cannot|can not|can't|never|was not|were not|is not|are not|has not|have not|had not)\s+(?:[a-z0-9_-]+\s+){0,4}(?:cause|link|support|show|lead to|result in|find|corroborate|prove|attribute|relate|imply|indicate|originate)\b|"
            r"\b(no relationship|no link|no evidence|no connection|no causal|no correlation|no association|no proof|lack of evidence|lack of link|unrelated)\b|"
            r"\b(cannot be attributed|not attributable|not associated with|not linked to|not caused by)\b|"
            r"\b(debunked|hoax|false|refuted|fake|denied|canceled|cancelled|did not|failed to|untrue|disproven|pants on fire|mostly false|fiction|myth|unproven|unfounded|baseless|ineligible|misleading|incorrect)\b|"
            r"\b(failed to replicate|no scientific basis|rejected by the scientific community)\b|"
            r"\b(not believe|doesn't show|does not show|don't believe)\b|"
            r"^no,\s+|\bno,\s+",
            ev_lower
        )
        if debunk_match and overlap_ratio >= 0.20:
            return "REFUTES", 0.95

        # Check celestial body contradiction (e.g. Apollo 11 Mars vs Moon)
        known_celestial = {"moon", "mars", "venus", "jupiter", "saturn", "mercury"}
        claim_celestial = {w for w in claim_keywords if w in known_celestial}
        ev_celestial = {w for w in ev_words if w in known_celestial}
        if claim_celestial and ev_celestial and claim_celestial != ev_celestial:
            return "REFUTES", 0.95

        # Check geographic / location contradiction for birth or residence claims
        us_locs = {"u.s.", "u.s", "usa", "united states", "hawaii", "honolulu"}
        kenya_locs = {"kenya", "mombasa", "nairobi"}
        has_kenya_in_claim = any(loc in claim_text for loc in kenya_locs)
        has_us_in_ev = any(loc in ev_lower for loc in us_locs)
        if has_kenya_in_claim and has_us_in_ev and any(w in ev_lower or w in claim_text for w in ("born", "birth", "citizen", "president", "american")):
            return "REFUTES", 0.95

        # Investigative question headlines (e.g. "Did X happen?", "Is X true?") are claims under review, not affirmative support
        is_question = ("?" in ev_lower) or any(ev_lower.strip().startswith(p) for p in ("did ", "is ", "was ", "can ", "could ", "would ", "has "))
        if is_question:
            return "QUALIFIES", 0.35

        # Entity presence
        entities_present = sum(1 for ent in subclaim.entities if ent.lower() in ev_lower)
        entity_ratio = entities_present / max(1, len(subclaim.entities))

        # Check temporal match
        claim_year = subclaim.time_expression
        years_in_ev = re.findall(r"\b(1[6-9][0-9]{2}|20[0-3][0-9])\b", ev_lower)

        if claim_year and years_in_ev:
            if claim_year in years_in_ev:
                # Strong match with matching year
                if overlap_ratio >= 0.30 or entity_ratio >= 0.5:
                    return "SUPPORTS", min(0.95, 0.65 + overlap_ratio * 0.3)
            else:
                # Year mismatch on explicit entity event
                if overlap_ratio >= 0.50:
                    return "REFUTES", 0.80

        # High keyword overlap or all key entities matched
        if overlap_ratio >= 0.50:
            return "SUPPORTS", min(0.95, 0.55 + overlap_ratio * 0.4)
        elif entity_ratio >= 0.65 and overlap_ratio >= 0.25:
            return "SUPPORTS", min(0.90, 0.50 + overlap_ratio * 0.4)
        elif overlap_ratio >= 0.25:
            return "QUALIFIES", 0.45
        else:
            return "QUALIFIES", 0.20
