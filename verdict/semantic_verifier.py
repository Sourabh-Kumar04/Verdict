"""
Semantic Stance Verification Layer using local Ollama LLM.
Hardens stance evaluation against false-positive lexical matching by verifying
whether evidence that lexically scored as SUPPORTS actually supports the claim semantically.
"""

from __future__ import annotations
import os
import re
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional, Any

import requests

from verdict.claim_decomposer import SubClaim

logger = logging.getLogger("verdict.semantic_verifier")

CONTRADICTION_PATTERN = re.compile(
    r'\b(contradict|contradicts|contradicting|contradiction|refute|refutes|refuting|disprove|disproved|disproven|debunk|debunked|false|untrue|myth|hoax|denies|denied)\b',
    re.IGNORECASE
)

NATIONALITY_CLUSTERS = [
    {"american", "united states", "usa", "u.s.", "u.s", "us"},
    {"indian", "india"},
    {"british", "uk", "united kingdom", "england", "english", "britain"},
    {"french", "france"},
    {"german", "germany"},
    {"chinese", "china"},
    {"japanese", "japan"},
    {"russian", "russia"},
    {"canadian", "canada"},
    {"australian", "australia"},
    {"italian", "italy"},
    {"spanish", "spain"},
    {"kenyan", "kenya", "mombasa", "nairobi"},
    {"israeli", "israel"},
    {"brazilian", "brazil"},
    {"mexican", "mexico"},
    {"korean", "korea", "south korea", "north korea"},
]


def detect_nationality_contradiction(claim_text: str, ev_text: str) -> bool:
    """Detects whether evidence states a different nationality/country than asserted in the claim."""
    claim_lower = claim_text.lower()
    ev_lower = ev_text.lower()

    claim_cluster_idx = -1
    for idx, cluster in enumerate(NATIONALITY_CLUSTERS):
        if any(re.search(r'\b' + re.escape(word) + r'\b', claim_lower) for word in cluster):
            claim_cluster_idx = idx
            break

    if claim_cluster_idx == -1:
        return False

    claim_cluster = NATIONALITY_CLUSTERS[claim_cluster_idx]
    if any(re.search(r'\b' + re.escape(word) + r'\b', ev_lower) for word in claim_cluster):
        return False

    for idx, cluster in enumerate(NATIONALITY_CLUSTERS):
        if idx != claim_cluster_idx:
            if any(re.search(r'\b' + re.escape(word) + r'\b', ev_lower) for word in cluster):
                return True

    return False


class SemanticStanceVerifier:
    """
    Verifies semantic stance using local Ollama LLM.
    Caches results in memory for sub-millisecond repeat queries.
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 3.0,
        enabled: bool = True
    ):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = timeout
        self.enabled = enabled
        self._cache: Dict[str, Tuple[bool, str, str]] = {}
        self._lock = threading.Lock()

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def _build_prompt(self, claim: str, evidence_snippet: str) -> str:
        return (
            "You are an objective fact-checking stance verifier. "
            "Your task is to determine whether the given evidence supports the factual claim. "
            "If the evidence explicitly states or confirms the key fact (such as a date, completion, event, or entity property), answer YES. "
            "If the evidence is unrelated, talks about a different subject/person, or contradicts the claim, answer NO.\n\n"
            f"Claim: {claim}\n"
            f"Evidence: {evidence_snippet}\n\n"
            "Does this evidence actually support the claim? Answer YES/NO with one sentence why."
        )

    def verify_single(self, claim: str, evidence_snippet: str) -> Tuple[bool, str, str]:
        """
        Verifies whether an individual evidence snippet actually supports the claim.
        Returns: (is_supported: bool, target_stance: str, reason: str)
        """
        cache_key = f"{claim.strip().lower()} ||| {evidence_snippet.strip().lower()}"
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        if not self.enabled:
            return True, "SUPPORTS", "Semantic verification disabled."

        prompt = self._build_prompt(claim, evidence_snippet)
        try:
            resp = requests.post(
                f"{self.ollama_url.rstrip('/')}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            if resp.status_code != 200:
                logger.warning("Ollama status %s: %s", resp.status_code, resp.text)
                return True, "SUPPORTS", "Ollama returned non-200; keeping original stance."

            ans = resp.json().get("response", "").strip()
            ans_clean = re.sub(r'[*_#`]', '', ans).strip()

            # Check if answer starts with YES or NO
            first_word_match = re.match(r'^(YES|NO)\b', ans_clean, re.IGNORECASE)
            is_no = False
            is_yes = False

            if first_word_match:
                word = first_word_match.group(1).upper()
                if word == "NO":
                    is_no = True
                elif word == "YES":
                    is_yes = True
            else:
                first_part = ans_clean[:50].lower()
                if "no," in first_part or "no." in first_part or "does not support" in first_part or "does not substantiate" in first_part:
                    is_no = True
                elif "yes," in first_part or "yes." in first_part or "supports the claim" in first_part:
                    is_yes = True

            if is_no or not is_yes:
                has_contradiction_keyword = bool(CONTRADICTION_PATTERN.search(ans_clean))
                has_nationality_mismatch = detect_nationality_contradiction(claim, evidence_snippet)

                if has_contradiction_keyword or has_nationality_mismatch:
                    target_stance = "REFUTES"
                else:
                    target_stance = "QUALIFIES"

                res = (False, target_stance, ans_clean)
            else:
                res = (True, "SUPPORTS", ans_clean)

            with self._lock:
                self._cache[cache_key] = res
            return res

        except Exception as e:
            logger.warning("Semantic verification failed (Ollama error/timeout): %s", e)
            return True, "SUPPORTS", f"Verification bypassed due to exception: {e}"

    def verify_supporting_evidence(
        self,
        subclaim: SubClaim,
        evidence_items: List[Any],
        top_n: int = 3
    ) -> None:
        """
        Batches verification of the top-N evidence items that lexically scored as SUPPORTS.
        Modifies evidence items in-place if downgraded.
        """
        supporting_items = [e for e in evidence_items if e.stance == "SUPPORTS"]
        if not supporting_items:
            return

        to_verify = sorted(supporting_items, key=lambda e: getattr(e, "weight", 0.0), reverse=True)[:top_n]

        def _get_ev_text(ev: Any) -> str:
            txt = ev.text or getattr(ev, "snippet", "")
            pub = getattr(ev, "published_date", None)
            if pub and str(pub).strip():
                txt = f"{txt} (Published: {pub})"
            return txt

        max_workers = min(3, len(to_verify))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                (ev, executor.submit(self.verify_single, subclaim.text, _get_ev_text(ev)))
                for ev in to_verify
            ]

            for ev, fut in futures:
                try:
                    is_sup, new_stance, reason = fut.result()
                    if not is_sup:
                        logger.info(
                            "Downgraded stance for evidence '%s' from SUPPORTS to %s: %s",
                            getattr(ev, "id", "ev"), new_stance, reason
                        )
                        ev.stance = new_stance
                        if new_stance == "REFUTES":
                            ev.weight = max(getattr(ev, "weight", 0.70), 0.90)
                        elif new_stance == "QUALIFIES":
                            ev.weight = min(getattr(ev, "weight", 0.70), 0.35)
                except Exception as ex:
                    logger.warning("Verification future failed: %s", ex)
