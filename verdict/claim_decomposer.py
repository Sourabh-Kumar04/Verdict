"""
Claim Decomposer: Splits complex factual claims into atomic, verifiable sub-claims.
Extracts entities, keywords, and temporal expressions.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional

from verdict.sanitizer import sanitize_claim


@dataclass
class SubClaim:
    id: str
    text: str
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    time_expression: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "keywords": self.keywords,
            "entities": self.entities,
            "time_expression": self.time_expression,
        }


class ClaimDecomposer:
    """
    Deterministic factual claim decomposer.
    Decomposes multi-part statements into atomic sub-claims while resolving
    subject ellipses and extracting key entities for graph indexing and retrieval.
    """

    # Common coordinating conjunctions and clause separators
    CONJUNCTION_PATTERN = re.compile(
        r'\s+(?:and(?:\s+also)?|as well as|while|but)\s+|;\s*',
        re.IGNORECASE
    )

    # Year or date patterns (e.g., 1889, 1969, 2024, March 1889)
    YEAR_PATTERN = re.compile(r'\b(1[6-9][0-9]{2}|20[0-3][0-9])\b')

    # Capitalized entity phrases (e.g., "The Eiffel Tower", "Apollo 11", "Marie Curie")
    ENTITY_PATTERN = re.compile(
        r'\b(?:[A-Z][a-zA-Z0-9]*(?:\s+(?:[A-Z][a-zA-Z0-9]*|\d+|of|the|and|de|von|van|la|le)\b)*(?:\s+[A-Z0-9][a-zA-Z0-9]*)*)\b'
    )

    STOPWORDS = {
        "the", "a", "an", "is", "was", "were", "are", "been", "be", "in", "on", "at",
        "to", "for", "of", "by", "with", "from", "that", "which", "it", "its", "and",
        "or", "as", "also", "has", "had", "have", "first", "second"
    }

    QUESTION_AUXILIARIES = {
        "did", "does", "do", "was", "were", "is", "are", "has", "have", "had",
        "can", "could", "will", "would", "should"
    }

    def __init__(self, prefix: str = "subclaim"):
        self.prefix = prefix

    def _normalize_question_to_claim(self, text: str) -> str:
        """
        Normalizes judge questions into declarative factual assertions.
        e.g., 'did apollo 11 land on mars?' -> 'Apollo 11 landed on Mars.'
              'is the earth flat?' -> 'The Earth is flat.'
              'was Marie Curie born in France?' -> 'Marie Curie was born in France.'
        """
        cleaned = text.strip()
        if not cleaned:
            return cleaned

        is_question = cleaned.endswith("?") or bool(
            re.match(r"^(?:did|does|do|was|were|is|are|has|have|had|can|could|will|would)\s+", cleaned, re.IGNORECASE)
        )
        if not is_question:
            return cleaned

        cleaned = cleaned.rstrip("?").strip()

        # 1. Did <subject> <verb> <rest>? -> <subject> <past_verb> <rest>
        m = re.match(
            r"^did\s+(.+?)\s+(land|cause|discover|invent|build|create|write|reach|visit|destroy|assassinate|start|end|win|lose|die|kill|make|form|find)\b(.*)$",
            cleaned,
            re.IGNORECASE
        )
        if m:
            subj = m.group(1).strip()
            verb = m.group(2).lower()
            rest = m.group(3).strip()
            irregulars = {
                "land": "landed", "cause": "caused", "discover": "discovered", "invent": "invented",
                "build": "built", "create": "created", "write": "wrote", "reach": "reached",
                "visit": "visited", "destroy": "destroyed", "assassinate": "assassinated",
                "start": "started", "end": "ended", "win": "won", "lose": "lost",
                "die": "died", "kill": "killed", "make": "made", "find": "found"
            }
            past = irregulars.get(verb, verb + "ed")
            res = f"{subj} {past} {rest}".strip()
            return res[0].upper() + res[1:]

        # 2. Was / Were <subject> <predicate>? -> <subject> was/were <predicate>
        m = re.match(
            r"^(was|were)\s+(.+?)\s+(born|built|completed|designed|founded|created|located|written|elected|killed|first|part|a|an|the|in|on|at|true|false)\b(.*)$",
            cleaned,
            re.IGNORECASE
        )
        if m:
            aux = m.group(1).lower()
            subj = m.group(2).strip()
            first_p = m.group(3).strip()
            rest = m.group(4).strip()
            res = f"{subj} {aux} {first_p} {rest}".strip()
            return res[0].upper() + res[1:]

        # 3. Is / Are <subject> <predicate>? -> <subject> is/are <predicate>
        m = re.match(
            r"^(is|are)\s+(.+?)\s+(flat|round|safe|dangerous|real|fake|true|false|in|on|at|a|an|the|connected|related|alive|dead)\b(.*)$",
            cleaned,
            re.IGNORECASE
        )
        if m:
            aux = m.group(1).lower()
            subj = m.group(2).strip()
            first_p = m.group(3).strip()
            rest = m.group(4).strip()
            res = f"{subj} {aux} {first_p} {rest}".strip()
            return res[0].upper() + res[1:]

        # 4. Does <subject> <verb> <rest>? -> <subject> <verb_s> <rest>
        m = re.match(r"^does\s+(.+?)\s+(cause|have|lead to|contain)\b(.*)$", cleaned, re.IGNORECASE)
        if m:
            subj = m.group(1).strip()
            verb = m.group(2).lower()
            rest = m.group(3).strip()
            v_s = "causes" if verb == "cause" else ("has" if verb == "have" else (verb + "s"))
            res = f"{subj} {v_s} {rest}".strip()
            return res[0].upper() + res[1:]

        cleaned_no_aux = re.sub(
            r"^(?:did|does|do|was|were|is|are|has|have|had|can|could|will|would)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE
        ).strip()
        if cleaned_no_aux:
            return cleaned_no_aux[0].upper() + cleaned_no_aux[1:]
        return cleaned

    def decompose(self, claim: str) -> List[SubClaim]:
        """
        Decomposes a factual claim into a list of atomic SubClaim objects.
        Normalizes questions and handles compound conjunctions.
        """
        text = sanitize_claim(claim)
        if not text:
            return []

        # Normalize question to declarative statement
        text = self._normalize_question_to_claim(text)

        # Remove trailing period for processing
        cleaned = text.rstrip(".")

        # Extract primary subject from first clause (first capitalized entity or initial noun chunk)
        primary_subject = self._extract_subject(cleaned)

        # Split clauses by conjunctions
        raw_parts = self.CONJUNCTION_PATTERN.split(cleaned)
        subclaims: List[SubClaim] = []

        for idx, part in enumerate(raw_parts):
            part_str = part.strip()
            if not part_str:
                continue

            # If this is a subsequent clause lacking a subject, carry over primary_subject
            resolved_text = part_str
            if idx > 0 and primary_subject:
                # Check if clause starts with a verb (e.g. "won two Nobel Prizes", "died in 1954")
                if self._starts_with_predicate(part_str):
                    resolved_text = f"{primary_subject} {part_str}"

            # Ensure proper capitalization and punctuation
            resolved_text = resolved_text[0].upper() + resolved_text[1:]
            if not resolved_text.endswith("."):
                resolved_text += "."

            time_expr = self._extract_time(resolved_text)
            entities = self._extract_entities(resolved_text)
            keywords = self._extract_keywords(resolved_text)

            subclaim = SubClaim(
                id=f"{self.prefix}_{idx + 1}",
                text=resolved_text,
                keywords=keywords,
                entities=entities,
                time_expression=time_expr,
            )
            subclaims.append(subclaim)

        # If splitting resulted in no valid parts, fallback to single claim
        if not subclaims:
            time_expr = self._extract_time(text)
            entities = self._extract_entities(text)
            keywords = self._extract_keywords(text)
            subclaims.append(
                SubClaim(
                    id=f"{self.prefix}_1",
                    text=text if text.endswith(".") else f"{text}.",
                    keywords=keywords,
                    entities=entities,
                    time_expression=time_expr,
                )
            )

        return subclaims

    def _extract_subject(self, text: str) -> Optional[str]:
        """Finds the main subject of the opening clause."""
        matches = self.ENTITY_PATTERN.findall(text)
        if matches:
            for m in matches:
                cand = m.strip()
                if cand.lower() not in self.QUESTION_AUXILIARIES and len(cand) > 2:
                    return cand

        # Fallback: take first 2-3 words before verb
        words = text.split()
        if len(words) >= 2:
            return " ".join(words[:2])
        return None

    def _starts_with_predicate(self, clause: str) -> bool:
        """Checks if clause appears to start with a verb/predicate rather than a noun phrase."""
        first_word = clause.split()[0].lower()
        common_verbs = {
            "is", "was", "were", "are", "became", "won", "died", "lived", "landed",
            "built", "completed", "constructed", "discovered", "invented", "wrote",
            "received", "graduated", "served", "held", "founded", "created"
        }
        if first_word in common_verbs:
            return True
        if first_word.endswith("ed") or first_word.endswith("s"):
            return True
        return False

    def _extract_time(self, text: str) -> Optional[str]:
        match = self.YEAR_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_entities(self, text: str) -> List[str]:
        raw_matches = self.ENTITY_PATTERN.findall(text)
        entities = []
        for m in raw_matches:
            cleaned = m.strip()
            cleaned = re.sub(r'^(?:Did|Does|Do|Was|Were|Is|Are|Has|Have|Had)\s+', '', cleaned, flags=re.IGNORECASE).strip()
            if cleaned.lower() not in self.STOPWORDS and cleaned.lower() not in self.QUESTION_AUXILIARIES and len(cleaned) > 1:
                if cleaned not in entities:
                    entities.append(cleaned)

        if len(entities) < 2:
            words = re.findall(r'\b[a-zA-Z0-9_-]+\b', text)
            for w in words:
                wl = w.lower()
                if (
                    wl not in self.STOPWORDS
                    and wl not in self.QUESTION_AUXILIARIES
                    and wl not in {"cause", "causes", "caused", "land", "lands", "landed"}
                    and len(wl) > 3
                ):
                    cand = w[0].upper() + w[1:]
                    if cand not in entities and w not in entities:
                        entities.append(cand)

        return entities

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z0-9_-]+\b', text)
        keywords = []
        for w in words:
            wl = w.lower()
            if wl not in self.STOPWORDS and wl not in self.QUESTION_AUXILIARIES and len(wl) > 2:
                if w not in keywords:
                    keywords.append(w)
        return keywords
