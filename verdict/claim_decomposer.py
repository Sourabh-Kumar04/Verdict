"""
Claim Decomposer: Splits complex factual claims into atomic, verifiable sub-claims.
Extracts entities, keywords, and temporal expressions.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional


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

    def __init__(self, prefix: str = "subclaim"):
        self.prefix = prefix

    def decompose(self, claim: str) -> List[SubClaim]:
        """
        Decomposes a factual claim into a list of atomic SubClaim objects.
        """
        text = claim.strip()
        if not text:
            return []

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
            first_entity = matches[0].strip()
            if len(first_entity) > 2:
                return first_entity

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
            if cleaned.lower() not in self.STOPWORDS and len(cleaned) > 1:
                if cleaned not in entities:
                    entities.append(cleaned)
        return entities

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z0-9_-]+\b', text)
        keywords = []
        for w in words:
            wl = w.lower()
            if wl not in self.STOPWORDS and len(wl) > 2:
                if w not in keywords:
                    keywords.append(w)
        return keywords
