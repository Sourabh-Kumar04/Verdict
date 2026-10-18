"""
Proof Path: Traverses and formats ranked proof paths from FalkorDB Cypher queries.
Extracts the exact chain of evidence from authoritative sources to claims.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from verdict.graph_store import GraphStore


@dataclass
class ProofStep:
    claim_id: str
    claim_text: str
    evidence_id: str
    evidence_text: str
    stance: str
    weight: float
    source_name: str
    source_url: str
    source_pagerank: float
    rank_score: float
    published_date: str = ""
    snippet: str = ""

    def to_line(self) -> str:
        symbol = "✓" if self.stance == "SUPPORTS" else ("✗" if self.stance == "REFUTES" else "~")
        pub = f" [published: {self.published_date}]" if self.published_date else ""
        return (
            f"[{symbol} {self.stance}] (score={self.rank_score:.2f}, trust={self.source_pagerank:.2f}) "
            f'"{self.evidence_text}" — Source: {self.source_name} ({self.source_url}){pub}'
        )

    def to_one_line_summary(self) -> str:
        pub = f" ({self.published_date})" if self.published_date else ""
        return (
            f"[{self.stance} rank={self.rank_score:.2f}] '{self.evidence_text[:65]}...' "
            f"via {self.source_name}{pub} (PageRank: {self.source_pagerank:.2f})"
        )


class ProofPathEngine:
    """
    Constructs proof paths by traversing FalkorDB evidence graph relationships.
    Orders steps by PageRank * Stance Weight.
    """

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def get_ranked_proof_path(self, root_claim_id: str, limit: int = 5) -> List[ProofStep]:
        """
        Executes Cypher path traversal from Source -> Evidence -> Claim.
        Returns ranked ProofStep items.
        """
        g = self.graph_store.graph

        q = """
        MATCH (c:Claim)
        WHERE c.id = $claim_id OR c.root_id = $claim_id
        MATCH (e:Evidence)-[r]->(c)
        WHERE type(r) IN ['SUPPORTS', 'REFUTES', 'QUALIFIES']
        MATCH (e)-[:EXTRACTED_FROM]->(s:Source)
        RETURN c.id AS claim_id,
               c.text AS claim_text,
               e.id AS ev_id,
               e.text AS ev_text,
               type(r) AS stance,
               r.weight AS weight,
               s.name AS source_name,
               s.url AS source_url,
               coalesce(s.pagerank, 0.5) AS pagerank,
               (coalesce(s.pagerank, 0.5) * r.weight) AS rank_score,
               coalesce(e.published_date, '') AS published_date,
               coalesce(e.snippet, '') AS snippet
        ORDER BY rank_score DESC
        LIMIT $limit
        """
        try:
            res = g.query(q, {"claim_id": root_claim_id, "limit": limit})
            steps: List[ProofStep] = []
            for row in res.result_set:
                cid, ctext, eid, etext, stance, weight, sname, surl, pr, rscore, pub_date, snip = row
                steps.append(ProofStep(
                    claim_id=str(cid),
                    claim_text=str(ctext),
                    evidence_id=str(eid),
                    evidence_text=str(etext),
                    stance=str(stance),
                    weight=float(weight),
                    source_name=str(sname),
                    source_url=str(surl),
                    source_pagerank=float(pr),
                    rank_score=round(float(rscore), 4),
                    published_date=str(pub_date or ""),
                    snippet=str(snip or "")
                ))
            return steps
        except Exception:
            return []

    def format_proof_path(self, steps: List[ProofStep]) -> str:
        """Formats the proof path into a human-readable proof chain."""
        if not steps:
            return "No evidence path found in graph (Abstained)."

        lines = ["Ranked Proof Path:"]
        for idx, step in enumerate(steps, 1):
            lines.append(f"  {idx}. {step.to_line()}")
        return "\n".join(lines)

    def get_one_line_summary(self, steps: List[ProofStep]) -> str:
        """Returns a concise one-line proof path for reporting."""
        if not steps:
            return "No evidence path available (Abstained)."
        return steps[0].to_one_line_summary()
