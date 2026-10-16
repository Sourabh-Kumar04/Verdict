"""
Verdict: Graph-driven verdict computation engine for Verdict.
HARD RULE: Load-bearing Cypher queries against FalkorDB determine the verdict.
No direct LLM call may bypass the graph.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from verdict.graph_store import GraphStore


@dataclass
class SubVerdict:
    subclaim_id: str
    text: str
    verdict: str  # "SUPPORTED" | "REFUTED" | "UNVERIFIED"
    support_score: float
    refute_score: float
    qualify_score: float
    evidence_count: int


@dataclass
class VerdictResult:
    claim_id: str
    claim_text: str
    verdict: str  # "SUPPORTED" | "REFUTED" | "UNVERIFIED"
    confidence: float
    reason: str
    subverdicts: List[SubVerdict] = field(default_factory=list)
    total_evidence_nodes: int = 0
    proof_chain: List[str] = field(default_factory=list)

    def is_abstention(self) -> bool:
        return self.verdict == "UNVERIFIED"


class VerdictEngine:
    """
    Computes verdicts strictly by evaluating graph topology and Cypher query results.
    Weighs stance edges by the PageRank authority of their origin source.
    Honors strict abstention when evidence is insufficient or conflicting.
    """

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def evaluate_claim(self, claim_id: str) -> VerdictResult:
        """
        Evaluates a claim by executing Cypher graph queries on FalkorDB.
        Fails or abstains if graph is empty or disconnected.
        """
        g = self.graph_store.graph

        # Step 1: Query root claim and all attached subclaims
        claim_q = """
        MATCH (root:Claim {id: $claim_id})
        OPTIONAL MATCH (root)-[:HAS_SUBCLAIM]->(sub:Claim)
        RETURN root.id, root.text, collect(sub.id), collect(sub.text)
        """
        res = g.query(claim_q, {"claim_id": claim_id}).result_set
        if not res:
            # If the claim does not exist in the graph, ABSTAIN / fail load-bearing check
            return VerdictResult(
                claim_id=claim_id,
                claim_text="",
                verdict="UNVERIFIED",
                confidence=0.0,
                reason="Load-bearing failure: Claim node not found in FalkorDB graph."
            )

        root_id, root_text, sub_ids, sub_texts = res[0]
        sub_ids = [s for s in sub_ids if s]
        sub_texts = [s for s in sub_texts if s]

        # Target nodes: if subclaims exist, evaluate them; otherwise evaluate root
        target_ids = sub_ids if sub_ids else [root_id]
        target_map = dict(zip(sub_ids, sub_texts)) if sub_ids else {root_id: root_text}

        # Step 2: Cypher aggregation query combining Evidence, Edge Weight, and Source PageRank
        agg_q = """
        MATCH (c:Claim)
        WHERE c.id IN $target_ids
        OPTIONAL MATCH (e:Evidence)-[r]->(c)
        WHERE type(r) IN ['SUPPORTS', 'REFUTES', 'QUALIFIES']
        OPTIONAL MATCH (e)-[:EXTRACTED_FROM]->(s:Source)
        RETURN c.id AS claim_id,
               type(r) AS stance,
               sum(r.weight * coalesce(s.pagerank, 0.5)) AS weighted_score,
               count(e) AS ev_count
        """
        agg_rows = g.query(agg_q, {"target_ids": target_ids}).result_set

        # Parse per-target scores
        target_scores: Dict[str, Dict[str, float]] = {tid: {"SUPPORTS": 0.0, "REFUTES": 0.0, "QUALIFIES": 0.0, "count": 0} for tid in target_ids}
        total_ev_count = 0

        for row in agg_rows:
            cid, stance, score, count = row
            if cid in target_scores and stance:
                target_scores[cid][stance] = float(score or 0.0)
                target_scores[cid]["count"] += int(count or 0)
                total_ev_count += int(count or 0)

        # Step 3: Compute subverdicts
        subverdicts: List[SubVerdict] = []
        has_refutation = False
        all_supported = True
        has_abstention = False

        for tid in target_ids:
            scores = target_scores[tid]
            sup = scores["SUPPORTS"]
            ref = scores["REFUTES"]
            qua = scores["QUALIFIES"]
            cnt = scores["count"]
            t_text = target_map.get(tid, "")

            # Insufficient evidence -> ABSTAIN on this subclaim
            if cnt == 0 or (sup < 0.20 and ref < 0.20):
                sv_verdict = "UNVERIFIED"
                has_abstention = True
                all_supported = False
            elif ref > (sup * 1.15) and ref >= 0.25:
                sv_verdict = "REFUTED"
                has_refutation = True
                all_supported = False
            elif sup > (ref * 1.25) and sup >= 0.25:
                sv_verdict = "SUPPORTED"
            else:
                # Ambiguous or contradictory
                sv_verdict = "UNVERIFIED"
                has_abstention = True
                all_supported = False

            subverdicts.append(SubVerdict(
                subclaim_id=tid,
                text=t_text,
                verdict=sv_verdict,
                support_score=round(sup, 4),
                refute_score=round(ref, 4),
                qualify_score=round(qua, 4),
                evidence_count=cnt
            ))

        # Step 4: Overall verdict synthesis
        if total_ev_count == 0:
            final_verdict = "UNVERIFIED"
            confidence = 0.0
            reason = "Abstained: Insufficient public evidence retrieved to corroborate or refute claim."
        elif has_refutation:
            final_verdict = "REFUTED"
            confidence = min(0.99, max(sv.refute_score for sv in subverdicts) / 1.5)
            reason = "Refuted: Contradicted by authoritative evidence graph sources."
        elif all_supported and not has_abstention:
            final_verdict = "SUPPORTED"
            confidence = min(0.99, min(sv.support_score for sv in subverdicts) / 1.5)
            reason = "Supported: All claim components verified by authoritative evidence graph sources."
        else:
            final_verdict = "UNVERIFIED"
            confidence = 0.35
            reason = "Abstained: Evidence is partial, conflicting, or incomplete across sub-claims."

        # Step 5: Update the root Claim node in FalkorDB with the computed verdict
        update_q = """
        MATCH (root:Claim {id: $claim_id})
        SET root.verdict = $verdict,
            root.confidence = $confidence,
            root.reason = $reason,
            root.evidence_count = $total_ev_count,
            root.resolved_at = timestamp()
        RETURN root.verdict
        """
        g.query(update_q, {
            "claim_id": claim_id,
            "verdict": final_verdict,
            "confidence": round(confidence, 3),
            "reason": reason,
            "total_ev_count": total_ev_count
        })

        return VerdictResult(
            claim_id=claim_id,
            claim_text=root_text,
            verdict=final_verdict,
            confidence=round(confidence, 3),
            reason=reason,
            subverdicts=subverdicts,
            total_evidence_nodes=total_ev_count
        )
