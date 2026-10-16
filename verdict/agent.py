"""
Verdict Agent: Main investigation orchestrator connecting all modules.
Pipeline:
  Claim -> Decompose -> Retrieve Evidence & Citations -> Build FalkorDB Graph
        -> Compute PageRank Trust -> Evaluate Verdict via Cypher -> Extract Proof Path
"""

from __future__ import annotations
import uuid
from typing import Dict, Any, Optional

from verdict.claim_decomposer import ClaimDecomposer
from verdict.retriever import EvidenceRetriever
from verdict.graph_store import GraphStore
from verdict.trust import TrustEngine
from verdict.verdict import VerdictEngine, VerdictResult
from verdict.proof_path import ProofPathEngine, ProofStep


class VerdictAgent:
    """
    Autonomous claim-investigation agent powered by FalkorDB.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = "verdict_investigation"
    ):
        self.graph_store = GraphStore(host=host, port=port, graph_name=graph_name)
        self.decomposer = ClaimDecomposer()
        self.retriever = EvidenceRetriever()
        self.trust_engine = TrustEngine(self.graph_store)
        self.verdict_engine = VerdictEngine(self.graph_store)
        self.proof_path_engine = ProofPathEngine(self.graph_store)

    def investigate(self, claim: str, clear_graph: bool = True) -> Dict[str, Any]:
        """
        Executes end-to-end investigation on a claim string.
        """
        if clear_graph:
            self.graph_store.delete_graph()
            self.graph_store.init_schema()

        claim_id = f"claim_{uuid.uuid4().hex[:8]}"

        # Step 1: Add root claim to FalkorDB
        self.graph_store.add_root_claim(claim_id=claim_id, text=claim)

        # Step 2: Decompose claim into atomic subclaims
        subclaims = self.decomposer.decompose(claim)
        for sub in subclaims:
            self.graph_store.add_subclaim(root_id=claim_id, subclaim=sub)

        # Step 3: Retrieve evidence & cross-source citations from public Wikipedia & Wikidata APIs
        for sub in subclaims:
            evidence_items, citations = self.retriever.retrieve_evidence_for_subclaim(sub, top_k=3)
            # Add citations to FalkorDB for PageRank
            for cite in citations:
                self.graph_store.add_citation(cite.source_from_url, cite.source_to_url, cite.relationship)
            # Add evidence to FalkorDB
            for ev in evidence_items:
                self.graph_store.add_evidence(ev, target_claim_id=sub.id)

        # Step 4: Compute PageRank trust across sources in FalkorDB
        trust_scores = self.trust_engine.compute_source_trust()

        # Step 5: Compute verdict strictly through Cypher graph queries on FalkorDB
        verdict_res: VerdictResult = self.verdict_engine.evaluate_claim(claim_id=claim_id)

        # Step 6: Extract ranked proof path
        proof_steps = self.proof_path_engine.get_ranked_proof_path(root_claim_id=claim_id)
        proof_path_text = self.proof_path_engine.format_proof_path(proof_steps)
        one_line_proof = self.proof_path_engine.get_one_line_summary(proof_steps)

        graph_counts = self.graph_store.get_counts()

        return {
            "claim_id": claim_id,
            "claim": claim,
            "verdict": verdict_res.verdict,
            "confidence": verdict_res.confidence,
            "reason": verdict_res.reason,
            "subverdicts": [
                {
                    "subclaim_id": sv.subclaim_id,
                    "text": sv.text,
                    "verdict": sv.verdict,
                    "support_score": sv.support_score,
                    "refute_score": sv.refute_score,
                    "evidence_count": sv.evidence_count
                }
                for sv in verdict_res.subverdicts
            ],
            "graph_counts": graph_counts,
            "trust_scores": trust_scores,
            "proof_steps": proof_steps,
            "proof_path": proof_path_text,
            "one_line_proof": one_line_proof
        }
