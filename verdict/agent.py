"""
Verdict Agent: Main investigation orchestrator connecting all modules.
Pipeline:
  Claim -> Decompose -> Retrieve Evidence & Citations -> Build FalkorDB Graph
        -> Compute PageRank Trust -> Evaluate Verdict via Cypher -> Extract Proof Path
"""

from __future__ import annotations
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Callable, Generator

from verdict.claim_decomposer import ClaimDecomposer
from verdict.retriever import EvidenceRetriever
from verdict.graph_store import GraphStore, FalkorDBConnectionError
from verdict.trust import TrustEngine
from verdict.verdict import VerdictEngine, VerdictResult
from verdict.proof_path import ProofPathEngine, ProofStep
from verdict.sanitizer import sanitize_claim


class VerdictAgent:
    """
    Autonomous claim-investigation agent powered by FalkorDB.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = "verdict_investigation",
        serpapi_key: Optional[str] = None,
        news_mode: str = "on_gap",
        fact_check_mode: str = "auto"
    ):
        self.graph_store = GraphStore(host=host, port=port, graph_name=graph_name)
        self.decomposer = ClaimDecomposer()
        self.retriever = EvidenceRetriever(
            serpapi_key=serpapi_key,
            news_mode=news_mode,
            fact_check_mode=fact_check_mode
        )
        self.trust_engine = TrustEngine(self.graph_store)
        self.verdict_engine = VerdictEngine(self.graph_store)
        self.proof_path_engine = ProofPathEngine(self.graph_store)

    def check_connection(self) -> bool:
        """Verifies FalkorDB instance is alive and reachable."""
        return self.graph_store.check_connection()

    def clear_cache(self) -> None:
        """Clears in-memory evidence caches."""
        self.retriever.clear_cache()

    def investigate(
        self,
        claim: str,
        clear_graph: bool = True,
        on_stage: Optional[Callable[[int, str, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end investigation on a claim string.
        Hardened against unreachable FalkorDB, empty evidence, and invalid/adversarial claims.
        """
        # Step 0: Input sanitization & connection health check
        sanitized = sanitize_claim(claim)
        if not sanitized:
            return {
                "claim_id": "claim_empty",
                "claim": claim or "",
                "verdict": "UNVERIFIED",
                "confidence": 0.0,
                "reason": "Abstained: Empty or invalid claim provided for investigation.",
                "subverdicts": [],
                "graph_counts": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "claims": 0,
                    "evidence": 0,
                    "sources": 0
                },
                "trust_scores": {},
                "proof_steps": [],
                "proof_path": "No evidence path found in graph (Abstained).",
                "one_line_proof": "No evidence path available (Abstained)."
            }

        # Verify FalkorDB is live before starting investigation
        self.check_connection()

        if clear_graph:
            self.graph_store.delete_graph()
            self.graph_store.init_schema()

        claim_id = f"claim_{uuid.uuid4().hex[:8]}"

        # Step 1: Add root claim to FalkorDB
        if on_stage:
            on_stage(1, "Decomposing Claim", "Decomposing claim into atomic sub-claims & entities...")
        self.graph_store.add_root_claim(claim_id=claim_id, text=sanitized)

        # Step 2: Decompose claim into atomic subclaims
        subclaims = self.decomposer.decompose(sanitized)
        for sub in subclaims:
            self.graph_store.add_subclaim(root_id=claim_id, subclaim=sub)

        # Step 3: Retrieve evidence & cross-source citations concurrently
        if on_stage:
            on_stage(2, "Retrieving Evidence", f"Querying Wikipedia, Wikidata & Fact-Checkers for {len(subclaims)} subclaim(s)...")

        if len(subclaims) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(subclaims))) as executor:
                retrieval_results = list(
                    executor.map(lambda s: (s, self.retriever.retrieve_evidence_for_subclaim(s, top_k=3)), subclaims)
                )
        else:
            retrieval_results = [
                (subclaims[0], self.retriever.retrieve_evidence_for_subclaim(subclaims[0], top_k=3))
            ] if subclaims else []

        for sub, (evidence_items, citations) in retrieval_results:
            for cite in citations:
                self.graph_store.add_citation(cite.source_from_url, cite.source_to_url, cite.relationship)
            for ev in evidence_items:
                self.graph_store.add_evidence(ev, target_claim_id=sub.id)

        # Step 4: Compute PageRank trust across sources in FalkorDB
        if on_stage:
            on_stage(3, "Building Graph", "Populating FalkorDB evidence graph & computing PageRank trust...")
        trust_scores = self.trust_engine.compute_source_trust()

        # Step 5: Compute verdict strictly through Cypher graph queries on FalkorDB
        if on_stage:
            on_stage(4, "Computing Verdict", "Executing Cypher verdict graph query against FalkorDB...")
        verdict_res: VerdictResult = self.verdict_engine.evaluate_claim(claim_id=claim_id)

        # Step 6: Extract ranked proof path
        if on_stage:
            on_stage(5, "Proof Path Extraction", "Traversing and ranking epistemic proof paths in FalkorDB...")
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

    def investigate_stream(
        self,
        claim: str,
        clear_graph: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Executes investigation yielding real-time stage progress events,
        culminating in the final verdict payload.
        """
        sanitized = sanitize_claim(claim)
        if not sanitized:
            empty_res = self.investigate(claim)
            yield {"type": "complete", "result": empty_res}
            return

        self.check_connection()

        if clear_graph:
            self.graph_store.delete_graph()
            self.graph_store.init_schema()

        claim_id = f"claim_{uuid.uuid4().hex[:8]}"
        self.graph_store.add_root_claim(claim_id=claim_id, text=sanitized)

        # Stage 1: Decomposing
        subclaims = self.decomposer.decompose(sanitized)
        for sub in subclaims:
            self.graph_store.add_subclaim(root_id=claim_id, subclaim=sub)

        sub_desc = f"{len(subclaims)} atomic sub-claim(s)" if len(subclaims) > 1 else f"'{subclaims[0].text[:40]}...'" if subclaims else "atomic components"
        yield {
            "type": "stage",
            "stage": 1,
            "name": "Decomposing Claim",
            "detail": f"Decomposed into {sub_desc} & extracted entities."
        }

        # Stage 2: Evidence Retrieval
        yield {
            "type": "stage",
            "stage": 2,
            "name": "Retrieving Evidence",
            "detail": f"Querying Wikipedia, Wikidata & Fact-Checkers concurrently..."
        }

        if len(subclaims) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(subclaims))) as executor:
                retrieval_results = list(
                    executor.map(lambda s: (s, self.retriever.retrieve_evidence_for_subclaim(s, top_k=3)), subclaims)
                )
        else:
            retrieval_results = [
                (subclaims[0], self.retriever.retrieve_evidence_for_subclaim(subclaims[0], top_k=3))
            ] if subclaims else []

        wiki_count = 0
        news_count = 0
        for sub, (evidence_items, citations) in retrieval_results:
            for cite in citations:
                self.graph_store.add_citation(cite.source_from_url, cite.source_to_url, cite.relationship)
            for ev in evidence_items:
                if ev.source_type in ("wikipedia", "wikidata"):
                    wiki_count += 1
                else:
                    news_count += 1
                self.graph_store.add_evidence(ev, target_claim_id=sub.id)

        # Stage 3: Building FalkorDB Graph
        trust_scores = self.trust_engine.compute_source_trust()
        graph_counts = self.graph_store.get_counts()
        yield {
            "type": "stage",
            "stage": 3,
            "name": "Building FalkorDB Graph",
            "detail": f"Graph: {graph_counts['total_nodes']} nodes, {graph_counts['total_edges']} edges. PageRank computed."
        }

        # Stage 4: Computing Verdict via Cypher
        verdict_res: VerdictResult = self.verdict_engine.evaluate_claim(claim_id=claim_id)
        yield {
            "type": "stage",
            "stage": 4,
            "name": "Computing Verdict",
            "detail": f"Verdict: {verdict_res.verdict} (Confidence: {verdict_res.confidence * 100:.1f}%)"
        }

        # Stage 5: Proof Path Extraction
        proof_steps = self.proof_path_engine.get_ranked_proof_path(root_claim_id=claim_id)
        proof_path_text = self.proof_path_engine.format_proof_path(proof_steps)
        one_line_proof = self.proof_path_engine.get_one_line_summary(proof_steps)
        yield {
            "type": "stage",
            "stage": 5,
            "name": "Proof Path Extraction",
            "detail": f"Extracted {len(proof_steps)} ranked proof step(s) in FalkorDB."
        }

        final_result = {
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

        yield {"type": "complete", "result": final_result}
