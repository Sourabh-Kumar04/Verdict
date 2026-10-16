"""
Trust: Source trust computation using FalkorDB's native PageRank algorithm.
Calculates authority scores over the source citation network.
"""

from __future__ import annotations
from typing import Dict, Any
from verdict.graph_store import GraphStore


class TrustEngine:
    """
    Computes PageRank authority scores for Source nodes in the FalkorDB graph.
    Propagates citation credibility so authoritative, highly-cited references
    carry proportionally greater weight in the final verdict determination.
    """

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def compute_source_trust(self) -> Dict[str, float]:
        """
        Executes native FalkorDB PageRank on (:Source)-[:CITES]->(:Source) network.
        Updates s.pagerank directly on the graph nodes via Cypher.
        Returns a mapping of source URL to computed PageRank score.
        """
        g = self.graph_store.graph

        # Check if any Source nodes exist
        count_res = g.query("MATCH (s:Source) RETURN count(s)").result_set
        source_count = count_res[0][0] if count_res else 0
        if source_count == 0:
            return {}

        scores: Dict[str, float] = {}

        try:
            # Native FalkorDB 6.0.1 PageRank algorithm procedure
            res = g.query("CALL algo.pageRank('Source', 'CITES') YIELD node, score RETURN node.url, score")
            for row in res.result_set:
                url = row[0]
                score = float(row[1])
                scores[url] = round(score, 4)

                # Persist computed trust back into the node properties in FalkorDB
                g.query(
                    "MATCH (s:Source {url: $url}) SET s.pagerank = $score",
                    {"url": url, "score": score}
                )
        except Exception as e:
            # Fallback in case of sparse or procedure error: assign equal baseline distribution
            baseline = round(1.0 / max(1, source_count), 4)
            res = g.query("MATCH (s:Source) RETURN s.url")
            for row in res.result_set:
                url = row[0]
                scores[url] = baseline
                g.query(
                    "MATCH (s:Source {url: $url}) SET s.pagerank = $score",
                    {"url": url, "score": baseline}
                )

        return scores
