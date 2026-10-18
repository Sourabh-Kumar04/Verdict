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
        # Normalization factor: maps baseline (1/N) to 0.5 so source authority is invariant to total source count
        scale = max(2, source_count) * 0.5

        try:
            # Native FalkorDB 6.0.1 PageRank algorithm procedure
            res = g.query("CALL algo.pageRank('Source', 'CITES') YIELD node, score RETURN node.url, score")
            for row in res.result_set:
                url = row[0]
                norm_score = round(float(row[1]) * scale, 4)
                scores[url] = norm_score

                # Persist computed trust back into the node properties in FalkorDB
                g.query(
                    "MATCH (s:Source {url: $url}) SET s.pagerank = $score",
                    {"url": url, "score": norm_score}
                )
        except Exception as e:
            # Fallback in case of sparse or procedure error: assign equal baseline distribution (0.5)
            baseline = 0.5
            res = g.query("MATCH (s:Source) RETURN s.url")
            for row in res.result_set:
                url = row[0]
                scores[url] = baseline
                g.query(
                    "MATCH (s:Source {url: $url}) SET s.pagerank = $score",
                    {"url": url, "score": baseline}
                )

        # Boost fact-checker sources to higher authority in the graph
        try:
            boost_res = g.query("""
                MATCH (s:Source)
                WHERE s.source_type = 'fact_checker'
                   OR s.url CONTAINS 'snopes.com'
                   OR s.url CONTAINS 'politifact.com'
                   OR s.url CONTAINS 'factcheck.org'
                SET s.pagerank = CASE
                    WHEN coalesce(s.pagerank, 0.5) < 0.85 THEN 0.90
                    ELSE s.pagerank * 1.15
                END
                RETURN s.url, s.pagerank
            """).result_set
            for brow in boost_res:
                scores[brow[0]] = round(float(brow[1]), 4)
        except Exception:
            pass

        return scores
