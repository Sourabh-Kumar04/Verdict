"""
Graph Store: FalkorDB 6.0.1 graph persistence and querying for Verdict.
Implements load-bearing graph schema for Claims, Evidence, and Sources.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from falkordb import FalkorDB, Graph

from verdict.claim_decomposer import SubClaim
from verdict.retriever import EvidenceItem, CitationLink


def format_vector(vector: List[float]) -> str:
    """FalkorDB 6.0.1 pitfall helper: Wrap raw embedding lists in vecf32([...])."""
    vals = ", ".join(f"{v:.6f}" for v in vector)
    return f"vecf32([{vals}])"


def flatten_properties(props: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """FalkorDB 6.0.1 pitfall helper: Flatten nested maps into top-level primitive properties."""
    flat: Dict[str, Any] = {}
    for k, v in props.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            flat.update(flatten_properties(v, prefix=key))
        elif isinstance(v, (list, tuple)):
            # FalkorDB allows lists of primitives (strings, ints, floats)
            flat[key] = [str(item) for item in v]
        elif v is None:
            flat[key] = ""
        else:
            flat[key] = v
    return flat


class GraphStore:
    """
    Manages the FalkorDB evidence graph.
    All claims, subclaims, evidence snippets, and sources are stored as native graph elements.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = "verdict_graph",
        password: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self.client = FalkorDB(host=self.host, port=self.port, password=password)
        self.graph: Graph = self.client.select_graph(self.graph_name)

    def delete_graph(self) -> bool:
        """
        Safely deletes the graph.
        Guards against FalkorDB 6.0.1 pitfall where deleting an empty graph raises an error.
        """
        try:
            self.graph.delete()
            return True
        except Exception as e:
            msg = str(e).lower()
            if "empty key" in msg or "not found" in msg or "invalid graph operation" in msg:
                return False
            # If it is any other exception, raise
            raise

    def init_schema(self) -> None:
        """Initializes graph schema and indexes."""
        index_queries = [
            "CREATE INDEX FOR (c:Claim) ON (c.id)",
            "CREATE INDEX FOR (s:Source) ON (s.url)",
            "CREATE INDEX FOR (e:Evidence) ON (e.id)",
        ]
        for q in index_queries:
            try:
                self.graph.query(q)
            except Exception:
                # Index may already exist
                pass

    def add_root_claim(self, claim_id: str, text: str) -> None:
        """Creates the root claim node."""
        q = """
        MERGE (c:Claim {id: $id})
        ON CREATE SET c.text = $text, c.is_root = true, c.verdict = 'PENDING', c.created_at = timestamp()
        ON MATCH SET c.text = $text, c.is_root = true
        RETURN c.id
        """
        self.graph.query(q, {"id": claim_id, "text": text})

    def add_subclaim(self, root_id: str, subclaim: SubClaim) -> None:
        """
        Creates a subclaim node and attaches it to the root claim via HAS_SUBCLAIM.
        Includes 'WITH' clause between MATCH/MERGE as per Cypher best practices.
        """
        q = """
        MATCH (root:Claim {id: $root_id})
        WITH root
        MERGE (sub:Claim {id: $sub_id})
        ON CREATE SET sub.text = $text,
                      sub.is_root = false,
                      sub.time_expr = $time_expr,
                      sub.root_id = $root_id
        WITH root, sub
        MERGE (root)-[:HAS_SUBCLAIM]->(sub)
        RETURN sub.id
        """
        self.graph.query(q, {
            "root_id": root_id,
            "sub_id": subclaim.id,
            "text": subclaim.text,
            "time_expr": subclaim.time_expression or ""
        })

    def add_source(
        self,
        url: str,
        name: str,
        source_type: str,
        initial_pagerank: float = 0.5
    ) -> None:
        """Creates or updates a Source node."""
        q = """
        MERGE (s:Source {url: $url})
        ON CREATE SET s.name = $name,
                      s.source_type = $source_type,
                      s.pagerank = $pagerank
        ON MATCH SET s.name = $name
        RETURN s.url
        """
        self.graph.query(q, {
            "url": url,
            "name": name,
            "source_type": source_type,
            "pagerank": float(initial_pagerank)
        })

    def add_citation(self, from_url: str, to_url: str, relationship: str = "CITES") -> None:
        """
        Creates a citation relationship between two Source nodes.
        Used for computing PageRank trust centrality over the citation network.
        """
        if from_url == to_url:
            return

        # Ensure both sources exist
        self.add_source(url=from_url, name=from_url, source_type="unknown")
        self.add_source(url=to_url, name=to_url, source_type="unknown")

        q = """
        MATCH (s1:Source {url: $from_url})
        MATCH (s2:Source {url: $to_url})
        WITH s1, s2
        MERGE (s1)-[:CITES]->(s2)
        RETURN count(*)
        """
        self.graph.query(q, {"from_url": from_url, "to_url": to_url})

    def add_evidence(self, item: EvidenceItem, target_claim_id: str) -> None:
        """
        Adds an Evidence node, links it to its Source node via EXTRACTED_FROM,
        and links it to the target Claim node via SUPPORTS, REFUTES, or QUALIFIES.
        """
        # Ensure source exists
        self.add_source(url=item.source_url, name=item.source_name, source_type=item.source_type)

        stance = item.stance.upper()
        if stance not in {"SUPPORTS", "REFUTES", "QUALIFIES"}:
            stance = "QUALIFIES"

        q = f"""
        MATCH (c:Claim {{id: $claim_id}})
        WITH c
        MATCH (s:Source {{url: $source_url}})
        WITH c, s
        MERGE (e:Evidence {{id: $eid}})
        ON CREATE SET e.text = $text,
                      e.stance = $stance,
                      e.weight = $weight,
                      e.snippet = $snippet
        WITH c, s, e
        MERGE (e)-[:EXTRACTED_FROM]->(s)
        WITH c, e
        MERGE (e)-[:{stance} {{weight: $weight}}]->(c)
        RETURN e.id
        """
        self.graph.query(q, {
            "claim_id": target_claim_id,
            "source_url": item.source_url,
            "eid": item.id,
            "text": item.text,
            "stance": stance,
            "weight": float(item.weight),
            "snippet": item.snippet or ""
        })

    def get_counts(self) -> Dict[str, int]:
        """Returns node and edge counts in the graph."""
        res_nodes = self.graph.query("MATCH (n) RETURN count(n)").result_set
        node_count = res_nodes[0][0] if res_nodes else 0

        res_edges = self.graph.query("MATCH ()-[r]->() RETURN count(r)").result_set
        edge_count = res_edges[0][0] if res_edges else 0

        res_claims = self.graph.query("MATCH (c:Claim) RETURN count(c)").result_set
        claim_count = res_claims[0][0] if res_claims else 0

        res_ev = self.graph.query("MATCH (e:Evidence) RETURN count(e)").result_set
        ev_count = res_ev[0][0] if res_ev else 0

        res_sources = self.graph.query("MATCH (s:Source) RETURN count(s)").result_set
        source_count = res_sources[0][0] if res_sources else 0

        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "claims": claim_count,
            "evidence": ev_count,
            "sources": source_count
        }

    def run_query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Executes arbitrary Cypher query with optional parameters."""
        return self.graph.query(cypher, params)
