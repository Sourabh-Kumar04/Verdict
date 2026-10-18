# Verdict: Proof-Carrying Claim-Investigation Agent

> **Track 1: Agents That Act on Connected Data**  
> *Built with FalkorDB 6.0.1 for the Graph Hacks Hackathon*

Verdict is an autonomous, proof-carrying claim-investigation agent that verifies factual claims by constructing an evidence knowledge graph in **FalkorDB**, computing source authority with native **PageRank**, and evaluating verdicts (`SUPPORTED`, `REFUTED`, or `UNVERIFIED`) exclusively via Cypher graph queries.

If evidence is insufficient, contradictory, or absent, Verdict **honestly abstains**—it never hallucinates or guesses.

---

## Architecture

```
                       +---------------------------------------+
                       |             Input Claim               |
                       | "The Eiffel Tower completed in 1889"  |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |        Claim Decomposer (NLP)         |
                       |    - Atomic Sub-claims & Entities     |
                       |    - Temporal & Predicate Extraction  |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |     Two-Tier Evidence Pipeline        |
                       |  PRIMARY: Wikipedia / Wikidata APIs   |
                       |  SECONDARY: Live Web/News Retriever   |
                       |    - SerpApi Google News (if key set) |
                       |    - Google News RSS (zero-config)    |
                       |    - Publication Date Recency Decay   |
                       +-------------------+-------------------+
                                           |
                                           v
       ===================== FALKORDB 6.0.1 LOAD-BEARING GRAPH =====================
      ||                                                                          ||
      ||   (:Claim {id, text, is_root})                                           ||
      ||        ^                                                                 ||
      ||        |-- [:SUPPORTS | :REFUTES | :QUALIFIES {weight}]                  ||
      ||   (:Evidence {id, text, snippet, published_date})                        ||
      ||        |                                                                 ||
      ||        |-- [:EXTRACTED_FROM]                                             ||
      ||        v                                                                 ||
      ||   (:Source {url, name, source_type, pagerank})                           ||
      ||        |                                                                 ||
      ||        +==== [:CITES] ====> (:Source)  [Citation Network]                ||
      ||                                                                          ||
      ||   >>> FalkorDB Graph Analytics:                                          ||
      ||       CALL algo.pageRank('Source', 'CITES') YIELD node, score            ||
      ||       SET s.pagerank = score                                             ||
      ||                                                                          ||
      ||   >>> Cypher Verdict Query:                                              ||
      ||       MATCH (c:Claim)<-[r:SUPPORTS|REFUTES|QUALIFIES]-(e:Evidence)       ||
      ||       MATCH (e)-[:EXTRACTED_FROM]->(s:Source)                            ||
      ||       RETURN sum(r.weight * s.pagerank) AS stance_score                  ||
      ||                                                                          ||
       =============================================================================
                                           |
                                           v
                       +---------------------------------------+
                       |          Proof Path Engine            |
                       |    - Ranked Evidence Traversal        |
                       |    - Audit Trail & Citations          |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |             Final Verdict             |
                       |  [SUPPORTED / REFUTED / UNVERIFIED]   |
                       |       + Ranked Proof Chain            |
                       +---------------------------------------+
```

---

## Anti-Decoration Guarantee

### Why FalkorDB is Strictly Load-Bearing

In many "graph RAG" prototypes, the graph database is purely decorative: claims are fed directly to an LLM prompt, and the database is only queried for context strings or bypassed entirely.

**In Verdict, FalkorDB is mathematically and architecturally load-bearing. There is no direct LLM bypass.**

### What Breaks if FalkorDB is Removed?

1. **Verdict Resolution Fails Immediately**:
   - The verdict (`SUPPORTED` / `REFUTED` / `UNVERIFIED`) is computed directly from Cypher query aggregations (`verdict/verdict.py`):
     ```cypher
     MATCH (c:Claim)
     WHERE c.id IN $target_ids
     OPTIONAL MATCH (e:Evidence)-[r]->(c)
     WHERE type(r) IN ['SUPPORTS', 'REFUTES', 'QUALIFIES']
     OPTIONAL MATCH (e)-[:EXTRACTED_FROM]->(s:Source)
     RETURN c.id AS claim_id,
            type(r) AS stance,
            sum(r.weight * coalesce(s.pagerank, 0.5)) AS weighted_score,
            count(e) AS ev_count
     ```
   - If FalkorDB is removed or disconnected, no verdict can be generated. The agent halts with an explicit load-bearing failure.
2. **Source Authority Collapses**:
   - Source credibility is computed using FalkorDB 6.0.1's native GraphBLAS PageRank algorithm:
     ```cypher
     CALL algo.pageRank('Source', 'CITES') YIELD node, score
     ```
   - Without FalkorDB's graph topology and citation traversal, trust scores cannot be propagated from primary sources to derivative articles.
3. **Proof Traversal Disappears**:
   - The ranked proof path is extracted via multi-hop Cypher traversal (`(:Source)<-[:EXTRACTED_FROM]-(:Evidence)-[r]->(:Claim)`).
   - Without the graph engine, the structured audit chain cannot be verified.
4. **State Persistence**:
   - Each claim node in FalkorDB is updated with its timestamped verdict, confidence, and evidence counts via Cypher `SET` operations.

*This invariant is automatically verified by `tests/test_anti_decoration.py`.*

---

## FalkorDB 6.0.1 Best Practices & Pitfalls Avoided

1. **Vector Wrapping**: Embedding vectors are wrapped using `vecf32([...])` in Cypher queries (`format_vector`). Raw arrays are never passed.
2. **Flat Properties**: All node and relationship properties are flattened into primitives; nested dictionaries are never passed (`flatten_properties`).
3. **WITH Clause Chaining**: All multi-statement write/read Cyphers put `WITH` clauses between `CREATE`/`MERGE` and `MATCH`.
4. **Guarded Empty Deletes**: `graph.delete()` on an empty key throws in FalkorDB 6.0.1 (`Invalid graph operation on empty key`); `delete_graph()` safely catches and guards this exception.

---

## How to Run

### 1. Prerequisites & Start FalkorDB 6.0.1

Start the FalkorDB 6.0.1 Docker container:

```bash
docker run -d -p 6379:6379 -p 3000:3000 --name falkordb-verdict falkordb/falkordb:6.0.1
```

Verify it is running:
```bash
docker ps | grep falkordb
```

### 2. Environment Setup

Install dependencies into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Or using `uv`: `uv venv && uv pip install -r requirements.txt`)*

### 3. Run the Test Suite

Execute the full 25-test suite (including CRUD, PageRank, mocked APIs, anti-decoration, live E2E, SerpApi/RSS news retrievers, recency weighting, and gap fallback):

```bash
pytest -v
```

### 4. Run the CLI Demo

Run an investigation on historical or breaking/recent claims:

```bash
# Historical claim (Verified SUPPORTED via Primary Wikipedia/Wikidata)
python demo.py "The Eiffel Tower was completed in 1889"

# False historical claim (Verified REFUTED via Primary Wikipedia)
python demo.py "Apollo 11 landed on Mars in 1969"

# Breaking / Recent news claim (Verified SUPPORTED via Secondary Live News & Recency Weighting)
python demo.py "Replit makes OpenAI its default over Anthropic"

# Recent event claim (Verified SUPPORTED via Live News Evidence Graph)
python demo.py "Tesla unveiled the Cybercab robotaxi in October 2024"

# Unknown/unprovable claim (Verified ABSTAIN / UNVERIFIED)
python demo.py "The Loch Ness monster was photographed by Isaac Newton in 1675"
```

---

## Two-Tier Evidence Pipeline & Recency Weighting

Verdict employs a two-tier evidence architecture:
1. **PRIMARY (Wikipedia & Wikidata)**:
   - Evaluated first for established factual and historical grounding.
   - Provides structured entity data and peer-reviewed citation networks.
2. **SECONDARY (Live Web/News)**:
   - Invoked automatically when Wikipedia has gaps (zero evidence, low-weight mentions, or purely qualifying stance) or when claims target recent/breaking events (>= 2024).
   - **SerpApi Google News**: Automatically utilized if `SERPAPI_API_KEY` is configured in the environment.
   - **Google News RSS**: Zero-config, free public news feed utilized when no API key is set or as reliable fallback.
   - **FalkorDB Evidence Nodes**: Web news evidence nodes store `published_date`, `snippet`, and `source_url`.
   - **Recency Decay Weighting**: Stance edge weights are weighted by publication recency:
     - $\le 7$ days: $1.05\times$ (breaking news boost)
     - $\le 30$ days: $1.00\times$ (fresh event)
     - $\le 90$ days: $0.90\times$ (recent quarter)
     - $\le 365$ days: $0.80\times$ (past year)
     - $> 365$ days: $0.65\times$ (older news)
   - **Citation PageRank**: Web news items link to publisher domains via `[:CITES]` edges, allowing native FalkorDB PageRank to compute relative authority across wire services and media outlets.

---

## Honest Limitations

1. **Public API Rate Limits & Latency**:
   - Retrieval relies on public Wikipedia, Wikidata, and news RSS APIs. Highly complex claims requiring dozens of sub-claim expansions must be metered to respect public server politeness limits.
2. **Linguistic Stance Heuristics**:
   - The stance evaluator uses lexical-semantic overlap, entity matching, and entity-predicate contradiction matching (e.g. celestial body substitution, date mismatch). Highly nuanced rhetorical claims (e.g. counterfactual statements or allegories) may benefit from fine-grained NLI models.
3. **Citation Density**:
   - PageRank relies on citation links between sources. If the retrieved evidence set spans only isolated pages without cross-links, PageRank assigns equal baseline distribution (0.50).
4. **Knowledge Scope**:
   - Verdict verifies claims against public knowledge in Wikipedia, Wikidata, and live web/news search. Private internal data or paywalled subscriber content not exposed via search snippets will be reported as `UNVERIFIED` (honest abstention).
