#!/usr/bin/env python3
"""
Verdict CLI Demo
Usage:
  python demo.py "The Eiffel Tower was completed in 1889"
"""

import sys
import json
from verdict.agent import VerdictAgent
from verdict.graph_store import FalkorDBConnectionError
from verdict.sanitizer import sanitize_claim


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py \"<claim>\"")
        print("Example: python demo.py \"The Eiffel Tower was completed in 1889\"")
        sys.exit(1)

    raw_input = sys.argv[1]
    claim = sanitize_claim(raw_input)
    if not claim:
        print("Error: Claim cannot be empty or contain only invalid/control characters.")
        sys.exit(1)

    print("=" * 70)
    print(" VERDICT: Proof-Carrying Claim Investigation Agent (FalkorDB)")
    print("=" * 70)
    print(f"Investigating Claim:\n  \"{claim}\"\n")

    try:
        agent = VerdictAgent()
        agent.check_connection()
    except (FalkorDBConnectionError, ConnectionError, Exception) as e:
        err_msg = str(e)
        if "falkordb" in err_msg.lower() or "connection" in err_msg.lower() or "refused" in err_msg.lower():
            print("\n" + "=" * 70)
            print(" [ERROR] FalkorDB Connection Failed")
            print("=" * 70)
            print("Could not connect to FalkorDB at localhost:6379.")
            print("\nVerdict requires FalkorDB 6.0.1 as a load-bearing graph database.")
            print("No LLM bypass or decorative fallback is permitted.")
            print("\nTo start FalkorDB in Docker, run:")
            print("  docker run -d -p 6379:6379 -p 3000:3000 --name falkordb-verdict falkordb/falkordb:6.0.1")
            print("=" * 70)
            sys.exit(1)
        raise

    print("[1/5] Decomposing claim into atomic sub-claims...")
    subclaims = agent.decomposer.decompose(claim)
    for sc in subclaims:
        print(f"  -> Sub-claim: {sc.text} (Entities: {sc.entities})")

    print("\n[2/5] Retrieving evidence from Wikipedia/Wikidata (Primary) & News/Fact-Checkers (Secondary)...")
    print("[3/5] Populating FalkorDB evidence graph & computing PageRank trust...")
    try:
        result = agent.investigate(claim)
    except (FalkorDBConnectionError, ConnectionError) as e:
        print(f"\n[ERROR] FalkorDB operation failed: {e}")
        sys.exit(1)

    counts = result["graph_counts"]
    print(f"  -> Graph populated: {counts['total_nodes']} nodes, {counts['total_edges']} relationships")
    print(f"     (Claims: {counts['claims']}, Evidence: {counts['evidence']}, Sources: {counts['sources']})")

    fc_count = sum(1 for p in result["proof_steps"] if "FactCheck" in getattr(p, "source_name", ""))
    if fc_count > 0:
        print(f"  -> Fact-checker evidence integrated: {fc_count} verification(s) from Snopes / PolitiFact / FactCheck.org")

    if counts["evidence"] == 0:
        print("  -> Note: All evidence sources returned empty. Graph correctly executed strict abstention.")

    if result["trust_scores"]:
        print("  -> PageRank Source Trust:")
        for url, score in list(result["trust_scores"].items())[:3]:
            print(f"     * {url}: {score:.4f}")

    print("\n[4/5] Executing Cypher verdict graph query against FalkorDB...")
    verdict = result["verdict"]
    confidence = result["confidence"]
    reason = result["reason"]

    verdict_badge = f"[{verdict}]"
    print("\n" + "=" * 70)
    print(f" FINAL VERDICT: {verdict_badge} (Confidence: {confidence:.2f})")
    print(f" Reason: {reason}")
    print("=" * 70)

    print("\n[5/5] Ranked Proof Path:")
    print(result["proof_path"])
    print("\nOne-Line Proof Summary:")
    print(f"  {result['one_line_proof']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
