#!/usr/bin/env python3
"""
Verdict CLI Demo
Usage:
  python demo.py "The Eiffel Tower was completed in 1889"
"""

import sys
import json
from verdict.agent import VerdictAgent


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py \"<claim>\"")
        print("Example: python demo.py \"The Eiffel Tower was completed in 1889\"")
        sys.exit(1)

    claim = sys.argv[1].strip()
    if not claim:
        print("Error: Claim cannot be empty.")
        sys.exit(1)

    print("=" * 70)
    print(" VERDICT: Proof-Carrying Claim Investigation Agent (FalkorDB)")
    print("=" * 70)
    print(f"Investigating Claim:\n  \"{claim}\"\n")

    agent = VerdictAgent()

    print("[1/5] Decomposing claim into atomic sub-claims...")
    subclaims = agent.decomposer.decompose(claim)
    for sc in subclaims:
        print(f"  -> Sub-claim: {sc.text} (Entities: {sc.entities})")

    print("\n[2/5] Retrieving evidence from Wikipedia & Wikidata APIs...")
    print("[3/5] Populating FalkorDB evidence graph & computing PageRank trust...")
    result = agent.investigate(claim)

    counts = result["graph_counts"]
    print(f"  -> Graph populated: {counts['total_nodes']} nodes, {counts['total_edges']} relationships")
    print(f"     (Claims: {counts['claims']}, Evidence: {counts['evidence']}, Sources: {counts['sources']})")

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
