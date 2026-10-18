"""
Verdict FastAPI Backend Server
Provides REST and Server-Sent Events (SSE) endpoints for claim investigation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from verdict.agent import VerdictAgent
from verdict.graph_store import FalkorDBConnectionError

logger = logging.getLogger("verdict.server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Verdict API",
    description="Proof-Carrying Claim-Investigation Agent powered by FalkorDB",
    version="0.1.0"
)

# Enable CORS for developer ease
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Agent singleton instance
_agent: Optional[VerdictAgent] = None


def get_agent() -> VerdictAgent:
    global _agent
    if _agent is None:
        _agent = VerdictAgent()
    return _agent


class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="The factual claim to investigate")


def serialize_proof_step(step: Any) -> Dict[str, Any]:
    """Serializes a ProofStep dataclass or dict into a JSON-safe dictionary."""
    if hasattr(step, "__dict__"):
        return {
            "claim_id": getattr(step, "claim_id", ""),
            "claim_text": getattr(step, "claim_text", ""),
            "evidence_id": getattr(step, "evidence_id", ""),
            "evidence_text": getattr(step, "evidence_text", ""),
            "stance": getattr(step, "stance", "UNVERIFIED"),
            "weight": float(getattr(step, "weight", 0.0)),
            "source_name": getattr(step, "source_name", ""),
            "source_url": getattr(step, "source_url", ""),
            "source_pagerank": float(getattr(step, "source_pagerank", 0.0)),
            "rank_score": float(getattr(step, "rank_score", 0.0)),
            "published_date": getattr(step, "published_date", ""),
            "snippet": getattr(step, "snippet", "")
        }
    elif isinstance(step, dict):
        return step
    return {"raw": str(step)}


def serialize_verdict(result: Dict[str, Any]) -> Dict[str, Any]:
    """Prepares verdict dictionary for clean JSON API response."""
    confidence = float(result.get("confidence", 0.0))
    steps = [serialize_proof_step(s) for s in result.get("proof_steps", [])]
    return {
        "claim_id": result.get("claim_id", ""),
        "claim": result.get("claim", ""),
        "verdict": result.get("verdict", "UNVERIFIED"),
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.1f}%",
        "reason": result.get("reason", ""),
        "subverdicts": result.get("subverdicts", []),
        "graph_counts": result.get("graph_counts", {
            "total_nodes": 0, "total_edges": 0, "claims": 0, "evidence": 0, "sources": 0
        }),
        "trust_scores": {k: round(float(v), 4) for k, v in result.get("trust_scores", {}).items()},
        "proof_steps": steps,
        "proof_path": result.get("proof_path", ""),
        "one_line_proof": result.get("one_line_proof", "")
    }


@app.get("/", response_class=HTMLResponse)
def get_index():
    """Serves the main Verdict UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found")
    return index_file.read_text(encoding="utf-8")


@app.get("/api/health")
def health_check():
    """Checks service health and FalkorDB connectivity."""
    agent = get_agent()
    falkordb_ok = False
    error_msg = None
    try:
        falkordb_ok = agent.check_connection()
    except Exception as e:
        error_msg = str(e)

    return {
        "status": "healthy" if falkordb_ok else "degraded",
        "falkordb_connected": falkordb_ok,
        "graph_name": agent.graph_store.graph_name,
        "error": error_msg,
        "falkordb_docker_command": "docker run -d -p 6379:6379 -p 3000:3000 --name falkordb-verdict falkordb/falkordb:6.0.1"
    }


@app.post("/api/investigate")
def investigate_claim(req: ClaimRequest):
    """
    Synchronous POST endpoint to investigate a factual claim.
    Returns complete verdict payload.
    """
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    agent = get_agent()

    # Pre-check FalkorDB connectivity
    try:
        agent.check_connection()
    except Exception as e:
        logger.error(f"FalkorDB connection failure: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "FalkorDB Connection Failed",
                "detail": "Could not connect to FalkorDB at localhost:6379. Make sure the 'falkordb-verdict' docker container is running.",
                "falkordb_docker_command": "docker run -d -p 6379:6379 -p 3000:3000 --name falkordb-verdict falkordb/falkordb:6.0.1"
            }
        )

    try:
        result = agent.investigate(claim)
        return serialize_verdict(result)
    except (FalkorDBConnectionError, ConnectionError) as e:
        logger.error(f"FalkorDB runtime error: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "FalkorDB Operation Failed",
                "detail": str(e),
                "falkordb_docker_command": "docker run -d -p 6379:6379 -p 3000:3000 --name falkordb-verdict falkordb/falkordb:6.0.1"
            }
        )
    except Exception as e:
        logger.exception("Unexpected error during investigation")
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")


@app.get("/api/investigate/stream")
def stream_investigate_claim(claim: str = Query(..., min_length=1)):
    """
    Server-Sent Events (SSE) endpoint providing live 5-stage pipeline progress
    followed by the final complete verdict result.
    """
    agent = get_agent()

    def event_generator():
        try:
            agent.check_connection()
        except Exception as e:
            err_payload = json.dumps({
                "type": "error",
                "error": "FalkorDB Connection Failed",
                "detail": "Could not connect to FalkorDB at localhost:6379. Make sure container 'falkordb-verdict' is running.",
                "falkordb_docker_command": "docker run -d -p 6379:6379 -p 3000:3000 --name falkordb-verdict falkordb/falkordb:6.0.1"
            })
            yield f"data: {err_payload}\n\n"
            return

        try:
            for item in agent.investigate_stream(claim):
                if item["type"] == "stage":
                    payload = json.dumps({
                        "type": "stage",
                        "stage": item["stage"],
                        "name": item["name"],
                        "detail": item["detail"]
                    })
                    yield f"data: {payload}\n\n"
                elif item["type"] == "complete":
                    serialized = serialize_verdict(item["result"])
                    payload = json.dumps({
                        "type": "complete",
                        "result": serialized
                    })
                    yield f"data: {payload}\n\n"
        except Exception as e:
            logger.exception("Stream error during investigation")
            err_payload = json.dumps({
                "type": "error",
                "error": "Investigation Failed",
                "detail": str(e)
            })
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
