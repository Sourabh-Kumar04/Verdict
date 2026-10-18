#!/usr/bin/env python3
"""
Verdict Web Server Runner
Usage:
  python server.py
  python server.py --host 0.0.0.0 --port 8000
"""

import sys
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the Verdict Web Application")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"Starting Verdict Web Server at http://{args.host}:{args.port}")
    uvicorn.run("verdict.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
