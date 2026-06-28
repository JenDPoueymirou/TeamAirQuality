"""
evals/run_evals.py
------------------
Automated eval runner for the NYC Pollution Chatbot.

Reads evals/golden_set.json, calls POST /chat for each active question,
and checks whether all expected_keywords appear in the answer (case-insensitive).

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --url http://127.0.0.1:8001
    python evals/run_evals.py --verbose

Target: 10/15 for MVP sign-off. Run time: under 2 minutes.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
DEFAULT_URL = "http://127.0.0.1:8001"
REQUEST_DELAY = 3  # seconds between requests — free tier allows 20/min

# ANSI colour codes (fall back gracefully on Windows without VT mode)
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

RESET = "\033[0m"
GREEN = "\033[32m"
RED   = "\033[31m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def call_chat(server_url: str, question: str) -> dict:
    data = json.dumps({"message": question, "history": []}).encode()
    req = urllib.request.Request(
        f"{server_url}/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def evaluate_one(server_url: str, entry: dict, verbose: bool) -> bool:
    question = entry["question"]
    keywords = entry["expected_keywords"]
    qid      = entry["id"]

    try:
        resp = call_chat(server_url, question)
        answer = resp.get("answer", "")
        model  = resp.get("model_used", "unknown")
        rows   = resp.get("rows_retrieved", 0)

        missing = [kw for kw in keywords if kw.lower() not in answer.lower()]
        passed  = len(missing) == 0

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  [{qid:>2}] {status}  {question}")
        print(f"        model={model}  rows={rows}  missing_keywords={missing or 'none'}")

        if verbose or not passed:
            preview = answer[:300].replace("\n", " ")
            print(f"        answer: {preview}")
            if len(answer) > 300:
                print(f"                ... ({len(answer)} chars total)")

        print()
        return passed

    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"  [{qid:>2}] {RED}ERROR{RESET}  {question}")
        print(f"        HTTP {e.code}: {body}")
        print()
        return False

    except Exception as exc:
        print(f"  [{qid:>2}] {RED}ERROR{RESET}  {question}")
        print(f"        {type(exc).__name__}: {exc}")
        print()
        return False


def check_server(url: str) -> None:
    """Verify the server is up and the LLM key is configured. Exit if not."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
            health = json.loads(r.read())
    except Exception as exc:
        print(f"{RED}Cannot reach server at {url}: {exc}{RESET}")
        print("Start it first:  uvicorn chatbot.main:app --host 127.0.0.1 --port 8001")
        sys.exit(1)

    warnings = health.get("config_warnings", [])
    key_missing = any("GEMINI_API_KEY" in w for w in warnings)
    if key_missing:
        print(f"{RED}GEMINI_API_KEY is not set — /chat will return 503{RESET}")
        print("Add the key to backend/.env and restart the server.")
        sys.exit(1)

    csv_rows = health.get("csv_rows", 0)
    chroma   = health.get("chroma_vectors", 0)
    print(f"Server OK  csv_rows={csv_rows}  chroma_vectors={chroma}")


def main() -> int:
    parser = argparse.ArgumentParser(description="NYC Chatbot eval runner")
    parser.add_argument("--url",     default=DEFAULT_URL, help="Server base URL")
    parser.add_argument("--verbose", action="store_true", help="Print full answers")
    args = parser.parse_args()

    if not GOLDEN_SET_PATH.exists():
        print(f"golden_set.json not found at {GOLDEN_SET_PATH}", file=sys.stderr)
        sys.exit(1)

    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    questions  = data["questions"]
    pending    = data.get("pending_questions", [])
    meta       = data.get("meta", {})
    mvp_target = meta.get("mvp_target", 10)
    total      = len(questions)

    print(f"\n{BOLD}NYC Pollution Chatbot — Eval Suite{RESET}")
    print(f"Server : {args.url}")
    print(f"Target : {mvp_target}/{total} to pass MVP")
    print()

    check_server(args.url)
    print()
    print("-" * 64)

    passed = 0
    for i, entry in enumerate(questions):
        if evaluate_one(args.url, entry, args.verbose):
            passed += 1
        if i < total - 1:
            time.sleep(REQUEST_DELAY)

    print("-" * 64)
    score_color = GREEN if passed >= mvp_target else RED
    print(f"\n{BOLD}Score: {score_color}{passed}/{total}{RESET}{BOLD} questions passed{RESET}")

    if passed >= mvp_target:
        print(f"{GREEN}MVP target ({mvp_target}/{total}) met — chatbot is grounded.{RESET}")
    else:
        gap = mvp_target - passed
        print(f"{RED}MVP target not met — {gap} more needed ({mvp_target}/{total}).{RESET}")
        print("Tune expected_keywords in golden_set.json if Llama paraphrases values.")

    if pending:
        print(f"\n{DIM}Skipped {len(pending)} pending questions (Issues #11/#12 required):{RESET}")
        for p in pending:
            print(f"{DIM}  [{p['id']}] {p['question']}{RESET}")

    print()
    return 0 if passed >= mvp_target else 1


if __name__ == "__main__":
    sys.exit(main())
