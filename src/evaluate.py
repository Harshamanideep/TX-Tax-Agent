"""Run the test questions through the agent and score the answers.

Run:  python -m src.evaluate                     all questions
      python -m src.evaluate --limit 5           first 5 only
      python -m src.evaluate --category search   one category
      python -m src.evaluate --delay 4           wait 4s between questions (rate limits)
      python -m src.evaluate --resume eval/results/run_XXXX.jsonl
                                                 continue a run that stopped

Each question is checked on:
  keywords   the answer contains the expected facts (and none of the forbidden ones)
  citation   the answer cites at least one expected page (if pages are given)
  tools      the agent called every expected tool (if tools are given)
A question PASSES only if every check that applies passes.
"""
import argparse
import json
import os
import re
import time
import unicodedata
import uuid
from collections import defaultdict
from datetime import datetime

from src.agent import run

QUESTIONS_FILE = "eval/questions.json"
RESULTS_DIR = "eval/results"


def normalize(text: str) -> str:
    # Some models write non-breaking hyphens/spaces ("TC‑40W", "April 15"),
    # which look identical but don't match a normal "-" or " ".
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[\u00a0\u202f\u2009\u200b]", " ", text)
    text = text.lower().replace("**", "").replace("’", "'")
    text = re.sub(r"(\d)\s+¢", r"\1¢", text).replace("¢", " cents")  # "50 ¢" -> "50 cents"
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 2,340 -> 2340
    text = re.sub(r"\.0+\b", "", text)              # 2340.0 -> 2340
    return text


def contains(text: str, phrase: str) -> bool:
    """Whole-word match, so 'no' doesn't match 'not' and 'valid' doesn't match 'invalid'."""
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def cited_pages(answer: str) -> set[int]:
    """Find page numbers in citations such as [UT TC-40-instructions, p.4, p.5],
    (TC-40 instructions, page 4) or 【UT TC-40, p. 4-5】."""
    pages = set()
    for citation in re.findall(r"[\[\(【]([^\]\)】]*)[\]\)】]", normalize(answer)):
        for start, end in re.findall(r"(?:pp?\.|pages?)\s*(\d+)(?:\s*-\s*(\d+))?", citation):
            pages.update(range(int(start), int(end or start) + 1))
        # extra pages in a list: "p.4, 5" or "p.4, p.5"
        if re.search(r"(?:pp?\.|pages?)\s*\d", citation):
            pages.update(int(n) for n in re.findall(r",\s*(\d+)\b", citation))
    return pages


def grade(case: dict, answer: str, tools_used: list[str]) -> dict:
    text = normalize(answer)
    checks = {}

    missing = [group for group in case.get("must_include", [])
               if not any(contains(text, alt) for alt in group)]
    forbidden = [p for p in case.get("must_exclude", []) if contains(text, p)]
    checks["keywords"] = not missing and not forbidden

    if case.get("pages"):
        checks["citation"] = bool(cited_pages(answer) & set(case["pages"]))

    if case.get("tools"):
        called = {t.split("(")[0] for t in tools_used}
        checks["tools"] = all(t in called for t in case["tools"])

    return {
        "checks": checks,
        "passed": all(checks.values()),
        "missing": [" / ".join(g) for g in missing],
        "forbidden_found": forbidden,
    }


def summarize(results: list[dict]) -> str:
    lines = []
    passed = sum(r["passed"] for r in results)
    lines.append(f"\nOVERALL: {passed}/{len(results)} passed ({100 * passed / len(results):.0f}%)")

    for check in ("keywords", "citation", "tools"):
        scored = [r["checks"][check] for r in results if check in r["checks"]]
        if scored:
            lines.append(f"  {check:<9} {sum(scored)}/{len(scored)} ({100 * sum(scored) / len(scored):.0f}%)")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["passed"])
    lines.append("\nBy category:")
    for cat, vals in sorted(by_cat.items()):
        lines.append(f"  {cat:<13} {sum(vals)}/{len(vals)}")

    ok = [r for r in results if "seconds" in r]
    if ok:
        avg = sum(r["seconds"] for r in ok) / len(ok)
        calls = sum(len(r["tools_used"]) for r in ok) / len(ok)
        lines.append(f"\nAvg latency: {avg:.1f}s   Avg tool calls: {calls:.1f}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        lines.append("\nFailures:")
        for r in failures:
            failed = [c for c, ok_ in r["checks"].items() if not ok_]
            detail = f" missing: {r['missing']}" if r["missing"] else ""
            detail += f" forbidden: {r['forbidden_found']}" if r["forbidden_found"] else ""
            lines.append(f"  {r['id']:<20} failed {', '.join(failed) or 'error'}{detail}")
            if r.get("error"):
                lines.append(f"      error: {r['error'][:150]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--resume", help="results .jsonl file to continue")
    parser.add_argument("--rescore", help="re-grade a results .jsonl file without calling the LLM")
    parser.add_argument("--show", action="store_true", help="print the answers of failed questions")
    args = parser.parse_args()

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        cases = json.load(f)
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]

    if args.rescore:
        rescore(args.rescore, {c["id"]: c for c in cases}, args.show)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.resume or os.path.join(
        RESULTS_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}.jsonl")

    results, done = [], set()
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("error"):  # retry questions that errored
                    results.append(r)
                    done.add(r["id"])
        print(f"Resuming: {len(done)} questions already done.")

    threads = {}  # cases sharing a "thread" key run in one conversation
    todo = [c for c in cases if c["id"] not in done]
    print(f"Running {len(todo)} questions -> {out_path}\n")

    for i, case in enumerate(todo, 1):
        thread = case.get("thread")
        thread_id = threads.setdefault(thread, str(uuid.uuid4())) if thread else str(uuid.uuid4())

        record = {"id": case["id"], "category": case["category"], "question": case["question"]}
        start = time.time()
        try:
            out = run(case["question"], thread_id)
            record.update(answer=out["answer"], tools_used=out["tools_used"],
                          seconds=round(time.time() - start, 1))
            record.update(grade(case, out["answer"], out["tools_used"]))
        except Exception as e:
            record.update(answer="", tools_used=[], error=str(e), checks={},
                          passed=False, missing=[], forbidden_found=[])

        status = "PASS" if record["passed"] else ("ERROR" if record.get("error") else "FAIL")
        print(f"[{i}/{len(todo)}] {status:<5} {case['id']}")
        results.append(record)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if record.get("error") and "429" in record["error"]:
            print("\nRate limit hit. Stopping. Continue later with:")
            print(f"  python -m src.evaluate --resume {out_path}")
            break
        if args.delay and i < len(todo):
            time.sleep(args.delay)

    summary = summarize(results)
    print(summary)
    if args.show:
        print(show_failures(results))
    with open(out_path.replace(".jsonl", "_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)


def show_failures(results: list[dict]) -> str:
    lines = ["\n=== Failed answers ==="]
    for r in results:
        if not r["passed"]:
            lines.append(f"\n[{r['id']}] {r['question']}")
            lines.append(f"tools: {r.get('tools_used')}")
            lines.append(r.get("answer") or r.get("error", ""))
    return "\n".join(lines)


def rescore(path: str, cases_by_id: dict, show: bool):
    """Re-grade saved answers with the current rules. Costs no LLM requests."""
    latest = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            latest[r["id"]] = r  # keep the last attempt of each question
    results = []
    for r in latest.values():
        case = cases_by_id.get(r["id"])
        if case and not r.get("error"):
            r.update(grade(case, r["answer"], r["tools_used"]))
        results.append(r)
    print(f"Re-scored {len(results)} answers from {path}")
    print(summarize(results))
    if show:
        print(show_failures(results))


if __name__ == "__main__":
    main()