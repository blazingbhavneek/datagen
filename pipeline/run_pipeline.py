"""
run_pipeline.py — Async master orchestrator.

Phases run sequentially (each depends on the prior's output).
Within each phase, individual work units run concurrently,
gated by that phase's semaphore.

Resume works the same way: re-run this script, already-done
work units are skipped at the task level before they hit
the semaphore.

Tune concurrency per phase via env vars:
    CONCURRENCY_PHASE1=10
    CONCURRENCY_PHASE2=5
    CONCURRENCY_PHASE4=8
    CONCURRENCY_PHASE5=10

Usage:
    python run_pipeline.py --md path/to/doc.md
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from config import load_intermediate, INTERMEDIATES_DIR


def print_header(phase_num, title):
    print(f"\n{'═'*60}")
    print(f"  PHASE {phase_num}: {title}")
    print(f"{'═'*60}\n")


def print_summary():
    files = sorted(INTERMEDIATES_DIR.glob("*.json"))
    if not files:
        print("  (no intermediates yet)")
        return
    print(f"\n  Intermediates on disk ({len(files)} files):")
    for f in files:
        size = f.stat().st_size
        print(f"    {f.name:<55} {size:>6} bytes")


async def run(md_path: str):
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        RAG BENCHMARK PIPELINE — ASYNC FULL RUN            ║")
    print(f"║  MD:  {Path(md_path).name:<53}║")
    print(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<49}║")
    print("╚════════════════════════════════════════════════════════════╝")

    # ─────────────────────────────────────────
    # PHASE 1 — reads markdown, chunks it, concurrent extraction
    # ─────────────────────────────────────────
    print_header(1, "Chunk Extraction — relationship maps per section")
    from phase1_chunk_extraction import run as phase1_run
    await phase1_run(md_path)
    print_summary()

    # ─────────────────────────────────────────
    # PHASE 2 — concurrent batches per pass, sequential passes
    # ─────────────────────────────────────────
    print_header(2, "Cross-Chunk Stitching — dependency graph")
    from phase2_cross_chunk_stitching import run as phase2_run
    await phase2_run()
    print_summary()

    # ─────────────────────────────────────────
    # PHASE 3 — single call
    # ─────────────────────────────────────────
    print_header(3, "Gap Identification — invisible edges")
    from phase3_gap_identification import run as phase3_run
    await phase3_run()
    print_summary()

    # ─────────────────────────────────────────
    # PHASE 4 — concurrent gaps (reads chunk text from phase1 manifest)
    # ─────────────────────────────────────────
    print_header(4, "Question Generation — Japanese benchmark questions")
    from phase4_question_generation import run as phase4_run
    await phase4_run()
    print_summary()

    # ─────────────────────────────────────────
    # PHASE 5 — concurrent questions (reads chunk text from phase1 manifest)
    # ─────────────────────────────────────────
    print_header(5, "Validation — filtering questions that break RAG")
    from phase5_validation import run as phase5_run
    final = await phase5_run()
    print_summary()

    # ─────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  EXPORTING FINAL BENCHMARK")
    print("═" * 60 + "\n")

    output_path = Path(__file__).parent / "benchmark_output.json"
    output_path.write_text(json.dumps(final, ensure_ascii=False, indent=2))
    print(f"  Saved: {output_path}")
    print(f"\n  Summary:")
    for k, v in final.get("summary", {}).items():
        print(f"    {k}: {v}")

    print(f"\n  ── Kept Questions (break RAG) ──\n")
    for q in final.get("kept_questions", []):
        print(f"    [{q['question_id']}]")
        print(f"      Q: {q['question_ja']}")
        print(f"      A: {q['correct_answer']}")
        print(f"      Model answered: {q['model_answer']}")
        print()

    print("═" * 60)
    print("  PIPELINE COMPLETE")
    print("═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Benchmark Pipeline")
    parser.add_argument("--md", required=True, help="Path to the markdown document")
    args = parser.parse_args()

    if not Path(args.md).exists():
        print(f"ERROR: Markdown file not found at {args.md}")
        sys.exit(1)

    asyncio.run(run(args.md))
