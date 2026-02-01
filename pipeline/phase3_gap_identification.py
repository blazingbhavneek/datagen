"""
phase3_gap_identification.py

Takes the stitched dependency graph from Phase 2 and identifies
the GAPS — places where information is missing, implicit, or only
derivable by connecting multiple pieces.

Single LLM call — no internal parallelism needed. Async interface
kept uniform with the rest of the pipeline.

Usage:
    python phase3_gap_identification.py
"""

import asyncio
import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import (
    get_llm,
    save_intermediate,
    load_intermediate,
    async_retry,
    GapAnalysis,
)


# ─────────────────────────────────────────────
# PROMPT & CHAIN
# ─────────────────────────────────────────────
PHASE3_PROMPT = """You are analyzing a dependency graph extracted from a large Japanese technical document.

The graph contains edges between components. Some edges are marked is_explicit=true (the doc states the connection directly). Some are is_explicit=false (the connection was inferred — the doc never says it).

There are also unresolved_references: components that are used but never fully defined.

Your job: identify GAPS. A gap is a place where understanding the system requires information that is NOT directly retrievable from any single section of the document.

TARGET THREE TYPES OF GAPS:

1. EMERGENT BEHAVIOR GAPS: Two or more implicit edges combine to produce a behavior that no single section documents. E.g., Component A silently modifies state that Component C reads, but neither A nor C mentions the other.

2. NEGATIVE SPACE GAPS: Things the system does NOT do, constraints that are never stated but are structurally enforced. E.g., Component X can never receive input Y because of how Z initializes — but this is never written anywhere.

3. PREREQUISITE CHAIN GAPS: A dependency chain where the middle links are implicit. The doc explains the start and end but not the path between them. Even if you retrieve the start and end chunks, you can't connect them without the middle.

For each gap, identify:
- Which chunks and components are involved
- Why no single chunk contains the full picture
- Which implicit edges from the graph are involved
- WHY retrieval (even multi-pass) will fail here — what textual signal is missing?

--- STITCHED DEPENDENCY GRAPH ---
{graph_json}
--- END GRAPH ---

{format_instructions}"""


def build_gap_chain():
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=GapAnalysis)
    prompt = ChatPromptTemplate.from_template(PHASE3_PROMPT)
    return prompt | llm | parser, parser


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def run():
    print("[Phase 3] Gap identification")

    graph = load_intermediate("phase2", "final_graph")
    if not graph:
        raise RuntimeError("Phase 2 final_graph not found. Run phase2 first.")

    # Skip check
    existing = load_intermediate("phase3", "gap_analysis")
    if existing:
        print("  [skip] Gap analysis already exists")
        return existing

    print(f"  Analyzing graph with {len(graph.get('edges', []))} edges...")
    print(f"  Implicit edges: {sum(1 for e in graph.get('edges', []) if not e.get('is_explicit'))}")

    chain, parser = build_gap_chain()

    async def _call():
        result = await chain.ainvoke({
            "graph_json": json.dumps(graph, ensure_ascii=False, indent=2),
            "format_instructions": parser.get_format_instructions(),
        })
        for i, gap in enumerate(result.gaps):
            if not gap.gap_id:
                gap.gap_id = f"gap_{i:03d}"
        return result

    result = await async_retry(_call)
    await save_intermediate("phase3", "gap_analysis", result.model_dump())

    print(f"[Phase 3] Done. {len(result.gaps)} gaps identified.")
    for gap in result.gaps:
        print(f"  - {gap.gap_id}: {gap.description[:80]}...")

    return result.model_dump()


if __name__ == "__main__":
    asyncio.run(run())
