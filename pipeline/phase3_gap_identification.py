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

Your job: identify GAPS. A gap is a place where understanding the system requires information that is NOT directly retrievable from any single section of the document, AND where even multi-pass RAG will struggle because the connections require deep inference.

TARGET SEVEN TYPES OF GAPS (prioritize complexity):

1. EMERGENT BEHAVIOR GAPS: Three or more implicit edges combine to produce a behavior that no single section documents. The behavior only emerges when you trace the full chain. E.g., Component A modifies state X, which triggers B's condition, which enables C's output mode — but no section mentions this ABC sequence.

2. NEGATIVE SPACE GAPS: Things the system does NOT do, constraints that are never stated but are structurally enforced through the absence of connections. E.g., Component X can never receive input Y because Z initializes first and consumes all Y, but this mutual exclusion is never documented.

3. PREREQUISITE CHAIN GAPS: A dependency chain where MULTIPLE middle links are implicit. The doc explains the start and end but not the 2-3 intermediate transformations. Even retrieving start and end chunks won't reveal the hidden transformations.

4. TEMPORAL DEPENDENCY GAPS: Ordering constraints that are implied by scattered initialization sequences, lifecycle mentions, or state transitions across different sections. The "must happen before" relationship exists but is never stated. Requires assembling a timeline from fragments.

5. CONSTRAINT INTERACTION GAPS: Two separately-documented constraints that, when combined, create a third unstated constraint or capability limit. E.g., "Max users: 100" in one section + "Each user spawns 5 threads" in another = unstated 500 thread limit that might cause issues.

6. CROSS-DOMAIN INFERENCE GAPS: A property in one domain (e.g., network timing) affects behavior in another domain (e.g., data consistency) but the document never bridges these domains. Requires knowledge of both domains AND the implicit connection.

7. MULTI-LAYER ABSTRACTION GAPS: Implementation detail in one section affects behavior at a higher abstraction level discussed in a different section, but the document never connects the layers. E.g., a low-level buffer size determines high-level transaction batching, but these are discussed separately without linking them.

For each gap, identify:
- Which chunks and components are involved (need at least 3 chunks for complexity)
- Why no single chunk OR simple two-chunk retrieval contains the full picture
- Which implicit edges from the graph are involved (need at least 2 implicit edges)
- The INFERENCE PATH: what mental steps are required to bridge the gap
- WHY multi-pass RAG will fail: what textual signals are missing, what connections require domain knowledge or logical inference that can't be retrieved

PRIORITIZE GAPS WHERE:
- The answer requires combining 3+ chunks
- At least 2 implicit edges are involved
- The connection requires inference, not just concatenation
- Different terminology is used in different chunks for related concepts
- Temporal or causal reasoning is needed
- The answer contradicts naive assumptions

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
    print(f"  Unresolved references: {len(graph.get('unresolved_references', []))}")

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
        print(f"    └─ Involves {len(gap.involved_chunks)} chunks, {len(gap.involved_components)} components")

    return result.model_dump()


if __name__ == "__main__":
    asyncio.run(run())
