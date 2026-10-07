"""Combine vector retrieval, graph retrieval and symbolic reasoning into one sectioned context."""
from __future__ import annotations

from src.graph_retriever import GraphRetriever, format_graph_context
from src.models import DocHit, GraphResult, Inference
from src.symbolic_rules import format_symbolic_context, run_rules
from src.vector_store import VectorStore

DOC_HEADER, GRAPH_HEADER, SYM_HEADER = "[DOCUMENT EVIDENCE]", "[GRAPH EVIDENCE]", "[SYMBOLIC REASONING]"


def format_doc_context(hits: list[DocHit]) -> str:
    return "\n".join(f"[D{h.rank}] ({h.doc_id}) {h.title}: {h.text}" for h in hits) or "No documents retrieved."


def build_context(hits: list[DocHit] | None, graph: GraphResult | None, inferences: list[Inference] | None) -> str:
    parts = []
    if hits is not None:
        parts.append(f"{DOC_HEADER}\n{format_doc_context(hits)}")
    if graph is not None:
        parts.append(f"{GRAPH_HEADER}\n{format_graph_context(graph)}")
    if inferences is not None:
        parts.append(f"{SYM_HEADER}\n{format_symbolic_context(inferences)}")
    return "\n\n".join(parts)


class HybridRetriever:
    def __init__(self, vectors: VectorStore, graph: GraphRetriever, top_k: int = 4):
        self.vectors, self.graph, self.top_k = vectors, graph, top_k

    def retrieve(self, question: str, use_vector=True, use_graph=True, use_rules=True) -> dict:
        seeds = self.graph.linker.link(question)  # shared question-understanding step
        hits = self.vectors.search(question, self.top_k) if use_vector else None
        graph = self.graph.retrieve(question, seeds) if use_graph else None
        inferences = run_rules(self.graph.store, seeds) if use_rules else None
        return dict(seeds=seeds, hits=hits, graph=graph, inferences=inferences,
                    context=build_context(hits, graph, inferences))
