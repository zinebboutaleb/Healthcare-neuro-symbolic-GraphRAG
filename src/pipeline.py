"""End-to-end pipeline.   python -m src.pipeline "your question" --mode hybrid"""
from __future__ import annotations

import argparse
import time

from src.embeddings import get_embedder
from src.graph_retriever import EntityLinker, GraphRetriever
from src.graph_store import GraphStore, get_store
from src.hybrid_retriever import HybridRetriever
from src.llm import LLM
from src.models import RAGResponse
from src.symbolic_rules import RULE_DESCRIPTIONS
from src.vector_store import VectorStore

# mode -> (vector, graph, symbolic rules). The first three are the project's headline modes.
MODES = {
    "vector": (True, False, False),
    "graph": (False, True, False),
    "hybrid": (True, True, True),
    # ablations (used only with `evaluate --ablation`)
    "graph+rules": (False, True, True),
    "vector+graph": (True, True, False),
}


class Pipeline:
    def __init__(self, store: GraphStore | None = None, llm: LLM | None = None, top_k: int = 4, max_hops: int = 4):
        self.store = store or get_store()
        self.embedder = get_embedder()
        self.llm = llm or LLM()
        self.vectors = VectorStore(self.embedder)
        self.graph = GraphRetriever(self.store, EntityLinker(self.store), max_hops=max_hops)
        self.hybrid = HybridRetriever(self.vectors, self.graph, top_k=top_k)
        self.embedder.embed(["warm-up"])  # exclude one-off model load from latency numbers
        if self.llm.provider == "hf":  # load weights now so latency numbers exclude model loading
            self.llm.answer("warm-up", "warm-up")

    @property
    def backends(self) -> dict[str, str]:
        return {"graph_backend": self.store.backend, "embedding_backend": self.embedder.name, "llm": self.llm.name}

    def answer(self, question: str, mode: str = "hybrid") -> RAGResponse:
        use_vector, use_graph, use_rules = MODES[mode]
        t0 = time.perf_counter()
        r = self.hybrid.retrieve(question, use_vector, use_graph, use_rules)
        t1 = time.perf_counter()
        answer = self.llm.answer(question, r["context"])
        t2 = time.perf_counter()
        return RAGResponse(question=question, mode=mode, answer=answer, context=r["context"], linked_entities=r["seeds"],
                           doc_hits=r["hits"] or [], graph=r["graph"], inferences=r["inferences"] or [],
                           latency_retrieval_ms=(t1 - t0) * 1000, latency_generation_ms=(t2 - t1) * 1000, backends=self.backends)


def explain(resp: RAGResponse) -> str:
    """Human-readable evidence trail: entities -> graph paths -> rules (source facts -> rule -> conclusion)."""
    lines = [f"Mode: {resp.mode}   Backends: {resp.backends}"]
    lines.append("Linked entities: " + (", ".join(f"{e.name} ({e.type})" for e in resp.linked_entities) or "-"))
    if resp.doc_hits:
        lines.append("\nRetrieved document chunks:")
        lines += [f"  [D{h.rank}] {h.doc_id} score={h.score:.3f}  {h.text[:110]}..." for h in resp.doc_hits]
    if resp.graph and resp.graph.paths:
        lines.append(f"\nGraph reasoning paths (longest = {resp.graph.max_hops} hops):")
        lines += [f"  [G{i}] {p.render()}" for i, p in enumerate(resp.graph.paths, 1)]
    if resp.inferences:
        lines.append("\nSymbolic inferences (source facts -> rule -> conclusion):")
        for inf in resp.inferences:
            lines.append(f"  {inf.rule_id} {inf.rule_name}: {inf.conclusion}")
            lines += [f"      premise: {p}" for p in inf.premises]
            lines.append(f"      rule:    {RULE_DESCRIPTIONS[inf.rule_id]}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", choices=list(MODES), default="hybrid")
    ap.add_argument("--show-context", action="store_true")
    a = ap.parse_args()
    pipe = Pipeline()
    print(f"[config] graph={pipe.backends['graph_backend']}  embeddings={pipe.backends['embedding_backend']}  llm={pipe.backends['llm']}")
    resp = pipe.answer(a.question, a.mode)
    if pipe.llm.provider == "offline":
        print("ANSWER: (offline mode: LLM_PROVIDER resolved to 'offline'. Check that .env exists in the project root "
              "and contains LLM_PROVIDER=hf)")
    else:
        print("ANSWER:\n" + resp.answer)
    print("\n" + "=" * 80 + "\nEVIDENCE\n" + "=" * 80)
    print(explain(resp))
    if a.show_context:
        print("\n" + "=" * 80 + "\nCONTEXT SENT TO LLM\n" + "=" * 80 + "\n" + resp.context)
    print(f"\nlatency: retrieval {resp.latency_retrieval_ms:.1f} ms, generation {resp.latency_generation_ms:.1f} ms")


if __name__ == "__main__":
    main()