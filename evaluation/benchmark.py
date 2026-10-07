"""Benchmark loading and scoring. Ground truth lives in data/benchmark_questions.csv (hand-defined)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.config import DATA_DIR
from src.graph_retriever import EntityLinker
from src.graph_store import GraphStore
from src.models import RAGResponse


@dataclass
class Question:
    id: str
    question: str
    hop_type: str          # single | multi
    gold_hops: int
    expected: list[str]    # entities that must appear in the ANSWER
    evidence: list[str]    # entities that must appear in the retrieved CONTEXT (incl. intermediate nodes)
    note: str


def load_benchmark(path=None) -> list[Question]:
    df = pd.read_csv(path or DATA_DIR / "benchmark_questions.csv", dtype=str).fillna("")
    return [Question(r["id"], r["question"], r["hop_type"], int(r["gold_hops"]), r["expected_answer"].split("|"),
                     r["required_evidence"].split("|"), r["reasoning_note"]) for r in df.to_dict("records")]


class Scorer:
    """String-match scoring (a deliberately crude, transparent proxy - see README limitations)."""

    def __init__(self, store: GraphStore):
        self.linker = EntityLinker(store)
        self.entities = {e.name: e for e in store.all_entities()}
        self._pat = {}

    def mentioned(self, text: str, name: str) -> bool:
        if name not in self._pat:
            ent = self.entities[name]
            alts = sorted({name.lower(), *[a.lower() for a in ent.aliases]}, key=len, reverse=True)
            self._pat[name] = re.compile(r"(?<![a-z0-9])(?:" + "|".join(map(re.escape, alts)) + r")(?![a-z0-9])")
        return bool(self._pat[name].search(text.lower()))

    def score(self, q: Question, resp: RAGResponse) -> dict:
        ans_hits = [e for e in q.expected if self.mentioned(resp.answer, e)]
        ev_hits = [e for e in q.evidence if self.mentioned(resp.context, e)]
        in_question = {e.name for e in self.linker.link(q.question)}
        in_context = {e.name for e in self.linker.link(resp.context)}
        unsupported = sorted({e.name for e in self.linker.link(resp.answer)} - in_context - in_question)
        return dict(
            answer_correct=len(ans_hits) == len(q.expected),
            answer_recall=len(ans_hits) / len(q.expected),
            evidence_complete=len(ev_hits) == len(q.evidence),
            evidence_recall=len(ev_hits) / len(q.evidence),
            retrieval_hops=resp.graph.max_hops if resp.graph else 0,
            n_doc_chunks=len(resp.doc_hits),
            n_graph_paths=len(resp.graph.paths) if resp.graph else 0,
            n_rules_fired=len(resp.inferences),
            context_chars=len(resp.context),
            unsupported_count=len(unsupported),
            unsupported_entities="|".join(unsupported),
        )
