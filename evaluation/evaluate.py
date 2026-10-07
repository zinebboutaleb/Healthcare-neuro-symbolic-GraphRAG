"""Run the toy benchmark in every retrieval mode.   python -m evaluation.evaluate [--ablation]

Writes evaluation/results.csv (one row per question x mode) and evaluation/results.md (summary tables).
Nothing is hard-coded: every number comes from running the pipeline.
"""
from __future__ import annotations

import argparse
import statistics as st

import pandas as pd

from evaluation.benchmark import Scorer, load_benchmark
from src.config import EVAL_DIR
from src.pipeline import Pipeline

LABEL = {"vector": "Vector RAG", "graph": "GraphRAG", "hybrid": "Hybrid (graph+vector+rules)",
         "graph+rules": "ablation: Graph + rules", "vector+graph": "ablation: Vector + graph"}


def run(modes: list[str]) -> tuple[pd.DataFrame, dict]:
    pipe = Pipeline()
    scorer = Scorer(pipe.store)
    rows = []
    for q in load_benchmark():
        for mode in modes:
            resp = pipe.answer(q.question, mode)
            rows.append(dict(question_id=q.id, hop_type=q.hop_type, gold_hops=q.gold_hops, mode=mode, **scorer.score(q, resp),
                             latency_retrieval_ms=round(resp.latency_retrieval_ms, 2),
                             latency_generation_ms=round(resp.latency_generation_ms, 2),
                             latency_total_ms=round(resp.latency_retrieval_ms + resp.latency_generation_ms, 2),
                             answer=resp.answer.replace("\n", " \\n ")[:600]))
    return pd.DataFrame(rows), pipe.backends


def summarize(df: pd.DataFrame, modes: list[str]) -> pd.DataFrame:
    out = []
    for m in modes:
        d = df[df["mode"] == m]
        s, mh = d[d.hop_type == "single"], d[d.hop_type == "multi"]
        out.append({
            "Method": LABEL[m],
            "Single-hop answer correct": f"{int(s.answer_correct.sum())}/{len(s)}",
            "Multi-hop answer correct": f"{int(mh.answer_correct.sum())}/{len(mh)}",
            "Evidence complete (all)": f"{int(d.evidence_complete.sum())}/{len(d)}",
            "Evidence complete (multi-hop)": f"{int(mh.evidence_complete.sum())}/{len(mh)}",
            "Mean evidence recall": f"{d.evidence_recall.mean():.2f}",
            "Unsupported entities": int(d.unsupported_count.sum()),
            "Median context chars": int(st.median(d.context_chars)),
            "Median latency ms": f"{st.median(d.latency_total_ms):.1f}",
        })
    return pd.DataFrame(out)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(df: pd.DataFrame, backends: dict, modes: list[str]) -> str:
    summary = summarize(df, modes)
    per_q = df.pivot(index="question_id", columns="mode", values=["answer_correct", "evidence_complete"])
    q_rows = []
    for qid in sorted(df.question_id.unique()):
        d = df[df.question_id == qid]
        row = {"Q": qid, "hops": int(d.gold_hops.iloc[0])}
        for m in modes:
            r = d[d["mode"] == m].iloc[0]
            row[LABEL[m]] = f"{'Y' if r.answer_correct else 'n'} / {'Y' if r.evidence_complete else 'n'} ({r.evidence_recall:.2f})"
        q_rows.append(row)
    offline = backends["llm"].startswith("offline")
    md = [
        "# Evaluation results (toy benchmark)", "",
        f"Backends used for this run: `{backends}`", "",
        "**This is a toy benchmark: 14 hand-written questions over a 113-entity synthetic KB. "
        "Numbers illustrate the methodology; they are not evidence of general or clinical performance.**", "",
        "## Summary", "", to_markdown(summary), "",
        "## Per question  (answer correct / evidence complete (evidence recall))", "", to_markdown(pd.DataFrame(q_rows)), "",
        "## How to read this", "",
        "- *Answer correct*: every expected entity is mentioned in the answer (string match, aliases allowed).",
        "- *Evidence complete / recall*: share of the required entities (answer + intermediate nodes) present in the retrieved context.",
        "- *Unsupported entities*: KB entities named in the answer that are in neither the context nor the question (hallucination proxy).",
        "- *Median context chars*: size of the context sent to the LLM. Larger contexts mechanically raise evidence recall, so compare with care.",
    ]
    if offline:
        md += ["", "> **Offline mode:** no LLM was configured, so the 'answer' is the retrieved context itself. "
               "Answer-correctness therefore equals evidence coverage and the unsupported-entity count is zero by construction. "
               "Set an LLM provider (see `.env.example`) to measure real answer quality and hallucination."]
    if "hashing" in backends["embedding_backend"]:
        md += ["", "> **Embedding fallback:** the hashing embedder is a lexical bag-of-words model, not MiniLM. "
               "Vector-RAG numbers will differ with `all-MiniLM-L6-v2`."]
    text = "\n".join(md) + "\n"
    (EVAL_DIR / "results.md").write_text(text, encoding="utf-8")
    return text


def k_sweep(ks=(4, 8, 12, 16)) -> pd.DataFrame:
    """Control experiment: does Vector RAG catch up when it is given a context budget comparable to GraphRAG?"""
    pipe = Pipeline()
    scorer = Scorer(pipe.store)
    rows = []
    for k in ks:
        pipe.hybrid.top_k = k
        res = [(q, pipe.answer(q.question, "vector")) for q in load_benchmark()]
        sc = [scorer.score(q, r) for q, r in res]
        multi = [s for (q, _), s in zip(res, sc) if q.hop_type == "multi"]
        rows.append(dict(top_k=k, evidence_complete=f"{sum(s['evidence_complete'] for s in sc)}/{len(sc)}",
                         evidence_complete_multihop=f"{sum(s['evidence_complete'] for s in multi)}/{len(multi)}",
                         mean_evidence_recall=round(sum(s["evidence_recall"] for s in sc) / len(sc), 2),
                         median_context_chars=int(st.median(s["context_chars"] for s in sc))))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true", help="also run graph+rules and vector+graph")
    ap.add_argument("--k-sweep", action="store_true", help="vector-RAG control: vary top-k to match context size")
    a = ap.parse_args()
    if a.k_sweep:
        sweep = k_sweep()
        sweep.to_csv(EVAL_DIR / "vector_k_sweep.csv", index=False)
        print(to_markdown(sweep))
        return
    modes = ["vector", "graph", "hybrid"] + (["graph+rules", "vector+graph"] if a.ablation else [])
    df, backends = run(modes)
    df.to_csv(EVAL_DIR / "results.csv", index=False)
    print(write_report(df, backends, modes))
    print(f"Wrote {EVAL_DIR / 'results.csv'} and {EVAL_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
