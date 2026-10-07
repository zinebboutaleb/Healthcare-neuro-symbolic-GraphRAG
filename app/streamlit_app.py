"""Minimal demo UI.   streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from evaluation.benchmark import load_benchmark
from src.pipeline import MODES, Pipeline
from src.symbolic_rules import RULE_DESCRIPTIONS

st.set_page_config(page_title="Neuro-symbolic GraphRAG", layout="wide")
st.title("Healthcare Knowledge Graph - GraphRAG demo")
st.warning("Educational research prototype over a SYNTHETIC knowledge base. Not medical advice; not for clinical use.")


@st.cache_resource(show_spinner="Loading graph, embeddings and index...")
def get_pipeline() -> Pipeline:
    return Pipeline()


pipe = get_pipeline()
with st.sidebar:
    st.header("Retrieval mode")
    mode = st.radio("Mode", ["vector", "graph", "hybrid"],
                    format_func={"vector": "Vector RAG", "graph": "GraphRAG", "hybrid": "Hybrid (graph + vector + rules)"}.get)
    st.caption(f"Graph backend: **{pipe.backends['graph_backend']}**  \nEmbeddings: **{pipe.backends['embedding_backend']}**  \nLLM: **{pipe.backends['llm']}**")
    examples = {f"{q.id}: {q.question}": q.question for q in load_benchmark()}
    pick = st.selectbox("Benchmark questions", ["(write your own)"] + list(examples))

question = st.text_input("Question", value=examples.get(pick, "A pregnant patient has Hypertension. Which medication should not be recommended?"))
if st.button("Ask", type="primary") and question.strip():
    resp = pipe.answer(question, mode)
    st.subheader("Answer")
    if pipe.llm.provider == "offline":
        st.info("No LLM configured (offline mode): the answer below is the retrieved context itself. Set a provider in `.env` to generate answers.")
    st.write(resp.answer if pipe.llm.provider != "offline" else "_see the context tab_")
    st.caption(f"retrieval {resp.latency_retrieval_ms:.1f} ms | generation {resp.latency_generation_ms:.1f} ms")

    t_docs, t_ent, t_rel, t_path, t_rules, t_ctx = st.tabs(
        ["Retrieved documents", "Graph entities", "Graph relationships", "Reasoning paths", "Symbolic rules", "Context sent to LLM"])
    with t_docs:
        if not resp.doc_hits:
            st.write("This mode does not use vector retrieval.")
        for h in resp.doc_hits:
            st.markdown(f"**[D{h.rank}] {h.title}** (`{h.doc_id}`, score {h.score:.3f})  \n{h.text}")
    g = resp.graph
    with t_ent:
        if g:
            st.dataframe(pd.DataFrame([e.model_dump(include={"name", "type", "description"}) for e in g.seeds]), width="stretch")
        else:
            st.write("This mode does not use the knowledge graph.")
    with t_rel:
        if g:
            st.dataframe(pd.DataFrame([{"source": e.src, "relation": e.rel, "target": e.dst, "note": e.note} for e in g.triples]), width="stretch")
    with t_path:
        if g and g.paths:
            st.caption(f"{len(g.paths)} paths, longest = {g.max_hops} hops")
            for i, p in enumerate(g.paths, 1):
                st.code(f"[G{i}] {p.render()}", language=None)
            dot = ["digraph G { rankdir=LR; node [shape=box, fontsize=10];"]
            for e in g.triples:
                dot.append(f'"{e.src}" -> "{e.dst}" [label="{e.rel}", fontsize=8];')
            st.graphviz_chart("\n".join(dot) + "}")
    with t_rules:
        if not resp.inferences:
            st.write("No symbolic rules were applied or none fired." if mode == "hybrid" else "This mode does not use symbolic rules.")
        for inf in resp.inferences:
            with st.expander(f"{inf.rule_id} - {inf.conclusion}", expanded=inf.rule_id in ("RULE_01", "RULE_03", "RULE_04")):
                st.markdown("**Source facts**")
                for p in inf.premises:
                    st.markdown(f"- {p}")
                st.markdown(f"**Rule:** {RULE_DESCRIPTIONS[inf.rule_id]}")
                st.markdown(f"**Conclusion:** {inf.conclusion}")
    with t_ctx:
        st.code(resp.context, language=None)
