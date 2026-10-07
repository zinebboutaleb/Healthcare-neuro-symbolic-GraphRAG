"""Offline, deterministic tests (NetworkX graph + hashing embedder + offline LLM).

    pytest -q
The Neo4j parity test is skipped automatically when no Neo4j instance is reachable.
"""
import os

os.environ["GRAPH_BACKEND"] = "networkx"
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["LLM_PROVIDER"] = "offline"

import pytest

from evaluation.benchmark import Scorer, load_benchmark
from src.graph_retriever import EntityLinker, GraphRetriever
from src.graph_store import NetworkXStore, load_kb
from src.hybrid_retriever import DOC_HEADER, GRAPH_HEADER, SYM_HEADER
from src.ontology import SCHEMA, validate_triple
from src.pipeline import Pipeline
from src.symbolic_rules import run_rules


@pytest.fixture(scope="module")
def store():
    return NetworkXStore()


@pytest.fixture(scope="module")
def pipe():
    return Pipeline()


# ---------------------------------------------------------------- ontology / data
def test_every_relationship_respects_the_ontology():
    _, rels = load_kb()
    for r in rels.to_dict("records"):
        validate_triple(r["source_type"], r["relation"], r["target_type"])


def test_ontology_rejects_invalid_triples():
    with pytest.raises(ValueError):
        validate_triple("Symptom", "TREATED_BY", "Disease")
    with pytest.raises(ValueError):
        validate_triple("Disease", "MADE_UP_REL", "Symptom")


def test_dataset_size_matches_spec():
    ents, rels = load_kb()
    counts = ents.type.value_counts().to_dict()
    assert counts["Disease"] == 10 and counts["Symptom"] >= 20 and counts["Medication"] >= 10
    assert set(rels.relation) == set(SCHEMA)  # every ontology relation is exercised


# ---------------------------------------------------------------- linking & graph retrieval
def test_entity_linking_aliases_and_longest_match(store):
    linker = EntityLinker(store)
    names = [e.name for e in linker.link("A pregnant patient with high blood pressure and low blood pressure")]
    assert names == ["Pregnancy", "Hypertension", "Low Blood Pressure"]  # 'blood pressure' alone is not an entity


def test_graph_retrieval_preserves_multihop_path(store):
    res = GraphRetriever(store).retrieve("Which treatment for Hypertension is contraindicated in a patient with Low Blood Pressure?")
    rendered = [p.render() for p in res.paths]
    assert "Hypertension -[TREATED_BY]-> Diurex -[CONTRAINDICATED_BY]-> Low Blood Pressure" in rendered


def test_graph_retrieval_finds_four_hop_path(store):
    q = "A patient reports Increased Thirst and Frequent Urination. Which medication has a side effect that affects Elderly Frailty?"
    res = GraphRetriever(store).retrieve(q)
    assert res.max_hops == 4
    assert any(p.hops == 4 and p.nodes[-1] == "Elderly Frailty" and "Sulfaglide" in p.nodes for p in res.paths)


# ---------------------------------------------------------------- symbolic rules
def test_rule01_is_traceable_to_source_facts(store):
    ents = EntityLinker(store).link("A pregnant patient has Hypertension.")
    r1 = [i for i in run_rules(store, ents) if i.rule_id == "RULE_01"]
    assert [i.conclusion.split()[0] for i in r1] == ["Cardivex"]
    joined = " ".join(r1[0].premises)
    assert "Hypertension -[TREATED_BY]-> Cardivex" in joined and "Cardivex -[CONTRAINDICATED_BY]-> Pregnancy" in joined


def test_rule02_ranks_candidate_disease_by_symptom_matches(store):
    ents = EntityLinker(store).link("Patient reports Shortness of Breath and Swelling in Legs")
    r2 = [i for i in run_rules(store, ents) if i.rule_id == "RULE_02"]
    assert r2[0].conclusion.startswith("Heart Failure")  # 2/2 matches, listed before 1/2 candidates (none shown)
    assert len(r2) == 1  # Asthma and CKD match only one symptom -> not candidates


def test_rule03_04_renal_scenario(store):
    ents = EntityLinker(store).link("A patient with Severe Renal Impairment has Heart Failure.")
    out = {(i.rule_id, i.conclusion.split()[0]) for i in run_rules(store, ents)}
    assert ("RULE_01", "Cardivex") in out       # blocked
    assert ("RULE_03", "Diurex") in out         # Dehydration AFFECTS Severe Renal Impairment -> review
    assert ("RULE_04", "Diurex") in out         # alternative to the blocked drug


def test_rules_are_deterministic(store):
    ents = EntityLinker(store).link("A pregnant patient has Hypertension.")
    assert [i.model_dump() for i in run_rules(store, ents)] == [i.model_dump() for i in run_rules(store, ents)]


# ---------------------------------------------------------------- vector store & pipeline
def test_vector_store_returns_ranked_chunks(pipe):
    hits = pipe.vectors.search("Which test is used for Hypothyroidism?", 4)
    assert len(hits) == 4 and hits[0].score >= hits[-1].score
    assert all(h.doc_id and h.text for h in hits)


def test_hybrid_context_has_three_separated_sections(pipe):
    resp = pipe.answer("A pregnant patient has Hypertension. Which medication should not be recommended?", "hybrid")
    assert resp.context.index(DOC_HEADER) < resp.context.index(GRAPH_HEADER) < resp.context.index(SYM_HEADER)
    assert resp.graph.paths and resp.inferences and resp.doc_hits


def test_modes_only_use_their_own_evidence(pipe):
    q = "Which diagnostic test is used for Hypothyroidism?"
    v, g = pipe.answer(q, "vector"), pipe.answer(q, "graph")
    assert v.graph is None and not v.inferences and GRAPH_HEADER not in v.context
    assert not g.doc_hits and DOC_HEADER not in g.context and SYM_HEADER not in g.context


# ---------------------------------------------------------------- benchmark integrity
def test_benchmark_gold_answers_are_consistent_with_the_kb(store):
    """Guards against typos in hand-written ground truth: every gold entity exists, and the
    single-hop gold answers can be re-derived directly from the graph."""
    qs = {q.id: q for q in load_benchmark()}
    names = {e.name for e in store.all_entities()}
    for q in qs.values():
        assert set(q.expected) <= names and set(q.evidence) <= names, q.id
        assert set(q.expected) <= set(q.evidence) | {"Heart Failure"}, q.id
    out = lambda n, rel: {e.dst for e in store.out_edges(n) if e.rel == rel}
    assert set(qs["Q01"].expected) == out("Asthma", "HAS_SYMPTOM")
    assert set(qs["Q02"].expected) == out("Type 2 Diabetes", "TREATED_BY")
    assert set(qs["Q03"].expected) == out("Hypothyroidism", "DIAGNOSED_BY")
    assert set(qs["Q04"].expected) == out("Migraine", "HAS_RISK_FACTOR")
    assert set(qs["Q05"].expected) == out("Respirol", "CONTRAINDICATED_BY")
    # multi-hop gold answers via explicit graph queries
    treats = lambda d: out(d, "TREATED_BY")
    assert {m for m in treats("Hypertension") if store.has_edge(m, "CONTRAINDICATED_BY", "Low Blood Pressure")} == set(qs["Q06"].expected)
    assert {m for m in treats("Hypertension") if store.has_edge(m, "CONTRAINDICATED_BY", "Pregnancy")} == set(qs["Q07"].expected)
    assert {m for m in treats("Hypertension") if store.has_edge(m, "HAS_SIDE_EFFECT", "Dry Cough")
            and store.has_edge(m, "CONTRAINDICATED_BY", "Pregnancy")} == set(qs["Q08"].expected)
    assert {m for m in treats("Type 2 Diabetes") if store.has_edge(m, "HAS_SIDE_EFFECT", "Hypoglycemia")} == {"Sulfaglide"}
    assert out("Sulfaglide", "CONTRAINDICATED_BY") == {"Recurrent Hypoglycemia History", "Elderly Frailty"}
    assert {m for m in treats("Osteoarthritis") if store.has_edge(m, "HAS_SIDE_EFFECT", "Stomach Irritation")} == {"Arthrocalm"}


def test_scorer_counts_unsupported_entities(pipe):
    scorer = Scorer(pipe.store)
    q = next(q for q in load_benchmark() if q.id == "Q03")
    resp = pipe.answer(q.question, "graph")
    resp.answer = "TSH Blood Test. Also consider Spirometry."  # Spirometry is not in the retrieved context
    s = scorer.score(q, resp)
    assert s["answer_correct"] and s["unsupported_entities"] == "Spirometry"


# ---------------------------------------------------------------- Neo4j parity (skipped without a server)
def test_neo4j_matches_networkx(store):
    try:
        from src.neo4j_client import Neo4jStore

        neo = Neo4jStore.connect()
        if not neo.all_entities():
            pytest.skip("Neo4j reachable but empty - run `python -m src.graph_builder`")
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {exc}")
    for ent in store.all_entities():
        assert sorted(e.key for e in neo.out_edges(ent.name)) == sorted(e.key for e in store.out_edges(ent.name)), ent.name
        assert sorted(e.key for e in neo.in_edges(ent.name)) == sorted(e.key for e in store.in_edges(ent.name)), ent.name
