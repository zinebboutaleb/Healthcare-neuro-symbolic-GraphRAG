"""Graph access layer.

All retrieval / reasoning code talks to a `GraphStore` (4 methods), so the same logic runs on:
  * Neo4jStore     (src/neo4j_client.py)  - the primary, Cypher-backed store
  * NetworkXStore  (this file)            - in-memory, built from the CSVs; used for tests and as a
                                            fallback when no Neo4j instance is reachable.
"""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from src.config import DATA_DIR, env
from src.models import Edge, Entity
from src.ontology import validate_triple

log = logging.getLogger(__name__)


def load_kb(data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir or DATA_DIR)
    ents = pd.read_csv(d / "entities.csv", dtype=str).fillna("")
    rels = pd.read_csv(d / "relationships.csv", dtype=str).fillna("")
    return ents, rels


def row_to_entity(r: dict) -> Entity:
    aliases = r.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [a for a in aliases.split("|") if a]
    return Entity(id=r.get("id") or "", name=r["name"], type=r["type"],
                  description=r.get("description") or "", aliases=list(aliases))


class GraphStore:
    backend = "abstract"

    def get_node(self, name: str) -> Entity | None:
        raise NotImplementedError

    def out_edges(self, name: str) -> list[Edge]:
        raise NotImplementedError

    def in_edges(self, name: str) -> list[Edge]:
        raise NotImplementedError

    def nodes_of_type(self, type_: str) -> list[Entity]:
        raise NotImplementedError

    def all_entities(self) -> list[Entity]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def has_edge(self, src: str, rel: str, dst: str) -> Edge | None:
        for e in self.out_edges(src):
            if e.rel == rel and e.dst == dst:
                return e
        return None


class NetworkXStore(GraphStore):
    backend = "networkx"

    def __init__(self, data_dir: Path | str | None = None):
        ents, rels = load_kb(data_dir)
        self.G = nx.MultiDiGraph()
        self._ents: dict[str, Entity] = {}
        for r in ents.to_dict("records"):
            e = row_to_entity(r)
            self._ents[e.name] = e
            self.G.add_node(e.name, type=e.type)
        for r in rels.to_dict("records"):
            validate_triple(r["source_type"], r["relation"], r["target_type"])
            self.G.add_edge(r["source"], r["target"], key=r["relation"], rel=r["relation"], note=r["note"])

    def _edge(self, u: str, v: str, d: dict) -> Edge:
        return Edge(src=u, src_type=self._ents[u].type, rel=d["rel"], dst=v, dst_type=self._ents[v].type, note=d.get("note", ""))

    def get_node(self, name):
        return self._ents.get(name)

    def out_edges(self, name):
        if name not in self.G:
            return []
        return sorted((self._edge(u, v, d) for u, v, d in self.G.out_edges(name, data=True)), key=lambda e: (e.rel, e.dst))

    def in_edges(self, name):
        if name not in self.G:
            return []
        return sorted((self._edge(u, v, d) for u, v, d in self.G.in_edges(name, data=True)), key=lambda e: (e.rel, e.src))

    def nodes_of_type(self, type_):
        return sorted((e for e in self._ents.values() if e.type == type_), key=lambda e: e.name)

    def all_entities(self):
        return list(self._ents.values())


def get_store(backend: str | None = None) -> GraphStore:
    """GRAPH_BACKEND = neo4j | networkx | auto (default: try Neo4j, fall back to NetworkX)."""
    backend = (backend or env("GRAPH_BACKEND", "auto")).lower()
    if backend in ("neo4j", "auto"):
        try:
            from src.neo4j_client import Neo4jStore

            store = Neo4jStore.connect()
            if not store.all_entities():
                raise RuntimeError("Neo4j is reachable but empty - run `python -m src.graph_builder`")
            return store
        except Exception as exc:  # noqa: BLE001
            if backend == "neo4j":
                raise
            log.warning("Neo4j unavailable (%s) - falling back to in-memory NetworkX graph.", exc)
    return NetworkXStore()
