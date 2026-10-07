"""Thin Neo4j wrapper + a Cypher-backed GraphStore. All values are passed as parameters."""
from __future__ import annotations

from src.config import env
from src.graph_store import GraphStore, row_to_entity
from src.models import Edge, Entity
from src.ontology import ENTITY_TYPES


class Neo4jClient:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        from neo4j import GraphDatabase  # imported lazily so the package is optional for offline use

        self.driver = GraphDatabase.driver(
            uri or env("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or env("NEO4J_USER", "neo4j"), password or env("NEO4J_PASSWORD", "healthcare123")),
        )

    def verify(self) -> None:
        self.driver.verify_connectivity()

    def run(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            return [r.data() for r in session.run(query, **params)]

    def close(self) -> None:
        self.driver.close()


_OUT = """
MATCH (a {name: $name})-[r]->(b)
RETURN a.name AS src, labels(a)[0] AS src_type, type(r) AS rel,
       b.name AS dst, labels(b)[0] AS dst_type, coalesce(r.note, '') AS note
ORDER BY rel, dst
"""
_IN = """
MATCH (a)-[r]->(b {name: $name})
RETURN a.name AS src, labels(a)[0] AS src_type, type(r) AS rel,
       b.name AS dst, labels(b)[0] AS dst_type, coalesce(r.note, '') AS note
ORDER BY rel, src
"""
_NODE_RETURN = "n.id AS id, n.name AS name, labels(n)[0] AS type, n.description AS description, n.aliases AS aliases"


class Neo4jStore(GraphStore):
    backend = "neo4j"

    def __init__(self, client: Neo4jClient):
        self.client = client
        self._cache_out: dict[str, list[Edge]] = {}
        self._cache_in: dict[str, list[Edge]] = {}

    @classmethod
    def connect(cls) -> "Neo4jStore":
        client = Neo4jClient()
        client.verify()
        return cls(client)

    def get_node(self, name):
        rows = self.client.run(f"MATCH (n {{name: $name}}) RETURN {_NODE_RETURN}", name=name)
        return row_to_entity(rows[0]) if rows else None

    def out_edges(self, name):
        if name not in self._cache_out:
            self._cache_out[name] = [Edge(**r) for r in self.client.run(_OUT, name=name)]
        return self._cache_out[name]

    def in_edges(self, name):
        if name not in self._cache_in:
            self._cache_in[name] = [Edge(**r) for r in self.client.run(_IN, name=name)]
        return self._cache_in[name]

    def nodes_of_type(self, type_):
        assert type_ in ENTITY_TYPES  # label cannot be a query parameter -> whitelist it
        return [row_to_entity(r) for r in self.client.run(f"MATCH (n:{type_}) RETURN {_NODE_RETURN} ORDER BY n.name")]

    def all_entities(self):
        return [row_to_entity(r) for r in self.client.run(f"MATCH (n) RETURN {_NODE_RETURN}")]

    def close(self):
        self.client.close()
