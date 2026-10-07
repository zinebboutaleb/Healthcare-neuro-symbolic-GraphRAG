"""Pydantic data models shared across the pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Entity(BaseModel):
    id: str = ""
    name: str
    type: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)


class Edge(BaseModel):
    src: str
    src_type: str
    rel: str
    dst: str
    dst_type: str
    note: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.rel, self.dst)

    def __str__(self) -> str:
        return f"{self.src} -[{self.rel}]-> {self.dst}"


class GraphPath(BaseModel):
    """A traversal path. `forward[i]` is False when edge i was walked against its direction."""

    nodes: list[str]
    edges: list[Edge]
    forward: list[bool]
    score: float = 0.0

    @property
    def hops(self) -> int:
        return len(self.edges)

    def render(self) -> str:
        out = self.nodes[0]
        for edge, fwd in zip(self.edges, self.forward):
            out += f" -[{edge.rel}]-> {edge.dst}" if fwd else f" <-[{edge.rel}]- {edge.src}"
        return out


class DocHit(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


class Inference(BaseModel):
    """A derived conclusion with full provenance: source facts -> rule -> conclusion."""

    rule_id: str
    rule_name: str
    premises: list[str]
    conclusion: str
    triples: list[Edge] = Field(default_factory=list)


class GraphResult(BaseModel):
    seeds: list[Entity] = Field(default_factory=list)
    paths: list[GraphPath] = Field(default_factory=list)
    triples: list[Edge] = Field(default_factory=list)
    node_notes: dict[str, str] = Field(default_factory=dict)
    max_hops: int = 0


class RAGResponse(BaseModel):
    question: str
    mode: str
    answer: str
    context: str
    linked_entities: list[Entity] = Field(default_factory=list)
    doc_hits: list[DocHit] = Field(default_factory=list)
    graph: GraphResult | None = None
    inferences: list[Inference] = Field(default_factory=list)
    latency_retrieval_ms: float = 0.0
    latency_generation_ms: float = 0.0
    backends: dict[str, str] = Field(default_factory=dict)
