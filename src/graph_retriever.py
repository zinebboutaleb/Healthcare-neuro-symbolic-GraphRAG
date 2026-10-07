"""GraphRAG retrieval: question -> linked seed entities -> multi-hop traversal -> ranked paths -> text.

Design choices (kept deliberately simple and inspectable):
  * Entity linking = longest-match lookup of entity names/aliases (no ML, fully deterministic).
  * Traversal = bounded DFS from every seed. It follows edge direction; the first hop may also go
    backwards (so a *symptom* question can start at a disease), and ASSOCIATED_WITH / ALTERNATIVE_TO
    may be walked backwards at any hop (ontology.INVERSE_TRAVERSAL).
  * Ranking = paths that connect several question entities score highest, plus a bonus for edges whose
    relation matches the intent expressed in the question ("side effect" -> HAS_SIDE_EFFECT, ...).
  * The paths themselves are preserved and are the explanation shown to the user.
"""
from __future__ import annotations

import re

from src.graph_store import GraphStore
from src.models import Edge, Entity, GraphPath, GraphResult
from src.ontology import INVERSE_TRAVERSAL, LINKABLE_TYPES

# keyword in the question -> relation it hints at
INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "HAS_SYMPTOM": ("symptom", "sign", "reporting", "reports"),
    "HAS_RISK_FACTOR": ("risk",),
    "TREATED_BY": ("treat", "therap", "medication", "drug", "manage", "option", "used for"),
    "CONTRAINDICATED_BY": ("contraindic", "inappropriate", "avoid", "not be recommended", "should not", "problematic", "unsafe"),
    "HAS_SIDE_EFFECT": ("side effect", "adverse", "can cause"),
    "DIAGNOSED_BY": ("diagnos", "test", "assess"),
    "REQUIRES": ("monitor", "guideline", "require"),
    "ALTERNATIVE_TO": ("alternative", "instead", "substitute"),
    "AFFECTS": ("affect", "concern", "additional review"),
    "ASSOCIATED_WITH": ("comorbid",),
}


class EntityLinker:
    """Longest-match, word-boundary, case-insensitive linking against names and aliases."""

    def __init__(self, store: GraphStore):
        self.entities = [e for e in store.all_entities() if e.type in LINKABLE_TYPES]
        pairs = []
        for e in self.entities:
            for alias in {e.name.lower(), *[a.lower() for a in e.aliases]}:
                pairs.append((alias, e))
        pairs.sort(key=lambda p: (-len(p[0]), p[0]))
        self._pairs = [(re.compile(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])"), e) for a, e in pairs]

    def link(self, text: str) -> list[Entity]:
        low = text.lower()
        taken: list[tuple[int, int]] = []
        found: list[tuple[int, Entity]] = []
        for pat, ent in self._pairs:
            for m in pat.finditer(low):
                s, e = m.span()
                if any(s < te and e > ts for ts, te in taken):
                    continue  # overlaps with a longer, already-linked mention
                taken.append((s, e))
                found.append((s, ent))
        found.sort(key=lambda x: x[0])
        seen, out = set(), []
        for _, ent in found:
            if ent.name not in seen:
                seen.add(ent.name)
                out.append(ent)
        return out


def detect_intents(question: str) -> set[str]:
    q = question.lower()
    return {rel for rel, kws in INTENT_KEYWORDS.items() if any(k in q for k in kws)}


class GraphRetriever:
    def __init__(self, store: GraphStore, linker: EntityLinker | None = None, max_hops: int = 4, max_paths: int = 20):
        self.store = store
        self.linker = linker or EntityLinker(store)
        self.max_hops = max_hops
        self.max_paths = max_paths

    # ---- traversal ------------------------------------------------------------------------------
    def _neighbors(self, node: str, depth: int) -> list[tuple[Edge, bool]]:
        out = [(e, True) for e in self.store.out_edges(node)]
        back = [(e, False) for e in self.store.in_edges(node) if depth == 0 or e.rel in INVERSE_TRAVERSAL]
        return out + back

    def _expand(self, seed: str) -> list[GraphPath]:
        paths: list[GraphPath] = []

        def dfs(node: str, nodes: list[str], edges: list[Edge], fwd: list[bool]) -> None:
            if len(edges) >= self.max_hops:
                return
            for edge, forward in self._neighbors(node, len(edges)):
                nxt = edge.dst if forward else edge.src
                if nxt in nodes:  # no cycles
                    continue
                n2, e2, f2 = nodes + [nxt], edges + [edge], fwd + [forward]
                paths.append(GraphPath(nodes=n2, edges=e2, forward=f2))
                dfs(nxt, n2, e2, f2)

        dfs(seed, [seed], [], [])
        return paths

    # ---- ranking --------------------------------------------------------------------------------
    @staticmethod
    def _score(path: GraphPath, seeds: set[str], intents: set[str]) -> float:
        extra_seeds = sum(1 for n in path.nodes[1:] if n in seeds)
        intent_hits = sum(1 for e in path.edges if e.rel in intents)
        return 3.0 * extra_seeds + 1.0 * intent_hits - 0.4 * path.hops

    @staticmethod
    def _keep(path: GraphPath, seeds: set[str], intents: set[str]) -> bool:
        """Tail trimming: a multi-hop path must end on another question entity or on an edge whose
        relation matches the question intent; otherwise it is an uninformative extension."""
        if path.hops == 1 or not intents:
            return True
        return path.nodes[-1] in seeds or path.edges[-1].rel in intents

    def retrieve(self, question: str, seeds: list[Entity] | None = None) -> GraphResult:
        seeds = seeds if seeds is not None else self.linker.link(question)
        if not seeds:
            return GraphResult()
        seed_names = {s.name for s in seeds}
        intents = detect_intents(question)

        candidates: dict[tuple, GraphPath] = {}
        for s in seeds:
            for p in self._expand(s.name):
                if not self._keep(p, seed_names, intents):
                    continue
                p.score = self._score(p, seed_names, intents)
                candidates[p.render()] = p
        ranked = sorted(candidates.values(), key=lambda p: (-p.score, -p.hops, p.render()))

        chosen: list[GraphPath] = []
        covered: set[tuple] = set()
        for p in ranked:  # greedy: keep a path only if it adds at least one new edge
            keys = {e.key for e in p.edges}
            if keys <= covered:
                continue
            chosen.append(p)
            covered |= keys
            if len(chosen) >= self.max_paths:
                break
        # drop paths that are strict prefixes of another chosen path
        rendered = [p.render() for p in chosen]
        chosen = [p for p, r in zip(chosen, rendered) if not any(o != r and o.startswith(r) for o in rendered)]

        triples: dict[tuple, Edge] = {}
        for p in chosen:
            for e in p.edges:
                triples[e.key] = e
        notes: dict[str, str] = {}
        for name in {n for p in chosen for n in p.nodes} | seed_names:
            node = self.store.get_node(name)
            if node and node.description and (name in seed_names or node.type in ("Contraindication", "ClinicalGuideline")):
                notes[name] = f"({node.type}) {node.description}"
        return GraphResult(seeds=seeds, paths=chosen, triples=list(triples.values()), node_notes=notes,
                           max_hops=max((p.hops for p in chosen), default=0))


def format_graph_context(result: GraphResult) -> str:
    """Turn the graph neighbourhood into text, preserving every traversed path."""
    if not result.seeds:
        return "No knowledge-graph entities could be linked to the question."
    lines = ["Linked entities: " + "; ".join(f"{s.name} ({s.type})" for s in result.seeds), "Paths:"]
    lines += [f"[G{i}] {p.render()}" for i, p in enumerate(result.paths, 1)]
    notes = [f"- {e.src} -[{e.rel}]-> {e.dst}: {e.note}" for e in result.triples if e.note]
    if notes:
        lines += ["Edge notes:"] + notes
    if result.node_notes:
        lines += ["Entity descriptions:"] + [f"- {n} {d}" for n, d in sorted(result.node_notes.items())]
    return "\n".join(lines)
