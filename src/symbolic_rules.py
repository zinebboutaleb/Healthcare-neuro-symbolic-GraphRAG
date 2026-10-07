"""Deterministic, explainable symbolic rules over the knowledge graph.

Every rule is a plain Python function  (store, ctx) -> list[Inference].  An Inference records:
    premises   - the source facts (graph edges / statements taken from the question)
    rule_id    - which rule fired
    conclusion - what was derived
No randomness, no learning: the same question + KG always yields the same conclusions.

Scenario assumptions (stated openly because they are modelling choices):
  * Entities mentioned in the question are the "patient scenario". PatientConditions mentioned = conditions
    the (synthetic) patient has. Symptoms / RiskFactors mentioned = observed. Diseases mentioned = diseases
    under consideration.
  * Candidate medications = medications mentioned + medications that TREAT a mentioned disease.
  * Closed-world assumption for rule R04: a missing CONTRAINDICATED_BY edge means "not contraindicated
    in this KB" (negation as failure).
"""
from __future__ import annotations

from collections import defaultdict

from src.graph_store import GraphStore
from src.models import Edge, Entity, Inference

RULE_DESCRIPTIONS = {
    "RULE_01": "IF patient has condition C AND medication M is a candidate AND M -CONTRAINDICATED_BY-> C THEN M should not be recommended.",
    "RULE_02": "IF >=2 reported symptoms AND disease D -HAS_SYMPTOM-> each of them THEN D is a candidate explanation (ranked by matches).",
    "RULE_03": "IF candidate M -HAS_SIDE_EFFECT-> S AND S -AFFECTS-> condition C of the patient AND M is not already blocked for C THEN flag M for additional review.",
    "RULE_04": "IF M is blocked by RULE_01 AND M' -ALTERNATIVE_TO- M AND M' treats the same disease AND M' has no CONTRAINDICATED_BY edge to the patient's conditions THEN M' is an alternative option.",
    "RULE_05": "IF M (candidate, not blocked) or disease D is in scope AND guideline G -ASSOCIATED_WITH-> M/D AND G -REQUIRES-> test T THEN T is required for monitoring.",
    "RULE_06": "IF patient has risk factor R AND disease D -HAS_RISK_FACTOR-> R THEN D is a higher-risk consideration.",
    "RULE_07": "IF disease D1 is in scope AND D1 -ASSOCIATED_WITH- D2 THEN D2 should be considered (comorbidity awareness).",
}


class Ctx:
    """Patient scenario derived from the entities linked in the question."""

    def __init__(self, store: GraphStore, entities: list[Entity]):
        self.store = store
        by_type: dict[str, list[str]] = defaultdict(list)
        for e in entities:
            by_type[e.type].append(e.name)
        self.diseases = by_type["Disease"]
        self.symptoms = by_type["Symptom"]
        self.risk_factors = by_type["RiskFactor"]
        self.conditions = by_type["PatientCondition"]
        self.mentioned_meds = by_type["Medication"]
        self._candidates: dict[str, list[Edge]] | None = None
        self._blocked: dict[tuple[str, str], Edge] | None = None

    def candidate_meds(self) -> dict[str, list[Edge]]:
        """medication -> edges explaining why it is a candidate ([] when mentioned explicitly)."""
        if self._candidates is None:
            c: dict[str, list[Edge]] = {m: [] for m in self.mentioned_meds}
            for d in self.diseases:
                for e in self.store.out_edges(d):
                    if e.rel == "TREATED_BY" and e.dst_type == "Medication":
                        c.setdefault(e.dst, []).append(e)
            self._candidates = dict(sorted(c.items()))
        return self._candidates

    def blocked(self) -> dict[tuple[str, str], Edge]:
        """(medication, condition) -> CONTRAINDICATED_BY edge, for all candidates vs. patient conditions."""
        if self._blocked is None:
            self._blocked = {}
            for m in self.candidate_meds():
                for c in self.conditions:
                    edge = self.store.has_edge(m, "CONTRAINDICATED_BY", c)
                    if edge:
                        self._blocked[(m, c)] = edge
        return self._blocked


def fact(edge: Edge) -> str:
    return f"{edge}" + (f"  [note: {edge.note}]" if edge.note else "")


def rule_01(store: GraphStore, ctx: Ctx) -> list[Inference]:
    out = []
    for (med, cond), edge in sorted(ctx.blocked().items()):
        premises = [f"Scenario: patient has {cond}"]
        premises += [f"KG fact: {e}" for e in ctx.candidate_meds()[med]] or [f"Scenario: {med} is under consideration"]
        premises.append(f"KG fact: {fact(edge)}")
        out.append(Inference(rule_id="RULE_01", rule_name="Contraindication block", premises=premises,
                             conclusion=f"{med} should NOT be recommended for a patient with {cond} (synthetic scenario).",
                             triples=ctx.candidate_meds()[med] + [edge]))
    return out


def rule_02(store: GraphStore, ctx: Ctx) -> list[Inference]:
    if len(ctx.symptoms) < 2:
        return []
    scored = []
    for d in store.nodes_of_type("Disease"):
        edges = [e for s in ctx.symptoms if (e := store.has_edge(d.name, "HAS_SYMPTOM", s))]
        if len(edges) >= 2:
            scored.append((-len(edges), d.name, edges))
    out = []
    for neg, name, edges in sorted(scored):
        out.append(Inference(rule_id="RULE_02", rule_name="Symptom-based candidate disease",
                             premises=[f"Scenario: patient reports {', '.join(ctx.symptoms)}"] + [f"KG fact: {e}" for e in edges],
                             conclusion=f"{name} is a candidate explanation ({-neg}/{len(ctx.symptoms)} reported symptoms match). Not a diagnosis.",
                             triples=edges))
    return out


def rule_03(store: GraphStore, ctx: Ctx) -> list[Inference]:
    out = []
    for med in ctx.candidate_meds():
        for se in [e for e in store.out_edges(med) if e.rel == "HAS_SIDE_EFFECT"]:
            for cond in ctx.conditions:
                aff = store.has_edge(se.dst, "AFFECTS", cond)
                if aff and (med, cond) not in ctx.blocked():
                    out.append(Inference(rule_id="RULE_03", rule_name="Side-effect review flag",
                                         premises=[f"Scenario: patient has {cond}", f"KG fact: {se}", f"KG fact: {aff}"],
                                         conclusion=f"{med} should be flagged for additional review for a patient with {cond} ({se.dst} is of particular concern).",
                                         triples=[se, aff]))
    return out


def rule_04(store: GraphStore, ctx: Ctx) -> list[Inference]:
    out, seen = [], set()
    for (med, cond), block_edge in sorted(ctx.blocked().items()):
        alts = [(e.dst, e) for e in store.out_edges(med) if e.rel == "ALTERNATIVE_TO"]
        alts += [(e.src, e) for e in store.in_edges(med) if e.rel == "ALTERNATIVE_TO"]
        for alt, alt_edge in sorted(alts, key=lambda x: x[0]):
            if ctx.diseases and alt not in ctx.candidate_meds():
                continue  # must treat a disease under consideration
            if any(store.has_edge(alt, "CONTRAINDICATED_BY", c) for c in ctx.conditions):
                continue
            if (med, alt) in seen:
                continue
            seen.add((med, alt))
            out.append(Inference(rule_id="RULE_04", rule_name="Alternative option",
                                 premises=[f"Derived: RULE_01 blocks {med} for {cond}", f"KG fact: {alt_edge}",
                                           f"Closed-world check: no CONTRAINDICATED_BY edge from {alt} to {', '.join(ctx.conditions)}"],
                                 conclusion=f"{alt} is an alternative to {med} with no recorded contraindication for the stated conditions.",
                                 triples=[block_edge, alt_edge]))
    return out


def rule_05(store: GraphStore, ctx: Ctx) -> list[Inference]:
    targets = [m for m in ctx.candidate_meds() if not any(m == b[0] for b in ctx.blocked())] + list(ctx.diseases)
    out = []
    for t in targets:
        for g in [e for e in store.in_edges(t) if e.rel == "ASSOCIATED_WITH" and e.src_type == "ClinicalGuideline"]:
            for req in [e for e in store.out_edges(g.src) if e.rel == "REQUIRES"]:
                out.append(Inference(rule_id="RULE_05", rule_name="Guideline monitoring requirement",
                                     premises=[f"KG fact: {g}", f"KG fact: {req}"],
                                     conclusion=f"{req.dst} is required for monitoring in connection with {t} (guideline: {g.src}).",
                                     triples=[g, req]))
    return out


def rule_06(store: GraphStore, ctx: Ctx) -> list[Inference]:
    out = []
    diseases = ctx.diseases or [d.name for d in store.nodes_of_type("Disease")]
    for rf in ctx.risk_factors:
        for d in diseases:
            edge = store.has_edge(d, "HAS_RISK_FACTOR", rf)
            if edge:
                out.append(Inference(rule_id="RULE_06", rule_name="Risk-factor match",
                                     premises=[f"Scenario: patient has risk factor {rf}", f"KG fact: {edge}"],
                                     conclusion=f"{d} is a higher-risk consideration for a patient with {rf}.", triples=[edge]))
    return out


def rule_07(store: GraphStore, ctx: Ctx) -> list[Inference]:
    out = []
    for d in ctx.diseases:
        pairs = [(e.dst, e) for e in store.out_edges(d) if e.rel == "ASSOCIATED_WITH" and e.dst_type == "Disease"]
        pairs += [(e.src, e) for e in store.in_edges(d) if e.rel == "ASSOCIATED_WITH" and e.src_type == "Disease"]
        for other, edge in sorted(pairs, key=lambda x: x[0]):
            out.append(Inference(rule_id="RULE_07", rule_name="Comorbidity awareness",
                                 premises=[f"Scenario: {d} is under consideration", f"KG fact: {edge}"],
                                 conclusion=f"{other} is associated with {d} and should be kept in mind (comorbidity).", triples=[edge]))
    return out


RULES = [rule_01, rule_02, rule_03, rule_04, rule_05, rule_06, rule_07]


def run_rules(store: GraphStore, entities: list[Entity]) -> list[Inference]:
    ctx = Ctx(store, entities)
    results: list[Inference] = []
    for rule in RULES:
        results.extend(rule(store, ctx))
    return results


def format_symbolic_context(inferences: list[Inference]) -> str:
    if not inferences:
        return "No symbolic rule fired for this question."
    lines = []
    for i, inf in enumerate(inferences, 1):
        lines.append(f"[R{i}] {inf.rule_id} ({inf.rule_name}) => {inf.conclusion}")
        lines += [f"     because: {p}" for p in inf.premises]
    return "\n".join(lines)
