"""Create the Neo4j schema, load the CSV knowledge base, and verify it with multi-hop Cypher.

    python -m src.graph_builder                 # init + load + verify
    python -m src.graph_builder --reset         # wipe the database first
    python -m src.graph_builder --verify        # only run the verification queries
"""
from __future__ import annotations

import argparse

from src.config import CYPHER_DIR
from src.graph_store import load_kb
from src.neo4j_client import Neo4jClient
from src.ontology import ENTITY_TYPES, validate_triple


def init_schema(client: Neo4jClient) -> None:
    """One uniqueness constraint (and implicit index) per ontology class."""
    for label in ENTITY_TYPES:  # labels come from the ontology whitelist, never from user input
        client.run(f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")


def load_graph(client: Neo4jClient, reset: bool = False) -> dict:
    ents, rels = load_kb()
    if reset:
        client.run("MATCH (n) DETACH DELETE n")
    for type_, grp in ents.groupby("type"):
        assert type_ in ENTITY_TYPES, f"unknown entity type {type_}"
        rows = [{"name": r["name"], "props": {"id": r["id"], "description": r["description"],
                                              "aliases": [a for a in r["aliases"].split("|") if a]}}
                for r in grp.to_dict("records")]
        client.run(f"UNWIND $rows AS row MERGE (n:{type_} {{name: row.name}}) SET n += row.props", rows=rows)
    for (st, rel, tt), grp in rels.groupby(["source_type", "relation", "target_type"]):
        validate_triple(st, rel, tt)  # also whitelists the relationship type used in the f-string
        rows = [{"source": r["source"], "target": r["target"], "note": r["note"]} for r in grp.to_dict("records")]
        client.run(
            f"UNWIND $rows AS row MATCH (a:{st} {{name: row.source}}) MATCH (b:{tt} {{name: row.target}}) "
            f"MERGE (a)-[r:{rel}]->(b) SET r.note = row.note", rows=rows)
    return {"nodes": len(ents), "relationships": len(rels)}


def verify(client: Neo4jClient) -> None:
    print("\n== Node counts by label ==")
    for r in client.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label"):
        print(f"  {r['label']:<18}{r['count']}")
    print("\n== Relationship counts by type ==")
    for r in client.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count ORDER BY rel"):
        print(f"  {r['rel']:<20}{r['count']}")
    text = (CYPHER_DIR / "multihop_queries.cypher").read_text(encoding="utf-8")
    for block in text.split(";"):
        block = block.strip()
        if not block:
            continue
        title = block.splitlines()[0].lstrip("/ ").strip()
        print(f"\n== {title} ==")
        for row in client.run(block)[:8]:
            print("  ", row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", action="store_true")
    p.add_argument("--load", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--reset", action="store_true", help="delete all nodes before loading")
    args = p.parse_args()
    run_all = not (args.init or args.load or args.verify)
    client = Neo4jClient()
    try:
        client.verify()
        if run_all or args.init:
            init_schema(client); print("Schema constraints created.")
        if run_all or args.load:
            print("Loaded:", load_graph(client, reset=args.reset))
        if run_all or args.verify:
            verify(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
