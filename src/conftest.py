"""Force deterministic offline settings for tests, even when a developer .env selects Neo4j / an LLM."""
import os

import src.config  # noqa: F401  (loads .env first, so the overrides below win)

os.environ.update(GRAPH_BACKEND="networkx", EMBEDDING_BACKEND="hashing", LLM_PROVIDER="offline")