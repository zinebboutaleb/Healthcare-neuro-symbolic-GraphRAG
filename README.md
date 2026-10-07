# Healthcare Neuro-Symbolic GraphRAG

A compact research prototype for question answering over a synthetic healthcare knowledge graph. The project combines graph retrieval, symbolic reasoning, vector retrieval, and an LLM into one evidence-first pipeline.

This repo is designed to study a simple but important idea: multi-hop medical questions are often easier to answer and explain when relevant facts are connected by explicit relationships, not just retrieved as semantically similar text chunks.

> Warning: this is a synthetic educational demo, not a medical system. The data is fictional and should not be treated as clinical advice.

## What the project does

The system answers questions such as:

- “A pregnant patient has Hypertension. Which medication should not be recommended?”
- “A patient with Severe Renal Impairment has Heart Failure. Which treatment should not be recommended, and which alternative needs additional review?”

It does this by combining:

- a structured knowledge graph
- document retrieval for supporting context
- deterministic symbolic rules for derived inferences
- an LLM that answers only from the supplied evidence

## Architecture

```text
Question
  ↓
Entity linking (names + aliases)
  ↓
+-----------------------+
| Vector retrieval      |
| Graph retrieval       |
| Symbolic rules        |
+-----------------------+
  ↓
Combined context
  ↓
LLM answer grounded in evidence
  ↓
Answer + graph paths + rule trace + document hits
```

The repo includes three retrieval modes:

- Vector RAG: semantic document retrieval only
- GraphRAG: multi-hop graph traversal and path retrieval
- Hybrid: graph + vector + symbolic reasoning together

## Why it exists

Large language models are good at fluent generation, but they often fail when the answer depends on several related facts spread across different sources. This project is a small, interpretable proof of concept for combining:

- explicit graph structure
- deterministic rule-based reasoning
- text retrieval for context
- explainable evidence trails

## Repository layout

```text
.
├── app/
│   └── streamlit_app.py
├── cypher/
│   └── multihop_queries.cypher
├── data/
│   ├── benchmark_questions.csv
│   ├── entities.csv
│   ├── relationships.csv
│   └── documents/
├── evaluation/
│   ├── benchmark.py
│   ├── evaluate.py
│   ├── results.csv
│   ├── results.md
│   └── vector_k_sweep.csv
├── ontology/
│   ├── ontology.md
│   ├── ontology_diagram.png
│   └── make_diagram.py
├── src/
│   ├── config.py
│   ├── data_generation.py
│   ├── embeddings.py
│   ├── graph_builder.py
│   ├── graph_retriever.py
│   ├── graph_store.py
│   ├── hybrid_retriever.py
│   ├── llm.py
│   ├── models.py
│   ├── neo4j_client.py
│   ├── ontology.py
│   ├── pipeline.py
│   ├── symbolic_rules.py
│   ├── vector_store.py
│   └── __init__.py
├── tests/
│   └── test_pipeline.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── README.md
├── requirements.txt
└── LICENSE (if added later)
```

## Setup

### 1) Create the environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2) Start Neo4j

```bash
docker compose up -d
```

The default Neo4j credentials are set in [.env.example](.env.example):

- username: `neo4j`
- password: `healthcare123`

### 3) Generate the synthetic dataset and load the graph

```bash
python -m src.data_generation
python -m src.graph_builder
```

If you want to wipe the previous graph first:

```bash
python -m src.graph_builder --reset
```

## Run the demo

```bash
python -m src.pipeline "A pregnant patient has Hypertension. Which medication should not be recommended?" --mode hybrid
```

You can also run the UI:

```bash
streamlit run app/streamlit_app.py
```

## Evaluation results

The benchmark in this repo was run with:

- graph backend: Neo4j
- embedding backend: all-MiniLM-L6-v2
- LLM: local HF model (Qwen/Qwen2.5-1.5B-Instruct)

The benchmark is intentionally small and synthetic: 14 hand-written questions over a 113-entity healthcare KB.

These numbers are taken from [evaluation/results.md](evaluation/results.md):

| Method | Single-hop correct | Multi-hop correct | Evidence complete (all) | Evidence complete (multi-hop) | Mean evidence recall | Unsupported entities | Median context chars | Median latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vector RAG | 5/5 | 7/9 | 11/14 | 6/9 | 0.90 | 1 | 1073 | 20186.6 |
| GraphRAG | 3/5 | 8/9 | 14/14 | 9/9 | 1.00 | 0 | 2674 | 28588.2 |
| Hybrid (graph+vector+rules) | 5/5 | 8/9 | 14/14 | 9/9 | 1.00 | 0 | 4826 | 31310.4 |
| Graph + rules | 4/5 | 8/9 | 14/14 | 9/9 | 1.00 | 0 | 3656 | 23140.5 |
| Vector + graph | 5/5 | 9/9 | 14/14 | 9/9 | 1.00 | 0 | 3723 | 30896.1 |

### How to interpret these results

- This is a toy benchmark, not a clinical validation.
- The data is synthetic and intentionally designed to favor structured, explicit reasoning.
- Graph-based retrieval performs especially well on multi-hop evidence coverage.
- The repo’s goal is methodology and explainability, not medical-grade performance.

## Important limitations

- The knowledge base is synthetic and very small.
- Entity linking is exact, alias-based matching and does not handle broad natural-language paraphrase well.
- The benchmark is hand-authored and uses simple string matching for correctness.
- The symbolic rules are deterministic but not clinically validated.
- This project is not a medical diagnosis tool and should not be used for patient care.

## GitHub readiness

For a public GitHub repository, keep the following in mind:

- do not commit secret keys or real credentials
- keep [.env.example](.env.example) as the template, not the actual runtime config
- use the repository’s `.gitignore` to avoid committing local virtual environments and environment files

## License

This repo does not currently include a license file. If you plan to publish it publicly on GitHub, add a license such as MIT before opening the repository to the public.

## Acknowledgement

This project is a research-oriented educational prototype for studying hybrid retrieval and structured reasoning in a healthcare domain.

