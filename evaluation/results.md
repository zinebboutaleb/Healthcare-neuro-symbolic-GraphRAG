# Evaluation results (toy benchmark)

Backends used for this run: `{'graph_backend': 'neo4j', 'embedding_backend': 'all-MiniLM-L6-v2', 'llm': 'hf:Qwen/Qwen2.5-1.5B-Instruct'}`

**This is a toy benchmark: 14 hand-written questions over a 113-entity synthetic KB. Numbers illustrate the methodology; they are not evidence of general or clinical performance.**

## Summary

| Method | Single-hop answer correct | Multi-hop answer correct | Evidence complete (all) | Evidence complete (multi-hop) | Mean evidence recall | Unsupported entities | Median context chars | Median latency ms |
|---|---|---|---|---|---|---|---|---|
| Vector RAG | 5/5 | 7/9 | 11/14 | 6/9 | 0.90 | 1 | 1073 | 20186.6 |
| GraphRAG | 3/5 | 8/9 | 14/14 | 9/9 | 1.00 | 0 | 2674 | 28588.2 |
| Hybrid (graph+vector+rules) | 5/5 | 8/9 | 14/14 | 9/9 | 1.00 | 0 | 4826 | 31310.4 |
| ablation: Graph + rules | 4/5 | 8/9 | 14/14 | 9/9 | 1.00 | 0 | 3656 | 23140.5 |
| ablation: Vector + graph | 5/5 | 9/9 | 14/14 | 9/9 | 1.00 | 0 | 3723 | 30896.1 |

## Per question  (answer correct / evidence complete (evidence recall))

| Q | hops | Vector RAG | GraphRAG | Hybrid (graph+vector+rules) | ablation: Graph + rules | ablation: Vector + graph |
|---|---|---|---|---|---|---|
| Q01 | 1 | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q02 | 1 | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q03 | 1 | Y / Y (1.00) | n / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q04 | 1 | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q05 | 1 | Y / Y (1.00) | n / Y (1.00) | Y / Y (1.00) | n / Y (1.00) | Y / Y (1.00) |
| Q06 | 2 | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q07 | 2 | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q08 | 2 | Y / Y (1.00) | Y / Y (1.00) | n / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q09 | 2 | n / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | n / Y (1.00) | Y / Y (1.00) |
| Q10 | 2 | Y / n (0.50) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q11 | 3 | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q12 | 3 | Y / Y (1.00) | n / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q13 | 3 | Y / n (0.60) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |
| Q14 | 4 | n / n (0.50) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) | Y / Y (1.00) |

## How to read this

- *Answer correct*: every expected entity is mentioned in the answer (string match, aliases allowed).
- *Evidence complete / recall*: share of the required entities (answer + intermediate nodes) present in the retrieved context.
- *Unsupported entities*: KB entities named in the answer that are in neither the context nor the question (hallucination proxy).
- *Median context chars*: size of the context sent to the LLM. Larger contexts mechanically raise evidence recall, so compare with care.
