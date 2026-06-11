# BASH-RAG: Bridge Augmented Swift Hop for Retrieval-Augmented Generation

BASH-RAG is a graph-based retrieval framework for multi-hop question answering.
It builds on HopRAG-style document graphs, but focuses on two practical changes:

1. **Bridge augmentation**: add generated bridge edges between semantically related graph components.
2. **Swift hop retrieval**: replace expensive LLM-guided traversal with a deterministic priority-queue traversal strategy, HopQ.

The goal is to retrieve complete multi-hop evidence more accurately and much faster than LLM-heavy graph traversal baselines.

## Overview

Traditional RAG retrieves passages from a flat index. BASH-RAG instead retrieves over a Neo4j graph:

- nodes are passage chunks with text, keywords, and dense embeddings;
- edges are answerable relations between chunks;
- bridge edges optionally connect distant but semantically related graph regions;
- HopQ expands the graph using a hybrid sparse+dense explore-exploit score.

For a query `q`, HopQ scores each neighbor `v'` from current node `v` as:

```text
score(v') = epsilon * SIM(v, v') + (1 - epsilon) * SIM(q, v')
```

where `SIM` combines dense cosine similarity and sparse keyword overlap. This keeps traversal deterministic and avoids LLM calls during neighbor selection.

## Current Experimental Results

We evaluate retrieval on the first 1,000 HotpotQA examples using a 40,454-node graph.
The main metric is **All Supports Hit**, the number of questions for which all gold supporting facts are retrieved. We also report support fact recall and average latency per query.

| Graph | Method | All Supports Hit | Any Support Hit | Support Fact Recall | F1 Proxy | sec/query |
|---|---:|---:|---:|---:|---:|---:|
| Original | Hybrid top-k | 485 | 942 | 0.7127 | 0.2103 | 0.231 |
| Original | Paper BFS | 587 | 971 | 0.7859 | 0.1760 | 22.630 |
| Original | HopQ, epsilon=0.2 | **642** | **981** | **0.8210** | 0.1790 | **1.263** |
| LLM-Augmented | Paper BFS | 581 | 973 | 0.7871 | 0.1763 | 22.908 |
| LLM-Augmented | HopQ, epsilon=0.2 | **642** | **980** | **0.8206** | 0.1789 | **1.313** |

Key takeaways:

- HopQ improves complete-support retrieval over Paper BFS on the original graph: `642` vs. `587` all-support hits.
- HopQ is much faster than Paper BFS: about `1.26s/query` vs. `22.63s/query`.
- The current LLM-augmented graph greatly improves connectivity, but does not materially improve HopQ retrieval quality in this setting.

Graph statistics:

| Graph | Nodes | Edges | Augmented Edges | Weak Components | Largest Component |
|---|---:|---:|---:|---:|---:|
| Original | 40,454 | 170,178 | 0 | 933 | 274 |
| LLM-Augmented | 40,454 | 172,290 | 2,112 | 46 | 39,021 |
| Fast-Augmented | 40,454 | 173,874 | 3,696 | 1 | 40,454 |

## Repository Layout

Important files:

- `HopRetriever.py`: retrieval strategies, including Paper BFS and HopQ.
- `HopQStrategy.py`: deterministic HopQ priority-queue traversal.
- `HopBuilder.py`: graph node/edge construction utilities.
- `HopGenerator.py`: end-to-end retrieval-augmented generation entry point.
- `graph_augment.py`: graph cloning and bridge-edge augmentation.
- `eval_retrieval.py`: retrieval-only benchmark runner used for the reported experiments.

## Setup

Requirements:

- Python 3.10+
- Neo4j Community Edition 5.x
- Python dependencies from `requirements.txt` or the project environment

Install dependencies:

```bash
git clone https://github.com/minsoo0926/HopRAG.git
cd HopRAG
pip install -r requirements.txt
```

Configure your local `config.py` before running experiments. Do not commit private credentials or machine-specific paths:

- Neo4j connection: `neo4j_url`, `neo4j_user`, `neo4j_password`, `neo4j_dbname`
- graph labels and indexes: `node_name`, `edge_name`, dense/sparse index names
- embedding model: `embed_model`, `embed_dim`
- local or API LLM settings: `local_model_name`, `query_generator_model`, `traversal_model`

## How To Run

### 1. Build or load a graph

Use `HopBuilder.py` for offline cache construction, Neo4j upload, relationship upload, and index creation.
The default script path runs all stages on the quickstart HotpotQA sample:

```bash
python HopBuilder.py
```

For larger runs, execute the same pipeline one stage at a time so the local cache can be reused:

```bash
HOPRAG_BUILD_STAGE=offline HOPRAG_BUILD_SPAN=100 python HopBuilder.py
HOPRAG_BUILD_STAGE=offline_edges python HopBuilder.py
HOPRAG_BUILD_STAGE=upload python HopBuilder.py
HOPRAG_BUILD_STAGE=upload_edges python HopBuilder.py
```

Useful environment variables:

- `HOPRAG_DOCS_DIR`: source document directory.
- `HOPRAG_PROBLEMS_PATH`: HotpotQA or MuSiQue jsonl file used to create edges.
- `HOPRAG_OFFLINE_CACHE_DIR`: local node/edge cache path.
- `HOPRAG_ONLINE_CACHE_DIR`: online upload cache path.
- `HOPRAG_BUILD_START_INDEX`, `HOPRAG_BUILD_SPAN`: document range for node construction.

### 2. Clone and augment a graph

Clone a graph into a separate label/relationship pair:

```bash
python graph_augment.py clone \
  --source-label hotpot_bgeen_qwen2_5_1b5 \
  --source-relationship pen2ans_hotpot_bgeen_qwen2_5_1b5 \
  --target-label hotpot_bgeen_qwen2_5_1b5_spacy_full_aug_r2_k2_llmq_v2 \
  --target-relationship pen2ans_hotpot_bgeen_qwen2_5_1b5_spacy_full_aug_r2_k2_llmq_v2
```

Add bridge edges using a question cache or an LLM:

```bash
python graph_augment.py augment \
  --label hotpot_bgeen_qwen2_5_1b5_spacy_full_aug_r2_k2_llmq_v2 \
  --relationship pen2ans_hotpot_bgeen_qwen2_5_1b5_spacy_full_aug_r2_k2_llmq_v2 \
  --r 2 \
  --k 2 \
  --generate-questions \
  --question-cache outputs/augmentation/bash_rag_bridge_questions.json
```

### 3. Run retrieval-only evaluation

HopQ on the original graph:

```bash
python eval_retrieval.py \
  --label hotpot_bgeen_qwen2_5_1b5 \
  --data dataset/hotpot.jsonl \
  --limit 1000 \
  --traversal hopq \
  --entry-type node \
  --hybrid \
  --topk 20 \
  --max-hop 4 \
  --epsilon 0.2 \
  --output outputs/retrieval/final_paper/original_hopq_eps0_2_limit1000.json
```

Paper BFS baseline:

```bash
python eval_retrieval.py \
  --label hotpot_bgeen_qwen2_5_1b5 \
  --data dataset/hotpot.jsonl \
  --limit 1000 \
  --traversal paper_bfs \
  --entry-type node \
  --hybrid \
  --topk 20 \
  --max-hop 4 \
  --output outputs/retrieval/final_paper/original_paper_bfs_limit1000.json
```

### 4. Run end-to-end generation

Use `HopGenerator.py` when answer generation is needed:

```bash
python HopGenerator.py \
  --model_name gpt-3.5-turbo \
  --data_path quickstart_dataset/hotpot_example.jsonl \
  --save_dir quickstart_dataset/hotpot_output \
  --retriever_name HopRetriever \
  --max_hop 4 \
  --topk 20 \
  --traversal hopq \
  --epsilon 0.2 \
  --hybrid \
  --entry_type node \
  --mode common
```

## Notes

- Internal retrieval scores are not used as final quality metrics because sparse full-text scores and dense/traversal scores may have different scales.
- For multi-hop retrieval, `All Supports Hit` is the strictest reported retrieval metric and should be read alongside support fact recall.
- The current bridge augmentation improves graph connectivity but should be evaluated carefully; more edges do not automatically imply better retrieval.

## Citation

This repository builds on HopRAG. If you use the original HopRAG components, cite:

```bibtex
@article{liu2025hoprag,
  title={{HopRAG}: Multi-hop reasoning for logic-aware retrieval-augmented generation},
  author={Liu, Hao and Wang, Zhengren and Chen, Xi and Li, Zhiyu and Xiong, Feiyu and Yu, Qinhan and Zhang, Wentao},
  journal={arXiv preprint arXiv:2502.12442},
  year={2025}
}
```
