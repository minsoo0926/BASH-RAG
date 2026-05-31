# Final Paper Experiment Plan

## Scope

Run retrieval-only experiments for the HopRAG paper comparison on a fixed HotpotQA setup.

Dataset:
- `dataset/hotpot.jsonl`
- Main runs use the first 1000 questions.
- Epsilon sweeps use the first 100 questions.

Shared retrieval settings:
- `topk=20`
- `max_hop=4`
- `entry_type=node`
- `hybrid=True` for graph traversal strategies
- same query order, embedding model, Neo4j snapshot, and evaluation script

## Graphs

1. Original graph
   - label: `hotpot_bgeen_qwen2_5_1b5`
   - relationship: `pen2ans_hotpot_bgeen_qwen2_5_1b5`

2. LLM-augmented graph
   - label: `hotpot_bgeen_qwen2_5_1b5_spacy_full_aug_r2_k2_llmq_v2`
   - relationship: `pen2ans_hotpot_bgeen_qwen2_5_1b5_spacy_full_aug_r2_k2_llmq_v2`

3. Fast augmented graph, diagnostic only
   - label: `hotpot_bgeen_qwen2_5_1b5_spacy_full_aug_r2_k2_fast`
   - relationship: `pen2ans_hotpot_bgeen_qwen2_5_1b5_spacy_full_aug_r2_k2_fast`

## Methods

1. Hybrid top-k baseline
   - No graph traversal.
   - Uses the initial hybrid sparse+dense node retrieval path.
   - Command uses `--mock-dense` for dense-only and `--mock-sparse` for sparse-only diagnostic baselines.

2. Paper BFS
   - `--traversal paper_bfs`
   - Uses the implemented paper-style LLM traversal and helpfulness pruning.

3. HopQ
   - `--traversal hopq`
   - Current implementation: NER-count sparse retrieval plus hybrid sparse+dense SIM for traversal.
   - Epsilon grid: `0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0`.

## Metrics

Main table metrics:
- support fact hit rate
- questions with all supports hit
- questions with any support hit
- precision proxy
- recall proxy
- F1 proxy
- average seconds per query

Graph diagnostics:
- nodes
- relationships
- augmented relationships
- weakly connected components
- largest component size
- augmented edge kinds

## Execution Order

1. Collect graph statistics for all three graphs.
2. Run HopQ epsilon sweeps on original and LLM-augmented graphs, 100 questions each.
3. Select the main HopQ epsilon using the sweep, preferring the best non-degenerate setting when results are close.
4. Run 1000-question full evaluations:
   - original dense top-k baseline
   - original sparse top-k diagnostic baseline
   - original paper BFS
   - original HopQ
   - LLM-augmented paper BFS
   - LLM-augmented HopQ
5. Keep fast augmented graph results as diagnostic/appendix only.
6. Generate a pre-publication LaTeX report from frozen outputs under `outputs/retrieval/final_paper/`.
