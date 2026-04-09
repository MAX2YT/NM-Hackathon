# Neuro-Vault Clinical Intelligence Engine (Simple Demo)

A compact, local-first demo of the "Neuro-Vault" architecture for a zero-trust healthcare environment.

## What this demo includes

- Local semantic retrieval over clinical records (no cloud dependency)
- Clinical abbreviation normalization (`SOB`, `GRBS`, `CABG`, etc.)
- Entity-aware chunking that keeps key clinical atoms together
- Neural-style reranking of top chunks for better relevance
- Deterministic abstention (`Not Found`) below a confidence threshold
- Source-to-answer provenance watermarking (`DOC#chunk`)
- Optional local Ollama generation (`--use-ollama`) with safe fallback

## Architecture mapping

| Neuro-Vault Layer | Demo Implementation |
|---|---|
| Hybrid Quantized Model (HQM) | Optional local Ollama call (e.g., quantized Llama/Mistral) with deterministic local fallback |
| Domain-Specific Embeddings | Clinical abbreviation/synonym expansion + local semantic vector search |
| Neural Reranker | Secondary rerank stage combining lexical, entity, and KG alignment scores |
| Clinical Entity-Aware Chunking | Regex-based NER for medications, dosages, dates, times before chunking |
| Knowledge Graph Integration | Local `knowledge_graph.json` used as contextual hints during answer synthesis |
| Automated Medical Normalization | Abbreviation map and synonym expansion during indexing and query time |
| Air-Gapped Execution | Runs fully local; optional `docker-compose.airgap.yml` with `network_mode: none` |
| Cryptographic Source Provenance | Every output sentence is watermarked with `[source: DOC#chunk]` or `KG#id` |
| Deterministic Abstention Protocol | Hard threshold gate via `--abstain-threshold` |
| Local Vector Store | Disk-local JSON docs indexed into an in-process TF-IDF vector store |

## Quick start

From project root:

```bash
python3 neuro_vault_demo.py --query "Patient has SOB after CABG with low SpO2"
```

Show debug scores:

```bash
python3 neuro_vault_demo.py \
  --query "GRBS 298 mg/dL management" \
  --show-debug
```

Trigger abstention for uncertain queries:

```bash
python3 neuro_vault_demo.py \
  --query "Treatment protocol for condition not in records" \
  --abstain-threshold 0.35
```

Use local Ollama if running at `http://127.0.0.1:11434`:

```bash
python3 neuro_vault_demo.py \
  --query "Post-op DVT prophylaxis dose" \
  --use-ollama \
  --ollama-model "mistral:7b-instruct-q4_K_M"
```

## Air-gapped container demo (optional)

```bash
docker compose -f docker-compose.airgap.yml run --rm neuro-vault-demo
```

This demo container is configured with `network_mode: none` to simulate strict zero-network execution.

## Important note

This is a hackathon demo and **not** a clinical decision system. Do not use it for real patient care.
