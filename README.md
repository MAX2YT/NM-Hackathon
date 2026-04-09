# Neuro-Vault Clinical Intelligence Engine (Simple Demo)

A compact, local-first demo of the "Neuro-Vault" architecture for a zero-trust healthcare environment.

## Judge quick walkthrough (2 minutes)

Use this section during evaluation.

### 1. Run the demo

From the project root:

```bash
python neuro_vault_demo.py --query "Patient has SOB after CABG with low SpO2"
```

Windows (if `python` is not available in PATH):

```powershell
C:/Users/Username/AppData/Local/Programs/Python/Python310/python.exe neuro_vault_demo.py --query "Patient has SOB after CABG with low SpO2"
```

### 2. Sample input

```text
Patient has SOB after CABG with low SpO2
```

### 3. Sample output (real run)

```text
Neuro-Vault Clinical Intelligence Engine (Demo)
========================================================================
Query: Patient has SOB after CABG with low SpO2
Expanded Query: Patient has SOB (shortness of breath) after CABG (coronary artery bypass graft) with low SpO2 (oxygen saturation) shortness of breath sob dyspnea breathlessness coronary artery bypass graft cabg bypass surgery oxygen saturation spo2 o2 saturation
Decision: ANSWER | Confidence=0.584 | Threshold=0.230
Generation Mode: Deterministic fallback

Answer:

Patient admitted on 08/04/2026 after CABG (coronary artery bypass graft) performed 6 days ago.
[source: TN-GH-OP-001#chunk-001] Guideline hint: For post-CABG breathlessness, evaluate fluid
overload and pulmonary edema, monitor oxygen saturation trends, and repeat ECG if symptoms
persist. [source: KG#KG-001]

Source Trace:
- TN-GH-OP-001#chunk-001 score=0.584 (base=0.536, lex=0.714, entity=0.667, kg=0.207)
```


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
python neuro_vault_demo.py --query "Patient has SOB after CABG with low SpO2"
```

Show debug scores:

```bash
python neuro_vault_demo.py \
  --query "GRBS 298 mg/dL management" \
  --show-debug
```

Trigger abstention for uncertain queries:

```bash
python neuro_vault_demo.py \
  --query "Treatment protocol for condition not in records" \
  --abstain-threshold 0.35
```

Use local Ollama if running at `http://127.0.0.1:11434`:

```bash
python neuro_vault_demo.py \
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
