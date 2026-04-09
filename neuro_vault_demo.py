#!/usr/bin/env python3
"""
Neuro-Vault Clinical Intelligence Engine (Simple Demo)

This demo is fully local and CPU-friendly. It illustrates:
- Clinical abbreviation normalization
- Entity-aware chunking
- Local semantic retrieval + reranking
- Deterministic abstention when confidence is low
- Source-to-answer provenance tagging
- Optional local Ollama generation fallback
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

ABBREVIATION_MAP: Dict[str, str] = {
    "CABG": "coronary artery bypass graft",
    "SOB": "shortness of breath",
    "GRBS": "capillary blood glucose",
    "RBS": "random blood sugar",
    "BP": "blood pressure",
    "HR": "heart rate",
    "RR": "respiratory rate",
    "SpO2": "oxygen saturation",
    "ECG": "electrocardiogram",
    "NPO": "nil per os",
}

SYNONYM_MAP: Dict[str, List[str]] = {
    "shortness of breath": ["sob", "dyspnea", "breathlessness"],
    "capillary blood glucose": ["grbs", "rbs", "random blood sugar"],
    "coronary artery bypass graft": ["cabg", "bypass surgery"],
    "myocardial infarction": ["mi", "heart attack"],
    "oxygen saturation": ["spo2", "o2 saturation"],
}

MEDICATION_TERMS = {
    "aspirin",
    "metformin",
    "insulin",
    "atorvastatin",
    "clopidogrel",
    "heparin",
    "furosemide",
    "salbutamol",
    "ipratropium",
    "ceftriaxone",
    "azithromycin",
    "enoxaparin",
}

CONDITION_TERMS = {
    "cabg",
    "shortness of breath",
    "dyspnea",
    "chest pain",
    "hyperglycemia",
    "dvt",
    "asthma",
    "pneumonia",
    "ischemia",
}

DATE_PATTERN = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
TIME_PATTERN = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?(?:AM|PM))?\b", re.IGNORECASE)
DOSE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units|l/min|iu|mg/dl)\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:/[a-z0-9]+)?", text.lower())


@dataclass
class ClinicalChunk:
    doc_id: str
    doc_title: str
    chunk_id: str
    text: str
    entities: Dict[str, List[str]]


@dataclass
class RetrievalHit:
    chunk: ClinicalChunk
    base_score: float
    rerank_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class QueryResponse:
    query: str
    expanded_query: str
    abstained: bool
    confidence: float
    answer: str
    citations: List[str]
    used_ollama: bool
    debug_hits: List[Tuple[str, float, float]]


class MedicalNormalizer:
    def __init__(self) -> None:
        self.abbreviation_map = ABBREVIATION_MAP
        self.synonym_map = SYNONYM_MAP

    def normalize_for_index(self, text: str) -> str:
        normalized = text
        for short, full in self.abbreviation_map.items():
            pattern = re.compile(rf"\b{re.escape(short)}\b", re.IGNORECASE)
            normalized = pattern.sub(lambda m: f"{m.group(0)} ({full})", normalized)
        return normalized

    def expand_query(self, query: str) -> str:
        base = self.normalize_for_index(query)
        lowered = base.lower()
        expansion_terms: List[str] = []

        for canonical, variants in self.synonym_map.items():
            if canonical in lowered or any(variant in lowered for variant in variants):
                expansion_terms.append(canonical)
                expansion_terms.extend(variants)

        seen = set()
        deduped = []
        for token in expansion_terms:
            if token not in seen:
                deduped.append(token)
                seen.add(token)

        if deduped:
            return f"{base} {' '.join(deduped)}"
        return base


class ClinicalEntityExtractor:
    def extract(self, text: str) -> Dict[str, List[str]]:
        lowered = text.lower()

        meds = sorted(
            term for term in MEDICATION_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)
        )
        conditions = sorted(
            term for term in CONDITION_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)
        )
        dates = sorted(set(DATE_PATTERN.findall(text)))
        times = sorted(set(match.upper() for match in TIME_PATTERN.findall(text)))
        dosages = sorted(set(match.lower() for match in DOSE_PATTERN.findall(text)))

        return {
            "medications": meds,
            "conditions": conditions,
            "dates": dates,
            "times": times,
            "dosages": dosages,
        }


class ClinicalEntityAwareChunker:
    def __init__(self, extractor: ClinicalEntityExtractor, max_chars: int = 380) -> None:
        self.extractor = extractor
        self.max_chars = max_chars

    def chunk_document(self, doc_id: str, doc_title: str, text: str) -> List[ClinicalChunk]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if not sentences:
            return []

        chunks: List[ClinicalChunk] = []
        current_sentences: List[str] = []
        current_entities = self._empty_entity_sets()

        for sentence in sentences:
            sent_entities = self.extractor.extract(sentence)
            candidate = (" ".join(current_sentences + [sentence])).strip()

            if current_sentences and len(candidate) > self.max_chars:
                if not self._should_keep_together(current_entities, sent_entities, len(candidate)):
                    chunks.append(
                        self._build_chunk(doc_id, doc_title, len(chunks) + 1, current_sentences, current_entities)
                    )
                    current_sentences = [sentence]
                    current_entities = self._to_entity_sets(sent_entities)
                    continue

            current_sentences.append(sentence)
            self._merge_entities(current_entities, sent_entities)

        if current_sentences:
            chunks.append(
                self._build_chunk(doc_id, doc_title, len(chunks) + 1, current_sentences, current_entities)
            )

        return chunks

    def _build_chunk(
        self,
        doc_id: str,
        doc_title: str,
        index: int,
        sentences: List[str],
        entity_sets: Dict[str, set],
    ) -> ClinicalChunk:
        entities = {k: sorted(v) for k, v in entity_sets.items()}
        return ClinicalChunk(
            doc_id=doc_id,
            doc_title=doc_title,
            chunk_id=f"chunk-{index:03d}",
            text=" ".join(sentences),
            entities=entities,
        )

    @staticmethod
    def _empty_entity_sets() -> Dict[str, set]:
        return {
            "medications": set(),
            "conditions": set(),
            "dates": set(),
            "times": set(),
            "dosages": set(),
        }

    def _to_entity_sets(self, entities: Dict[str, List[str]]) -> Dict[str, set]:
        entity_sets = self._empty_entity_sets()
        self._merge_entities(entity_sets, entities)
        return entity_sets

    @staticmethod
    def _merge_entities(target: Dict[str, set], source: Dict[str, List[str]]) -> None:
        for key, values in source.items():
            target[key].update(values)

    @staticmethod
    def _is_critical(entities: Dict[str, List[str]] | Dict[str, set]) -> bool:
        return bool(
            entities.get("medications")
            or entities.get("dosages")
            or entities.get("dates")
            or entities.get("times")
        )

    def _should_keep_together(
        self,
        current_entities: Dict[str, set],
        incoming_entities: Dict[str, List[str]],
        candidate_len: int,
    ) -> bool:
        return (
            candidate_len <= int(self.max_chars * 1.35)
            and self._is_critical(current_entities)
            and self._is_critical(incoming_entities)
        )


class LocalVectorStore:
    def __init__(self) -> None:
        self.chunks: List[ClinicalChunk] = []
        self.doc_freq: Counter[str] = Counter()
        self.vectors: List[Dict[str, float]] = []
        self.norms: List[float] = []
        self.total_docs: int = 0

    def build(self, chunks: List[ClinicalChunk]) -> None:
        self.chunks = chunks
        self.doc_freq = Counter()
        self.vectors = []
        self.norms = []
        self.total_docs = len(chunks)

        tokenized_docs: List[List[str]] = []
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tokenized_docs.append(tokens)
            for term in set(tokens):
                self.doc_freq[term] += 1

        for tokens in tokenized_docs:
            vec = self._tfidf(tokens)
            self.vectors.append(vec)
            self.norms.append(self._norm(vec))

    def search(self, query: str, top_k: int) -> List[RetrievalHit]:
        q_vec = self._tfidf(tokenize(query))
        q_norm = self._norm(q_vec)
        if q_norm == 0.0:
            return []

        hits: List[RetrievalHit] = []
        for idx, chunk in enumerate(self.chunks):
            score = self._cosine_similarity(q_vec, q_norm, self.vectors[idx], self.norms[idx])
            if score > 0.0:
                hits.append(RetrievalHit(chunk=chunk, base_score=score))

        hits.sort(key=lambda hit: hit.base_score, reverse=True)
        return hits[:top_k]

    def _tfidf(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total_terms = sum(tf.values()) or 1

        vector: Dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((1 + self.total_docs) / (1 + self.doc_freq.get(term, 0))) + 1
            vector[term] = (count / total_terms) * idf
        return vector

    @staticmethod
    def _norm(vector: Dict[str, float]) -> float:
        return math.sqrt(sum(weight * weight for weight in vector.values()))

    @staticmethod
    def _cosine_similarity(
        query_vec: Dict[str, float],
        query_norm: float,
        doc_vec: Dict[str, float],
        doc_norm: float,
    ) -> float:
        if query_norm == 0.0 or doc_norm == 0.0:
            return 0.0

        common = set(query_vec).intersection(doc_vec)
        dot_product = sum(query_vec[t] * doc_vec[t] for t in common)
        return dot_product / (query_norm * doc_norm)


class KnowledgeGraph:
    def __init__(self, concepts: List[Dict[str, object]]) -> None:
        self.concepts = concepts

    @classmethod
    def from_json(cls, path: Path) -> "KnowledgeGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data.get("concepts", []))

    def find_hints(self, query: str) -> List[Dict[str, object]]:
        query_lower = query.lower()
        hints: List[Dict[str, object]] = []
        for concept in self.concepts:
            triggers = [str(t).lower() for t in concept.get("triggers", [])]
            if any(trigger in query_lower for trigger in triggers):
                hints.append(concept)
        return hints


class NeuralReranker:
    def __init__(self, extractor: ClinicalEntityExtractor) -> None:
        self.extractor = extractor

    def rerank(
        self,
        query: str,
        hits: List[RetrievalHit],
        knowledge_hints: List[Dict[str, object]],
        top_k: int,
    ) -> List[RetrievalHit]:
        query_tokens = set(tokenize(query))
        query_entities = self.extractor.extract(query)
        kg_tokens = set(tokenize(" ".join(str(h.get("recommendation", "")) for h in knowledge_hints)))

        reranked: List[RetrievalHit] = []
        for hit in hits[: max(top_k, 5)]:
            chunk_tokens = set(tokenize(hit.chunk.text))
            lexical_overlap = self._overlap_ratio(query_tokens, chunk_tokens)
            entity_overlap = self._entity_overlap(query_entities, hit.chunk.entities)
            kg_alignment = self._overlap_ratio(kg_tokens, chunk_tokens) if kg_tokens else 0.0

            score = (
                0.55 * hit.base_score
                + 0.25 * lexical_overlap
                + 0.15 * entity_overlap
                + 0.05 * kg_alignment
            )

            reasons = [
                f"base={hit.base_score:.3f}",
                f"lex={lexical_overlap:.3f}",
                f"entity={entity_overlap:.3f}",
            ]
            if kg_tokens:
                reasons.append(f"kg={kg_alignment:.3f}")

            reranked.append(
                RetrievalHit(
                    chunk=hit.chunk,
                    base_score=hit.base_score,
                    rerank_score=score,
                    reasons=reasons,
                )
            )

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _overlap_ratio(left: set, right: set) -> float:
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / len(left)

    @staticmethod
    def _entity_overlap(
        query_entities: Dict[str, List[str]],
        chunk_entities: Dict[str, List[str]],
    ) -> float:
        query_terms = set()
        chunk_terms = set()

        for values in query_entities.values():
            query_terms.update(v.lower() for v in values)
        for values in chunk_entities.values():
            chunk_terms.update(v.lower() for v in values)

        if not query_terms:
            return 0.0
        return len(query_terms.intersection(chunk_terms)) / len(query_terms)


class NeuroVaultEngine:
    def __init__(self, docs_path: Path, kg_path: Path) -> None:
        self.normalizer = MedicalNormalizer()
        self.extractor = ClinicalEntityExtractor()
        self.chunker = ClinicalEntityAwareChunker(self.extractor)
        self.vector_store = LocalVectorStore()
        self.knowledge_graph = KnowledgeGraph.from_json(kg_path)

        self._index_documents(docs_path)

    def _index_documents(self, docs_path: Path) -> None:
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        all_chunks: List[ClinicalChunk] = []

        for doc in docs:
            doc_id = str(doc["id"])
            title = str(doc.get("title", doc_id))
            text = str(doc["text"])

            normalized_text = self.normalizer.normalize_for_index(text)
            chunks = self.chunker.chunk_document(doc_id, title, normalized_text)
            all_chunks.extend(chunks)

        self.vector_store.build(all_chunks)

    def answer(
        self,
        query: str,
        top_k: int,
        abstain_threshold: float,
        use_ollama: bool,
        ollama_model: str,
    ) -> QueryResponse:
        expanded_query = self.normalizer.expand_query(query)
        hints = self.knowledge_graph.find_hints(expanded_query)

        initial_hits = self.vector_store.search(expanded_query, top_k=max(8, top_k))
        reranked_hits = self.reranker.rerank(expanded_query, initial_hits, hints, top_k=top_k)

        best_score = reranked_hits[0].rerank_score if reranked_hits else 0.0
        trusted_cutoff = max(abstain_threshold, best_score * 0.55)
        trusted_hits = [hit for hit in reranked_hits if hit.rerank_score >= trusted_cutoff]
        if not trusted_hits and reranked_hits:
            trusted_hits = [reranked_hits[0]]

        if best_score < abstain_threshold:
            abstain_answer = (
                "Not Found: confidence below deterministic safety threshold; "
                "no clinically reliable context was retrieved."
            )
            citations = [
                f"{hit.chunk.doc_id}#{hit.chunk.chunk_id} score={hit.rerank_score:.3f}"
                for hit in reranked_hits[:3]
            ]
            debug_hits = [
                (f"{hit.chunk.doc_id}#{hit.chunk.chunk_id}", hit.base_score, hit.rerank_score)
                for hit in reranked_hits
            ]
            return QueryResponse(
                query=query,
                expanded_query=expanded_query,
                abstained=True,
                confidence=best_score,
                answer=abstain_answer,
                citations=citations,
                used_ollama=False,
                debug_hits=debug_hits,
            )

        answer_text, used_ollama = self._generate_answer(
            query=query,
            hits=trusted_hits,
            hints=hints,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
        )

        citations = [
            f"{hit.chunk.doc_id}#{hit.chunk.chunk_id} score={hit.rerank_score:.3f} ({', '.join(hit.reasons)})"
            for hit in trusted_hits[:top_k]
        ]
        debug_hits = [
            (f"{hit.chunk.doc_id}#{hit.chunk.chunk_id}", hit.base_score, hit.rerank_score)
            for hit in reranked_hits
        ]

        return QueryResponse(
            query=query,
            expanded_query=expanded_query,
            abstained=False,
            confidence=best_score,
            answer=answer_text,
            citations=citations,
            used_ollama=used_ollama,
            debug_hits=debug_hits,
        )

    @property
    def reranker(self) -> NeuralReranker:
        return NeuralReranker(self.extractor)

    def _generate_answer(
        self,
        query: str,
        hits: List[RetrievalHit],
        hints: List[Dict[str, object]],
        use_ollama: bool,
        ollama_model: str,
    ) -> Tuple[str, bool]:
        if use_ollama:
            prompt = self._build_prompt(query, hits, hints)
            llm_text = self._call_ollama(prompt, ollama_model)
            if llm_text:
                return self._watermark_sentences(llm_text, hits), True

        return self._fallback_answer(hits, hints), False

    def _build_prompt(
        self,
        query: str,
        hits: List[RetrievalHit],
        hints: List[Dict[str, object]],
    ) -> str:
        context_lines = []
        for hit in hits[:3]:
            source = f"{hit.chunk.doc_id}#{hit.chunk.chunk_id}"
            context_lines.append(f"[{source}] {hit.chunk.text}")

        kg_lines = [
            f"[{item.get('id', 'KG')}] {item.get('recommendation', '')}" for item in hints[:2]
        ]

        return (
            "You are a clinical assistant in a zero-trust environment. "
            "Use only provided context. If uncertain, say Not Found. "
            "Every sentence must end with [source: DOC#chunk or KG#id].\n\n"
            f"Question: {query}\n\n"
            "Context:\n"
            + "\n".join(context_lines)
            + "\n\nKnowledge Graph Hints:\n"
            + "\n".join(kg_lines)
        )

    def _call_ollama(self, prompt: str, model: str) -> str | None:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 250,
            },
        }

        request = urllib.request.Request(
            url="http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            return str(parsed.get("response", "")).strip() or None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def _watermark_sentences(self, text: str, hits: List[RetrievalHit]) -> str:
        source_cycle = [f"{hit.chunk.doc_id}#{hit.chunk.chunk_id}" for hit in hits[:3]]
        if not source_cycle:
            source_cycle = ["UNKNOWN#chunk-000"]

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        stamped: List[str] = []

        for idx, sentence in enumerate(sentences):
            if "[source:" in sentence.lower():
                stamped.append(sentence)
            else:
                source = source_cycle[min(idx, len(source_cycle) - 1)]
                stamped.append(f"{sentence} [source: {source}]")

        return " ".join(stamped)

    def _fallback_answer(
        self,
        hits: List[RetrievalHit],
        hints: List[Dict[str, object]],
    ) -> str:
        lines: List[str] = []

        for hit in hits[:2]:
            source = f"{hit.chunk.doc_id}#{hit.chunk.chunk_id}"
            sentence = self._first_sentence(hit.chunk.text)
            lines.append(f"{sentence} [source: {source}]")

        if hints:
            hint = hints[0]
            kg_id = hint.get("id", "KG")
            recommendation = str(hint.get("recommendation", ""))
            lines.append(f"Guideline hint: {recommendation} [source: KG#{kg_id}]")

        if not lines:
            return "Not Found: no evidence retrieved. [source: UNKNOWN#chunk-000]"

        return " ".join(lines)

    @staticmethod
    def _first_sentence(text: str) -> str:
        first = re.split(r"(?<=[.!?])\s+", text.strip())[0].strip()
        return first if len(first) <= 240 else first[:237] + "..."


def build_parser(default_docs: Path, default_kg: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neuro-Vault Clinical Intelligence Demo")
    parser.add_argument("--query", required=True, help="Clinical question to ask")
    parser.add_argument(
        "--docs", default=str(default_docs), help="Path to local clinical documents JSON"
    )
    parser.add_argument(
        "--kg", default=str(default_kg), help="Path to local medical knowledge graph JSON"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks to keep")
    parser.add_argument(
        "--abstain-threshold",
        type=float,
        default=0.23,
        help="If reranker confidence is below this threshold, return Not Found",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use local Ollama generation (fallback remains local if unavailable)",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3:8b-instruct-q4_K_M",
        help="Local Ollama model name",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--show-debug", action="store_true", help="Show reranking debug details")
    return parser


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = build_parser(
        default_docs=root / "demo_data" / "clinical_docs.json",
        default_kg=root / "demo_data" / "knowledge_graph.json",
    )
    args = parser.parse_args()

    engine = NeuroVaultEngine(docs_path=Path(args.docs), kg_path=Path(args.kg))
    result = engine.answer(
        query=args.query,
        top_k=args.top_k,
        abstain_threshold=args.abstain_threshold,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "query": result.query,
                    "expanded_query": result.expanded_query,
                    "abstained": result.abstained,
                    "confidence": round(result.confidence, 4),
                    "used_ollama": result.used_ollama,
                    "answer": result.answer,
                    "citations": result.citations,
                    "debug_hits": result.debug_hits,
                },
                indent=2,
            )
        )
        return

    print("Neuro-Vault Clinical Intelligence Engine (Demo)")
    print("=" * 72)
    print(f"Query: {result.query}")
    print(f"Expanded Query: {result.expanded_query}")
    print(
        f"Decision: {'ABSTAIN (Not Found)' if result.abstained else 'ANSWER'} | "
        f"Confidence={result.confidence:.3f} | Threshold={args.abstain_threshold:.3f}"
    )
    print(f"Generation Mode: {'Ollama' if result.used_ollama else 'Deterministic fallback'}")
    print("\nAnswer:\n")
    print(textwrap.fill(result.answer, width=96))

    print("\nSource Trace:")
    for citation in result.citations:
        print(f"- {citation}")

    if args.show_debug:
        print("\nReranker Debug:")
        for source, base, rerank in result.debug_hits:
            print(f"- {source}: base={base:.3f}, rerank={rerank:.3f}")


if __name__ == "__main__":
    main()
