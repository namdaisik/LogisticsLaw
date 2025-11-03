import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - dependency missing at runtime only
    PdfReader = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - dependency missing at runtime only
    OpenAI = None


INDEX_FILENAME = "mcp_index.json"
EMBED_MATRIX_FILENAME = "mcp_embeddings.npy"

_EMBEDDING_ARRAY_CACHE: Dict[str, np.ndarray] = {}


@dataclass
class Chunk:
    id: str
    file: str
    page_start: int
    page_end: int
    text: str
    embedding: List[float]


def _ensure_pypdf() -> None:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Please install dependencies.")


def _ensure_openai() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai SDK is not installed. Please install dependencies.")
    return OpenAI()


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_pdfs(root: str) -> List[str]:
    pdfs: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, name))
    return sorted(pdfs)


def extract_pdf_text_with_pages(path: str) -> List[Tuple[int, str]]:
    _ensure_pypdf()
    reader = PdfReader(path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages


def chunk_pages(
    pages: Sequence[Tuple[int, str]],
    target_chars: int = 1500,
    overlap_chars: int = 150,
) -> List[Tuple[int, int, str]]:
    chunks: List[Tuple[int, int, str]] = []
    buffer: List[str] = []
    buffer_chars = 0
    start_page: int | None = None
    for page_num, text in pages:
        if start_page is None:
            start_page = page_num
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        for paragraph in paragraphs:
            if buffer and buffer_chars + len(paragraph) + 1 > target_chars:
                combined = "\n".join(buffer)
                end_page = page_num
                chunks.append((start_page, end_page, combined))
                overlap_text = combined[-overlap_chars:]
                buffer = [overlap_text]
                buffer_chars = len(overlap_text)
                start_page = page_num
            buffer.append(paragraph)
            buffer_chars += len(paragraph) + 1
    if buffer:
        combined = "\n".join(buffer)
        end_page = pages[-1][0] if pages else 1
        chunks.append((start_page or 1, end_page, combined))
    return [(s, e, t) for (s, e, t) in chunks if t.strip()]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def embed_texts(texts: Sequence[str], model: str, batch_size: int = 64) -> List[List[float]]:
    if not texts:
        return []
    client = _ensure_openai()
    vectors: List[List[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = list(texts[start:start + batch_size])
        response = client.embeddings.create(model=model, input=batch)
        for item in response.data:
            vectors.append(item.embedding)
    return vectors


def _chunk_embedding_matrix(index: Dict[str, Any]) -> np.ndarray:
    cached = index.get("_chunk_embedding_matrix")
    if cached is not None:
        return cached

    chunks = index.get("chunks") or []
    if not chunks:
        matrix = np.empty((0, 0), dtype=np.float32)
        index["_chunk_embedding_matrix"] = matrix
        return matrix

    sample = chunks[0]
    if "embedding" in sample:
        matrix = np.asarray([chunk["embedding"] for chunk in chunks], dtype=np.float32)
        index["_chunk_embedding_matrix"] = matrix
        return matrix

    if "embedding_index" in sample:
        root = index.get("root")
        if not root:
            raise RuntimeError("Index missing 'root' metadata required to load embeddings.")
        matrix_path = os.path.join(root, EMBED_MATRIX_FILENAME)
        cache_key = os.path.abspath(matrix_path)
        matrix_full = _EMBEDDING_ARRAY_CACHE.get(cache_key)
        if matrix_full is None:
            if not os.path.exists(matrix_path):
                raise FileNotFoundError(f"Embedding matrix file not found: {matrix_path}")
            matrix_full = np.load(matrix_path, allow_pickle=False)
            if matrix_full.dtype != np.float32:
                matrix_full = matrix_full.astype(np.float32)
            _EMBEDDING_ARRAY_CACHE[cache_key] = matrix_full

        indices = np.asarray(
            [chunk.get("embedding_index", -1) for chunk in chunks],
            dtype=np.int64,
        )
        if np.any(indices < 0):
            raise KeyError("embedding_index")
        max_index = int(indices.max()) if indices.size else -1
        if max_index >= matrix_full.shape[0]:
            raise ValueError(
                f"embedding_index {max_index} out of bounds for matrix with {matrix_full.shape[0]} rows"
            )
        matrix = matrix_full[indices]
        index["_chunk_embedding_matrix"] = matrix
        return matrix

    raise KeyError("embedding")


def build_or_load_index(root: str, embed_model: str, force_rebuild: bool) -> Dict[str, Any]:
    index_path = os.path.join(root, INDEX_FILENAME)
    existing: Dict[str, Any] = {}
    if os.path.exists(index_path) and not force_rebuild:
        with open(index_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    pdfs = find_pdfs(root)
    files_meta: Dict[str, str] = {p: sha1_file(p) for p in pdfs}

    reuse_ok = (
        existing.get("embed_model") == embed_model
        and existing.get("files") == files_meta
        and "chunks" in existing
    )
    if reuse_ok:
        return existing

    print(f"Indexing PDFs in: {root} (files: {len(pdfs)})")
    chunk_records: List[Chunk] = []
    per_file_chunks: List[Tuple[str, Tuple[int, int, str]]] = []

    for path in tqdm(pdfs, desc="Extracting", unit="file"):
        pages = extract_pdf_text_with_pages(path)
        for start, end, text in chunk_pages(pages):
            per_file_chunks.append((path, (start, end, text)))

    texts = [chunk[1][2] for chunk in per_file_chunks]
    vectors = embed_texts(texts, embed_model) if texts else []

    for (path, (start, end, text)), embedding in zip(per_file_chunks, vectors):
        chunk_id = hashlib.sha1((path + str(start) + str(end)).encode("utf-8")).hexdigest()[:16]
        chunk_records.append(
            Chunk(
                id=chunk_id,
                file=os.path.relpath(path, root),
                page_start=start,
                page_end=end,
                text=text,
                embedding=embedding,
            )
        )

    index = {
        "version": 1,
        "created": int(time.time()),
        "root": os.path.abspath(root),
        "embed_model": embed_model,
        "files": files_meta,
        "chunks": [asdict(chunk) for chunk in chunk_records],
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    return index


def search_index(index: Dict[str, Any], query: str, embed_model: str, top_k: int) -> List[Dict[str, Any]]:
    vectors = embed_texts([query], embed_model)
    if not vectors:
        return []
    if not index.get("chunks"):
        return []
    query_vec = np.array([vectors[0]], dtype=np.float32)
    matrix = _chunk_embedding_matrix(index)
    if matrix.size == 0:
        return []
    sims = cosine_sim(matrix, query_vec).reshape(-1)
    top_indices = np.argsort(-sims)[:top_k]
    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        chunk = dict(index["chunks"][int(idx)])
        chunk["score"] = float(sims[int(idx)])
        results.append(chunk)
    return results


SYSTEM_PROMPT = (
    "당신은 한국의 물류 관련 법령을 정확히 요약하는 전문 분석가입니다. "
    "모든 주장에 출처를 붙이고, 과장 없이 사실만 전달하세요."
)


def _format_context(context: Dict[str, Any]) -> str:
    return (
        f"[출처: {context['file']}, p.{context['page_start']}-{context['page_end']}, "
        f"점수 {context['score']:.3f}]\n{context['text']}"
    )


def build_user_prompt(query: str, contexts: Sequence[Dict[str, Any]]) -> str:
    context_blob = "\n\n".join(_format_context(ctx) for ctx in contexts)
    instructions = (
        "당신은 한국의 물류 관련 법령과 고시를 분석해 보고서를 작성합니다.\n"
        "- 보고서는 개요 → 핵심 조항 요약 → 시사점/실무 체크리스트 순으로 구성합니다.\n"
        "- 각 문장 뒤에는 [문서, 페이지] 형태로 출처를 표기합니다.\n"
        "- 제공된 근거가 부족하면 '추가 근거 필요'라고 명확히 밝힙니다.\n"
        "- 최신 개정일·고시번호가 보이면 함께 적어 주세요."
    )
    return (
        f"{instructions}\n\n[검색 질의]\n{query}\n\n[컨텍스트]\n{context_blob}"
    )


def chat_with_context(
    history_turns: Iterable[Tuple[str, str]],
    query: str,
    contexts: Sequence[Dict[str, Any]],
    chat_model: str,
    temperature: float = 0.2,
) -> str:
    client = _ensure_openai()
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_text, assistant_text in history_turns:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": build_user_prompt(query, contexts)})

    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def query_rag(
    index_root: str,
    query: str,
    embed_model: str,
    chat_model: str,
    top_k: int = 8,
    force_rebuild: bool = False,
    history_turns: Iterable[Tuple[str, str]] | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    history_turns = list(history_turns or [])
    index = build_or_load_index(index_root, embed_model, force_rebuild)
    contexts = search_index(index, query, embed_model, top_k)
    if not contexts:
        return "관련 컨텍스트를 찾지 못했습니다.", []
    answer = chat_with_context(history_turns, query, contexts, chat_model)
    return answer, contexts
