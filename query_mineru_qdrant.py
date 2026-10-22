# query_mineru_qdrant.py
import argparse
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

from llm_clients import call_local_llm, call_remote_llm
import base64
from pathlib import Path
from pydantic import BaseModel
# Memory store
from fastapi import FastAPI
from db.postgres import PostgresClient
from memory.manager import MemoryManager
from memory.sementic import SemanticMemory
import logging

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
COLLECTION_NAME = "mineru_docs"
MEM_COLLECTION_NAME = "memory"

import base64
from pathlib import Path

logger = logging.getLogger("uvicorn, error")

class ChatRequest(BaseModel):
    user_id: str
    message: str
    qdrant_url: str
    collection: str

def image_to_data_url(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return ""
    suffix = p.suffix.lower()
    mime = "image/jpeg"
    if suffix in [".png"]:
        mime = "image/png"
    elif suffix in [".webp"]:
        mime = "image/webp"
    elif suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"

    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def normalize_text_for_dedup(text: str) -> str:
    import re
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text

def retrieve(
    question: str,
    qdrant_url: str,
    collection_name: str,
    topk: int = 12,
) -> List[Dict[str, Any]]:
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    client = QdrantClient(url=qdrant_url)
    qvec = embedder.encode([question], normalize_embeddings=True)[0]

    try:
        resp = client.query_points(
            collection_name=collection_name,
            query=qvec.tolist(),
            limit=topk * 20,
            with_payload=True,
        )
        hits = getattr(resp, "points", resp)
    except AttributeError:
        hits = client.search(
            collection_name=collection_name,
            query_vector=qvec.tolist(),
            limit=topk * 20,
            with_payload=True,
        )

    out = []
    seen = set()
    for h in hits:
        payload = h.payload or {}
        if payload.get("type") in {"footer", "page_footer", "page_header", "page_number", "title"}:
            continue
        text = payload.get("cleaned_text") or payload.get("raw_text") or payload.get("text", "")
        norm = normalize_text_for_dedup(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append({
            "score": getattr(h, "score", None),
            "text": payload.get("cleaned_text") or payload.get("raw_text") or payload.get("text", ""),
            "metadata": payload,
        })

        if len(out) >= topk:
            break
    #print("out:", out)
    return out


def rerank(question: str, docs: List[Dict[str, Any]], final_topk: int = 12) -> List[Dict[str, Any]]:
    if not docs:
        return []

    model = CrossEncoder(RERANK_MODEL_NAME)
    pairs = [(question, d["text"]) for d in docs]
    scores = model.predict(pairs)

    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)

    docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    # rerank 后再去重一次
    final = []
    seen = set()
    for d in docs:
        norm = normalize_text_for_dedup(d["text"])
        if norm in seen:
            continue
        seen.add(norm)
        final.append(d)
        if len(final) >= final_topk:
            break

    return final


def format_doc_block(i: int, d: Dict[str, Any]) -> str:
    meta = d.get("metadata", {})
    text = d.get("text", "") or ""

    lines = [
        f"[{i}]",
        f"type={meta.get('type')}",
        f"page={meta.get('page')}",
    ]

    # 你当前数据里通常没有 file 字段，所以这里做兼容
    if meta.get("source_pdf"):
        lines.append(f"source_pdf={meta.get('source_pdf')}")
    elif meta.get("file"):
        lines.append(f"file={meta.get('file')}")

    if meta.get("image_path"):
        lines.append(f"image_path={meta.get('image_path')}")

    if meta.get("quality"):
        lines.append(f"quality={meta.get('quality')}")

    if meta.get("llm_clean_used") is not None:
        lines.append(f"llm_clean_used={meta.get('llm_clean_used')}")

    # 取更有用的正文
    body_parts = []
    cleaned_text = meta.get("cleaned_text")
    raw_text = meta.get("raw_text")

    if cleaned_text and cleaned_text.strip():
        body_parts.append("CLEANED_TEXT:")
        body_parts.append(cleaned_text.strip())

    # 如果 cleaned_text 不够，再补 raw_text
    if raw_text and raw_text.strip() and raw_text.strip() != cleaned_text.strip():
        body_parts.append("")
        body_parts.append("RAW_TEXT:")
        body_parts.append(raw_text.strip())

    # 兜底：retrieve() 的 d["text"]
    if not body_parts and text.strip():
        body_parts.append("TEXT:")
        body_parts.append(text.strip())

    lines.append("CONTENT:")
    lines.append("\n".join(body_parts).strip())

    return "\n".join(lines).strip()

def build_messages(question: str, docs: List[Dict[str, Any]], profile: dict, memories: str) -> List[Dict[str, Any]]:
    text_blocks = []
    image_urls = []

    for i, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        text = d.get("text", "") or ""
        header = [
            f"[{i}]",
            f"type={meta.get('type')}",
            f"page={meta.get('page')}",
        ]
        if meta.get("source_pdf"):
            header.append(f"source_pdf={meta.get('source_pdf')}")

        block = "\n".join([
            " | ".join(header),
            text,
        ])
        text_blocks.append(block)

        if meta.get("type") == "image" and meta.get("image_path"):
            data_url = image_to_data_url(meta["image_path"])
            if data_url:
                image_urls.append(data_url)

    context_text = "\n\n---\n\n".join(text_blocks)
    profile_text = "\n".join([f"{k}: {v}" for k, v in profile.items()])
    content = [
        {
            "type": "text",
            "text": f"""
You are a precise and professional assistant.
answer need be accurate and details.
Use the provided context as evidence.
Do not answer using only page numbers.
Prefer quoting or paraphrasing the actual content.
If the answer is uncertain, say so.
Cite sources as [1], [2], [3], [4], [5], [6] [7] [8] [9] [10] [11] [12] etc.
PROFILE:
{profile_text}

MEMORY:
{memories}

CONTEXT:
{context_text}

QUESTION:
{question}
""".strip()
        }
    ]

    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })
    return [{
        "role": "user",
        "content": content
    }]

def build_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    blocks = [format_doc_block(i, d) for i, d in enumerate(docs, 1)]
    context = "\n\n---\n\n".join(blocks)

    return f"""
You are a precise and professional assistant.
answer need be accurate and details.
Use the provided context as evidence.
Do not answer using only page numbers.
Prefer quoting or paraphrasing the actual content.
If the answer is uncertain, say so.
Cite sources as [1], [2], [3], [4], [5], [6] [7] [8] etc.

CONTEXT:
{context}

QUESTION:
{question}
""".strip()



def ask(question: str, qdrant_url: str, collection_name: str, req: ChatRequest, pg: PostgresClient, mem: MemoryManager):
    user_id = req.user_id
    message = req.message
    # Profile
    profile = pg.get(user_id)

    # Memory
    memories = mem.semantic.search(user_id, message, MEM_COLLECTION_NAME, topk=5)

    docs = retrieve(question, qdrant_url, collection_name, topk=12)
    docs = rerank(question, docs, final_topk=12)

    prompt = build_messages(question, docs, profile, memories)
    #answer = call_local_llm(prompt)
    answer = call_remote_llm(prompt)

    mem.store(user_id, message + "\n" + answer)

    return {
        "prompt": prompt[0]["content"][0]["text"],
        "answer": answer,
        "docs": docs,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--qdrant_url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    args = parser.parse_args()

    result = ask(args.question, args.qdrant_url, args.collection)
    print("\n=== prompt ===\n")
    print(result["prompt"])
    print("\n=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== SOURCES ===\n")
    for i, d in enumerate(result["docs"], 1):
        meta = d["metadata"]
        print(f"[{i}] type={meta.get('type')} file={meta.get('source_pdf')} page={meta.get('page')}")
        if meta.get("image_path"):
            print(f"    image_path={meta.get('image_path')}")
        print(f"    rerank_score={d.get('rerank_score')}")
        #print(f"    text={d['text'][:300]}")
        print()


if __name__ == "__main__":
    main()