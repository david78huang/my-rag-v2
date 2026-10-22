#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ingest_mineru_qdrant_full.py

读取 MinerU 已经解析完成的输出目录，完成：
1) 解析 markdown / content_list / images / metadata
2) 文本脏数据检测与清洗
3) 可选调用本地 LLM 做二次清洗
4) 使用 BGE-M3 向量化
5) 写入 Qdrant
6) 保留 raw_text / image_path / source files metadata，便于后续追查与展示

适配场景：
- 情况 B：MinerU 输出文件都在同一目录，且以 pdf 文件名为前缀
  例如：
    test1.md
    test1_content_list.json
    test1_content_list_v2.json
    test1_middle.json
    test1_model.json
    test1_layout.pdf
    test1.pdf
    images/

运行示例：
python ingest_mineru_qdrant_full.py ^
  --mineru_dir D:\docker_hub\AI_Project\pdf_output ^
  --qdrant_url http://127.0.0.1:6333 ^
  --collection mineru_docs ^
  --embed_model BAAI/bge-m3 ^
  --llm_api http://127.0.0.1:18081/v1/chat/completions ^
  --llm_model gemma-4-E4B ^
  --enable_llm_clean

依赖：
pip install qdrant-client sentence-transformers requests
"""

import argparse
import hashlib
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


# -----------------------------
# 基础工具
# -----------------------------
def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def safe_read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(errors="ignore")


def safe_read_json(path: Path) -> Any:
    try:
        return json.loads(safe_read_text(path))
    except Exception:
        return None


def normalize_path_str(p: Path) -> str:
    return str(p.resolve())


# -----------------------------
# 脏文本检测 / 清洗
# -----------------------------
RE_REPEAT_CHAR = re.compile(r"(.)\1{5,}")
RE_SPACED_LETTERS = re.compile(r"\b(?:[A-Za-z0-9]\s+){3,}[A-Za-z0-9]\b")
RE_NON_WORD_HEAVY = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff\s\.,;:\-\(\)\[\]/_%#&]+")


COMMON_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "will", "node",
    "rack", "network", "figure", "table", "system", "power", "switch", "gpu",
    "cpu", "memory", "storage", "overview", "introduction", "specifications",
    "configuration", "component", "vera", "rubin", "compute", "cable", "tray",
    "internal", "port", "transceiver", "nvl72"
}


def fix_spaced_letters(text: str) -> str:
    def repl(match):
        raw = match.group(0)
        chars = raw.split()
        return "".join(chars)
    return RE_SPACED_LETTERS.sub(repl, text)


def collapse_repeated_chars(text: str, keep: int = 2) -> str:
    def repl(match):
        ch = match.group(1)
        return ch * keep
    return RE_REPEAT_CHAR.sub(repl, text)


def basic_clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = fix_spaced_letters(text)
    text = collapse_repeated_chars(text, keep=2)

    # 去掉控制字符
    text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", " ", text)

    # 多空格规整
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 常见目录点线/噪声
    text = re.sub(r"\.{6,}", " ...... ", text)
    text = re.sub(r":{4,}", " : ", text)

    return text.strip()


def tokenize_rough(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text.lower())


def calc_text_quality(text: str) -> Dict[str, Any]:
    raw = text or ""
    if not raw.strip():
        return {
            "length": 0,
            "alpha_num_ratio": 0.0,
            "repeat_runs": 0,
            "spaced_letter_hits": 0,
            "non_word_ratio": 1.0,
            "readable_word_ratio": 0.0,
            "avg_token_len": 0.0,
            "is_dirty": True,
            "reasons": ["empty"],
        }

    tokens = tokenize_rough(raw)
    text_len = len(raw)
    alnum_count = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", raw))
    repeat_runs = len(RE_REPEAT_CHAR.findall(raw))
    spaced_letter_hits = len(RE_SPACED_LETTERS.findall(raw))
    non_word_chars = len(RE_NON_WORD_HEAVY.findall(raw))
    common_hits = sum(1 for t in tokens if t in COMMON_WORDS or len(t) >= 3)
    avg_token_len = statistics.mean([len(t) for t in tokens]) if tokens else 0.0

    alpha_num_ratio = alnum_count / max(text_len, 1)
    non_word_ratio = non_word_chars / max(text_len, 1)
    readable_word_ratio = common_hits / max(len(tokens), 1)

    reasons = []
    if repeat_runs >= 2:
        reasons.append("too_many_repeated_chars")
    if spaced_letter_hits >= 1:
        reasons.append("letters_split_by_spaces")
    if alpha_num_ratio < 0.35:
        reasons.append("low_alnum_ratio")
    if non_word_ratio > 0.10:
        reasons.append("high_nonword_ratio")
    if len(tokens) >= 20 and readable_word_ratio < 0.35:
        reasons.append("few_readable_words")
    if len(raw) >= 120 and avg_token_len <= 1.8:
        reasons.append("tokens_too_fragmented")

    is_dirty = len(reasons) > 0
    return {
        "length": text_len,
        "alpha_num_ratio": round(alpha_num_ratio, 4),
        "repeat_runs": repeat_runs,
        "spaced_letter_hits": spaced_letter_hits,
        "non_word_ratio": round(non_word_ratio, 4),
        "readable_word_ratio": round(readable_word_ratio, 4),
        "avg_token_len": round(avg_token_len, 4),
        "is_dirty": is_dirty,
        "reasons": reasons,
    }


# -----------------------------
# 本地 LLM 清洗
# -----------------------------
def llm_clean_text(
    raw_text: str,
    llm_api: str,
    llm_model: str,
    timeout: int = 120,
) -> str:
    prompt = f"""
You are a document cleaning assistant.

Please clean the following OCR / extracted text.

Requirements:
1. Preserve factual meaning.
2. Remove obvious OCR garbage like repeated letters: sssssss
3. Merge split letters like: i t e m 1 -> item 1
4. Keep tables readable if possible.
5. Do NOT hallucinate missing facts.
6. Output only cleaned text, no explanation.

RAW TEXT:
{raw_text}
""".strip()

    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 4000,
    }
    headers = {"Content-Type": "application/json"}
    r = requests.post(llm_api, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return raw_text


# -----------------------------
# MinerU 输出定位
# -----------------------------
def find_mineru_primary_dir(mineru_dir: Path) -> Dict[str, Path]:
    """
    适配情况B：
      同目录下文件以 pdf 文件名为前缀
      例如 test1.md / test1_content_list.json / test1.pdf / images/

    返回：
      {
        "base_dir": ...,
        "prefix": ...,
        "markdown": ...,
        "content_list": ...,
        "content_list_v2": ...,
        "middle": ...,
        "model": ...,
        "layout_pdf": ...,
        "original_pdf": ...,
        "images_dir": ...
      }
    """
    mineru_dir = mineru_dir.resolve()
    if not mineru_dir.exists():
        raise FileNotFoundError(f"mineru_dir not found: {mineru_dir}")

    files = [p for p in mineru_dir.iterdir() if p.is_file()]
    images_dir = mineru_dir / "images"
    pdf_files = [p for p in files if p.suffix.lower() == ".pdf"]

    # 优先找“不是 layout.pdf 的原始 pdf”
    original_pdf = None
    for p in pdf_files:
        if not p.name.lower().endswith("_layout.pdf") and p.name.lower() != "layout.pdf":
            original_pdf = p
            break

    if original_pdf is None:
        # 退化情况：随便挑一个 pdf
        original_pdf = pdf_files[0] if pdf_files else None

    prefix = None
    if original_pdf is not None:
        prefix = original_pdf.stem

    # 如果目录中没有 original pdf，则尝试从 md 推断 prefix
    if prefix is None:
        md_files = [p for p in files if p.suffix.lower() == ".md"]
        if not md_files:
            raise RuntimeError(f"Cannot infer MinerU prefix in {mineru_dir}")
        prefix = md_files[0].stem

    def find_prefixed_file(name_suffix: str, exact_fallback: Optional[str] = None) -> Optional[Path]:
        cand = mineru_dir / f"{prefix}{name_suffix}"
        if cand.exists():
            return cand
        if exact_fallback:
            fallback = mineru_dir / exact_fallback
            if fallback.exists():
                return fallback
        for p in files:
            if p.name.startswith(prefix) and p.name.endswith(name_suffix):
                return p
        return None

    markdown = find_prefixed_file(".md")
    content_list = find_prefixed_file("_content_list.json", "content_list.json")
    content_list_v2 = find_prefixed_file("_content_list_v2.json", "content_list_v2.json")
    middle = find_prefixed_file("_middle.json", "middle.json")
    model = find_prefixed_file("_model.json", "model.json")
    layout_pdf = find_prefixed_file("_layout.pdf", "layout.pdf")

    if markdown is None and content_list is None and content_list_v2 is None:
        raise RuntimeError(f"No usable MinerU outputs found in {mineru_dir}")

    return {
        "base_dir": mineru_dir,
        "prefix": Path(prefix),
        "markdown": markdown,
        "content_list": content_list,
        "content_list_v2": content_list_v2,
        "middle": middle,
        "model": model,
        "layout_pdf": layout_pdf,
        "original_pdf": original_pdf,
        "images_dir": images_dir if images_dir.exists() else mineru_dir,
    }


# -----------------------------
# 内容解析
# -----------------------------
def split_markdown_to_chunks(markdown_text: str, max_chars: int = 1200) -> List[Dict[str, Any]]:
    markdown_text = markdown_text.strip()
    if not markdown_text:
        return []

    blocks = re.split(r"\n\s*\n", markdown_text)
    out = []
    chunk_index = 0

    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue

        if len(blk) <= max_chars:
            out.append({
                "type": "text",
                "text": blk,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
            continue

        start = 0
        while start < len(blk):
            piece = blk[start:start + max_chars]
            if start + max_chars < len(blk):
                cut = piece.rfind("\n")
                if cut < 200:
                    cut = piece.rfind(" ")
                if cut > 0:
                    piece = piece[:cut]
            out.append({
                "type": "text",
                "text": piece.strip(),
                "chunk_index": chunk_index,
            })
            chunk_index += 1
            start += max(len(piece), 1)

    return out


def parse_content_list_records(records: Any, images_dir: Path) -> List[Dict[str, Any]]:
    """
    兼容 MinerU content_list / content_list_v2 的不同结构
    """
    chunks: List[Dict[str, Any]] = []
    if not isinstance(records, list):
        return chunks

    chunk_index = 0
    for item in records:
        if not isinstance(item, dict):
            continue

        # 常见 page 字段
        page = item.get("page_idx", item.get("page", item.get("page_id")))

        # 文本
        text = item.get("text") or item.get("content") or item.get("md_content") or ""
        if isinstance(text, str) and text.strip():
            chunks.append({
                "type": item.get("type", "text"),
                "text": text.strip(),
                "page": page,
                "chunk_index": chunk_index,
                "raw_item": item,
            })
            chunk_index += 1

        # 表格
        table_text = item.get("table") or item.get("table_text") or item.get("table_caption")
        if isinstance(table_text, str) and table_text.strip():
            chunks.append({
                "type": "table",
                "text": table_text.strip(),
                "page": page,
                "chunk_index": chunk_index,
                "raw_item": item,
            })
            chunk_index += 1

        # 图片
        img_fields = [
            item.get("img_path"),
            item.get("image_path"),
            item.get("path"),
            item.get("figure_path"),
        ]
        for imgp in img_fields:
            if not imgp:
                continue
            img_path = Path(imgp)
            if not img_path.is_absolute():
                img_path = images_dir / img_path.name if (images_dir / img_path.name).exists() else images_dir / imgp
            chunks.append({
                "type": "image",
                "text": item.get("caption") or item.get("figure_caption") or item.get("text") or "",
                "page": page,
                "chunk_index": chunk_index,
                "image_path": normalize_path_str(img_path),
                "raw_item": item,
            })
            chunk_index += 1
            break

    return chunks


def parse_mineru_output(primary: Dict[str, Path]) -> List[Dict[str, Any]]:
    """
    返回统一 chunks:
      {
        type, text, raw_text, page, chunk_index, image_path?, ...
      }
    """
    chunks: List[Dict[str, Any]] = []

    markdown = primary.get("markdown")
    content_list = primary.get("content_list")
    content_list_v2 = primary.get("content_list_v2")
    images_dir = primary.get("images_dir")
    original_pdf = primary.get("original_pdf")

    # 优先用 content_list_v2
    if content_list_v2 and content_list_v2.exists():
        data = safe_read_json(content_list_v2)
        chunks.extend(parse_content_list_records(data, images_dir))

    # 再补 content_list
    if content_list and content_list.exists():
        data = safe_read_json(content_list)
        more = parse_content_list_records(data, images_dir)
        if more:
            chunks.extend(more)

    # 如果 json 没有足够内容，就用 md 补
    if markdown and markdown.exists():
        md_text = safe_read_text(markdown)
        md_chunks = split_markdown_to_chunks(md_text, max_chars=1200)
        if md_chunks:
            existing_hashes = {sha1_text(c.get("text", "")) for c in chunks if c.get("text")}
            for c in md_chunks:
                if sha1_text(c.get("text", "")) not in existing_hashes:
                    chunks.append(c)

    # 补基础元数据
    final_chunks = []
    for i, c in enumerate(chunks):
        raw_text = c.get("text", "") or ""
        final_chunks.append({
            "type": c.get("type", "text"),
            "text": raw_text,
            "raw_text": raw_text,
            "page": c.get("page"),
            "chunk_index": c.get("chunk_index", i),
            "image_path": c.get("image_path"),
            "source_pdf": normalize_path_str(original_pdf) if original_pdf else None,
            "mineru_markdown": normalize_path_str(markdown) if markdown and markdown.exists() else None,
            "mineru_content_list": normalize_path_str(content_list) if content_list and content_list.exists() else None,
            "mineru_content_list_v2": normalize_path_str(content_list_v2) if content_list_v2 and content_list_v2.exists() else None,
            "raw_item": c.get("raw_item"),
        })

    return final_chunks


# -----------------------------
# 文本后处理
# -----------------------------
def clean_chunk_text(
    raw_text: str,
    enable_llm_clean: bool = False,
    llm_api: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    basic = basic_clean_text(raw_text)
    quality = calc_text_quality(basic)

    cleaned = basic
    llm_used = False

    if quality["is_dirty"] and enable_llm_clean and llm_api and llm_model:
        try:
            llm_out = llm_clean_text(basic, llm_api=llm_api, llm_model=llm_model)
            llm_out = basic_clean_text(llm_out)
            if llm_out:
                cleaned = llm_out
                llm_used = True
                quality = calc_text_quality(cleaned)
        except Exception as e:
            quality["llm_clean_error"] = str(e)

    meta = {
        "quality": quality,
        "llm_clean_used": llm_used,
    }
    return cleaned, meta


# -----------------------------
# 向量存储
# -----------------------------
def ensure_qdrant_collection(client: QdrantClient, collection: str, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def build_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 16) -> List[List[float]]:
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vecs]


def make_point_id(source_pdf: str, chunk_type: str, page: Any, chunk_index: Any, text: str) -> str:
    raw = f"{source_pdf}::{chunk_type}::p{page}::c{chunk_index}::{sha1_text(text)}"
    return sha1_text(raw)


def ingest_to_qdrant(
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    chunks: List[Dict[str, Any]],
    batch_size: int = 16,
):
    if not chunks:
        print("[WARN] no chunks to ingest")
        return

    texts = [c["cleaned_text"] for c in chunks]
    vectors = embed_texts(embedder, texts, batch_size=batch_size)
    dim = len(vectors[0])
    ensure_qdrant_collection(client, collection, dim)

    points = []
    for c, v in zip(chunks, vectors):
        pid = make_point_id(
            c.get("source_pdf") or "",
            c.get("type") or "text",
            c.get("page"),
            c.get("chunk_index"),
            c.get("cleaned_text") or "",
        )
        payload = {
            "type": c.get("type"),
            "page": c.get("page"),
            "chunk_index": c.get("chunk_index"),
            "source_pdf": c.get("source_pdf"),
            "image_path": c.get("image_path"),
            "cleaned_text": c.get("cleaned_text"),
            "raw_text": c.get("raw_text"),
            "quality": c.get("quality"),
            "llm_clean_used": c.get("llm_clean_used", False),
            "mineru_markdown": c.get("mineru_markdown"),
            "mineru_content_list": c.get("mineru_content_list"),
            "mineru_content_list_v2": c.get("mineru_content_list_v2"),
        }
        points.append(PointStruct(id=pid, vector=v, payload=payload))

    client.upsert(collection_name=collection, points=points)
    print(f"[DONE] upserted {len(points)} points into collection={collection}")


# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mineru_dir", required=True, help="MinerU 输出目录")
    parser.add_argument("--qdrant_url", default="http://127.0.0.1:6333")
    parser.add_argument("--qdrant_api_key", default=None)
    parser.add_argument("--collection", default="mineru_docs")
    parser.add_argument("--embed_model", default="BAAI/bge-m3")
    parser.add_argument("--enable_llm_clean", action="store_true")
    parser.add_argument("--llm_api", default=None, help="OpenAI-compatible local API endpoint, e.g. http://127.0.0.1:18081/v1/chat/completions")
    parser.add_argument("--llm_model", default="gemma-4-E4B")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dump_clean_json", default=None, help="可选，导出清洗结果 json")
    args = parser.parse_args()

    mineru_dir = Path(args.mineru_dir)
    primary = find_mineru_primary_dir(mineru_dir)

    print("[INFO] detected MinerU outputs:")
    for k, v in primary.items():
        print(f"  - {k}: {v}")

    parsed_chunks = parse_mineru_output(primary)
    print(f"[INFO] parsed chunks: {len(parsed_chunks)}")

    processed = []
    for c in parsed_chunks:
        raw_text = c.get("raw_text", "") or ""
        chunk_type = c.get("type")

        # image 没文本时，至少给 caption 占位，确保可检索 & 可显示
        if chunk_type == "image" and not raw_text.strip():
            raw_text = f"[IMAGE] {Path(c.get('image_path') or '').name}"

        cleaned_text, clean_meta = clean_chunk_text(
            raw_text=raw_text,
            enable_llm_clean=args.enable_llm_clean,
            llm_api=args.llm_api,
            llm_model=args.llm_model,
        )

        # 太脏且清洗后为空，跳过
        if not cleaned_text.strip():
            continue

        item = dict(c)
        item["raw_text"] = raw_text
        item["cleaned_text"] = cleaned_text
        item["quality"] = clean_meta["quality"]
        item["llm_clean_used"] = clean_meta["llm_clean_used"]
        processed.append(item)

    print(f"[INFO] processed chunks: {len(processed)}")

    if args.dump_clean_json:
        dump_path = Path(args.dump_clean_json)
        dump_path.write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] dumped cleaned json -> {dump_path}")

    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
    )
    embedder = build_embedder(args.embed_model)

    ingest_to_qdrant(
        client=client,
        collection=args.collection,
        embedder=embedder,
        chunks=processed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()