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
import os.path as osp
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
import uuid
import base64
import time

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
    timeout: int = 360,
    image_path = None
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
    image_field = None
    try:
        p = Path(image_path)
        if p.exists() and p.is_file():
            # 以二进制方式读取，避免 UnicodeDecodeError
            with open(p, "rb") as f:
                b = f.read()
            b64 = base64.b64encode(b).decode()
            image_field = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            }
    except Exception as e:
        # 读取图片失败时记录但继续以文本方式调用 LLM
        print(f"[WARN] cannot read image file {image_path}: {e}")
        image_field = None

    # 构造 messages：总是包含清洗 prompt；如果 image_field 非空则一并发送
    content_list = [{"type": "text", "text": prompt}]
    if image_field:
        content_list.append(image_field)

    payload = {
            "model": llm_model,
            "messages": [{
                "role": "user",
                "content": content_list
            }],
            "temperature": 0.0,
            "stream": False,
            "max_tokens": 128000,
        }
    headers = {"Content-Type": "application/json"}
    r = requests.post(llm_api, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    print("payload:", payload)
    print("data:                 ", data)

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
    #print("mineru_dir when before:", mineru_dir)
    if not mineru_dir.exists():
        raise FileNotFoundError(f"mineru_dir not found: {mineru_dir}")
    mineru_dir = Path(osp.join(mineru_dir.resolve(), mineru_dir.name, "hybrid_auto"))
    #print("mineru_dir when after:", mineru_dir)
    files = [p for p in mineru_dir.iterdir() if p.is_file()]
    images_dir = mineru_dir / "images"
    pdf_files = [p for p in files if p.suffix.lower() == ".pdf"]
    #print("pdf files:", pdf_files)
    # 优先找“不是 layout.pdf 的原始 pdf”
    original_pdf = None
    for p in pdf_files:
        if not p.name.lower().endswith("_layout.pdf") and p.name.lower() != "layout.pdf":
            original_pdf = p
            break
    #print("original pdf name: {}".format(original_pdf))
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

    def find_prefixed_file(prefix: str, name_suffix: str, exact_fallback: Optional[str] = None) -> Optional[Path]:
        if prefix.endswith("origin"):
            prefix = prefix.replace("origin","")
        prefix = prefix[:-1]
        cand = mineru_dir / f"{prefix}{name_suffix}"
        #print("cand:  {}   ".format(cand))
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

    markdown = find_prefixed_file(prefix, ".md")
    content_list = find_prefixed_file(prefix, "_content_list.json", "content_list.json")
    content_list_v2 = find_prefixed_file(prefix,"_content_list_v2.json", "content_list_v2.json")
    middle = find_prefixed_file(prefix,"_middle.json", "middle.json")
    model = find_prefixed_file(prefix,"_model.json", "model.json")
    layout_pdf = find_prefixed_file(prefix,"_layout.pdf", "layout.pdf")

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

def split_structured_text(text: str, source_format: str = "markdown", max_chars: int = 1200) -> List[Dict[str, Any]]:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        HTMLHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    text = (text or "").strip()
    if not text:
        return []

    chunks: List[Dict[str, Any]] = []

    if source_format == "markdown":
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        docs = header_splitter.split_text(text)

        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=150,
        )

        chunk_index = 0
        for doc in docs:
            parts = char_splitter.split_text(doc.page_content)
            for part in parts:
                chunks.append({
                    "type": "text",
                    "text": part.strip(),
                    "raw_text": part.strip(),
                    "chunk_index": chunk_index,
                    "headers": dict(doc.metadata or {}),
                    "source_format": "markdown",
                })
                chunk_index += 1

    elif source_format == "html":
        headers_to_split_on = [
            ("h1", "h1"),
            ("h2", "h2"),
            ("h3", "h3"),
            ("h4", "h4"),
        ]
        header_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        docs = header_splitter.split_text(text)

        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=150,
        )

        chunk_index = 0
        for doc in docs:
            parts = char_splitter.split_text(doc.page_content)
            for part in parts:
                chunks.append({
                    "type": "text",
                    "text": part.strip(),
                    "raw_text": part.strip(),
                    "chunk_index": chunk_index,
                    "headers": dict(doc.metadata or {}),
                    "source_format": "html",
                })
                chunk_index += 1

    else:
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=150,
        )
        parts = char_splitter.split_text(text)
        for i, part in enumerate(parts):
            chunks.append({
                "type": "text",
                "text": part.strip(),
                "raw_text": part.strip(),
                "chunk_index": i,
                "headers": {},
                "source_format": source_format,
            })

    return chunks
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
        #print("blk:...", blk)
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

def parse_content_list_records_v1(records: Any, images_dir: Path) ->List[Dict[str, Any]]:
    """
        兼容 MinerU content_list / content_list_v2 的不同结构
        """
    chunks: List[Dict[str, Any]] = []
    if not isinstance(records, list):
        return chunks
    skip_type = ["header", "page_number", "footer"]
    chunk_index = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        if item.get("type", "") in skip_type:
            continue
        # 常见 page 字段
        page = item.get("page_idx", item.get("page", item.get("page_id")))
        if not page:
            continue
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


def parse_content_list_records(records: Any, images_dir: Path) -> List[Dict[str, Any]]:
    """
    兼容 MinerU content_list_v2:
    顶层是按页分组的 list[list[block]]
    也兼容旧格式 list[block]
    """
    chunks: List[Dict[str, Any]] = []
    #print("records", records)
    if not isinstance(records, list):
        return chunks
    chunk_index = 0
    # 判断是否为按页结构
    is_page_grouped = bool(records) and all(isinstance(page_blocks, list) for page_blocks in records)
    if is_page_grouped:
        page_iter = enumerate(records, start=1)
    else:
        page_iter = [(None, records)]
    for page_no, blocks in page_iter:
        if not isinstance(blocks, list):
            continue
        for item in blocks:
            if not isinstance(item, dict):
                continue

            block_type = item.get("type", "text")
            bbox = item.get("bbox")
            anchor = item.get("anchor")
            sub_type = item.get("sub_type")

            # 页码优先取 item 自带，否则取外层 page_no
            page = item.get("page_idx", item.get("page", item.get("page_id", page_no)))
            if not page:
                raise("Page number can't be empty!!!  ", item)
            text = extract_text_from_block_v2(item)
            #print("item: ", item)
            image_path = extract_image_path_from_block(item, images_dir)
            #print(images_dir, image_path)
            # 对 image/table/chart，就算文本很少，也尽量保留
            if text == "SKIP" or block_type in ["footer", "page_number"]:
                continue
            if not text:
                text = ""
            if text.strip() or image_path or block_type in {"image", "table", "chart", "equation_interline"}:
                chunks.append({
                    "type": block_type,
                    "text": text.strip(),
                    "page": page,
                    "chunk_index": chunk_index,
                    "image_path": image_path,
                    "bbox": bbox,
                    "anchor": anchor,
                    "sub_type": sub_type,
                    "raw_item": item,
                })
                chunk_index += 1
    return chunks

def parse_mineru_output(primary: Dict[str, Path]) -> List[Dict[str, Any]]:
    """
    返回统一 chunks:
      {type, text, raw_text, page, chunk_index, image_path?, ...}
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
        more = parse_content_list_records(data, images_dir)
        if more:
            chunks.extend(more)
            #print("content_list_v2 chunk:", more)

    # 再补 content_list
    if content_list and content_list.exists():
        data = safe_read_json(content_list)
        more = parse_content_list_records_v1(data, images_dir)
        if more:
            chunks.extend(more)
            #print("content_list_v1 chunk:", more)

    # 如果 json 没有足够内容，就用 md 补
    if markdown and markdown.exists():
        md_text = safe_read_text(markdown)
        #md_chunks = split_markdown_to_chunks(md_text, max_chars=1200)
        md_chunks = split_structured_text(md_text)
        if md_chunks:
            existing_hashes = {sha1_text(c.get("text", "")) for c in chunks if c.get("text")}
            for c in md_chunks:
                if sha1_text(c.get("text", "")) not in existing_hashes:
                    chunks.append(c)

    # 补基础元数据
    final_chunks = []
    for i, c in enumerate(chunks):
        raw_text = c.get("text", "") or ""
        #if c.get("type", "text") == "image":
            #print("raw_test:", raw_text)
        final_chunks.append({
            "type": c.get("type", "text"),
            "sub_type": c.get("sub_type"),
            "text": raw_text,
            "raw_text": raw_text,
            "page": c.get("page"),
            "chunk_index": c.get("chunk_index", i),
            "image_path": c.get("image_path"),
            "bbox": c.get("bbox"),
            "anchor": c.get("anchor"),
            "source_pdf": normalize_path_str(original_pdf) if original_pdf else None,
            "mineru_markdown": normalize_path_str(markdown) if markdown and markdown.exists() else None,
            "mineru_content_list": normalize_path_str(content_list) if content_list and content_list.exists() else None,
            "mineru_content_list_v2": normalize_path_str(
                content_list_v2) if content_list_v2 and content_list_v2.exists() else None,
            "raw_item": c.get("raw_item"),
        })

    return final_chunks

def extract_inline_text_v2(spans: Any) -> str:
    """提取 MinerU 行内 span 列表中的可读文本
    支持: - {"type": "text", "content": "..."}
        - {"type": "hyperlink", "content": "...", "children": [...]}
    """
    if not isinstance(spans, list):
        return ""
    parts = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        span_type = span.get("type")
        if span_type == "text":
            txt = span.get("content", "")
            if txt:
                parts.append(str(txt))
        elif span_type == "hyperlink":
            # 优先 children，因为说明里说 children 样式更精确
            children = span.get("children")
            if isinstance(children, list) and children:
                child_text = extract_inline_text_v2(children)
                if child_text:
                    parts.append(child_text)
                else:
                    txt = span.get("content", "")
                    if txt:
                        parts.append(str(txt))
            else:
                txt = span.get("content", "")
                if txt:
                    parts.append(str(txt))
        else:
            # 兜底：有 content 且是字符串就收
            txt = span.get("content")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
    return "".join(parts).strip()

def extract_text_from_block_v2(item: Dict[str, Any]) -> str:
    """根据 MinerU content_list_v2 的 type + content 结构提取文本
    """
    if not isinstance(item, dict):
        return ""
    block_type = item.get("type")
    content = item.get("content", {})
    if not isinstance(content, dict):
        return ""
    # 标题
    if block_type == "title":
        return extract_inline_text_v2(content.get("title_content", []))
    # 段落
    if block_type == "paragraph":
        return extract_inline_text_v2(content.get("paragraph_content", []))

    # 页脚注 / 页眉 / 页码 / aside / footnote
    if block_type in {
        "page_header", "page_footer", "page_number",
        "page_aside_text", "page_footnote", "footer"
    }:
        key_map = {
            "page_header": "page_header_content",
            "page_footer": "page_footer_content",
            "page_number": "page_number_content",
            "page_aside_text": "page_aside_text_content",
            "page_footnote": "page_footnote_content",
        }
        #return extract_inline_text_v2(content.get(key_map[block_type], []))
        return "SKIP"

    # 列表 / 索引
    if block_type in {"list", "index"}:
        items = content.get("list_items", [])
        texts = []
        if isinstance(items, list):
            for li in items:
                if isinstance(li, dict):
                    # 有些实现可能 item 内还有 content
                    txt = extract_inline_text_v2(li.get("item_content", []))
                    if not txt:
                        txt = extract_inline_text_v2(li.get("list_item_content", []))
                    if txt:
                        texts.append(txt)
                elif isinstance(li, str):
                    texts.append(li)
        return "\n".join(texts).strip()
    # 代码
    if block_type == "code":
        code_text = content.get("code_content", "")
        caption = extract_inline_text_v2(content.get("code_caption", []))
        footnote = extract_inline_text_v2(content.get("code_footnote", []))
        language = content.get("code_language", "")
        parts = []
        if caption:
            parts.append(f"Code: {caption}")
        if language:
            parts.append(f"Language: {language}")
        if code_text:
            parts.append(str(code_text))
        if footnote:
            parts.append(f"Note: {footnote}")
        return "\n".join(parts).strip()

    # 算法
    if block_type == "algorithm":
        algo_text = content.get("algorithm_content", "")
        caption = extract_inline_text_v2(content.get("algorithm_caption", []))
        footnote = extract_inline_text_v2(content.get("algorithm_footnote", []))
        parts = []
        if caption:
            parts.append(f"Algorithm: {caption}")
        if algo_text:
            parts.append(str(algo_text))
        if footnote:
            parts.append(f"Note: {footnote}")
        return "\n".join(parts).strip()

    # 行间公式
    if block_type == "equation_interline":
        math_text = content.get("math_content", "")
        math_type = content.get("math_type", "")
        if math_type:
            return f"[{math_type}] {math_text}".strip()
        return str(math_text).strip()

    # 表格
    if block_type == "table":
        parts = []

        # caption
        caption = extract_inline_text_v2(content.get("table_caption", []))
        if caption:
            parts.append(f"Table: {caption}")

        # html
        html = content.get("html", "")
        #print("html after extract:", html)
        if html:
            parts.append(f"html {html}.")
        # 如果有表格文本
        table_body = content.get("table_content")
        if isinstance(table_body, str) and table_body.strip():
            parts.append(table_body.strip())

        # footnote
        footnote = extract_inline_text_v2(content.get("table_footnote", []))
        if footnote:
            parts.append(f"Note: {footnote}")

        return "\n".join(parts).strip()
    # 图片 / 图表
    if block_type in {"image", "chart"}:
        parts = []
        #print("content:", content)
        image_content = content.get("content", "")
        if image_content and "David Huang" in image_content:
            image_content = image_content.replace("David Huang 103", "")
        if image_content:
            parts.append(f"Content: {image_content}")
        caption_keys = ["image_caption", "chart_caption", "figure_caption", "caption"]
        footnote_keys = ["image_footnote", "chart_footnote", "figure_footnote", "footnote"]

        for key in caption_keys:
            cap = content.get(key)
            if isinstance(cap, list):
                txt = extract_inline_text_v2(cap)
                if txt:
                    parts.append(f"Caption: {txt}")
                    break
            elif isinstance(cap, str) and cap.strip():
                parts.append(f"Caption: {cap.strip()}")
                break

        for key in footnote_keys:
            fn = content.get(key)
            if isinstance(fn, list):
                txt = extract_inline_text_v2(fn)
                if txt:
                    parts.append(f"Note: {txt}")
                    break
            elif isinstance(fn, str) and fn.strip():
                parts.append(f"Note: {fn.strip()}")
                break
        #print("parts:  ",parts)
        return "\n".join(parts).strip()
    # 兜底：尝试遍历 content 内所有 *_content 字段
    parts = []
    for k, v in content.items():
        if k.endswith("_content"):
            if isinstance(v, list):
                txt = extract_inline_text_v2(v)
                if txt:
                    parts.append(txt)
            elif isinstance(v, str) and v.strip():
                parts.append(v.strip())

    return "\n".join(parts).strip()

def extract_image_path_from_block(item: Dict[str, Any], images_dir: Path) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    content = item.get("content", {})
    if not isinstance(content, dict):
        return None
    candidates = [
        content.get("img_path"),
        content.get("image_path"),
        content.get("figure_path"),
        content.get("path"),
        item.get("img_path"),
        item.get("image_path"),
        item.get("figure_path"),
        item.get("path"),
        item.get("image_source").get("path") if "image_source" in item else None
    ]

    for imgp in candidates:
        if not imgp:
            continue

        p = Path(str(imgp))
        if p.is_absolute() and p.exists():
            return normalize_path_str(p)

        # 相对路径处理
        cand1 = images_dir / p.name
        if cand1.exists():
            return normalize_path_str(cand1)

        cand2 = images_dir / str(imgp)
        if cand2.exists():
            return normalize_path_str(cand2)

        # 即使不存在，也保留一个可追踪路径
        return normalize_path_str(cand2)

    return None

def judge_image_need_llm(image_path: str) -> dict:
    from PIL import Image
    from pathlib import Path

    p = Path(image_path)
    result = {
        "need_llm": True,
        "reason": [],
        "width": None,
        "height": None,
        "file_size": None,
    }

    if not p.exists():
        result["need_llm"] = False
        result["reason"].append("file_not_found")
        return result

    result["file_size"] = p.stat().st_size

    try:
        with Image.open(p) as img:
            w, h = img.size
            result["width"] = w
            result["height"] = h
    except Exception as e:
        result["need_llm"] = False
        result["reason"].append(f"cannot_open_image:{e}")
        return result

    # 规则1：太小的图标/缩略图
    if w < 96 or h < 96:
        result["need_llm"] = False
        result["reason"].append("too_small_pixels")

    # 规则2：总像素太少
    if w * h < 20000:
        result["need_llm"] = False
        result["reason"].append("too_few_pixels")

    # 规则3：文件太小
    if result["file_size"] is not None and result["file_size"] < 8000:
        result["need_llm"] = False
        result["reason"].append("too_small_file")

    # 规则4：疑似线条
    if min(w, h) < 20 and max(w, h) > 200:
        result["need_llm"] = False
        result["reason"].append("likely_line")

    return result

# -----------------------------
# 文本后处理
# -----------------------------
def clean_chunk_text(
    raw_text: str,
    enable_llm_clean: bool = True,
    llm_api: Optional[str] = None,
    llm_model: Optional[str] = None,
    chunk_type = None,
    image_path = None
) -> Tuple[str, Dict[str, Any]]:
    quality = {}
    basic = raw_text
    if chunk_type != "image":
        basic = basic_clean_text(raw_text)
        quality = calc_text_quality(basic)
        #print("quality before llm:", quality)
    else:
        if not image_path:
            #print("image path can't be empty.{}".format(image_path))
            raise Exception("image path can't be empty.", image_path)
        basic = basic_clean_text(raw_text)
        image_decision = judge_image_need_llm(image_path)
        quality["is_dirty"] = image_decision["need_llm"]
    cleaned = basic
    llm_used = False
    print("raw_text:  ", raw_text)
    if quality["is_dirty"] and enable_llm_clean and llm_api and llm_model and chunk_type == "image":
        print("Start doing clean on {} {}...".format(llm_api, llm_model))
        try:
            llm_out = llm_clean_text(basic, llm_api=llm_api, llm_model=llm_model, image_path=image_path)
            llm_out = basic_clean_text(llm_out)
            print("llm_out: {}".format(llm_out))
            if llm_out:
                cleaned = llm_out
                llm_used = True
                quality = calc_text_quality(cleaned)
                #print("quality after llm:", quality)
        except Exception as e:
            quality["llm_clean_error"] = str(e)
            print("Exception: {}".format(e))

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


def qdrant_collection_exists(client: QdrantClient, collection: str) -> bool:
    try:
        return collection in [c.name for c in client.get_collections().collections]
    except Exception as e:
        print(f"[WARN] cannot list Qdrant collections: {e}")
        return False


def ensure_source_pdf_payload_index(client: QdrantClient, collection: str):
    for field_name in ("source_pdf", "type", "ingest_status"):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def make_doc_manifest_id(source_pdf: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mineru-ingest-manifest::{source_pdf}"))


def source_pdf_filter(source_pdf_value: str, exclude_manifest: bool = False) -> models.Filter:
    conditions = [
        models.FieldCondition(
            key="source_pdf",
            match=models.MatchValue(value=source_pdf_value),
        )
    ]
    must_not = []
    if exclude_manifest:
        must_not.append(
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value="_ingest_manifest"),
            )
        )
    return models.Filter(must=conditions, must_not=must_not or None)


def qdrant_doc_manifest_complete(client: QdrantClient, collection: str, source_pdf: Optional[Path]) -> bool:
    if source_pdf is None:
        return False

    source_pdf_value = normalize_path_str(source_pdf)
    try:
        points = client.retrieve(
            collection_name=collection,
            ids=[make_doc_manifest_id(source_pdf_value)],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return False
        payload = points[0].payload or {}
        return payload.get("ingest_status") == "complete"
    except Exception as e:
        print(f"[WARN] manifest precheck failed for {source_pdf_value}: {e}")
        return False


def qdrant_source_pdf_chunk_count(client: QdrantClient, collection: str, source_pdf: Optional[Path]) -> Optional[int]:
    if source_pdf is None:
        return None

    source_pdf_value = normalize_path_str(source_pdf)
    try:
        result = client.count(
            collection_name=collection,
            count_filter=source_pdf_filter(source_pdf_value, exclude_manifest=True),
            exact=True,
        )
        return int(result.count)
    except Exception as e:
        print(f"[WARN] source_pdf count precheck failed for {source_pdf_value}: {e}")
        return None


def build_embedder(model_name: str) -> SentenceTransformer:
    import torch

    # 设置 PyTorch 内存分配策略
    if torch.cuda.is_available():
        # 设置显存占用比例
        #torch.cuda.set_per_process_memory_fraction(0.8, 0)  # 限制使用 50% 显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return SentenceTransformer(model_name, device='cpu')
        except torch.cuda.OutOfMemoryError:
            print("⚠️  GPU OOM, falling back to CPU")
            return SentenceTransformer(model_name, device='gpu')
    else:
        return SentenceTransformer(model_name, device='cpu')


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 4) -> List[List[float]]:
    import gc
    import torch

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        vecs = model.encode(
            batch_texts,
            batch_size=min(batch_size, len(batch_texts)),
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_vecs.extend([v.tolist() for v in vecs])

        # 释放显存
        del vecs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_vecs


def make_point_id(source_pdf: str, chunk_type: str, page: Any, chunk_index: Any, text: str) -> str:
    raw = f"{source_pdf}::{chunk_type}::p{page}::c{chunk_index}::{sha1_text(text)}"
    #return sha1_text(raw)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def get_existing_point_ids(
        client: QdrantClient,
        collection: str,
        chunks: List[Dict[str, Any]],
        retrieve_batch_size: int = 64,
) -> set[str]:
    """批量检查哪些点已经存在；失败时跳过去重，让 upsert 保持幂等。"""
    existing_ids = set()

    ids_to_check = [
        make_point_id(
            c.get("source_pdf") or "",
            c.get("type") or "text",
            c.get("page"),
            c.get("chunk_index"),
            c.get("cleaned_text") or ""
        )
        for c in chunks
    ]

    for ids_batch in batched(ids_to_check, retrieve_batch_size):
        try:
            points = client.retrieve(
                collection_name=collection,
                ids=ids_batch,
                with_payload=False,
                with_vectors=False,
            )
        except Exception:
            print("[WARN] Qdrant existing-id check timed out/failed; continuing with idempotent upsert")
            return existing_ids

        for point in points:
            existing_ids.add(str(point.id))

    return existing_ids

def ingest_to_qdrant(
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    chunks: List[Dict[str, Any]],
    batch_size: int = 6,
    upsert_batch_size: int = 18,
    skip_existing: bool = True,
):
    if not chunks:
        print("[WARN] no chunks to ingest")
        return

    # 先确保 collection 存在：需要先拿一个向量维度
    sample_vec = embedder.encode(
        [chunks[0]["cleaned_text"]],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    dim = len(sample_vec)
    ensure_qdrant_collection(client, collection, dim)
    ensure_source_pdf_payload_index(client, collection)

    # 获取已存在的点 ID（用于跳过）。如果 Qdrant 查询超时，后续 upsert 仍然是幂等的。
    existing_ids = set()
    if skip_existing:
        existing_ids = get_existing_point_ids(client, collection, chunks)
        if existing_ids:
            print(f"[INFO] Skipping {len(existing_ids)} existing points")

    import gc
    import torch
    points = []
    total = len(chunks)
    done = 0
    for i in range(0, total, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        texts = []
        valid_batch_chunks = []
        # 过滤掉已存在的
        for c in batch_chunks:
            pid = make_point_id(
                c.get("source_pdf") or "",
                c.get("type") or "text",
                c.get("page"),
                c.get("chunk_index"),
                c.get("cleaned_text") or "",
            )
            if pid not in existing_ids:
                valid_batch_chunks.append(c)
                texts.append(c["cleaned_text"])
        if not valid_batch_chunks:
            continue
        #texts = [c["cleaned_text"] for c in batch_chunks]
        vectors = embed_texts(embedder, texts, batch_size=min(batch_size, len(texts)))
        for c, v in zip(valid_batch_chunks, vectors):
            pid = make_point_id(
                c.get("source_pdf") or "",
                c.get("type") or "text",
                c.get("page"),
                c.get("chunk_index"),
                c.get("cleaned_text") or "",
            )
            _image_path = image_path = None
            if c.get("type") == "image":
                if c.get('image_path', None):
                    image_path = c.get('image_path')
                else:
                    _parent_path = c.get('mineru_content_list', None)
                    _image_path = c.get('raw_item', None)
                    if _image_path and _parent_path:
                        if "content" in _image_path and "image_source" in _image_path["content"]:
                            image_path = osp.join(osp.dirname(_parent_path), _image_path["content"]["image_source"]["path"])
                        else:
                            image_path = osp.join(osp.dirname(_parent_path),
                                                  _image_path["img_path"] if "img_path" in _image_path else _image_path[
                                                      "image_path"])

            payload = {
                "type": c.get("type"),
                "sub_type": c.get("sub_type"),
                "page": c.get("page"),
                "chunk_index": c.get("chunk_index"),
                "bbox": c.get("bbox"),
                "anchor": c.get("anchor"),
                "source_pdf": c.get("source_pdf"),
                "image_path": image_path,
                "cleaned_text": c.get("cleaned_text"),
                "raw_text": c.get("raw_text"),
                "quality": c.get("quality"),
                "llm_clean_used": c.get("llm_clean_used", False),
                "mineru_markdown": c.get("mineru_markdown"),
                "mineru_content_list": c.get("mineru_content_list"),
                "mineru_content_list_v2": c.get("mineru_content_list_v2"),
            }
            points.append(PointStruct(id=pid,vector=v,payload=payload))
        # 每批 upsert 一次，避免内存堆积
        for batch in batched(points, upsert_batch_size):
            client.upsert(collection_name=collection, points=batch)
            done += len(batch)
            print(f"[INFO] upserted batch: {done}/{total}")

        points.clear()
        del vectors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_pdf = chunks[0].get("source_pdf") or ""
    if source_pdf:
        manifest_payload = {
            "type": "_ingest_manifest",
            "source_pdf": source_pdf,
            "ingest_status": "complete",
            "expected_chunks": total,
            "completed_at": int(time.time()),
        }
        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=make_doc_manifest_id(source_pdf),
                    vector=[1.0] + ([0.0] * (dim - 1)),
                    payload=manifest_payload,
                )
            ],
        )
        print(f"[INFO] wrote ingest manifest for {source_pdf}")

    print(f"[DONE] upserted {done}/{total} new points into collection={collection}")

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
    parser.add_argument("--llm_model", default="Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--dump_clean_json", default=None, help="可选，导出清洗结果 json")
    parser.add_argument("--upsert_batch_size", type=int, default=18)
    parser.add_argument("--qdrant_timeout", type=float, default=120.0)
    parser.add_argument("--no_skip_existing", action="store_true")
    parser.add_argument("--no_doc_skip", action="store_true")
    args = parser.parse_args()
    mineru_dir = Path(args.mineru_dir)
    mineru_dir_list = [d for d in mineru_dir.iterdir() if d.is_dir()]

    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        timeout=args.qdrant_timeout,
    )
    collection_exists = qdrant_collection_exists(client, args.collection)
    if collection_exists:
        ensure_source_pdf_payload_index(client, args.collection)

    embedder = None
    for _mineru_dir in mineru_dir_list:
        try:
            # 获取所有文件
            _dir = Path(osp.join(_mineru_dir.resolve(), _mineru_dir.name, "hybrid_auto"))
            if not osp.exists(_dir):
                continue
            files = [p for p in _dir.iterdir() if p.is_file()]
            print(files)
        except PermissionError:
            print(f"Warning: Permission denied accessing: {mineru_dir}")
            continue
        except Exception as e:
            print(f"Error accessing {mineru_dir}: {e}")
            continue
        primary = find_mineru_primary_dir(_mineru_dir)

        print("[INFO] detected MinerU outputs:")
        for k, v in primary.items():
            print(f"  - {k}: {v}")

        if (
            not args.no_skip_existing
            and not args.no_doc_skip
            and collection_exists
            and qdrant_doc_manifest_complete(client, args.collection, primary.get("original_pdf"))
        ):
            print(f"[SKIP] complete ingest manifest exists: {primary.get('original_pdf')}")
            continue

        parsed_chunks = parse_mineru_output(primary)
        print(f"[INFO] parsed chunks: {len(parsed_chunks)}")

        if not args.no_skip_existing and not args.no_doc_skip and collection_exists:
            existing_count = qdrant_source_pdf_chunk_count(client, args.collection, primary.get("original_pdf"))
            if existing_count is not None and parsed_chunks and existing_count >= len(parsed_chunks):
                print(
                    f"[SKIP] existing Qdrant chunks look complete: "
                    f"{existing_count}/{len(parsed_chunks)} for {primary.get('original_pdf')}"
                )
                continue

        processed = []
        for c in parsed_chunks:
            raw_text = c.get("raw_text", "") or c.get("text", "") or ""
            chunk_type = c.get("type")
            image_path = None
            print("chunk slice:", c)
            # image 没文本时，至少给 caption 占位，确保可检索 & 可显示
            if chunk_type == "image":
                print("raw text..............{}".format(raw_text))
            if chunk_type == "image":
                #raw_text = f"[IMAGE] {Path(c.get('image_path') or '').name}"
                if c.get('image_path'):
                    image_path = c.get('image_path')
                else:
                    _parent_path = c.get('mineru_content_list', None)
                    _image_path = c.get('raw_item',None)
                    if _image_path and _parent_path:
                        if "content" in _image_path and "image_source" in _image_path["content"]:
                            image_path = osp.join(osp.dirname(_parent_path), _image_path["content"]["image_source"]["path"])
                        else:
                            image_path = osp.join(osp.dirname(_parent_path), _image_path["img_path"] if "img_path" in _image_path else _image_path["image_path"])
                if not raw_text.strip():
                    raw_text = image_path
                print("raw_text_2..............", raw_text)

            cleaned_text, clean_meta = clean_chunk_text(
                raw_text=raw_text,
                enable_llm_clean=args.enable_llm_clean,
                llm_api=args.llm_api,
                llm_model=args.llm_model,
                chunk_type=chunk_type,
                image_path= image_path
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

        if embedder is None:
            embedder = build_embedder(args.embed_model)

        ingest_to_qdrant(
            client=client,
            collection=args.collection,
            embedder=embedder,
            chunks=processed,
            batch_size=args.batch_size,
            upsert_batch_size=args.upsert_batch_size,
            skip_existing=not args.no_skip_existing,
        )
        collection_exists = True

if __name__ == "__main__":
    main()