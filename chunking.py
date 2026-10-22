# chunking.py
from typing import List, Dict, Any


def split_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]

        if end < n:
            last_break = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind("\n"))
            if last_break > int(chunk_size * 0.5):
                end = start + last_break + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def build_chunk_record(
    document: str,
    metadata: Dict[str, Any],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    out = []
    pieces = split_text(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for i, piece in enumerate(pieces):
        m = dict(metadata)
        m["chunk_index"] = i
        out.append({
            "text": piece,
            "metadata": m,
        })
    return out