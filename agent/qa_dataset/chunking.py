"""Source ingestion and chunking.

Reads documents from a source directory, splits them into overlapping,
paragraph-aware chunks sized by an approximate token budget, preserves tables
and figure captions, flags table-bearing chunks (for the table_qa strategy), and
drops tiny fragments. Deep extraction cleaning / OCR is intentionally out of
scope (see QA_PIPELINE_INTEGRATION_PLAN.md "Deferred").
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

_WORD = re.compile(r"\S+")
TEXT_EXTS = {".txt", ".md", ".markdown"}
DOC_EXTS = {".pdf", ".epub"}


def approx_tokens(text: str) -> int:
    """Approximate token count. Whitespace words scaled by 1.3 (good enough for sizing)."""
    return int(len(_WORD.findall(text)) * 1.3)


def looks_like_table(block: str) -> bool:
    """Heuristic: markdown pipe tables or repeated multi-column rows."""
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    pipey = sum(1 for ln in lines if ln.count("|") >= 2)
    if pipey >= 2:
        return True
    tabby = sum(1 for ln in lines if "\t" in ln or re.search(r"\S {2,}\S", ln))
    return tabby >= 3 and tabby >= 0.6 * len(lines)


def is_caption(block: str) -> bool:
    return bool(re.match(r"^\s*(figure|fig\.|table)\s*\d+", block, re.IGNORECASE))


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    location: str
    text: str
    topic: Optional[str] = None
    has_table: bool = False
    tokens: int = 0
    extra: dict = field(default_factory=dict)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_document(path: Path) -> str:
    """Extract plain text from PDF/EPUB via optional backends (lazy import)."""
    try:
        from markitdown import MarkItDown  # type: ignore

        return MarkItDown().convert(str(path)).text_content
    except Exception:
        pass
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            return "\n\n".join((pg.extract_text() or "") for pg in reader.pages)
        except Exception as e:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                f"Could not extract {path.name}; install markitdown or pypdf "
                f"(pip install 'ml-intern[qa]'). Underlying error: {e}"
            )
    raise RuntimeError(
        f"No extractor available for {path.name}; install markitdown "
        f"(pip install 'ml-intern[qa]')."
    )


def read_source(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return _read_text_file(path)
    if ext in DOC_EXTS:
        return _read_document(path)
    raise RuntimeError(f"Unsupported source type: {path.name}")


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines into paragraph blocks, trimming empties."""
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def chunk_text(
    text: str,
    source_id: str,
    *,
    chunk_tokens: int = 800,
    overlap: int = 100,
    min_chunk_chars: int = 200,
    preserve_tables: bool = True,
    keep_figure_captions: bool = True,
    start_index: int = 0,
) -> list[Chunk]:
    """Paragraph-aware, token-budgeted chunking with overlap.

    Tables (and optionally figure captions) are emitted as their own chunks and
    never split, so table_qa sees intact structures.
    """
    paras = split_paragraphs(text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    idx = start_index

    def flush(has_table: bool = False) -> None:
        nonlocal buf, buf_tokens, idx
        if not buf:
            return
        body = "\n\n".join(buf).strip()
        if len(body) >= min_chunk_chars or has_table:
            chunks.append(
                Chunk(
                    chunk_id=f"{source_id}::c{idx:05d}",
                    source_id=source_id,
                    location=f"chunk {idx}",
                    text=body,
                    has_table=has_table,
                    tokens=approx_tokens(body),
                )
            )
            idx += 1
        buf = []
        buf_tokens = 0

    for para in paras:
        is_tbl = preserve_tables and looks_like_table(para)
        if is_tbl:
            flush()  # close current text run
            chunks.append(
                Chunk(
                    chunk_id=f"{source_id}::c{idx:05d}",
                    source_id=source_id,
                    location=f"chunk {idx}",
                    text=para.strip(),
                    has_table=True,
                    tokens=approx_tokens(para),
                )
            )
            idx += 1
            continue
        if keep_figure_captions and is_caption(para) and buf:
            buf.append(para)  # attach caption to current run
            continue
        ptok = approx_tokens(para)
        if buf_tokens + ptok > chunk_tokens and buf:
            # carry overlap: keep tail paragraphs up to `overlap` tokens
            flush()
            carry: list[str] = []
            ctok = 0
            for prev in reversed(chunks[-1].text.split("\n\n")) if chunks else []:
                t = approx_tokens(prev)
                if ctok + t > overlap:
                    break
                carry.insert(0, prev)
                ctok += t
            buf = list(carry)
            buf_tokens = ctok
        buf.append(para)
        buf_tokens += ptok
    flush()
    return chunks


def chunk_sources(
    input_dir: str | Path,
    *,
    chunk_tokens: int = 800,
    overlap: int = 100,
    min_chunk_chars: int = 200,
    preserve_tables: bool = True,
    keep_figure_captions: bool = True,
) -> list[Chunk]:
    """Chunk every supported file under input_dir. source_id = file stem."""
    root = Path(input_dir)
    all_chunks: list[Chunk] = []
    files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in (TEXT_EXTS | DOC_EXTS)
    )
    for f in files:
        text = read_source(f)
        all_chunks.extend(
            chunk_text(
                text,
                source_id=f.stem,
                chunk_tokens=chunk_tokens,
                overlap=overlap,
                min_chunk_chars=min_chunk_chars,
                preserve_tables=preserve_tables,
                keep_figure_captions=keep_figure_captions,
            )
        )
    return all_chunks


def write_chunks(chunks: Iterable[Chunk], path: str | Path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_chunks(path: str | Path) -> list[Chunk]:
    out: list[Chunk] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Chunk(**json.loads(line)))
    return out
