# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""From Unlimited-OCR's tagged text to a document: parse, strip, merge, HTML.

The model emits one ``<|det|>label [x1, y1, x2, y2]<|/det|>`` per layout
block, coordinates on a 0-999 page box, pages introduced by ``<PAGE>``. What
it does *not* do is edit the document: running headers, footers and page
numbers are transcribed like everything else, and a table that continues
across a page break comes out as two tables. (Its multi-page training data
was single pages concatenated with a separator, so it has never seen a join.)
Both are ours, and both are decided here from labels and positions alone.

This module never touches the GPU and takes plain strings, so every rule can
be tested against synthetic fixtures — which matters, because the real
documents are the kind nobody should paste into a test file.

Labels observed on real documents: ``header``, ``footer``,
``page_number``, ``aside_text``, ``title``, ``text``, ``list``, ``table``,
``image``. Tables arrive as HTML (``<table><tr><td>``). ``list`` is a
container with no content of its own whose bbox spans the preceding items.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

DET_RE = re.compile(r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[([^\]]*)\]\s*<\|/det\|>", re.DOTALL)
PAGE_SEP = "<PAGE>"

# Page furniture: transcribed faithfully by the model, wanted by nobody.
FURNITURE = frozenset({"header", "footer", "page_number"})

_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh]\b", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
# A paragraph that stops mid-sentence: no terminal punctuation, or a hyphen
# where a word was broken at the margin.
_SENTENCE_END = re.compile(r"""[.!?:;…"”’)\]]\s*$""")

# Element per label. Anything unlisted becomes a div carrying the label.
_TAG = {"title": "h2", "text": "p", "aside_text": "aside", "list": "div", "footnote": "aside"}


@dataclass
class Block:
    label: str
    content: str
    page: int
    bbox: tuple[int, int, int, int] | None = None
    # Pages this block spans after a merge; the first is `page`.
    pages: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.pages:
            self.pages = [self.page]


# --- Parse -------------------------------------------------------------------


def _bbox(text: str) -> tuple[int, int, int, int] | None:
    try:
        nums = [int(float(x)) for x in text.replace(",", " ").split()]
    except ValueError:
        return None
    return (nums[0], nums[1], nums[2], nums[3]) if len(nums) >= 4 else None


def parse(raw: str) -> list[Block]:
    """Tagged model output → ordered blocks with page numbers (1-based).

    Text outside any tag (a page the model transcribed without layout, or a
    stray line) is kept as a ``text`` block rather than dropped: losing
    content silently is the one failure this pipeline must not have.
    """
    pages = raw.split(PAGE_SEP)
    if pages and not pages[0].strip():
        pages = pages[1:]
    if not pages:
        return []
    blocks: list[Block] = []
    for page_no, page_text in enumerate(pages, start=1):
        matches = list(DET_RE.finditer(page_text))
        lead = page_text[: matches[0].start()].strip() if matches else page_text.strip()
        if lead:
            blocks.append(Block("text", lead, page_no))
        for idx, m in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(page_text)
            content = page_text[m.end() : end].strip()
            blocks.append(Block(m.group(1), content, page_no, _bbox(m.group(2))))
    return blocks


def page_count(raw: str) -> int:
    return len([p for p in raw.split(PAGE_SEP) if p.strip()]) if raw.strip() else 0


# --- Strip -------------------------------------------------------------------


def strip_furniture(blocks: list[Block], labels: frozenset[str] = FURNITURE) -> list[Block]:
    """Drop headers, footers and page numbers by label.

    The model labels them on every page it was given (observed at y 36-94 and
    916-983 of 999), so a label filter is enough. A positional fallback for
    mislabelled furniture is deliberately not here yet: guessing from position
    alone would also eat a real first line on a short page.
    """
    return [b for b in blocks if b.label not in labels]


# --- Merge across page breaks ------------------------------------------------


def _rows(table_html: str) -> list[str]:
    return _ROW_RE.findall(table_html)


def _columns(row_html: str) -> int:
    return len(_CELL_RE.findall(row_html))


def _row_text(row_html: str) -> str:
    return " ".join(_TAG_RE.sub(" ", row_html).split()).lower()


def _merge_tables(a: Block, b: Block) -> Block | None:
    """Append b's rows to a when they are the same table continued.

    Same table means the same column count in the first row. A repeated
    header row (the continuation restating the column titles) is dropped once
    on the join. Anything malformed leaves both tables alone.
    """
    rows_a, rows_b = _rows(a.content), _rows(b.content)
    if not rows_a or not rows_b or "</table>" not in a.content.lower():
        return None
    if _columns(rows_a[0]) != _columns(rows_b[0]) or _columns(rows_a[0]) == 0:
        return None
    if _row_text(rows_b[0]) == _row_text(rows_a[0]):
        rows_b = rows_b[1:]
    close = a.content.lower().rfind("</table>")
    merged = a.content[:close] + "".join(rows_b) + a.content[close:]
    return Block("table", merged, a.page, a.bbox, pages=[*a.pages, *b.pages])


def _merge_paragraphs(a: Block, b: Block) -> Block | None:
    """Join a paragraph the page break cut in two.

    Conservative on purpose: the first half must lack terminal punctuation
    and the second must start lowercase (or the first must end in a hyphen
    from a word broken at the margin). German capitalises nouns, so "no
    punctuation" alone would join a heading-less new paragraph onto the old.
    """
    if not a.content or not b.content:
        return None
    if a.content.endswith("-") and b.content[0].isalpha():
        joined = a.content[:-1] + b.content
    elif not _SENTENCE_END.search(a.content) and b.content[0].islower():
        joined = a.content + " " + b.content
    else:
        return None
    return Block("text", joined, a.page, a.bbox, pages=[*a.pages, *b.pages])


def merge_pages(blocks: list[Block]) -> list[Block]:
    """Join content split by a page break: the last block of one page with
    the first of the next, when both are tables or both are paragraphs.

    Call after ``strip_furniture``: with the footer and the next header in
    between, nothing is ever adjacent.
    """
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if prev is not None and b.page > prev.pages[-1]:
            merged = None
            if prev.label == "table" and b.label == "table":
                merged = _merge_tables(prev, b)
            elif prev.label == "text" and b.label == "text":
                merged = _merge_paragraphs(prev, b)
            if merged is not None:
                out[-1] = merged
                continue
        out.append(b)
    return out


# --- HTML --------------------------------------------------------------------


def _attrs(b: Block) -> str:
    pages = ",".join(str(p) for p in b.pages)
    attrs = f' data-page="{pages}"'
    if b.bbox:
        attrs += f' data-bbox="{",".join(str(v) for v in b.bbox)}"'
    return attrs


def to_html(blocks: list[Block]) -> str:
    """Blocks → an HTML fragment. Tables pass through (the model wrote them
    as HTML); every other block's text is escaped. Empty containers vanish,
    except images, which leave a figure placeholder with their position."""
    parts: list[str] = []
    for b in blocks:
        if b.label == "table":
            content = b.content
            low = content.lower()
            i = low.find("<table")
            if i >= 0:
                j = i + len("<table")
                content = content[:j] + _attrs(b) + content[j:]
            parts.append(content)
        elif b.label == "image":
            parts.append(f"<figure{_attrs(b)}></figure>")
        elif b.content:
            tag = _TAG.get(b.label, "div")
            text = html.escape(" ".join(b.content.split()))
            parts.append(f'<{tag} class="{html.escape(b.label)}"{_attrs(b)}>{text}</{tag}>')
    return "\n".join(parts)


def blocks_json(blocks: list[Block]) -> list[dict]:
    return [
        {"page": b.page, "pages": b.pages, "label": b.label, "bbox": b.bbox, "content": b.content}
        for b in blocks
    ]


def render(raw: str, *, strip: bool = True, merge: bool = True) -> tuple[str, list[dict]]:
    """The whole pipeline: raw model text → (html, blocks)."""
    blocks = parse(raw)
    if strip:
        blocks = strip_furniture(blocks)
    if merge:
        blocks = merge_pages(blocks)
    return to_html(blocks), blocks_json(blocks)
