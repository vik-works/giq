# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq_ocr.ocrdoc on synthetic model output.

The fixtures are invented documents in the model's exact tag grammar: no real
page ever needs to be in a test file for the strip and merge rules to be
pinned down.
"""

from giq_ocr import ocrdoc

# Two pages. Page 1 ends with a table; page 2 opens (after its running
# header) with the same table continued, restating the column titles.
TABLE_ACROSS_PAGES = (
    "<PAGE>\n"
    "<|det|>header [100, 36, 400, 70]<|/det|>Example AG · Statement\n"
    "<|det|>title [100, 110, 500, 140]<|/det|>Positions\n"
    "<|det|>text [100, 160, 900, 200]<|/det|>Intro line.\n"
    "<|det|>table [100, 220, 900, 900]<|/det|>"
    "<table><tr><td>Item</td><td>Qty</td></tr><tr><td>Bolt</td><td>4</td></tr></table>\n"
    "<|det|>footer [100, 940, 400, 970]<|/det|>Confidential\n"
    "<|det|>page_number [900, 940, 950, 970]<|/det|>1\n"
    "<PAGE>\n"
    "<|det|>header [100, 36, 400, 70]<|/det|>Example AG · Statement\n"
    "<|det|>table [100, 110, 900, 400]<|/det|>"
    "<table><tr><td>Item</td><td>Qty</td></tr><tr><td>Nut</td><td>9</td></tr></table>\n"
    "<|det|>text [100, 420, 900, 460]<|/det|>Closing line.\n"
    "<|det|>page_number [900, 940, 950, 970]<|/det|>2\n"
)

PARAGRAPH_ACROSS_PAGES = (
    "<PAGE>\n"
    "<|det|>text [100, 800, 900, 900]<|/det|>The delivery was accepted on the con-\n"
    "<|det|>page_number [900, 940, 950, 970]<|/det|>1\n"
    "<PAGE>\n"
    "<|det|>header [100, 36, 400, 70]<|/det|>Example AG\n"
    "<|det|>text [100, 110, 900, 200]<|/det|>dition that the invoice follows.\n"
    "<|det|>text [100, 220, 900, 300]<|/det|>A new paragraph, unrelated\n"
    "<PAGE>\n"
    "<|det|>text [100, 110, 900, 200]<|/det|>Starts With Capital and is not a continuation.\n"
)


def test_parse_keeps_every_block_with_its_page():
    blocks = ocrdoc.parse(TABLE_ACROSS_PAGES)
    assert [b.page for b in blocks] == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    assert blocks[0].label == "header" and blocks[0].bbox == (100, 36, 400, 70)
    assert blocks[3].label == "table" and blocks[3].content.startswith("<table>")
    assert ocrdoc.page_count(TABLE_ACROSS_PAGES) == 2


def test_parse_keeps_text_the_model_did_not_tag():
    blocks = ocrdoc.parse("<PAGE>\nloose line\n<|det|>text [1, 2, 3, 4]<|/det|>tagged\n")
    assert [(b.label, b.content) for b in blocks] == [("text", "loose line"), ("text", "tagged")]
    # No separator at all: still one page, still nothing lost.
    assert [b.content for b in ocrdoc.parse("just text")] == ["just text"]
    assert ocrdoc.parse("") == []


def test_strip_drops_furniture_by_label_only():
    blocks = ocrdoc.strip_furniture(ocrdoc.parse(TABLE_ACROSS_PAGES))
    labels = {b.label for b in blocks}
    assert not labels & {"header", "footer", "page_number"}
    assert "Confidential" not in ocrdoc.to_html(blocks)
    assert "Example AG" not in ocrdoc.to_html(blocks)
    # Everything that is not furniture survives untouched.
    assert [b.label for b in blocks] == ["title", "text", "table", "table", "text"]


def test_table_continues_across_the_page_break():
    blocks = ocrdoc.merge_pages(ocrdoc.strip_furniture(ocrdoc.parse(TABLE_ACROSS_PAGES)))
    assert [b.label for b in blocks] == ["title", "text", "table", "text"]
    table = blocks[2]
    assert table.pages == [1, 2]
    rows = ocrdoc._rows(table.content)
    # Header row once, then the rows of both halves in order.
    assert [ocrdoc._row_text(r) for r in rows] == ["item qty", "bolt 4", "nut 9"]
    assert table.content.endswith("</table>")


def test_tables_with_different_shapes_stay_apart():
    raw = (
        "<PAGE>\n<|det|>table [1, 1, 2, 2]<|/det|><table><tr><td>a</td><td>b</td></tr></table>\n"
        "<PAGE>\n<|det|>table [1, 1, 2, 2]<|/det|><table><tr><td>a</td></tr></table>\n"
    )
    blocks = ocrdoc.merge_pages(ocrdoc.parse(raw))
    assert [b.label for b in blocks] == ["table", "table"]


def test_merge_only_joins_across_a_page_boundary():
    raw = (
        "<PAGE>\n<|det|>table [1, 1, 2, 2]<|/det|><table><tr><td>a</td></tr></table>\n"
        "<|det|>table [1, 3, 2, 4]<|/det|><table><tr><td>b</td></tr></table>\n"
    )
    # Two tables on one page are two tables.
    assert len(ocrdoc.merge_pages(ocrdoc.parse(raw))) == 2


def test_paragraph_broken_at_the_margin_is_rejoined():
    blocks = ocrdoc.merge_pages(ocrdoc.strip_furniture(ocrdoc.parse(PARAGRAPH_ACROSS_PAGES)))
    assert [b.label for b in blocks] == ["text", "text", "text"]
    assert (
        blocks[0].content == "The delivery was accepted on the condition that the invoice follows."
    )
    assert blocks[0].pages == [1, 2]
    # "A new paragraph, unrelated" lacks punctuation, but the next page starts
    # with a capital: not a continuation, left alone.
    assert blocks[1].pages == [2]
    assert blocks[2].content.startswith("Starts With Capital")


def test_lowercase_continuation_without_hyphen_is_joined():
    raw = (
        "<PAGE>\n<|det|>text [1, 1, 2, 2]<|/det|>ends without a stop\n"
        "<PAGE>\n<|det|>text [1, 1, 2, 2]<|/det|>and carries on.\n"
    )
    (block,) = ocrdoc.merge_pages(ocrdoc.parse(raw))
    assert block.content == "ends without a stop and carries on."


def test_html_escapes_text_and_passes_tables_through():
    raw = (
        "<PAGE>\n<|det|>title [1, 1, 2, 2]<|/det|>A & B <not a tag>\n"
        "<|det|>text [1, 3, 2, 4]<|/det|>line one\nline two\n"
        "<|det|>table [1, 5, 2, 6]<|/det|><table><tr><td>x</td></tr></table>\n"
        "<|det|>image [1, 7, 2, 8]<|/det|>\n"
        "<|det|>list [1, 3, 2, 4]<|/det|>\n"
        "<|det|>aside_text [1, 9, 2, 9]<|/det|>margin\n"
    )
    html = ocrdoc.to_html(ocrdoc.parse(raw))
    assert (
        '<h2 class="title" data-page="1" data-bbox="1,1,2,2">A &amp; B &lt;not a tag&gt;</h2>'
        in html
    )
    assert '<p class="text" data-page="1" data-bbox="1,3,2,4">line one line two</p>' in html
    assert '<table data-page="1" data-bbox="1,5,2,6"><tr><td>x</td></tr></table>' in html
    assert '<figure data-page="1" data-bbox="1,7,2,8"></figure>' in html
    assert "<aside" in html
    # An empty container leaves nothing behind.
    assert "list" not in html


def test_render_is_the_whole_pipeline():
    html, blocks = ocrdoc.render(TABLE_ACROSS_PAGES)
    assert html.count("<table") == 1 and 'data-page="1,2"' in html
    assert "Confidential" not in html
    assert [b["label"] for b in blocks] == ["title", "text", "table", "text"]
    html_raw, blocks_raw = ocrdoc.render(TABLE_ACROSS_PAGES, strip=False, merge=False)
    assert "Confidential" in html_raw and html_raw.count("<table") == 2
    assert len(blocks_raw) == 10
