# PDF Text Handling

This document describes how PDFs are parsed and converted into text blocks with positional metadata.

## Parser

- Implementation: `app/core/pdf_parser.py`
- Library: PyMuPDF (`fitz`)
- Output: `ParsedDocument` containing `TextBlock` entries

Each `TextBlock` includes:

- `text`: extracted text
- `page_number`: 1-indexed page number
- `x0, y0, x1, y1`: bounding box coordinates (as returned by PyMuPDF)
- `block_index`: position within the page
- `line_start`, `line_end`: document-wide line range

## Extraction behavior

- Uses `page.get_text("blocks")` and includes only text blocks (`block[6] == 0`).
- Trims whitespace and skips empty blocks.
- Counts lines using `text.count("\n") + 1` and assigns global line numbers across the entire document.
- The parser does not perform OCR; it only extracts text embedded in the PDF.

## How parsed blocks are used

During background processing (`app/workers/document_worker.py`):

1. The parser extracts all blocks and returns `ParsedDocument`.
2. Each block becomes a `document_chunks` row with its bounding box and line range.
3. Embeddings are generated per block and stored for retrieval.

## Storage notes

- `document_page_blocks` exists in the schema for block-level storage, but the current worker only persists blocks as `document_chunks`.
- Bounding boxes are stored with each chunk and returned in citations.
