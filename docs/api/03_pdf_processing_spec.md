# PDF Processing Specification

This document details the internal logic and data structures used for PDF parsing and processing. While not a public HTTP API, this "Internal API" is critical for the background worker system.

## 1. Parser Interface

The system uses `PyMuPDF` (fitz) to interact with PDF files.

- **Class:** `app.core.pdf_parser.PDFParser`
- **Primary Method:** `parse(file_path: str) -> ParsedDocument`

### Input
- `file_path`: Absolute path to the PDF file on the local filesystem.

### Output Data Structures

#### `ParsedDocument`
The top-level container for extraction results.

| Field | Type | Description |
|---|---|---|
| `filename` | string | Original filename |
| `page_count` | int | Total number of pages |
| `blocks` | List[TextBlock] | Extracted content blocks |
| `total_lines` | int | Total lines of text across all pages |
| `error` | string \| None | Error message if parsing failed |

#### `TextBlock`
Represents a distinct "paragraph" or block of text identified by the PDF engine.

| Field | Type | Description |
|---|---|---|
| `text` | string | The extracted text content (trimmed) |
| `page_number` | int | 1-based page index |
| `x0` | float | Left coordinate of bounding box |
| `y0` | float | Top coordinate of bounding box |
| `x1` | float | Right coordinate of bounding box |
| `y1` | float | Bottom coordinate of bounding box |
| `block_index` | int | Order of the block on the page (0-indexed) |
| `line_start` | int | Document-wide starting line number (1-indexed) |
| `line_end` | int | Document-wide ending line number |

## 2. Coordinate System

The extraction engine uses the standard PDF coordinate system as returned by PyMuPDF.

- **Origin (0,0):** Top-left corner of the page.
- **Units:** Points (1/72 inch).
- **Bounding Box:** `(x0, y0, x1, y1)` where:
  - `x0`: Left
  - `y0`: Top
  - `x1`: Right
  - `y1`: Bottom

## 3. Line Numbering

The system implements a **Global Line Numbering** scheme to allow for precise citations across the entire document, not just per page.

- **Calculation:** `line_count = text.count("\n") + 1`
- **Global Counter:** Maintains a running total `global_line_number` across all pages.
- **Assignment:**
  - `line_start = previous_global_total + 1`
  - `line_end = line_start + line_count - 1`

## 4. Limitations

- **Text Only:** Images within PDFs are ignored (`block_type="text"` check).
- **No OCR:** Only selectable text embedded in the PDF is extracted. Scanned documents (images of text) will result in empty blocks.
