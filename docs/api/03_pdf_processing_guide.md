# PDF Processing Guide

This guide explains how we turn a raw PDF file into structured, searchable data with precise location information.

## The Challenge

PDFs are primarily designed for printing, not data extraction. Text is often stored as loose commands ("draw 'H' at x,y", "draw 'e' at x+5,y") rather than coherent paragraphs. We need to reconstruct this into meaningful blocks while retaining the original layout for highlighting.

## Extraction Process

We use **PyMuPDF**, a high-performance rendering library, to analyze the visual structure of the page.

#### Flow Diagram: Parsing Logic

```mermaid
flowchart TD
    A[Start: Raw PDF File] --> B{File Exists?}
    B -- No --> C[Return Error]
    B -- Yes --> D[Open Document]
    
    D --> E[Iterate Pages (1..N)]
    
    subgraph Page Processing
    E --> F[Get 'Blocks']
    F --> G{Is Block Text?}
    G -- No (Image) --> H[Skip]
    G -- Yes --> I[Extract Coordinates (x0,y0,x1,y1)]
    I --> J[Calculate Line Counts]
    J --> K[Assign Global Line Numbers]
    K --> L[Add to Block List]
    end
    
    L --> M{More Pages?}
    M -- Yes --> E
    M -- No --> N[Return ParsedDocument]
```

## Spatial Extraction

One of the key features of this system is **Spatial Awareness**. We don't just extract text; we extract *where* it is.

### Bounding Boxes
Every piece of text is returned with a bounding box `[x0, y0, x1, y1]`.
- This allows the frontend to draw a highlight box over the exact source text when a user asks a question.
- **Coordinate System:** Origin `(0,0)` is at the **Top-Left**.

```
(0,0) ---------------------> X
  |
  |      [ Text Block ]
  |      (x0,y0)
  |         *-------*
  |         | Hello |
  |         *-------*
  |              (x1,y1)
  |
  v Y
```

## Global Line Numbering

To make citations precise, we treat the entire document as one continuous stream of lines, regardless of page breaks.

**Example:**
- **Page 1** has 50 lines.
- **Page 2** starts at line 51.

This abstraction helps the LLM (Language Model) understand the relative distance between two pieces of information, even if they are on different pages.

## Handling "Scanned" PDFs

**Current Limitation:** The parser currently **only** supports "native" PDFs (documents created digitally).
- If you upload a scanned image of a document (without an OCR layer), the parser will see it as an image block and skip it.
- **Future Upgrade:** We can integrate `Tesseract` or a similar OCR engine to handle image-based PDFs.
