"""
PDF parsing utilities using PyMuPDF (fitz).
Extracts text with spatial layout (coordinates) for precision citations.
"""

from dataclasses import dataclass
from typing import List, Optional
import os

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


@dataclass
class TextBlock:
    """A block of text extracted from a PDF with coordinates."""

    text: str
    page_number: int  # 1-indexed
    x0: float
    y0: float
    x1: float
    y1: float
    block_type: str = "text"  # 'text' or 'image'


@dataclass
class ParsedDocument:
    """Result of parsing a PDF document."""

    filename: str
    page_count: int
    blocks: List[TextBlock]
    error: Optional[str] = None


class PDFParser:
    """
    PDF parser using PyMuPDF to extract text with coordinates.
    
    The coordinates (x0, y0, x1, y1) represent the bounding box
    of each text block, enabling precision citations.
    """

    def __init__(self):
        if fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF parsing. "
                "Install with: poetry add pymupdf"
            )

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a PDF file and extract text blocks with coordinates.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ParsedDocument with extracted text blocks
        """
        if not os.path.exists(file_path):
            return ParsedDocument(
                filename=os.path.basename(file_path),
                page_count=0,
                blocks=[],
                error=f"File not found: {file_path}",
            )

        try:
            doc = fitz.open(file_path)
            blocks: List[TextBlock] = []

            for page_num, page in enumerate(doc, start=1):
                # Get text blocks with position info
                # Each block: (x0, y0, x1, y1, "text", block_no, block_type)
                page_blocks = page.get_text("blocks")

                for block in page_blocks:
                    if len(block) >= 5 and block[6] == 0:  # Text block (not image)
                        x0, y0, x1, y1, text = block[:5]
                        text = text.strip()

                        if text:  # Skip empty blocks
                            blocks.append(
                                TextBlock(
                                    text=text,
                                    page_number=page_num,
                                    x0=x0,
                                    y0=y0,
                                    x1=x1,
                                    y1=y1,
                                    block_type="text",
                                )
                            )

            doc.close()

            return ParsedDocument(
                filename=os.path.basename(file_path),
                page_count=len(doc) if doc else 0,
                blocks=blocks,
            )

        except Exception as e:
            return ParsedDocument(
                filename=os.path.basename(file_path),
                page_count=0,
                blocks=[],
                error=str(e),
            )

    def extract_text_only(self, file_path: str) -> str:
        """
        Extract plain text from PDF without coordinates.
        Useful for simple text extraction needs.
        """
        if not os.path.exists(file_path):
            return ""

        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception:
            return ""


# Singleton instance
pdf_parser = PDFParser() if fitz else None


def get_pdf_parser() -> PDFParser:
    """Get the PDF parser instance."""
    if pdf_parser is None:
        raise ImportError("PyMuPDF is not installed")
    return pdf_parser
