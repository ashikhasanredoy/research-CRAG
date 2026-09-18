from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import fitz  # PyMuPDF

class PDFPaperLoader:
    """Extracts clean text, pages, and detected sections from academic research paper PDFs."""

    SECTION_PATTERNS = [
        (r'^(?:[0-9IVXLCDM]+\.?\s*)?abstract\b', "Abstract"),
        (r'^(?:[0-9IVXLCDM]+\.?\s*)?introduction\b', "Introduction"),
        (r'^(?:[0-9IVXLCDM]+\.?\s*)?related\s+work\b', "Related Work"),
        (r'^(?:[0-9IVXLCDM]+\.?\s*)?(?:methodology|method|proposed\s+method|framework)\b', "Methodology"),
        (r'^(?:[0-9IVXLCDM]+\.?\s*)?(?:experiments|experimental\s+setup|evaluation)\b', "Experiments"),
        (r'^(?:[0-9IVXLCDM]+\.?\s*)?(?:results|results\s+and\s+discussion|discussion)\b', "Results"),
        (r'^(?:[0-9IVXLCDM]+\.?\s*)?(?:conclusion|conclusions|concluding\s+remarks)\b', "Conclusion"),
        (r'^(?:[0-9IVXLCDM]+\.?\s*)?references\b', "References"),
    ]

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.file_path}")

    def extract_paper(self) -> Dict[str, Any]:
        """Reads PDF and returns structured paper document with page contents and detected sections."""
        doc = fitz.open(self.file_path)
        total_pages = len(doc)
        pages_data: List[Dict[str, Any]] = []

        title = self._extract_title(doc)
        current_section = "Abstract / Intro"

        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            page_num = page_idx + 1
            raw_text = page.get_text("text")
            
            clean_text, section_found = self._clean_and_detect_section(raw_text, current_section)
            if section_found:
                current_section = section_found

            pages_data.append({
                "page_number": page_num,
                "text": clean_text,
                "section": current_section
            })

        doc.close()

        return {
            "filename": self.file_path.name,
            "title": title or self.file_path.stem.replace("_", " ").title(),
            "total_pages": total_pages,
            "pages": pages_data
        }

    def _clean_and_detect_section(self, text: str, current_section: str) -> tuple[str, Optional[str]]:
        lines = text.split("\n")
        cleaned_lines = []
        new_section = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if line indicates section header
            lower = stripped.lower()
            detected = None
            if len(stripped) < 80:  # Heading length constraint
                for pattern, sec_name in self.SECTION_PATTERNS:
                    if re.match(pattern, lower, re.IGNORECASE):
                        detected = sec_name
                        break

            if detected:
                new_section = detected
                cleaned_lines.append(f"\n### {new_section}\n")
            else:
                cleaned_lines.append(stripped)

        # Join text with proper paragraph spacing
        joined_text = " ".join(cleaned_lines)
        joined_text = re.sub(r'\s+', ' ', joined_text)
        joined_text = re.sub(r'\n### ([A-Za-z ]+)\n', r'\n\n[Section: \1]\n', joined_text)
        return joined_text.strip(), new_section

    def _extract_title(self, doc: fitz.Document) -> Optional[str]:
        # First check document metadata
        meta_title = doc.metadata.get("title")
        if meta_title and len(meta_title.strip()) > 5:
            return meta_title.strip()

        # Fallback to first bold or large text block from first page
        if len(doc) > 0:
            first_page = doc.load_page(0)
            blocks = first_page.get_text("blocks")
            if blocks:
                # First block usually contains paper title
                first_block_text = blocks[0][4].strip().replace("\n", " ")
                if len(first_block_text) > 5 and len(first_block_text) < 200:
                    return first_block_text
        return None
