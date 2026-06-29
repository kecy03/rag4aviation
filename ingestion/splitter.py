import re

from langchain_core.documents import Document

from config.settings import get_settings


class SemanticChunker:
    """Split documents by section/paragraph boundaries, falling back to fixed-size chunks."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        # Patterns for detecting section boundaries in Chinese aviation documents
        self._section_patterns = [
            re.compile(r"^第[一二三四五六七八九十\d]+章\s+"),  # 第三章
            re.compile(r"^第[一二三四五六七八九十\d]+节\s+"),  # 第三节
            re.compile(r"^\d+\.\d+(\.\d+)?\s+"),  # 3.1 or 3.1.1
            re.compile(r"^#{1,4}\s+"),  # Markdown headers
            re.compile(r"^[A-Z一-鿿]{4,}$"),  # ALL-CAPS or dense Chinese header lines
        ]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            doc_chunks = self._split_single(doc)
            chunks.extend(doc_chunks)
        return chunks

    def _split_single(self, doc: Document) -> list[Document]:
        text = doc.page_content
        if len(text) <= self.chunk_size:
            return [doc]

        sections = self._detect_sections(text)
        if not sections:
            # No section boundaries found, fall back to paragraph chunking
            return self._chunk_by_paragraphs(doc, text)

        chunks: list[Document] = []
        for start, end, section_title in sections:
            section_text = text[start:end].strip()
            if len(section_text) <= self.chunk_size:
                if section_text:
                    new_doc = Document(
                        page_content=section_text,
                        metadata={**doc.metadata, "section_title": section_title},
                    )
                    chunks.append(new_doc)
            else:
                sub_chunks = self._chunk_text(
                    section_text, doc.metadata, section_title
                )
                chunks.extend(sub_chunks)
        return chunks

    def _detect_sections(self, text: str) -> list[tuple[int, int, str]]:
        """Return list of (start, end, section_title) spans."""
        boundaries: list[tuple[int, str]] = []
        lines = text.split("\n")
        pos = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                pos += len(line) + 1
                continue
            for pattern in self._section_patterns:
                if pattern.match(stripped):
                    boundaries.append((pos, stripped[:60]))
                    break
            pos += len(line) + 1

        if not boundaries:
            return []

        sections: list[tuple[int, int, str]] = []
        for i, (start, title) in enumerate(boundaries):
            if i + 1 < len(boundaries):
                end = boundaries[i + 1][0]
            else:
                end = len(text)
            sections.append((start, end, title))
        return sections

    def _chunk_by_paragraphs(self, doc: Document, text: str) -> list[Document]:
        return self._chunk_text(text, doc.metadata, "")

    def _chunk_text(
        self, text: str, base_metadata: dict, section_title: str
    ) -> list[Document]:
        """Split text into chunks respecting paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks: list[Document] = []
        current_chunk = ""
        current_meta = {**base_metadata, "section_title": section_title}

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if (
                len(current_chunk) + len(para) + 2 <= self.chunk_size
            ):
                current_chunk = (current_chunk + "\n\n" + para).strip()
            else:
                if current_chunk:
                    chunks.append(
                        Document(page_content=current_chunk, metadata=current_meta)
                    )
                if len(para) > self.chunk_size:
                    # Paragraph itself is too long, forcibly split
                    for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                        sub = para[i : i + self.chunk_size]
                        if sub.strip():
                            chunks.append(
                                Document(page_content=sub.strip(), metadata=current_meta)
                            )
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(Document(page_content=current_chunk, metadata=current_meta))

        return chunks
