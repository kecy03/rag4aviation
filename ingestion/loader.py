import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document


def load_documents(source_dir: str) -> list[Document]:
    """Load all supported documents from a directory. Dispatches to format-specific loaders."""
    all_docs: list[Document] = []
    if not os.path.exists(source_dir):
        print(f"Directory not found: {source_dir}")
        return all_docs

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path):
            try:
                docs = _load_single_file(file_path)
                all_docs.extend(docs)
                print(f"  Loaded: {filename} ({len(docs)} pages/sections)")
            except Exception as e:
                print(f"  Skipped {filename}: {e}")
    return all_docs


def _load_single_file(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return _load_docx(file_path)
    elif ext in (".html", ".htm"):
        return _load_html(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        from ingestion.ocr import ocr_image

        text = ocr_image(file_path)
        if text:
            doc = Document(
                page_content=text,
                metadata={"source": file_path, "page": 0, "format": "image"},
            )
            return [doc]
        return []
    else:
        print(f"  Unsupported format: {ext}")
        return []


def _load_pdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["format"] = "pdf"
        # If the page has essentially no text, it may be scanned
        if len(doc.page_content.strip()) < 20:
            from ingestion.ocr import ocr_pdf_page

            page_num = doc.metadata.get("page", 0)
            text = ocr_pdf_page(file_path, page_num)
            if text:
                doc.page_content = text
                doc.metadata["ocr_applied"] = True
    return docs


def _load_docx(file_path: str) -> list[Document]:
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["format"] = "docx"
    return docs


def _load_html(file_path: str) -> list[Document]:
    from bs4 import BeautifulSoup

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Split into sections by heading markers
    docs: list[Document] = []
    current_title = os.path.basename(file_path)
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("第") and ("章" in line or "节" in line):
            if current_lines:
                docs.append(
                    Document(
                        page_content="\n".join(current_lines),
                        metadata={
                            "source": file_path,
                            "format": "html",
                            "section_title": current_title,
                        },
                    )
                )
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        docs.append(
            Document(
                page_content="\n".join(current_lines),
                metadata={
                    "source": file_path,
                    "format": "html",
                    "section_title": current_title,
                },
            )
        )
    return docs
