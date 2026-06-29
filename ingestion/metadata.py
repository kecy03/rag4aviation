import os
import re

from langchain_core.documents import Document


def enrich_metadata(documents: list[Document], source_dir: str) -> list[Document]:
    """Add document_type, aircraft_model, collection to each Document's metadata."""
    for doc in documents:
        source = doc.metadata.get("source", "")
        doc.metadata["document_type"] = infer_document_type(source, source_dir)
        doc.metadata["aircraft_model"] = infer_aircraft_model(
            source, doc.page_content
        )
        doc.metadata["collection"] = infer_collection(source_dir)
        doc.metadata["section_title"] = doc.metadata.get("section_title", "")
    return documents


def infer_document_type(file_path: str, source_dir: str) -> str:
    """Return 'manual' | 'regulation' | 'textbook' based on path heuristics."""
    path_lower = file_path.lower()
    dir_name = os.path.basename(source_dir).lower()

    if "regulation" in dir_name or "regulation" in path_lower or "法规" in path_lower:
        return "regulation"
    if "manual" in dir_name or "manual" in path_lower or "手册" in path_lower:
        return "manual"
    if "textbook" in dir_name or "textbook" in path_lower or "教材" in path_lower:
        return "textbook"
    if "structure" in dir_name or "构造" in path_lower:
        return "textbook"
    if "ops" in dir_name or "flight" in dir_name or "操作" in path_lower:
        return "manual"
    return "manual"


_AIRCRAFT_KEYWORDS: list[tuple[str, str]] = [
    ("B737", r"\bB737\b|波音\s*737|Boeing\s*737"),
    ("B747", r"\bB747\b|波音\s*747|Boeing\s*747"),
    ("B777", r"\bB777\b|波音\s*777|Boeing\s*777"),
    ("B787", r"\bB787\b|波音\s*787|Boeing\s*787"),
    ("A320", r"\bA320\b|空客\s*320|Airbus\s*A320"),
    ("A330", r"\bA330\b|空客\s*330|Airbus\s*A330"),
    ("A350", r"\bA350\b|空客\s*350|Airbus\s*A350"),
    ("A380", r"\bA380\b|空客\s*380|Airbus\s*A380"),
    ("C919", r"\bC919\b|商飞\s*C919|COMAC\s*C919"),
    ("ARJ21", r"\bARJ21\b|商飞\s*ARJ21"),
]


def infer_aircraft_model(file_path: str, text: str) -> str:
    """Return aircraft model from filename or text content using keyword matching."""
    combined = file_path + " " + text[:2000]
    for model, pattern in _AIRCRAFT_KEYWORDS:
        if re.search(pattern, combined, re.IGNORECASE):
            return model
    return "unknown"


def infer_collection(source_dir: str) -> str:
    """Map source directory name to a collection name."""
    dir_name = os.path.basename(source_dir).lower()
    if "structure" in dir_name or "构造" in dir_name:
        return "aircraft_structure"
    if "ops" in dir_name or "flight" in dir_name or "操作" in dir_name or "飞行" in dir_name:
        return "flight_ops"
    if "regulation" in dir_name or "法规" in dir_name:
        return "regulations"
    return "aircraft_structure"
