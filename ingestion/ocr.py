import os

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def ocr_image(image_path: str) -> str:
    """Run OCR on a single image file, return extracted text."""
    if not OCR_AVAILABLE:
        print(f"  OCR not available (install pytesseract, Pillow, pdf2image)")
        return ""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text.strip()
    except Exception as e:
        print(f"  OCR failed for {image_path}: {e}")
        return ""


def ocr_pdf_page(pdf_path: str, page_num: int) -> str:
    """Render a PDF page to image then OCR it. For scanned PDFs."""
    if not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_path(
            pdf_path, first_page=page_num + 1, last_page=page_num + 1
        )
        if images:
            text = pytesseract.image_to_string(images[0], lang="chi_sim+eng")
            return text.strip()
    except Exception as e:
        print(f"  OCR failed for {pdf_path} page {page_num}: {e}")
    return ""
