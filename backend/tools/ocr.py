import pytesseract
from PIL import Image
from pdf2image import convert_from_path


def extract_text_from_file(file_path):
    """
    Extract text via OCR from either an image (PNG/JPG) or a PDF.
    PDFs are first rendered page-by-page to images with pdf2image,
    then each page is OCR'd with the same pytesseract call used for
    plain images. Text from all pages is joined together.
    """
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path)
        text_parts = [pytesseract.image_to_string(page) for page in pages]
        return "\n".join(text_parts)
    else:
        return pytesseract.image_to_string(Image.open(file_path))