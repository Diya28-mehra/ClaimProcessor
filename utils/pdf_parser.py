import pdfplumber

def extract_pages(file):
    """
    Extracts text from a PDF file page by page.
    
    Args:
        file: A file-like object containing the PDF.
        
    Returns:
        A list of dictionaries, where each dictionary represents a page
        and contains "page_number" (1-indexed) and "text".
    """
    pages = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "page_number": i + 1,
                    "text": text
                })
            else:
                # Handle cases where no text is found (e.g., scanned images)
                # For this assignment, we might assume text is extractable
                pages.append({
                    "page_number": i + 1,
                    "text": "" 
                })
    return pages
