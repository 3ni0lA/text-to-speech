# app/parse_documents.py
import PyPDF2
from docx import Document

# PDF Parsing function
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

# DOCX Parsing function
def extract_text_from_docx(file_path):
    text = ""
    doc = Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Testing the functions
if __name__ == "__main__":
    # Test files (ensure you add some test documents in the app folder for testing)
    pdf_text = extract_text_from_pdf("sample.pdf")
    docx_text = extract_text_from_docx("sample.docx")

    print("PDF Text:", pdf_text)
    print("\nDOCX Text:", docx_text)
