import os
from PyPDF2 import PdfMerger
from tqdm import tqdm

def merge_pdfs_in_directory(directory_path, output_path):
    merger = PdfMerger()

    # Get a list of all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]

    # Sort the files to ensure they are merged in order
    pdf_files.sort()

    # Append each PDF to the merger
    for pdf in pdf_files:
        merger.append(os.path.join(directory_path, pdf))

    # Write out the merged PDF
    with open(output_path, 'wb') as output_pdf:
        merger.write(output_pdf)

    merger.close()

directory_path = 'C:/Users/mrami/OneDrive/Desktop/LLM/RAG/data_pdf'
output_path = 'C:/Users/mrami/OneDrive/Desktop/LLM/RAG/data_pdf/merged_pdf/merged_pdf.pdf'
merge_pdfs_in_directory(directory_path, output_path)
