import os
import pdfkit  
from tqdm import *

def html_to_pdf(paths):
    error_count = 0
    directory = "C:/Users/mrami/OneDrive/Desktop/LLM/RAG/data/"
    config = pdfkit.configuration(wkhtmltopdf = r"C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe")  
    for e,path in tqdm(enumerate(paths)):
        try:
            path = os.path.join(directory,path)
            pdfkit.from_file(path, f'data_pdf/notes_{e}.pdf', configuration = config)  
        except Exception as e:
            print("ERROR:",e)
            error_count += 1
    print(f"Total number of html files converted to pdf out of {len(paths)}: {len(paths) - error_count}")

if __name__ == '__main__':
    paths = os.listdir("data/")
    html_to_pdf(paths)
    print("Done")


# Total number of html files converted to pdf out of 131: 99