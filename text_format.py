import os
import requests
import pymupdf 
from tqdm.auto import tqdm
import pandas as pd
import random
import nltk
from tqdm import tqdm

nltk.download('punkt')

def text_formatter(text: str) -> str:
    """Performs formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() 

    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """Opens a PDF file and reads the text from each page."""
    doc = pymupdf.open(pdf_path) 
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)): 
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4, 
                                "text": text})
    return pages_and_texts


def preprocessing(pages_and_texts):
    for item in tqdm(pages_and_texts):
        item["sentences"] = nltk.sent_tokenize(item["text"])  

        item["page_sentence_count_spacy"] = len(item["sentences"]) 

    return pages_and_texts 
    

# function that recursively splits a list into desired sizes
def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

import re

def split_into_chunks(pages_and_texts):
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

