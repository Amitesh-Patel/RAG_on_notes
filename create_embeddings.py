from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
import os
import pandas as pd
import config

embedding_model = SentenceTransformer(model_name_or_path=config.EMBEDDING_MODEL,
                                      device="cuda")

from text_format import open_and_read_pdf , preprocessing , split_into_chunks , split_list

def get_embeddings_df():
    pdf_path = "data_pdf/merged_pdf/merged_pdf.pdf"

    if not os.path.exists(pdf_path):
        print("File doesn't exist, Please provide correct path")
    pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
    pages_and_texts = preprocessing(pages_and_texts=pages_and_texts)
    df = pd.DataFrame(pages_and_texts)
    num_sentence_chunk_size = 10
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                            slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    df = pd.DataFrame(pages_and_texts)

    pages_and_chunks = split_into_chunks(pages_and_texts=pages_and_texts)
    df = pd.DataFrame(pages_and_chunks)
    min_token_length = config.MIN_TOKEN_LENGTH
    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    print("after chunking ",len(pages_and_chunks))
    return pages_and_chunks_over_min_token_len


pages_and_chunks_over_min_token_len = get_embeddings_df()

for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])

text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]

text_chunk_embeddings = embedding_model.encode(text_chunks,
                                            batch_size=config.BATCH_SIZE, 
                                            convert_to_tensor=True)


text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False,escapechar='\\')

# text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
print("Done !!")