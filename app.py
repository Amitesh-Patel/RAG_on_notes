import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import textwrap
import google.generativeai as genai
from utils import retrieve_relevant_resources , prompt_formatter , generate_answer_api

@st.cache_resource
def load_data():
    df = pd.read_csv("text_chunks_and_embeddings_df.csv")
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return df

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

#local models - https://huggingface.co/unsloth


def main():
    st.title("Personal Notes Query System")

    df = load_data()
    pages_and_chunks = df.to_dict(orient="records")
    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)

    embedding_model_name = st.selectbox(
        "Choose an embedding model",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
    )

    api_key = st.text_input("Enter your Gemini API key:", type="password")
    if api_key:
        genai.configure(api_key=api_key)
    
    embedding_model = load_embedding_model(embedding_model_name)

    query = st.text_input("Enter your query:")

    if query and api_key:
        _ , indices = retrieve_relevant_resources(query, embeddings, embedding_model)
        context_items = [pages_and_chunks[i] for i in indices]
        
        print("Context items:", context_items)

        st.subheader("Extracted Context:")
        for item in context_items:
            st.write(f"- {str(item['sentence_chunk'])}")

        prompt = prompt_formatter(query, context_items)
        answer = generate_answer_api(prompt)

        st.subheader("Answer:")
        st.write(textwrap.fill(answer, 80))
    elif query and not api_key:
        st.error("Please enter your Gemini API key.")

if __name__ == "__main__":
    main()