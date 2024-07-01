import torch
import textwrap
import numpy as np

def prompt_formatter(query, context_items):
    context = "- " + "\n- ".join([str(item.get('sentence_chunk', item)) for item in context_items])
    prompt = f"""Based on the following context items, please answer the query.
    Context:
    {context}
    
    User query: {query}
    Answer:"""
    return prompt

def retrieve_relevant_resources(query, embeddings, model, n_resources_to_return=5):
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    
    # Resize query_embedding if necessary
    if query_embedding.shape[0] != embeddings.shape[1]:
        query_embedding = np.resize(query_embedding, (embeddings.shape[1],))
    
    dot_scores = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(dot_scores)[-n_resources_to_return:][::-1]
    top_scores = dot_scores[top_indices]
    return top_scores, top_indices

def generate_answer_api(prompt):
    generation_config = genai.GenerationConfig(
        stop_sequences=None,
        temperature=0.0,
        max_output_tokens=5000,
        top_p=0.0,
        top_k=0,
    )
    
    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
    return response.text

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def dot_product(vector1, vector2):
    return torch.dot(vector1, vector2)

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)

    # Get Euclidean/L2 norm of each vector (removes the magnitude, keeps direction)
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))

    return dot_product / (norm_vector1 * norm_vector2)
