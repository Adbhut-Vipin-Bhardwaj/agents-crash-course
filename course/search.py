import os
import json
import hashlib
import numpy as np
from tqdm.auto import tqdm
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer

from extract_docs import read_repo_data
from chunk_docs import (
    chunk_docs_using_sliding_window,
    chunk_docs_by_section,
    chunk_docs_using_llm
)

force_refresh_embeds = False
cache_dir = "./embeds_cache"
model_str = 'multi-qa-distilbert-cos-v1'

evidently_docs = read_repo_data("evidentlyai", "docs")
print(f"Evidently documents: {len(evidently_docs)}")

char_count_chunks = chunk_docs_using_sliding_window(evidently_docs)
print(f"Evidently char count chunks: {len(char_count_chunks)}")

index = Index(
    text_fields=["section", "title", "description", "filename"],
    keyword_fields=[],
)
index.fit(char_count_chunks)


def get_cache_key(docs, model):
    docs_str = json.dumps(docs, sort_keys=True)
    return hashlib.sha256(f"{model}:{docs_str}".encode()).hexdigest()


def get_cache_path(docs, model):
    cache_key = get_cache_key(docs, model)
    cache_file = os.path.join(cache_dir, f"{cache_key}.npy")
    return cache_file


def read_from_cache(docs, model):
    cache_file = get_cache_path(docs, model)
    if os.path.exists(cache_file):
        return np.load(cache_file)
    return None


def write_to_cache(docs, model, embeddings):
    cache_file = get_cache_path(docs, model)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_file, embeddings)


embedding_model = SentenceTransformer(model_str)

embeddings = []
if not force_refresh_embeds:
    embeddings = read_from_cache(char_count_chunks, model_str)

if embeddings is None or len(embeddings) == 0:
    if embeddings is None:
        embeddings = []

    for doc in tqdm(char_count_chunks, desc="Generating embeddings"):
        doc_title = doc.get("title", "")
        doc_section = doc.get("section", "")
        text = f"{doc_title}\n{doc_section}"
        embed = embedding_model.encode(text)
        embeddings.append(embed)
    embeddings = np.array(embeddings)
    write_to_cache(char_count_chunks, model_str, embeddings)

vector_index = VectorSearch()
vector_index.fit(embeddings, char_count_chunks)


def text_search(query):
    return index.search(query, num_results=5)


def vector_search(query):
    query_embedding = embedding_model.encode(query)
    return vector_index.search(query_embedding, num_results=5)


def hybrid_search(query):
    text_results = text_search(query)
    vector_results = vector_search(query)
    
    seen_sections = set()
    final_results = []

    for result in text_results + vector_results:
        section = result["section"]
        if section not in seen_sections:
            seen_sections.add(section)
            final_results.append(result)

    return final_results


if __name__ == "__main__":
    query = "What is Evidently?"
    print(f"Query: {query}")

    # Keyword search
    text_search_results = text_search(query)
    print(f"Text Search results: {len(text_search_results)}")

    # Vector search
    vector_search_results = vector_search(query)
    print(f"Vector Search results: {len(vector_search_results)}")

    # Hybrid search
    hybrid_search_results = hybrid_search(query)
    print(f"Hybrid search results: {len(hybrid_search_results)}")
