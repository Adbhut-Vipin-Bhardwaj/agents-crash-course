import os
import json
import hashlib
import numpy as np
from typing import Any
from tqdm.auto import tqdm
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer

from extract_docs import read_repo_data
from chunk_docs import (
    chunk_docs_using_sliding_window,
    chunk_docs_by_section,
    chunk_docs_using_llm
)

class SearchEngine:
    def __init__(
        self, 
        owner: str = "evidentlyai", 
        repo: str = "docs", 
        model_name: str = 'multi-qa-distilbert-cos-v1',
        cache_dir: str = "./embeds_cache"
    ):
        self.owner = owner
        self.repo = repo
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.vector_index = None
        self.chunks = None

    def initialize(self, force_refresh: bool = False):
        """Initialize both text and vector indices."""
        print(f"Initializing SearchEngine for {self.owner}/{self.repo}...")
        docs = read_repo_data(self.owner, self.repo)
        self.chunks = chunk_docs_using_sliding_window(docs)
        
        # Text Index
        self.index = Index(
            text_fields=["section", "title", "description", "filename"],
            keyword_fields=[],
        )
        self.index.fit(self.chunks)
        
        # Vector Index
        embeddings = None
        if not force_refresh:
            embeddings = self._read_from_cache(self.chunks)
            
        if embeddings is None or len(embeddings) == 0:
            embeddings = []
            for doc in tqdm(self.chunks, desc=f"Generating embeddings for {self.owner}/{self.repo}"):
                doc_title = doc.get("title", "")
                doc_section = doc.get("section", "")
                text = f"{doc_title}\n{doc_section}"
                embed = self.embedding_model.encode(text)
                embeddings.append(embed)
            embeddings = np.array(embeddings)
            self._write_to_cache(self.chunks, embeddings)
            
        self.vector_index = VectorSearch()
        self.vector_index.fit(embeddings, self.chunks)
        print("Initialization complete.")

    def _get_cache_key(self, docs):
        docs_str = json.dumps(docs, sort_keys=True)
        return hashlib.sha256(f"{self.model_name}:{docs_str}".encode()).hexdigest()

    def _get_cache_path(self, docs):
        cache_key = self._get_cache_key(docs)
        return os.path.join(self.cache_dir, f"{cache_key}.npy")

    def _read_from_cache(self, docs):
        cache_file = self._get_cache_path(docs)
        if os.path.exists(cache_file):
            return np.load(cache_file)
        return None

    def _write_to_cache(self, docs, embeddings):
        cache_file = self._get_cache_path(docs)
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(cache_file, embeddings)

    def text_search(self, query: str, num_results: int = 5) -> list[Any]:
        if not self.index:
            raise RuntimeError("SearchEngine not initialized. Call .initialize() first.")
        return self.index.search(query, num_results=num_results)

    def vector_search(self, query: str, num_results: int = 5) -> list[Any]:
        if not self.vector_index:
            raise RuntimeError("SearchEngine not initialized. Call .initialize() first.")
        query_embedding = self.embedding_model.encode(query)
        return self.vector_index.search(query_embedding, num_results=num_results)

    def hybrid_search(self, query: str, num_results: int = 5) -> list[Any]:
        text_results = self.text_search(query, num_results=num_results)
        vector_results = self.vector_search(query, num_results=num_results)
        
        seen_sections = set()
        final_results = []

        for result in text_results + vector_results:
            section = result["section"]
            if section not in seen_sections:
                seen_sections.add(section)
                final_results.append(result)
            if len(final_results) >= num_results:
                break

        return final_results

if __name__ == "__main__":
    # Test with default dataset
    engine = SearchEngine()
    engine.initialize()
    
    query = "What is Evidently?"
    print(f"\nQuery: {query}")

    # Keyword search
    text_results = engine.text_search(query)
    print(f"Text Search results: {len(text_results)}")

    # Vector search
    vector_results = engine.vector_search(query)
    print(f"Vector Search results: {len(vector_results)}")

    # Hybrid search
    hybrid_results = engine.hybrid_search(query)
    print(f"Hybrid search results: {len(hybrid_results)}")
