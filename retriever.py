# retriever.py
# Author: Humphrey Adjei-Kwarteng - 10022200164

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks = []
        self.index = None

    def build_index(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        print(f"Embedding {len(texts)} chunks... (this takes 2-3 mins)")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        faiss.write_index(self.index, "data/faiss.index")
        with open("data/chunks.json", "w") as f:
            json.dump(chunks, f)
        print("Index built and saved!")

    def load_index(self):
        self.index = faiss.read_index("data/faiss.index")
        with open("data/chunks.json") as f:
            self.chunks = json.load(f)
        print(f"Index loaded: {len(self.chunks)} chunks")

    def expand_query(self, query):
        expansions = {
            "election": "election voting results winner candidate votes NPP NDC year region",
            "budget": "budget expenditure revenue fiscal policy Ghana 2025",
            "ndc": "NDC National Democratic Congress John Mahama votes",
            "npp": "NPP New Patriotic Party Akufo Addo votes",
            "economy": "economy GDP growth inflation",
            "education": "education schools policy spending",
            "health": "health hospitals medical spending",
            "who won": "votes winner candidate NPP NDC election results",
            "winner": "votes winner candidate NPP NDC election results",
            "votes": "votes candidate party NPP NDC election year region",
            "results": "election results votes candidate party region year"
        }
        extra = []
        for keyword, expansion in expansions.items():
            if keyword.lower() in query.lower():
                extra.append(expansion)
        return (query + " " + " ".join(extra)).strip()

    def retrieve(self, query, top_k=5):
        expanded_query = self.expand_query(query)
        query_vec = self.model.encode([expanded_query]).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "original_query": query,
                    "expanded_query": expanded_query
                })
        return results

    def retrieve_with_fallback(self, query, top_k=5, threshold=0.25):
        results = self.retrieve(query, top_k)
        for r in results:
            r["low_confidence"] = results[0]["score"] < threshold
        return results