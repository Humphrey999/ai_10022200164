# pipeline.py
# Author: Humphrey Adjei-Kwarteng - 10022200164

import json
import os
from datetime import datetime
from retriever import Retriever
from generator import build_prompt, generate_response

os.makedirs("logs", exist_ok=True)
retriever = Retriever()

def initialize():
    if os.path.exists("data/faiss.index"):
        retriever.load_index()
    else:
        from ingest import build_corpus
        chunks = build_corpus()
        retriever.build_index(chunks)

def run_pipeline(query, top_k=5):
    timestamp = datetime.now().isoformat()
    log = {"timestamp": timestamp, "query": query, "stages": {}}

    # STAGE 1: RETRIEVAL
    results = retriever.retrieve_with_fallback(query, top_k=top_k)
    log["stages"]["retrieval"] = {
        "expanded_query": results[0]["expanded_query"] if results else query,
        "retrieved_chunks": [
            {
                "id": r["chunk"]["id"],
                "source": r["chunk"]["source"],
                "score": round(r["score"], 4),
                "preview": r["chunk"]["text"][:120] + "...",
                "low_confidence": r.get("low_confidence", False)
            }
            for r in results
        ]
    }

    # STAGE 2: PROMPT CONSTRUCTION
    prompt = build_prompt(query, results)
    log["stages"]["prompt"] = {
        "prompt_length": len(prompt),
        "full_prompt": prompt
    }

    # STAGE 3: LLM GENERATION
    response = generate_response(prompt)
    log["stages"]["generation"] = {"response": response}

    # SAVE LOG
    log_file = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)

    return {
        "query": query,
        "retrieved": log["stages"]["retrieval"]["retrieved_chunks"],
        "prompt": prompt,
        "response": response,
        "log_file": log_file
    }