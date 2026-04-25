# RAG System Architecture
**Name:** Humphrey Adjei-Kwarteng
**Index Number:** 10022200164

## Component Breakdown

**Data Ingestion Layer**
- `ingest.py` loads Ghana Election CSV and 2025 Budget PDF
- Text is cleaned, chunked (500 chars, 100 overlap) and saved to `chunks.json`
- Embeddings built using `all-MiniLM-L6-v2` and stored in FAISS index

**Retrieval Layer (retriever.py)**
- Query expansion adds domain keywords before embedding
- MiniLM encodes query to 384-dim vector
- FAISS cosine similarity returns top-5 chunks
- Low confidence threshold (0.25) flags poor matches

**Prompt Layer (generator.py)**
- Retrieved chunks injected into structured prompt template
- Hallucination controlled by strict context-only instruction
- Context window capped at 3000 chars, ranked by score

**Generation Layer**
- Groq LLaMA 3.1 8B at temperature 0.2 for factual responses
- Max 512 tokens per response

**Logging (pipeline.py)**
- Every stage logged to JSON in `/logs` folder
- Logs include: query, expanded query, chunks, scores, prompt, response