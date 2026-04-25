# ingest.py
# Author: Humphrey Adjei-Kwarteng - 10022200164
import pandas as pd
import requests
import PyPDF2
import io
import re
import json
import os

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    return text.strip()

def chunk_text(text, source):
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append({
                "id": f"{source}_{chunk_id}",
                "text": chunk.strip(),
                "source": source,
                "chunk_index": chunk_id
            })
        chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def load_election_csv():
    print("Loading election data...")
    url = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.dropna(how='all').fillna('Unknown')
    rows_as_text = []
    for _, row in df.iterrows():
        row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
        rows_as_text.append(row_text)
    full_text = clean_text(" | ".join(rows_as_text))
    chunks = chunk_text(full_text, "election_data")
    print(f"Election data: {len(chunks)} chunks")
    return chunks

def load_budget_pdf():
    print("Loading budget PDF... (this may take a minute)")
    url = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    response = requests.get(url, timeout=120)
    pdf_file = io.BytesIO(response.content)
    reader = PyPDF2.PdfReader(pdf_file)
    all_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        all_text += f" [Page {page_num+1}] {page_text}"
    all_text = clean_text(all_text)
    chunks = chunk_text(all_text, "budget_2025")
    print(f"Budget PDF: {len(chunks)} chunks")
    return chunks

def build_corpus():
    os.makedirs("data", exist_ok=True)
    election_chunks = load_election_csv()
    budget_chunks = load_budget_pdf()
    all_chunks = election_chunks + budget_chunks
    with open("data/chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"\nDone! Total chunks saved: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    build_corpus()