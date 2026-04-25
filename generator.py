# generator.py
# Author: Humphrey Adjei-Kwarteng - 10022200164

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_CONTEXT_CHARS = 3000

def build_prompt(query, retrieved_chunks):
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["score"], reverse=True)
    context_parts = []
    total_chars = 0
    for r in sorted_chunks:
        text = r["chunk"]["text"]
        source = r["chunk"]["source"]
        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(f"[Source: {source}]\n{text}")
        total_chars += len(text)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an AI assistant for Academic City University.
You answer questions about Ghana's 2025 Budget and Ghana Election Results.

IMPORTANT RULES:
- Only use information from the CONTEXT below to answer.
- If the context does not contain the answer, say: "I don't have enough information in my knowledge base to answer that."
- Do not make up facts, numbers, or names.
- Be concise and factual.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt

def generate_response(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=512
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error from Groq: {str(e)}"