# app.py
# Author: Humphrey Adjei-Kwarteng - 10022200164

import streamlit as st
from pipeline import initialize, run_pipeline

st.set_page_config(page_title="ACity RAG Chatbot", page_icon="🎓", layout="wide")

st.title("🎓 Academic City University RAG Chatbot")
st.caption("Ask questions about Ghana Election Results & the 2025 Budget Statement")

# Initialize on first load
if "initialized" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        initialize()
    st.session_state.initialized = True
    st.session_state.history = []

# Suggested questions
st.markdown("**Try asking:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Who won the 2020 Ghana election?"):
        st.session_state.suggested = "Who won the 2020 Ghana election?"
with col2:
    if st.button("What is Ghana's 2025 budget revenue?"):
        st.session_state.suggested = "What is Ghana's 2025 budget revenue?"
with col3:
    if st.button("How many votes did NDC get in Accra?"):
        st.session_state.suggested = "How many votes did NDC get in Accra?"

st.divider()

# Chat input
query = st.chat_input("Ask anything about Ghana elections or the 2025 budget...")

# Handle suggested question clicks
if "suggested" in st.session_state and st.session_state.suggested:
    query = st.session_state.suggested
    st.session_state.suggested = None

if query:
    with st.spinner("Searching knowledge base and generating response..."):
        result = run_pipeline(query)
    st.session_state.history.append(result)

# Display chat history
for item in reversed(st.session_state.history):
    st.chat_message("user").write(item["query"])

    with st.chat_message("assistant"):
        st.write(item["response"])

        with st.expander("🔍 Retrieved Chunks & Similarity Scores"):
            for chunk in item["retrieved"]:
                confidence = "⚠️ Low confidence" if chunk["low_confidence"] else "✅ Good match"
                st.markdown(f"""
**{confidence}** | **Source:** `{chunk['source']}` | **Score:** `{chunk['score']}`
> {chunk['preview']}
---
""")

        with st.expander("📋 Final Prompt Sent to LLM"):
            st.code(item["prompt"])

        with st.expander("📁 Log File"):
            st.write(item["log_file"])