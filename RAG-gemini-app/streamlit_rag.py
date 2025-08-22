import os
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-1.5-pro"

# ---------- Utils ----------
def chunk_text(text, size=500, overlap=100):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def embed(texts, task_type="retrieval_document"):
    """Embed a list of texts into vectors"""
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=[{"parts": [t]} for t in texts],
        task_type=task_type
    )

    # Case 1: batch embeddings
    if "embeddings" in resp:
        return np.array([e["values"] for e in resp["embeddings"]], dtype="float32")

    # Case 2: single embedding as dict with values
    if isinstance(resp.get("embedding"), dict) and "values" in resp["embedding"]:
        return np.array([resp["embedding"]["values"]], dtype="float32")

    # Case 3: single embedding as list
    if isinstance(resp.get("embedding"), list):
        emb = resp["embedding"]
        # Nested list: [[float, float, ...]]
        if len(emb) == 1 and isinstance(emb[0], list):
            return np.array(emb, dtype="float32")
        # Flat list: [float, float, ...]
        if all(isinstance(v, (int, float)) for v in emb):
            return np.array([emb], dtype="float32")

    raise ValueError(f"Unexpected response format from embed_content: {resp}")

def embed_query(query):
    """Embed a single query into a vector"""
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )

    # Case 1: dict with values
    if isinstance(resp.get("embedding"), dict) and "values" in resp["embedding"]:
        return np.array(resp["embedding"]["values"], dtype="float32")[None, :]

    # Case 2: list
    if isinstance(resp.get("embedding"), list):
        emb = resp["embedding"]
        # Nested list [[floats]]
        if len(emb) == 1 and isinstance(emb[0], list):
            return np.array(emb, dtype="float32")
        # Flat list [floats]
        if all(isinstance(v, (int, float)) for v in emb):
            return np.array(emb, dtype="float32")[None, :]

    # Case 3: batch embeddings
    if "embeddings" in resp:
        return np.array([e["values"] for e in resp["embeddings"]], dtype="float32")

    raise ValueError(f"Unexpected response format from embed_content: {resp}")

def read_file(uploaded_file):
    """Reads txt or pdf file and returns text"""
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    return ""

def build_index(files):
    """Build FAISS index from uploaded files"""
    docs, chunks = [], []
    for uploaded_file in files:
        text = read_file(uploaded_file)
        for c in chunk_text(text):
            docs.append({"text": c, "source": uploaded_file.name})
            chunks.append(c)

    X = embed(chunks)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    return index, docs

def retrieve(index, docs, query, k=3):
    """Retrieve top-k similar chunks"""
    qvec = embed_query(query)
    D, I = index.search(qvec, k)
    return [docs[i] for i in I[0]]

def answer(query, passages):
    """Generate answer using Gemini with retrieved passages"""
    context = "\n\n".join([f"Source: {p['source']}\n{p['text']}" for p in passages])
    prompt = f"Answer the question using only the sources.\n\nQuestion: {query}\n\nSources:\n{context}\n\nAnswer:"
    
    model = genai.GenerativeModel(LLM_MODEL)
    resp = model.generate_content(prompt)
    return resp.text

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="=")
st.title("Gemini RAG Chatbot")
st.caption("Upload your custom dataset (.txt / .pdf) and chat with Gemini + RAG")

# File uploader
uploaded_files = st.file_uploader("Upload text or PDF files", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    if "index" not in st.session_state:
        with st.spinner("Building vector index..."):
            st.session_state.index, st.session_state.docs = build_index(uploaded_files)
        st.success("Index built! You can now chat with your data.")

    query = st.text_input("Ask a question about your documents:")
    if st.button("Ask") and query.strip():
        with st.spinner("Thinking..."):
            passages = retrieve(st.session_state.index, st.session_state.docs, query, k=3)
            ans = answer(query, passages)

        st.markdown("### Answer")
        st.write(ans)

        st.markdown("### Sources")
        for i, p in enumerate(passages, 1):
            st.write(f"[{i}] **{p['source']}** — {p['text'][:150]}...")
else:
    st.info("Upload one or more `.txt` or `.pdf` files to start chatting.")
