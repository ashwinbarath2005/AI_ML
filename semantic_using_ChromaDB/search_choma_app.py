import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

# ==========================
# 1. Initialize ChromaDB
# ==========================
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="products")

# ==========================
# 2. Load embedding model
# ==========================
model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================
# 3. Page Config & CSS
# ==========================
st.set_page_config(page_title="E-commerce Semantic Search", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #f4f6f9;
    }
    .search-box {
        width: 60%;
        margin: 20px auto;
        position: relative;
    }
    .search-input {
        width: 100%;
        padding: 15px 20px;
        border-radius: 50px;
        border: 2px solid #4CAF50;
        font-size: 18px;
        outline: none;
        transition: 0.3s;
    }
    .search-input:focus {
        border-color: #2e7d32;
        box-shadow: 0 0 8px rgba(46,125,50,0.4);
    }
    .product-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        transition: transform 0.2s ease-in-out;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.15);
    }
    .product-title {
        font-size: 20px;
        font-weight: bold;
        color: #333;
        margin-bottom: 8px;
    }
    .product-desc {
        font-size: 16px;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# 4. CSV Upload
# ==========================
st.title("🛍️ E-commerce Semantic Search")
uploaded_file = st.file_uploader("Upload CSV (with `name` and `description`)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "name" not in df.columns or "description" not in df.columns:
        st.error("CSV must contain 'name' and 'description' columns!")
    else:
        st.success(f"✅ Uploaded {len(df)} products successfully!")

        # Store embeddings (batch for speed)
        embeddings = model.encode(df["description"].tolist(), batch_size=32, show_progress_bar=True)
        for idx, row in df.iterrows():
            collection.add(
                ids=[str(row.get("id", idx))],
                metadatas=[{"name": row["name"], "description": row["description"]}],
                embeddings=[embeddings[idx].tolist()],
            )

        st.success("✅ Product embeddings stored in VectorDB successfully!")

        # ==========================
        # 5. Semantic Search (Custom Input)
        # ==========================
        query = st.text_input("🔍 Search Products", key="search", label_visibility="collapsed")

        top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

        if query:
            query_emb = model.encode(query).tolist()
            results = collection.query(query_embeddings=[query_emb], n_results=top_k)

            st.markdown("### 🔎 Search Results")
            for i in range(len(results["ids"][0])):
                name = results["metadatas"][0][i]["name"]
                desc = results["metadatas"][0][i]["description"]

                st.markdown(
                    f"""
                    <div class="product-card">
                        <div class="product-title">{name}</div>
                        <div class="product-desc">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
