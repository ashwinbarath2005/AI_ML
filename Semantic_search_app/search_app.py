import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from streamlit_lottie import st_lottie

# --- Step 1: Product dataset ---
products = [
    {"name": "Nike Running Shoes", "image": "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.nike.com%2Fin%2Ft%2Finteract-run-easyon-road-running-shoes-6mKGXc&psig=AOvVaw3gTapGOtQ0e4U73vLhfgVB&ust=1755942723697000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCOjcyumSno8DFQAAAAAdAAAAABAE"},
    {"name": "Adidas Sneakers", "image": "https://upload.wikimedia.org/wikipedia/commons/2/20/Adidas_Stan_Smith_white_green.jpg"},
    {"name": "Puma Sports T-shirt", "image": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Puma_tshirt.jpg"},
    {"name": "Apple iPhone 14", "image": "https://upload.wikimedia.org/wikipedia/commons/f/fa/IPhone_14_Pro_vector.svg"},
    {"name": "Samsung Galaxy S23", "image": "https://upload.wikimedia.org/wikipedia/commons/8/8e/Samsung_Galaxy_S23.png"},
    {"name": "Sony Headphones", "image": "https://upload.wikimedia.org/wikipedia/commons/f/f9/Sony_WH-1000XM4.jpg"},
    {"name": "Levi’s Jeans", "image": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Levi%27s_501_jeans.jpg"},
    {"name": "Under Armour Hoodie", "image": "https://upload.wikimedia.org/wikipedia/commons/2/25/Under_Armour_Hoodie.jpg"},
]

# --- Step 2: Load embedding model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Precompute embeddings
product_names = [p["name"] for p in products]
product_embeddings = model.encode(product_names)

# --- Step 3: Semantic Search ---
def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, product_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(products[idx], similarities[idx]) for idx in top_indices]
    return results

# --- Step 4: Load Lottie Animation ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_dance = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_touohxv0.json")  # dancing animation

# --- UI Setup ---
st.set_page_config(page_title="E-Commerce Semantic Search", page_icon="🛒", layout="wide")

st.title("🛒 E-Commerce Semantic Search")
st.write("Find the most relevant products instantly! Type what you’re looking for 👇")

query = st.text_input("🔍 Search for products (e.g., 'sports shoes', 'latest iPhone')")

top_k = st.slider("Number of results", 1, 5, 3)

# --- Display results ---
if query:
    results = semantic_search(query, top_k=top_k)

    # 🎉 Fun animation
    st_lottie(lottie_dance, height=200, key="dance")

    st.subheader("✨ Search Results:")

    cols = st.columns(top_k)
    for i, (product, score) in enumerate(results):
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    border-radius: 20px;
                    padding: 15px;
                    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
                    text-align: center;
                    transition: transform 0.2s;
                " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    <img src="{product['image']}" style="width:100%; border-radius:15px;"/>
                    <h3 style="margin-top:10px;">{product['name']}</h3>
                    <p>🔗 Similarity Score: <b>{score:.2f}</b></p>
                    <a href="https://www.amazon.in/s?k={product['name'].replace(' ', '+')}" target="_blank">
                        <button style="
                            background:#FF4B4B;
                            color:white;
                            padding:10px 20px;
                            border:none;
                            border-radius:12px;
                            cursor:pointer;
                        ">🛒 Buy Now</button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
