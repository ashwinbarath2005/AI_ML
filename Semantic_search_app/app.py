from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Product dataset (no images needed)
products = [
    {"name": "Nike Running Shoes"},
    {"name": "Adidas Sneakers"},
    {"name": "Puma Sports T-shirt"},
    {"name": "Apple iPhone 14"},
    {"name": "Samsung Galaxy S23"},
    {"name": "Sony Headphones"},
    {"name": "Levi’s Jeans"},
    {"name": "Under Armour Hoodie"},
]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
product_names = [p["name"] for p in products]
product_embeddings = model.encode(product_names)

def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, product_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(products[idx], similarities[idx]) for idx in top_indices]
    return results

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form["query"]
        results = semantic_search(query, top_k=3)
    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)
