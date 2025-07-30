import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📚 Book Recommender System")

# Load data
books = pd.read_csv("books.csv")

# --- CONTENT-BASED ---
books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

# --- USER-BASED ---
ratings = books.drop(['Title', 'Author', 'Genre', 'features'], axis=1)
user_similarity = cosine_similarity(ratings.fillna(0))

# --- SELECT BOOK ---
book_titles = books['Title'].tolist()
selected = st.selectbox("Choose a Book", book_titles)
book_index = books[books['Title'] == selected].index[0]

if st.button("Recommend"):
    st.write("## 🔍 Recommended Books")

    # --- Content-Based ---
    st.subheader("📘 Content-Based")
    content_books = content_similarity[book_index].argsort()[::-1][1:4]
    for i in content_books:
        st.write("- " + books.iloc[i]['Title'])

    # --- User-Based ---
    st.subheader("👤 User-Based")
    book_ratings = ratings.iloc[book_index].fillna(0)
    user_scores = cosine_similarity([book_ratings], ratings.fillna(0))[0]
    user_books = user_scores.argsort()[::-1][1:4]
    for i in user_books:
        st.write("- " + books.iloc[i]['Title'])

    # --- Hybrid ---
    st.subheader("🔀 Hybrid")
    hybrid_scores = 0.5 * content_similarity[book_index] + 0.5 * user_scores
    hybrid_books = hybrid_scores.argsort()[::-1][1:4]
    for i in hybrid_books:
        st.write("- " + books.iloc[i]['Title'])
