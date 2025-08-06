import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper to read uploaded files
def read_file(file):
    return file.read().decode("utf-8")

st.set_page_config(page_title=" Resume Ranker", page_icon="📄")
st.title("Resume Ranker by Job Match")

# Upload resumes (multiple files)
resume_files = st.file_uploader("Upload Resumes (TXT or PDF)", type=["txt"], accept_multiple_files=True)

# Upload job description
job_desc = st.text_area("Paste Job Description")

# When both are provided
if st.button("Rank Resumes") and resume_files and job_desc:
    resume_texts = [read_file(file) for file in resume_files]
    names = [file.name for file in resume_files]
    documents = [job_desc] + resume_texts

    # TF-IDF and Cosine Similarity
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(documents)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    # Rank resumes
    ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)

    st.subheader("Ranked Resumes")
    for i, (name, score) in enumerate(ranked, 1):
        st.write(f"{i}. **{name}** - Similarity: `{score:.2f}`")
