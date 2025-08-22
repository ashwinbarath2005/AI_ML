#======================================================
# chatbot_streamlit.py — Streamlit chat UI
# Run: streamlit run chatbot_streamlit.py
#======================================================
import streamlit as st
from inference_utils import load_artifacts, greedy_decode

MODEL_DIR = 'artifacts'

@st.cache_resource
def _load():
    model, tok, meta = load_artifacts(MODEL_DIR)
    return model, tok, meta
 
st.set_page_config(page_title="GRU Chatbot", page_icon="        ")

st.title("GRU Chatbot")
st.caption("Seq2Seq with GRU — trained on your CSV (prompt,reply)")

model, tok, meta = _load()

if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type your message…")

chat = st.container()
with chat:
    for role, text in st.session_state.history:
        st.chat_message(role).markdown(text)

if user_input:
    st.session_state.history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)
    reply = greedy_decode(model, tok, meta, user_input)
    st.session_state.history.append(("assistant", reply))
    st.chat_message("assistant").markdown(reply)
