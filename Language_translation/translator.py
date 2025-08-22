import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
import requests

# Available translation models
translation_models = {
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Italian": "Helsinki-NLP/opus-mt-en-it",
    "Portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "Russian": "Helsinki-NLP/opus-mt-en-ru",
    "Chinese": "Helsinki-NLP/opus-mt-en-zh"
}

# Load animation from Lottie
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_translate = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_svy4ivvy.json")

# Streamlit page config
st.set_page_config(page_title="🌍 AI Translator", page_icon="🌐", layout="centered")

st.title("🌍 AI Translator")
st.markdown("### Translate English text into multiple languages ✨")

# Show animation
st_lottie(lottie_translate, height=200, key="translation")

add_vertical_space(1)

# User input
text = st.text_area("✍️ Enter your English text here:", height=120, placeholder="Type something like 'Hello, how are you?'...")

language = st.selectbox("🌐 Choose target language:", list(translation_models.keys()))

# Translation function
@st.cache_resource
def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Button
if st.button("✨ Translate Now"):
    if text.strip():
        tokenizer, model = load_model(translation_models[language])
        with st.spinner("Translating... ⏳"):
            result = translate(text, model, tokenizer)
        st.success(f"✅ Translation in {language}:")
        st.markdown(f"### {result}")
    else:
        st.warning("⚠️ Please enter some text to translate.")
