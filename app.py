
import streamlit as st
from hello_poetry.hello import personalized_hello

st.set_page_config(
    page_title="Hello Poetry",
    page_icon="👋",
    layout="centered",
)

st.markdown("""
<style>
    .block-container {
        max-width: 650px;
        padding-top: 5rem;
    }

    h1 {
        font-weight: 700;
        letter-spacing: -1px;
    }

    div.stButton > button {
        border-radius: 8px;
        height: 45px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("Hello, World! 👋")
st.caption("A simple Python app built with Poetry and Streamlit.")

st.divider()

name = st.text_input(
    "Your name",
    placeholder="Enter your name",
)

language = st.selectbox(
    "Language",
    ["English", "Arabic"],
)

if st.button("Say Hello", use_container_width=True):
    st.success(personalized_hello(name, language))

st.divider()

st.caption("Python · Poetry · Streamlit")
