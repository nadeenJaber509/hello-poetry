"""Minimal Streamlit Cloud diagnostic app."""

import sys

import streamlit as st

st.set_page_config(page_title="Cloud Check")

st.title("Cloud Environment Check")
st.success("Streamlit started successfully!")

st.write("Python:", sys.version)
st.write("Streamlit:", st.__version__)