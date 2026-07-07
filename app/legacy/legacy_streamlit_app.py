import streamlit as st
import requests

st.title("FireLens 🔥")
user_input = st.text_area(label="Send Message", label_visibility="hidden", placeholder="Chat with FireLens ...")
if st.button(label="Send"):
    resp = requests.post("http://localhost:8000/", json={"prompt": user_input})
    choices = resp.json()["choices"][0] 
    st.markdown(choices["message"]["content"])
