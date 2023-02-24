import time
import requests


import streamlit as st
import pandas as pd
import numpy as np


# TODO: step 0: Quick env setup.
# TODO: step 1: Streamlit 101.
# TODO: step 2: Emulate a model service using HuggingFace inference API (https://huggingface.co/valhalla/distilbart-mnli-12-3).
# TODO: step 3: Deploy the app (create GitHub repo (public/private), init local Git repo, secrets).

st.set_page_config(
    page_title="The AI Epiphany",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

API_KEY = st.secrets["API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
headers = {"Authorization": f"Bearer {API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

with st.form(key='my_form'):
    labels = st.multiselect('Choose your labels', ["bro", "sis"])
    inputs = st.text_area('Enter your text for classification', "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!")
    payload = {
        "inputs": inputs,
        "parameters": {"candidate_labels": labels},
    }
    submitted = st.form_submit_button('Classify!')
    if submitted:
        output = query(payload)
        st.write(output)
