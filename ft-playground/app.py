import streamlit as st
from datasets import load_dataset

st.title("Fine-tuning Playground")
st.write("Zero-code interface for fine-tuning `Transformers` models.")
st.logo("resources/logo.png", size='large', link="https://github.com/istat-methodology/fine-tuning-pipelines")

with st.sidebar:
    hf_token = st.text_input('Huggingface Token', type='password')
    data_id = st.text_input('Training Data')
    if st.button('Load'):
        dataset = load_dataset(data_id, token=hf_token)
    st.divider()
    st.selectbox('Task', options=['Classification', 'Generation'])
    st.selectbox('Text Feature', options=['text'])
    st.selectbox('Target Feature', options=['label'])

st.text(dataset)