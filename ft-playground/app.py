import streamlit as st
from resources import params
from datasets import load_dataset

st.title("Fine-tuning Playground")
st.write("Zero-code interface for fine-tuning `Transformers` models.")
st.logo("resources/logo.png", size='large', link="https://github.com/istat-methodology/fine-tuning-pipelines")

with st.sidebar:
    hf_token = st.text_input('Huggingface Token', type='password')
    st.subheader('Data')
    data_id = st.text_input('Training Data')
    if st.button('Load'):
        dataset = load_dataset(data_id, token=hf_token)
    st.selectbox('Text Feature', options=['text'])
    st.selectbox('Target Feature', options=['label'])
    st.subheader('Model')
    model = st.selectbox('Select model', options=params.MODELS.keys())
    task_list = params.MODELS[model]['tasks']
    st.selectbox('Select task', options=task_list)

tab1, tab2, tab3 = st.tabs(['Settings', 'Training Board', 'Results'])

with tab1:
    with st.expander("Training settings", expanded=False, icon=':material/rule_settings:'):
        if not st.toggle('Auto'):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, format="%0.5f", step=1e-5)
            with col2:
                st.selectbox("LR scheduler", options=['None', 'Linear'])
            with col3:
                st.number_input("Weight decay", min_value=0.0, max_value=1e-1, format="%0.5f", step=1e-5)
            with col4:
                st.number_input("Warmup ratio", min_value=0.00, max_value=0.99, format="%0.2f", step=0.01)
    with st.expander("Optimization", expanded=False, icon=':material/bolt:'):
        st.checkbox("4-bit quantization")
        if st.toggle("LoRA"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.number_input("Rank", step=1, value=8)
            with col2:
                st.number_input("Alpha", step=1, value=16)
            with col3:
                st.number_input("Dropout", min_value=0.0, max_value=0.99, value=0.05)
    st.button('Train')