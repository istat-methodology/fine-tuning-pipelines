import streamlit as st
from resources import params
from datasets import load_dataset

st.title("Fine-tuning Playground")
st.write("Zero-code interface for fine-tuning `Transformers` models.")
st.logo("resources/logo.png", size='medium', link="https://github.com/istat-methodology/fine-tuning-pipelines", icon_image="resources/logo-small.png")

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

        training_config = params.TRAINING_CONFIGS
        logging_config = params.LOGGING_CONFIGS
        if not st.toggle('Auto'):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.number_input(**training_config['LEARNING_RATE'])
            with col2:
                st.selectbox(**training_config['LR_SCHEDULER'])
            with col3:
                st.number_input(**training_config['WEIGHT_DECAY'])
            with col4:
                st.number_input(**training_config['WARMUP_RATIO'])

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.selectbox(**training_config['TRAIN_BS'])
            with col2:
                st.selectbox(**training_config['EVAL_BS'])
            with col3:
                st.selectbox(**training_config['GA_STEPS'])
            with col4:
                st.selectbox(**training_config['PRECISION'])
            with col5:
                st.number_input(**training_config['NUM_EPOCHS'])
            
            st.divider()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.number_input(**logging_config['LOGGING_STEPS'])
            with col2:
                st.selectbox(**logging_config['EVAL_STRATEGY'])
            with col3:
                st.selectbox(**logging_config['SAVE_STRATEGY'])
            
            st.checkbox(**training_config['LOAD_BEST_MODEL_AT_END'], disabled=False if st.session_state['eval_strategy'] == st.session_state['save_strategy'] else True)
            


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