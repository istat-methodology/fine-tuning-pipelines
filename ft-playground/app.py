import streamlit as st
from resources import params
from datasets import load_dataset

st.title("Fine-tuning Playground")
st.write("Zero-code interface for fine-tuning `Transformers` models.")
st.logo("resources/logo.png", size='medium', link="https://github.com/istat-methodology/fine-tuning-pipelines", icon_image="resources/logo-small.png")

data_config = params.DATA_CONFIGS
training_config = params.TRAINING_CONFIGS
logging_config = params.LOGGING_CONFIGS
optimization_config = params.OPTIMIZATION_CONFIGS

if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None

with st.sidebar:
    hf_token = st.text_input('Huggingface Token', type='password')
    st.subheader('Data')
    data_id = st.text_input('Training Data')
    if st.button('Load'):
        with st.spinner('Loading data...'):
            st.session_state['dataset'] = load_dataset(data_id, token=hf_token)
            st.toast('Data loaded succesfully!')
    st.subheader('Model')
    model = st.selectbox('Select model', options=params.MODELS.keys())
    task_list = params.MODELS[model]['tasks']
    st.selectbox('Select task', options=task_list, key='training_task')

tab1, tab2, tab3 = st.tabs(['Settings', 'Training Board', 'Results'])

with tab1:
    with st.expander("Data", expanded=False, icon=':material/database:'):
        if st.session_state['dataset']:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.selectbox(**data_config['TRAIN_SPLIT'], options=st.session_state['dataset'].keys())
                st.write(f"n = {len(st.session_state['dataset'][st.session_state[data_config['TRAIN_SPLIT']['key']]])}")
            with col2:
                val_options = list(st.session_state['dataset'].keys())
                val_options.append('-')
                st.selectbox(**data_config['VAL_SPLIT'], options=val_options)
                if st.session_state[data_config['VAL_SPLIT']['key']] != '-':
                    st.write(f"n = {len(st.session_state['dataset'][st.session_state[data_config['VAL_SPLIT']['key']]])}")
            with col3:
                st.selectbox(**data_config['TEST_SPLIT'], options=val_options)
                if st.session_state[data_config['TEST_SPLIT']['key']] != '-':
                    st.write(f"n = {len(st.session_state['dataset'][st.session_state[data_config['TEST_SPLIT']['key']]])}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.selectbox(**data_config['TEXT_FEATURE'], options=st.session_state['dataset'][st.session_state[data_config['TRAIN_SPLIT']['key']]].features)
            with col2:
                st.selectbox(**data_config['TARGET_FEATURE'], options=st.session_state['dataset'][st.session_state[data_config['TRAIN_SPLIT']['key']]].features)
        
        
    with st.expander("Training", expanded=False, icon=':material/rule_settings:'):
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
            
            training_settings = {
                'learning_rate': st.session_state[training_config['LEARNING_RATE']['key']],
                'lr_scheduler': st.session_state[training_config['LR_SCHEDULER']['key']],
                'weight_decay': st.session_state[training_config['WEIGHT_DECAY']['key']],
                'warmup_ratio': st.session_state[training_config['WARMUP_RATIO']['key']],
                'per_device_train_batch_size': st.session_state[training_config['TRAIN_BS']['key']],
                'per_device_eval_batch_size': st.session_state[training_config['EVAL_BS']['key']],
                'gradient_accumulation_steps': st.session_state[training_config['GA_STEPS']['key']],
                'precision': st.session_state[training_config['PRECISION']['key']],
                'num_train_epochs': st.session_state[training_config['NUM_EPOCHS']['key']],
            }
            
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
        st.checkbox(**optimization_config['4BIT_QUANTIZATION'])
        if st.toggle("LoRA"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.number_input(**optimization_config['LORA_R'])
            with col2:
                st.number_input(**optimization_config['LORA_ALPHA'])
            with col3:
                st.number_input(**optimization_config['LORA_DROPOUT'])
            with col4:
                st.selectbox(**optimization_config['TASK_TYPE'])
    st.button('Train')

    st.json(training_settings)