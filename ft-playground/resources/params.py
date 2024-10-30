MODELS = {
    'XLM RoBERTa (base)': {
        'model_id': 'FacebookAI/xlm-roberta-base',
        'tasks': ['sequence-classification']
    },
    'Llama 3.2 (1B)': {
        'model_id': 'meta-llama/Llama-3.2-1B',
        'tasks': ['text-classification', 'text-generation']
    }
}

DATA_CONFIGS = {
    'TRAIN_SPLIT': {
        'label': 'Training',
        'key': 'training_split',
        'help': ''
    },
    'VAL_SPLIT': {
        'label': 'Validation',
        'key': 'validation_split',
        'help': ''
    },
    'TEST_SPLIT': {
        'label': 'Test',
        'key': 'test_split',
        'help': ''
    },
    'TEXT_FEATURE': {
        'label': 'Text feature',
        'key': 'text_feature',
        'help': ''
    },
    'TARGET_FEATURE': {
        'label': 'Target feature',
        'key': 'target_feature',
        'help': ''
    }
}

TRAINING_CONFIGS = {
    'LEARNING_RATE': {
        'label': 'Learning rate',
        'min_value': 1e-6,
        'max_value': 0.99,
        'value': 1e-3,
        'format': '%0.5f',
        'step': 1e-5,
        'key': 'learning_rate',
        'help': ''
    },
    'LR_SCHEDULER': {
        'label': 'LR scheduler',
        'options': ['None', 'Linear'],
        'key': 'lr_scheduler',
        'help': ''
    },
    'WEIGHT_DECAY': {
        'label': 'Weight decay',
        'min_value': 0.0,
        'max_value': 0.99,
        'value': 1e-4,
        'format': '%0.5f',
        'step': 1e-5,
        'key': 'weight_decay',
        'help': ''
    },
    'WARMUP_RATIO': {
        'label': 'Warmup ratio',
        'min_value': 0.0,
        'max_value': 0.99,
        'value': 0.1,
        'format': '%0.2f',
        'step': 1e-2,
        'key': 'warmup_ratio',
        'help': ''
    },
    'TRAIN_BS': {
        'label': 'BS (train)',
        'options': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        'key': 'train_bs',
        'help': ''
    },
    'EVAL_BS': {
        'label': 'BS (eval)',
        'options': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        'key': 'eval_bs',
        'help': ''
    },
    'GA_STEPS': {
        'label': 'GA steps',
        'options': ['None', 1, 2, 4, 8, 16, 32, 64, 128, 256],
        'key': 'gradient_accumulation_steps',
        'help': ''
    },
    'PRECISION': {
        'label': 'Precision',
        'options': ['fp16', 'fp32'],
        'key': 'precision',
        'help': ''
    },
    'NUM_EPOCHS': {
        'label': 'Epochs',
        'min_value': 1,
        'value': 3,
        'key': 'num_epochs',
        'help': ''
    },
    'LOAD_BEST_MODEL_AT_END': {
        'label': 'Load best model at end',
        'key': 'load_best_model_at_end'
    }
}

LOGGING_CONFIGS = {
    'LOGGING_STEPS': {
        'label': 'Logging steps',
        'min_value': 1,
        'value': 50,
        'step': 5,
        'key': 'logging_steps',
        'help': ''
    },
    'EVAL_STRATEGY': {
        'label': 'Eval strategy',
        'options': ['epoch', 'steps'],
        'key': 'eval_strategy',
        'help': ''
    },
    'SAVE_STRATEGY': {
        'label': 'Save strategy',
        'options': ['epoch', 'steps'],
        'key': 'save_strategy',
        'help': ''
    }
}

OPTIMIZATION_CONFIGS = {
    '4BIT_QUANTIZATION': {
        'label': '4-bit quantization',
        'key': '4bit_quantization',
        'help': ''
    },
    'LORA_R': {
        'label': 'Rank',
        'min_value': 1,
        'value': 8,
        'key': 'r',
        'help': ''
    },
    'LORA_ALPHA': {
        'label': 'Alpha',
        'min_value': 1,
        'value': 16,
        'key': 'lora_alpha',
        'help': ''
    },
    'LORA_DROPOUT': {
        'label': 'Dropout',
        'min_value': 0.0,
        'max_value': 0.99,
        'value': 0.05,
        'step': 0.05,
        'key': 'lora_dropout',
        'help': ''
    },
    'TASK_TYPE': {
        'label': 'Task type',
        'options': ['SEQ_CLS'],
        'key': 'task_type',
        'help': ''
    }
}