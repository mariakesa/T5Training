import torch
import importlib
import sys
sys.path.append('../')
from lib import T5Train
from peft import TaskType

bitsandbytes_config= {
    'load_in_8bit': True
}

lora_config={
    'task_type' : TaskType.SEQ_2_SEQ_LM,
    # the dimension of the low-rank matrices
    'r':4,
    # the scaling factor for the low-rank matrices
    'lora_alpha':32,
    # the dropout probability of the LoRA layers
    'lora_dropout':0.01,
    'target_modules':["k","q","v","o"]
}

model_config={
    'pretrained_model_name_or_path': "google/long-t5-tglobal-base",
    'device_map': 'auto',
    'quantization_config': T5Train.construct_bitsandbytes_config(bitsandbytes_config)
}

train_config= {
    'output_dir': './t5_results',
    'auto_find_batch_size': True,
    'learning_rate': 1e-3,
    'num_train_epochs': 1,
    'per_device_train_batch_size': 10,
    'logging_strategy': 'epoch',
    'logging_steps': 500,
    'save_strategy': 'epoch',
    'push_to_hub': False,
    #'report_to': 'tensorboard',
    'remove_unused_columns': False
}
