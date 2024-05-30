from datasets import load_dataset
from transformers import AutoTokenizer
import sys
sys.path.append('../')
from configs.t5config import train_config, lora_config, bitsandbytes_config, model_config
import time
from lib import T5Train

def make_data():
    # New instruction dataset
    billsum_dataset = "billsum"



    dataset = load_dataset(billsum_dataset, split="train[0:1000]")

    print(dataset)
    max_input_length = 512
    max_target_length = 30


    def preprocess_function(examples):
        model_name="google/long-t5-tglobal-base"
        #model_name="google-t5/t5-small"
        tokenizer=AutoTokenizer.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            examples["summary"], max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs



    dat=dataset.map(preprocess_function, batched=True)

    tokenized_datasets = dat.remove_columns(
    dataset.column_names
    )
    #print(merged_dataset[:5])
    return tokenized_datasets

dataset=make_data()

start=time.time()
T5Train(train_config, lora_config, bitsandbytes_config, model_config).execute(dataset)
end=time.time()
print('Time taken: ', end-start)
