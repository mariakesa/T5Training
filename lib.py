from transformers import TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

class T5Train:
    def __init__(self, train_config, lora_config, bitsandbytes_config, model_config):

        self.train_config = self.construct_train_config(train_config)

        self.lora_config = self.construct_lora_config(lora_config)

        self.bitsandbytes_config = self.construct_bitsandbytes_config(bitsandbytes_config)

        self.model, self.tokenizer = self.make_t5_model(model_config)

        self.data_collator = self.make_data_collator()
        

    def construct_train_config(self, train_config):
        return Seq2SeqTrainingArguments(**train_config)

    def construct_lora_config(self, lora_config):
        return LoraConfig(**lora_config)
    
    @classmethod
    def construct_bitsandbytes_config(self, bitsandbytes_config):
        return BitsAndBytesConfig(**bitsandbytes_config)
    
    def make_t5_model(self, model_config):
        config = AutoConfig.from_pretrained(**model_config)
        model = AutoModelForSeq2SeqLM.from_config(config)   
        tokenizer = AutoTokenizer.from_pretrained(model_config['pretrained_model_name_or_path'], load_in_8bit=True, device_map="auto")
        model=prepare_model_for_kbit_training(model)
        model.enable_input_require_grads()
        peft_model=get_peft_model(model, self.lora_config)
        peft_model.print_trainable_parameters()
        peft_model.base_model.model.encoder.enable_input_require_grads()
        peft_model.base_model.model.decoder.enable_input_require_grads()
        peft_model.config.use_cache=False
        return peft_model, tokenizer
    
    def make_data_collator(self):
        # padding the sentence of the entire datasets
        label_pad_token_id=-100
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8
        )
        return data_collator
    
    def execute(self, dataset):
 
        trainer=Seq2SeqTrainer(
            model=self.model,
            args=self.train_config,
            data_collator=self.data_collator,
            train_dataset=dataset,
            #eval_dataset=eval_dataset,
            #compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.save_model()
