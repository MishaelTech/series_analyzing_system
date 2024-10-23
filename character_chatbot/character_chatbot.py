import torch
import huggingface_hub
import pandas as pd
import re
from datasets import Dataset
from transformers import (BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer)
from transformers.utils import quantization_config
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import gc


# Remove the character in bracket or action from transcript using re
def remove_paranthesis(text):
    result = re.sub(r'\(.*?\)', '', text)

    return result


class CharacterChatbot():
    def __init__(self,
                 model_path,
                 data_path="/content/series_analyzing_system/data/naruto.csv",
                 huggingface_token=None):

        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = "meta-llama/Meta-Llama-3-8B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        # If the model path exist we use it else we train, save and use it
        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            # Train and load the model
            print("Model not found in huggingface hub, we will train our own model")

            train_dataset = self.load_data()
            # Train

            # Load

    # Train Function
    def train(
        self, 
        base_model_name_or_path, 
        dataset,
        output_dir="./results", 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=1,
        optimizer="paged_adamw_32bit", 
        save_steps=200, 
        logging_steps=10, 
        learning_rate=2e-4, 
        max_grad_norm=0.3, 
        max_steps=300, 
        warmup_ratio=0.3, 
        learning_rate_schedular_type="constant",):

        # We load the model into 4bit instead of 32 or 64bit so it can fit into memory (We will lose accuracy). Therefore we will use bit and bite config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, 
            quantization_config=bnb_config,
            trust_remote_code=True
        )

        #Not using any cache but getting from the hugging face hub directly
        model.config.use_cache=False

        # Making use of the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        # Having the padding token
        tokenizer.psd_token = tokenizer.eos_token # Ending of state padding token EOS

        # Lora config that enables us to have somE additional width next to the model that will be trained and it will enhance the model instead of training the full model which will take alot of time
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r=64

        # We will use peft for lora config 
        peft_config=LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias=None,
            task_type="CASUAL_LM",
        )

        # Intializing the training argument
        training_arguments=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optimizer,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            fp16=True,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True, # Make it more optimized, similar length to each other
            lr_scheduler_type=learning_rate_schedular_type,
            report_to="none"
        )

        max_seq_len = 512

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=max_seq_len,
            tokenizer=tokenizer,
            args=training_arguments
        )

        trainer.train()

        # Save the Model we trained
        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")

        # Flushing the memory
        del trainer, model
        gc.collect()

        # Read the model and add the ckpt width we said
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
    
    # Load Function
    def load_data(self):
        naruto_transcript_df = pd.read_csv(self.data_path)
        naruto_transcript_df = naruto_transcript_df.dropna()

        # Remove paranthesis
        naruto_transcript_df['line'] = naruto_transcript_df['line'].apply(remove_paranthesis)

        # Calculate number of width
        # Get the response and make sure its not short
        naruto_transcript_df["number_of_words"] = naruto_transcript_df["line"].str.strip().str.split("")
        # Count it
        naruto_transcript_df["number_of_words"] = naruto_transcript_df["number_of_words"].apply(lambda x: len(x))

        # Naruto respond flag, so anything that has naruto and the word is greater than 5
        naruto_transcript_df['naruto_respond_flag'] = 0
        naruto_transcript_df.loc[(naruto_transcript_df['name'] == 'Naruto') & (
                    naruto_transcript_df["number_of_words"] > 5), "naruto_respond_flag"] = 1

        # Picking the indexes and Excluding the first row because its just intializing coversation
        indexes_to_take = list(naruto_transcript_df[(naruto_transcript_df['naruto_respond_flag'] == 1) & (
                    naruto_transcript_df.index > 0)].index)

        # Create the prompt we will feed through the chatbot so it can act as naruto, and give it the statement set to naruto and the response so it can imitate it

        system_prompt = """" Your are naruto from the anime "Naruto". Your responses should reflect his personalities and speech patterns \n"""
        prompts = []
        for ind in indexes_to_take:
            prompt = system_prompt

            # What was said
            prompt += naruto_transcript_df.iloc[ind - 1]["line"] + "\n"
            # Response
            prompt += naruto_transcript_df.iloc[ind]["line"]
            prompts.append(prompt)

        # Putting the prompts in pd df
        df = pd.DataFrame({"prompt": prompts})

        # Convert to huggingface dataset
        dataset = Dataset.from_pandas(df)

        return dataset
