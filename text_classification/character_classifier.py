import torch
import huggingface_hub
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments,
                          pipeline)
import pandas as pd
from .cleaner import Cleaner
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from .training_utils import get_class_weights, compute_metrics
from .custom_trainer import CustomTrainer
import gc


class JutsuClassifier():
    def __init__(self, model_path, data_path=None, text_column_name="text", label_column_name="jutsu",
                 model_name="distilbert/distilbert-base-uncased", text_size=0.2, num_labels=3, huggingface_token=None):

        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.text_size = text_size
        self.text_size = text_size
        self.num_labels = num_labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.huggingface_token = huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        self.tokenizer = self.load_tokenizer()

        # if we don't see the model_path then we train it
        if not huggingface_hub.repo_exists(self.model_path):

            # Check if data path is provided
            if data_path is None:
                raise ValueError(
                    "Data path is required to train the model, since the model path does not exist in huggingface hub")

            # Function to load the dataset
            train_data, test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()

            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data)

            self.train_model(train_data, test_data, class_weights)

        self.model = self.load_model(self.model_path)

    def load_model(self, model_path):
        model = pipeline("text_classification", model=model_path, return_all_scores=True)
        return model

    # Training of the model
    def train_model(self, train_data, test_data, class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels,
                                                                   id2label=self.label_dict, )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(output_dir=self.model_path, learning_rate=2e-4, per_device_train_batch_size=8,
                                          per_device_eval_batch_size=8, num_train_epochs=5, weight_decay=0.01,
                                          evaluation_strategy="epoch", logging_strategy="epoch", push_to_hub=True, )

        # Custom trainer that takes the class weight and multiple them with the actual loss
        trainer = CustomTrainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=test_data,
                                tokenizer=self.tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)

        trainer.set_device(self.device)
        trainer.set_class_weights(class_weights)

        trainer.train()

        # After it push to hub flush the memory
        del trainer, model
        gc.collect()  # gc is a gbarish collector

        if self.device == "cuda":
            torch.cuda.empty_cache()

    # Have few jutsu_type name using function
    def simplify_jutsu(self, jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"

    # Function that tokenizes the text
    def preprocess_function(self, tokenizer, examples):
        return tokenizer(examples["text_cleaned"],
                         truncation=True)  # If it something above the 512 token it truncates so it wouldn't break the model

    def load_data(self, data_path):
        df = pd.read_json(data_path, lines=True)
        df["jutsu_type_simplified"] = df["jutsu_type"].apply(self.simplify_jutsu)

        # Name and description in one column called text
        df["text"] = df["jutsu_name"] + ". " + df["jutsu_description"]
        df[self.label_column_name] = df["jutsu_type_simplified"]

        # Taking only the text and jutsu
        df = df[["text", self.label_column_name]]
        # Drop missing values
        df = df.dropna()

        # Clean the text
        cleaner = Cleaner()
        df["text_cleaned"] = df[self.text_column_name].apply(cleaner.clean_html)

        # Label Encoder
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(df[self.label_column_name].tolist())

        # One line for loop that return a dict
        label_dict = {index: label_name for index, label_name in enumerate(label_encoder.__dict__['classes_'].tolist())}
        self.label_dict = label_dict

        # Transform the jutsu to numerical type
        df["label"] = label_encoder.transform(df[self.label_column_name].tolist())

        # Train & Test Split
        test_size = 0.2
        train_df, test_df = train_test_split(df,
                                             test_size=test_size,
                                             stratify=df[
                                                 "label"], )  # stratify makes sure that each of the classes that 80% is in training and 20% is in the test

        # Run it on the whole dataset
        # Convert Pandas (df) to a hugging face dataset
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # tokenize the dataset
        tokenized_train_dataset = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                                    batched=True)

        tokenized_test_dataset = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                                  batched=True)

        return tokenized_train_dataset, tokenized_test_dataset

    # Check if we've trained the huggingface and saved it in huggingface_hub we use the model_path else we choose the model_name and train on it
    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return tokenizer

    def postprocess(self, model_output):
        output = []
        for pred in model_output:
            # get maximums score
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output

    def classify_jutsu(self, text):
        model_output = self.model(text)
        predictions = self.postprocess(model_output)
        return predictions

