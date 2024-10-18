import torch
import huggingface_hub
import pandas as pd
import re
from datasets import Dataset


# Remove the character in bracket or action from transcript using re
def remove_paranthesis(text):
    result = re.sub(r'\(.*?\)', '', text)

    return result


class CharacterChatbot():
    def __init__(self, model_path, data_path="/content/series_analyzing_system/data/naruto.csv",
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
