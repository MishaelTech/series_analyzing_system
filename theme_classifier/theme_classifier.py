from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import numpy as np
import os
import sys
import pathlib
# Get path
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,'../'))
from utils import load_subtitles_dataset
nltk.download('punkt')
nltk.download('punkt_tab')


class ThemeClassifier():
    def __init__(self, theme_list):
        # Load Model
        self.model_name = "facebook/bart-large-mnli"
        # Pick a device
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)

    # Function to load model
    def load_model(self,device):
        theme_classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )

        return theme_classifier

    def get_themes_inference(self, script):
        script_sentences = sent_tokenize(script)

        # Batch Sentence into 20
        sentence_batch_size=20
        script_batches = []
        for index in range(0,len(script_sentences),sentence_batch_size):
            # All the 20 sentence batch together in one
            sent = " ".join(script_sentences[index:index+sentence_batch_size])
            script_batches.append(sent)
        
        # Run Model
        theme_output = self.theme_classifier(
            script_batches,
            # script_batches[:2]
            self.theme_list,
            multi_label=True
        )

        # Wrangle the output
        # i.e battle: [0.8546878099441528, 0.6581301093101501]. The matrix/score for 2 batch
        themes = {}
        for output in theme_output:
            for label,score in zip(output['labels'],output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)

        themes = {key: np.mean(np.array(value)) for key,value in themes.items()}

        return themes

    def get_themes(self,dtaset_path, save_path=None):
        # Read Save Output if Exists
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            return df

        # load Dataset
        df = load_subtitles_dataset(dtaset_path)
        # df = df.head(2)

        # Run Inference
        output_themes = df['script'].apply(self.get_themes_inference)

        themes_df = pd.DataFrame(output_themes.tolist())
        df[themes_df.columns] = themes_df

        # Save output
        if save_path is not None:
            df.to_csv(save_path,index=False)
        return df
