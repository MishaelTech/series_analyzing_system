import pandas as pd
import spacy
from nltk import sent_tokenize
import nltk
from ast import literal_eval
import os
import sys
import pathlib

folder_path = pathlib.Path().parent.resolve()
# sys.path.append(os.path.join(folder_path, "series_analyzing_system"))
sys.path.append(os.path.join(folder_path, "../"))

from utils import load_subtitles_dataset

# Download the 'punkt' resource
nltk.download('punkt')


class NamedEntityRecognition:
    def __init__(self):
        self.nlp_model = self.load_model()
        pass

    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp

    # Getting only person
    # for entity in documents.ents:
    #     if entity.label_=="PERSON":
    #         print(entity,entity.label_)

    def get_ners_inference(self, script):
        script_sentences = sent_tokenize(script)

        ner_output = []

        for sentence in script_sentences:
            documents = self.nlp_model(sentence)
            ners = set()
            for entity in documents.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text
                    first_name = entity.text.split(" ")[0]
                    first_name = first_name.strip()
                    ners.add(first_name)
            ner_output.append(ners)

        return ner_output

    def get_ners(self, dataset_path, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df["ners"] = df["ners"].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df

        # Load Dataset
        df = load_subtitles_dataset(dataset_path)

        # Run Inference
        df["ners"] = df["script"].apply(self.get_ners_inference)

        if save_path is not None:
            df.to_csv(save_path, index=False)
















