import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognition, CharacterNetworkGenerator
from text_classification import JutsuClassifier


def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]

    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme', 'Score']

    return output_df


def get_character_network(subtitles_path, ner_path):
    ner = NamedEntityRecognition()
    ner_df = ner.get_ners(subtitles_path, ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html


def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    #huggingface_token = os.getenv("HUGGINGFACE_TOKEN", " ")
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    jutsu_classifier = JutsuClassifier(model_path=text_classification_model,
                                       data_path=text_classification_data_path,
                                       huggingface_token=huggingface_token)

    output = jutsu_classifier.classify_jutsu(text_to_classify)
    # remove the output from dict
    output = output[0]

    return output


def main():
    st.title("Text Analysis App")

    st.header("Theme Classification (Zero Shot Classifier)")
    theme_list = st.text_input("Themes (comma separated)")
    subtitles_path = st.text_input("Subtitles or script path")
    save_path = st.text_input("Save Path")

    if st.button("Get Themes"):
        if theme_list and subtitles_path and save_path:
            output_df = get_themes(theme_list, subtitles_path, save_path)
            st.bar_chart(output_df.set_index('Theme'))
        else:
            st.warning("Please fill all fields.")

    st.header("Character Network Generator (NERS & Graphs)")
    ner_subtitles_path = st.text_input("Subtitles or script path (for NER)")
    ner_path = st.text_input("NERs Save Path")

    if st.button("Get Character Network"):
        if ner_subtitles_path and ner_path:
            network_html = get_character_network(ner_subtitles_path, ner_path)
            st.components.v1.html(network_html, height=600)
        else:
            st.warning("Please fill all fields.")

    st.header("Text Classification with LLMs")
    text_classification_model = st.text_input("Model Path")
    text_classification_data_path = st.text_input("Data Path")
    text_to_classify = st.text_area("Text input")

    if st.button("Classify Text (Jutsu)"):
        if text_classification_model and text_classification_data_path and text_to_classify:
            classification_output = classify_text(text_classification_model, text_classification_data_path,
                                                  text_to_classify)
            st.text_area("Text Classification Output", value=classification_output, height=300)

        elif text_classification_model and text_to_classify:
            classification_output = classify_text(text_classification_model, text_classification_data_path,
                                                  text_to_classify)
            st.text_area("Text Classification Output", value=classification_output, height=300)

        else:
            st.warning("Please fill the required field")


if __name__ == "__main__":
    main()
