import streamlit as st
import gradio as gr
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognition, CharacterNetworkGenerator
from text_classification import JutsuClassifier
from character_chatbot import CharacterChatbot

huggingface_token = ""
#huggingface_token = os.getenv("HUGGINGFACE_TOKEN", " ")
#huggingface_token = os.getenv("HUGGINGFACE_TOKEN")


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

    jutsu_classifier = JutsuClassifier(model_path=text_classification_model,
                                       data_path=text_classification_data_path,
                                       huggingface_token=huggingface_token)

    output = jutsu_classifier.classify_jutsu(text_to_classify)
    # remove the output from dict
    output = output[0]

    return output


# def chat_with_character_chatbot(message, history):
#     character_chatbot = CharacterChatbot("MishaelTech1/Naruto_Llama-3-8B", huggingface_token=huggingface_token)

#     # Ensure history is formatted as a list of dictionaries with correct keys
#     formatted_history = [{"role": msg["role"], "content": msg["content"]} for msg in history]
#     output = character_chatbot.chat(message, formatted_history)
#     return output

def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatbot("MishaelTech1/Naruto_Llama-3-8B", huggingface_token=huggingface_token)

    output = character_chatbot.chat(message, history)
    output = output['content'].strip()
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


    # Chatbot header
    st.header("Character Chatbot")

    # Initialize chat history if it doesn't exist in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # This will store the history of chat messages

    # Display chat messages from the history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input text box
    user_input = st.text_input("You:")

    if user_input:
        # Prepare the message history for the chatbot
        history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display a loading spinner while generating a response
        with st.spinner("Assistant is typing..."):
            response = chat_with_character_chatbot(user_input, history)

        # Add chatbot's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chatbot's response
        with st.chat_message("assistant"):
            st.markdown(response)


    # st.header("Character Chatbot")
    # chat_interface = gr.ChatInterface(chat_with_character_chatbot)

    # # Launch the Gradio interface and get the URL
    # gradio_url = chat_interface.launch(share=True, inline=False)

    # # Display the Gradio interface in Streamlit
    # st.markdown(f'<iframe src="{gradio_url}" width="100%" height="500px"></iframe>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
