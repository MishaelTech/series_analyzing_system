import gradio as gr
import pandas as pd

from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognition, CharacterNetworkGenerator
from text_classification import JutsuClassifier
import os
# from dotenv import load_dotenv
# load_dotenv()


def get_themes(theme_list_str,subtitles_path,save_path):
    # Convert input string to a list of themes
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    # Call the theme classification method and get the output dataframe
    output_df = theme_classifier.get_themes(subtitles_path,save_path)

    # Remove dialogue from the theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]

    # Filter and aggregate the output data
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme','Score']

    # Create bar chart data
    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme","Score"],
        vertical=False,
        width=500,
        height=260
    )

    return output_chart


def get_character_network(subtitles_path, ner_path):
    ner=NamedEntityRecognition()
    ner_df=ner.get_ners(subtitles_path, ner_path)

    character_network_generator=CharacterNetworkGenerator()
    relationship_df=character_network_generator.generate_character_network(ner_df)
    # After getting then drawing
    html=character_network_generator.draw_network_graph(relationship_df)

    return html


def classify_text(text_classification_model, text_classification_data_path,text_to_classify):
    # huggingface_token = os.getenv("HUGGINGFACE_TOKEN", " ")
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')  # Replace 'your_default_token' with a default if desir

    jutsu_classifier = JutsuClassifier(model_path = text_classification_model, data_path = text_classification_data_path, huggingface_token = huggingface_token) # huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    # Run the classifier on the text and get the output
    output = jutsu_classifier.classify_jutsu(text_to_classify)

    return output



def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.Markdown("<h1>Theme Classification (Zero Shot Classifier)</h1>")
                with gr.Row():
                    with gr.Column():
                        # Create a placeholder for plot output
                        plot = gr.BarPlot(x="Theme", y="Score", title="Series Themes", tooltip=["Theme", "Score"],
                                          vertical=False)
                    # For Input
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes (comma separated)")
                        subtitles_path = gr.Textbox(label="Subtitles or script path")
                        save_path = gr.Textbox(label="Save Path")
                        # Button to trigger theme classification
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path],
                                                outputs=[plot])


        # Character Network Generator Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("<h1>Character Network Generator (NERS & Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        # Create a placeholder for plot output
                        network_html=gr.HTML()

                    # For Input
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or script path")
                        ner_path = gr.Textbox(label="NERs Save Path")

                        # Button to trigger theme classification
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path, ner_path], outputs=[network_html])


        # Text Classification with LLMs Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        # Output will be a text
                        text_classification_output = gr.Textbox(label="Text Classification Output")

                    # For Input
                    with gr.Column():
                        text_classification_model = gr.Textbox(label="Model Path")
                        text_classification_data_path = gr.Textbox(label="Data Path")
                        text_to_classify = gr.Textbox(label="Text input")

                        # Button to trigger theme classification
                        classify_text_button = gr.Button("Classify Text (Jutsu)")
                        classify_text_button.click(classify_text, inputs=[text_classification_model, text_classification_data_path, text_to_classify], outputs=[text_classification_output])


    iface.launch(share=True)


if __name__ == "__main__":
    main()
