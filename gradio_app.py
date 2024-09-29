import gradio as gr
from theme_classifier import ThemeClassifier
import pandas as pd


def get_themes(theme_list_str, subtitles_path, save_path):
    # Convert input string to a list of themes
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    # Call the theme classification method and get the output dataframe
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove 'dialogue' from theme list if it's present
    theme_list = [theme for theme in theme_list if theme != 'dialogue']

    # Filter and aggregate the output data
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme', 'Score']

    # Create bar chart data
    chart_data = pd.DataFrame(output_df)

    return chart_data


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
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes (comma separated)")
                        subtitles_path = gr.Textbox(label="Subtitles or script path")
                        save_path = gr.Textbox(label="Save Path")
                        # Button to trigger theme classification
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path],
                                                outputs=[plot])

    iface.launch(share=True)


if __name__ == "__main__":
    main()
