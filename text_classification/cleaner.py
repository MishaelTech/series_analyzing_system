from bs4 import BeautifulSoup


# Creating a function called Cleaner to remove HTML Tags
class Cleaner:
    def __init__(self):
        pass

    # Putting line break after each paragraph
    def put_line_break(self, text):
        return text.replace("<\p> ", "<\p>\n")

    # Remove html tags using beautifulsoup
    def remove_html_tags(self, text):
        clean_text = BeautifulSoup(text, "lxml").text
        return clean_text

    def clean_html(self, text):
        clean_text = self.remove_html_tags(text)
        clean_text = self.put_line_break(clean_text)
        clean_text = clean_text.strip()
        return clean_text
