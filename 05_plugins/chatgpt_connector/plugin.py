import pyperclip
import webbrowser


class ChatGPTConnector:

    def send(self, prompt: str):

        pyperclip.copy(prompt)

        webbrowser.open("https://chatgpt.com")

        print("=" * 60)
        print("Prompt copied to clipboard.")
        print("ChatGPT opened.")
        print("Press Cmd + V")
        print("=" * 60)