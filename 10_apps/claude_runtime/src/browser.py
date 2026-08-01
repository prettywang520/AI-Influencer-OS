from pathlib import Path
import webbrowser


CHATGPT_URL = "https://chatgpt.com"


class Browser:

    def open_chatgpt(self) -> None:
        webbrowser.open_new_tab(CHATGPT_URL)

    def open_prompt_file(self, prompt_path: str | Path) -> None:
        prompt_path = Path(prompt_path)

        if not prompt_path.exists():
            raise FileNotFoundError(prompt_path)

        print("=" * 60)
        print("Prompt File")
        print("=" * 60)
        print(prompt_path)
        print("=" * 60)

        self.open_chatgpt()