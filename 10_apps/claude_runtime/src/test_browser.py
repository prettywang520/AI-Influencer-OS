from pathlib import Path

from playwright.sync_api import sync_playwright


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = RUNTIME_ROOT / "browser_profile"


def main() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
        )

        page = context.pages[0] if context.pages else context.new_page()
        page.goto(
            "https://chatgpt.com",
            wait_until="domcontentloaded",
        )

        print("ChatGPT opened.")
        print("Please log in manually if required.")
        input("After login is complete, press ENTER to close...")

        context.close()


if __name__ == "__main__":
    main()