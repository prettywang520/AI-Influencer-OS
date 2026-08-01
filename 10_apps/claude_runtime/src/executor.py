import subprocess


class Executor:

    def run(self, today):

        theme = today["theme"]

        print("\nGenerating Feed...\n")

        subprocess.run(
            [
                "python3",
                "src/main.py",
                f"{theme}/{today['feed']}"
            ]
        )

        print("\nGenerating Stories...\n")

        for story in today["stories"]:

            subprocess.run(
                [
                    "python3",
                    "src/main.py",
                    f"{theme}/{story}"
                ]
            )

        print("\nDone.\n")