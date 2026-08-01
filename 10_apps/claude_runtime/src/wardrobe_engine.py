from __future__ import annotations

from pathlib import Path
import json


class WardrobeEngine:

    def __init__(self, history_path: Path):

        self.history_path = history_path

    def choose(

        self,

        city: str,

        weather: str,

        season: str,

        venue: str,

        activity: str,

    ) -> dict:

        return {

            "top": "fitted rib knit top",

            "bottom": "high waist denim skirt",

            "dress": None,

            "shoes": "white sneakers",

            "bag": "cream leather shoulder bag",

            "jewellery": "minimal gold jewellery",

            "outerwear": None,

            "hair": "long silky loose hair",

        }

    def save(

        self,

        date: str,

        outfit: dict,

    ):

        history = {}

        if self.history_path.exists():

            history = json.loads(

                self.history_path.read_text()

            )

        history[date] = outfit

        self.history_path.write_text(

            json.dumps(

                history,

                indent=2,

                ensure_ascii=False,

            )

        )