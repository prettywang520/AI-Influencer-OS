from __future__ import annotations

from datetime import date

from .content_models import (
    ContentMoment,
    DailyContentPlan,
    ReelScene,
)


class ContentPlanner:
    """
    Creates one complete daily story before prompts are generated.

    Rules:
    - behavior first
    - every image has one main emotion
    - every image has social interaction
    - camera angles must differ
    - story must progress from arrival to leaving
    """

    def create_plan(
        self,
        *,
        target_date: str | None = None,
    ) -> DailyContentPlan:
        production_date = target_date or date.today().isoformat()

        feed = ContentMoment(
            content_id=f"{production_date}-feed-01",
            content_type="feed",
            title="Unexpected Book Discovery",
            location="Daikanyama T-Site, Tokyo, Japan",
            real_life=(
                "Aiko discovers a large travel photography book while "
                "sheltering from an unexpected afternoon rainstorm."
            ),
            behavior="comparing two travel photography books",
            emotion="inspired",
            interaction=(
                "A bookstore employee opens one book to a staff-favourite "
                "photo essay and explains why it is special."
            ),
            body_motion=(
                "Aiko supports one heavy book against her forearm, turns "
                "a page with her thumb and leans closer while listening."
            ),
            camera_story=(
                "Hidden documentary view through a narrow gap between "
                "two bookshelves."
            ),
            daily_details=[
                "wet transparent umbrella",
                "camera strap across her shoulder",
                "leather handbag",
                "hardcover photography books",
                "rain droplets on the window",
            ],
            story_stage="hero_moment",
            japanese_detail="A bookmark reads「旅は人生そのもの」",
        )

        stories = [
            ContentMoment(
                content_id=f"{production_date}-story-01",
                content_type="story",
                title="Escaping the Rain",
                location="Daikanyama T-Site entrance",
                real_life=(
                    "Aiko arrives during sudden heavy rain and prepares "
                    "to enter the bookstore."
                ),
                behavior="folding and securing a wet umbrella",
                emotion="relieved",
                interaction=(
                    "Another customer holds the glass door while Aiko "
                    "thanks them with a small bow."
                ),
                body_motion=(
                    "She presses the umbrella strap closed, shakes away "
                    "water carefully and steps across the wet entrance mat."
                ),
                camera_story=(
                    "Low-angle view from inside the entrance looking "
                    "through rain-covered glass."
                ),
                daily_details=[
                    "transparent umbrella",
                    "wet pavement",
                    "door reflection",
                    "handbag strap",
                    "entrance mat",
                ],
                story_stage="arrival",
            ),
            ContentMoment(
                content_id=f"{production_date}-story-02",
                content_type="story",
                title="Finding the Travel Section",
                location="Daikanyama T-Site information desk",
                real_life=(
                    "Aiko cannot find the photography section and asks "
                    "for help."
                ),
                behavior="showing a book title saved on her phone",
                emotion="curious",
                interaction=(
                    "A staff member points to the upper floor and marks "
                    "the correct section on a small store map."
                ),
                body_motion=(
                    "Aiko holds her phone beside the paper map, follows "
                    "the staff member's finger and nods attentively."
                ),
                camera_story=(
                    "Over-the-shoulder view from behind the information "
                    "desk employee."
                ),
                daily_details=[
                    "phone screen",
                    "paper store map",
                    "information counter",
                    "camera strap",
                    "floor directory",
                ],
                story_stage="discovery",
            ),
            ContentMoment(
                content_id=f"{production_date}-story-03",
                content_type="story",
                title="Writing New Travel Ideas",
                location="Bookstore café window table",
                real_life=(
                    "The photography book inspires Aiko to reorganise "
                    "ideas for her next journey."
                ),
                behavior="writing three destination ideas in her journal",
                emotion="thoughtful",
                interaction=(
                    "The barista quietly refills her water glass and "
                    "places a fresh napkin beside the journal."
                ),
                body_motion=(
                    "Her pen moves across the page while one elbow rests "
                    "beside the open book and she pauses to reread a note."
                ),
                camera_story=(
                    "Top-down documentary frame centred on the journal, "
                    "book and moving hand."
                ),
                daily_details=[
                    "travel journal",
                    "fountain pen",
                    "open photography book",
                    "water glass",
                    "rain outside the window",
                ],
                story_stage="quiet_moment",
                japanese_detail="The journal heading reads「次の旅」",
            ),
            ContentMoment(
                content_id=f"{production_date}-story-04",
                content_type="story",
                title="Taking the Book Home",
                location="Daikanyama T-Site cashier",
                real_life=(
                    "Aiko purchases the photography book before the "
                    "rain stops."
                ),
                behavior="receiving a wrapped book and checking the receipt",
                emotion="satisfied",
                interaction=(
                    "The cashier wraps the book in protective paper and "
                    "hands it over with both hands."
                ),
                body_motion=(
                    "Aiko receives the package with both hands, glances "
                    "at the receipt and gives a polite bow."
                ),
                camera_story=(
                    "Close over-the-counter documentary view focused on "
                    "the exchange."
                ),
                daily_details=[
                    "wrapped book",
                    "paper shopping bag",
                    "receipt",
                    "credit card",
                    "wooden checkout counter",
                ],
                story_stage="leaving",
                japanese_detail="The receipt reads「ありがとうございました」",
            ),
        ]

        reel_scenes = [
            ReelScene(
                scene_number=1,
                title="Rain Changes the Plan",
                behavior="closing a wet umbrella",
                emotion="relieved",
                interaction="a customer holds the entrance door",
                camera_story="low entrance angle through rainy glass",
                body_motion="quick hand movement securing the umbrella strap",
                daily_details=[
                    "umbrella",
                    "rain",
                    "wet shoes",
                ],
            ),
            ReelScene(
                scene_number=2,
                title="Asking for Directions",
                behavior="showing a saved book title on her phone",
                emotion="curious",
                interaction="staff member marks the store map",
                camera_story="employee over-the-shoulder shot",
                body_motion="Aiko compares the phone screen with the map",
                daily_details=[
                    "phone",
                    "map",
                    "floor directory",
                ],
            ),
            ReelScene(
                scene_number=3,
                title="The Discovery",
                behavior="opening a large photography book",
                emotion="inspired",
                interaction="staff member recommends a photo essay",
                camera_story="hidden bookshelf-gap shot",
                body_motion="page turns while Aiko leans toward the image",
                daily_details=[
                    "hardcover book",
                    "camera strap",
                    "wooden shelf",
                ],
            ),
            ReelScene(
                scene_number=4,
                title="New Travel Ideas",
                behavior="writing destinations in a journal",
                emotion="thoughtful",
                interaction="barista refills her water",
                camera_story="top-down table shot",
                body_motion="pen pauses before circling one destination",
                daily_details=[
                    "journal",
                    "pen",
                    "water glass",
                ],
            ),
            ReelScene(
                scene_number=5,
                title="A Story to Take Home",
                behavior="receiving the wrapped book",
                emotion="satisfied",
                interaction="cashier hands the package over with both hands",
                camera_story="close counter-level exchange",
                body_motion="Aiko receives it and gives a small bow",
                daily_details=[
                    "wrapped book",
                    "receipt",
                    "shopping bag",
                ],
            ),
        ]

        return DailyContentPlan(
            date=production_date,
            country="Japan",
            city="Tokyo",
            venue="Daikanyama T-Site",
            theme="Rainy Bookstore Afternoon",
            story_summary=(
                "A sudden rainstorm changes Aiko's plans and leads her "
                "to a photography book that inspires her next journey."
            ),
            feed=feed,
            stories=stories,
            reel_scenes=reel_scenes,
        )