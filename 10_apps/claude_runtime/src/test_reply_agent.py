from reply_agent import build_agent
from reply_models import CommentInput, ContentContext


def main() -> None:
    agent = build_agent()

    comment = CommentInput(
        request_id="test-001",
        comment_id="comment-001",
        follower_id="sample-follower",
        text="beautiful photo! where is this?",
        platform="instagram_feed",
        content_id="feed-2026-07-26",
    )

    context = ContentContext(
        content_type="feed",
        topic="bookstore",
        venue="Daikanyama T-Site",
        city="Tokyo",
        country="Japan",
        location="Daikanyama, Tokyo",
        activity="browsing a photography book",
        emotion="inspired",
        ai_generated=True,
    )

    result = agent.process(comment, context)
    print(result.to_dict())


if __name__ == "__main__":
    main()