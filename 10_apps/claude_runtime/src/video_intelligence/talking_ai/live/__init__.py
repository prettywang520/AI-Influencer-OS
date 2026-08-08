"""Phase 12D.3 -- Talking AI Live Learning Adapter.

Bridges Instagram Reel *research* evidence -- from
creator_research/instagram's connector, or an operator who watched a
public Reel by hand -- into talking_ai's existing
TalkingAIEvidence/SpeechSegment schema, then drives it through the
existing, unmodified talking_ai.workflow.TalkingAIWorkflow. Builds no
new analyzer (the 9 domain analyzers, naturalness.py, and
production_dna.py from Phase 12D.2 are reused completely unmodified)
and builds no new Instagram collector -- creator_research/instagram's
connector remains the only thing in this codebase that ever opens a
browser; this package imports only its stable, already-public
dataclass types (ReelRecord), never its connector/observer/navigator
modules. Performs NO live Instagram access of its own -- every path
through this package consumes evidence already collected or hand-built
in memory. See docs/video_intelligence/talking_ai_live_adapter.md.
"""
