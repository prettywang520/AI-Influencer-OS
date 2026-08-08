"""Phase 12D.1 -- Instagram Reels Learning Adapter.

A thin adapter and orchestration layer that normalizes Instagram Reel
evidence into video_intelligence's existing VideoEvidence schema and
drives it through the existing, unmodified
video_intelligence.workflow.VideoIntelligenceWorkflow for many Reels
at once. Builds no new analyzer, duplicates no Video Intelligence
logic, and performs no Instagram collection of its own -- live
acquisition remains owned by creator_research/instagram. No browser,
no scraping, no network, no Whisper anywhere in this package.
"""
