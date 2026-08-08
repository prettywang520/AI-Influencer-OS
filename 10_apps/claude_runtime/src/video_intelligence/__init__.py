"""Phase 12D.0 -- Video Intelligence OS.

A platform-neutral analysis package scoped around *a video* (not a
creator): from operator-supplied VideoEvidence about an observed
reference video, it derives a VideoDNA (13 trait fields covering hook,
pacing, speech, lip sync, gesture, camera, framing, editing,
subtitles, audio, storytelling, CTA, emotion) and folds many videos'
VideoDNA records into a de-identified, cross-video pattern knowledge
base. This package performs NO video production, rendering, or
generation of any kind, and NO automated evidence collection --
never Whisper, never an external API, never a browser, never ffmpeg/
ffprobe. It is fully decoupled from creator_intelligence/
creator_research/creator_learning and from the existing Timeline/
Subtitle/Overlay/Renderer engines -- nothing is imported in either
direction.
"""
