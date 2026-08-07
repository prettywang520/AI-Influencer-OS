"""Phase 12B.1 -- Creator Research Agent.

Orchestrates creator research through an abstract Connector interface:
ResearchAgent -> Connector -> Evidence Collector -> Evidence Bundle ->
Creator Intelligence -> Creator DNA. This package contains NO
platform-specific scraping logic, never logs in, never automates a
browser, and never bypasses an access control. The only connector
implementation anywhere in this package is the synthetic
DummyConnector used in tests/demos. See docs/creator_research/
for the architecture, workflow, and the contract a future real
connector must follow.
"""
