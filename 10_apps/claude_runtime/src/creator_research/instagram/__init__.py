"""Phase 12B.2 -- Instagram Research Connector.

A read-only, observation-only implementation of
`creator_research.connector.BaseConnector` for Instagram. It does not
scrape, does not log in, does not automate around access controls, and
does not perform any posting/engagement action. See
docs/creator_research/instagram_connector.md for the architecture and
docs/creator_research/instagram_live_research_runbook.md for how a
future, separately-authorized live run would be prepared and executed.

Nothing in this package is registered or executed automatically on
import -- `connector.register_instagram_connector()` must be called
explicitly.
"""
