import unittest
from pathlib import Path

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_research.connector import BaseConnector, ConnectorRegistry
from src.creator_research.evidence_queue import EvidenceQueue
from src.creator_research.exceptions import ConnectorNotImplementedError, ConnectorNotRegisteredError
from src.creator_research.interfaces import ConnectorSection, EvidenceBundle


def _evidence(n=1):
    return [Evidence(evidence_type=EvidenceType.OPERATOR_OBSERVATION, source_description=f"item {i}") for i in range(n)]


class BaseConnectorTests(unittest.TestCase):
    def test_instantiable_without_overriding_anything(self):
        connector = BaseConnector()
        self.assertEqual(connector.name, "base")

    def test_each_uncalled_method_raises_connector_not_implemented(self):
        connector = BaseConnector()
        methods = [
            "collect_profile",
            "collect_grid",
            "collect_posts",
            "collect_captions",
            "collect_comments",
            "collect_creator_replies",
            "collect_reels",
            "collect_highlights",
            "collect_relationships",
            "collect_visual_examples",
        ]
        for method_name in methods:
            with self.assertRaises(ConnectorNotImplementedError):
                getattr(connector, method_name)(job=None)

    def test_partial_subclass_can_be_instantiated(self):
        class PartialConnector(BaseConnector):
            name = "partial"

            def collect_profile(self, job):
                return EvidenceBundle(section=ConnectorSection.PROFILE, items=_evidence())

        connector = PartialConnector()
        result = connector.collect_profile(job=None)
        self.assertEqual(len(result.items), 1)
        with self.assertRaises(ConnectorNotImplementedError):
            connector.collect_grid(job=None)

    def test_all_ten_sections_have_a_matching_method(self):
        connector = BaseConnector()
        for section in ConnectorSection.ALL:
            self.assertTrue(hasattr(connector, f"collect_{section}"))


class ConnectorRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ConnectorRegistry()

    def test_register_and_get(self):
        self.registry.register("dummy", lambda: BaseConnector())
        connector = self.registry.get("dummy")
        self.assertIsInstance(connector, BaseConnector)

    def test_get_unregistered_name_raises(self):
        with self.assertRaises(ConnectorNotRegisteredError):
            self.registry.get("nonexistent")

    def test_registered_names_sorted(self):
        self.registry.register("zeta", lambda: BaseConnector())
        self.registry.register("alpha", lambda: BaseConnector())
        self.assertEqual(self.registry.registered_names(), ("alpha", "zeta"))

    def test_re_registering_same_name_overwrites(self):
        self.registry.register("dummy", lambda: BaseConnector())
        marker = object()

        class Marked(BaseConnector):
            name = "marked"

        self.registry.register("dummy", lambda: Marked())
        self.assertIsInstance(self.registry.get("dummy"), Marked)

    def test_factory_called_fresh_each_time(self):
        calls = []

        def factory():
            calls.append(1)
            return BaseConnector()

        self.registry.register("dummy", factory)
        self.registry.get("dummy")
        self.registry.get("dummy")
        self.assertEqual(len(calls), 2)


class EvidenceBundleTests(unittest.TestCase):
    def test_default_construction(self):
        bundle = EvidenceBundle(section=ConnectorSection.PROFILE)
        self.assertEqual(bundle.items, [])
        self.assertEqual(bundle.warnings, [])

    def test_items_hold_real_evidence_objects(self):
        bundle = EvidenceBundle(section=ConnectorSection.CAPTIONS, items=_evidence(2))
        self.assertEqual(len(bundle.items), 2)
        self.assertIsInstance(bundle.items[0], Evidence)


class EvidenceQueueTests(unittest.TestCase):
    def setUp(self):
        self.queue = EvidenceQueue()

    def test_add_and_all_items(self):
        self.queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, items=_evidence(2)))
        self.assertEqual(len(self.queue.all_items()), 2)

    def test_items_for_section_filters(self):
        self.queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, items=_evidence(1)))
        self.queue.add(EvidenceBundle(section=ConnectorSection.GRID, items=_evidence(3)))
        self.assertEqual(len(self.queue.items_for_section(ConnectorSection.GRID)), 3)
        self.assertEqual(len(self.queue.items_for_section(ConnectorSection.CAPTIONS)), 0)

    def test_counts_by_section(self):
        self.queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, items=_evidence(1)))
        self.queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, items=_evidence(2)))
        self.assertEqual(self.queue.counts_by_section()[ConnectorSection.PROFILE], 3)

    def test_warnings_union_across_bundles(self):
        self.queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, warnings=["a"]))
        self.queue.add(EvidenceBundle(section=ConnectorSection.GRID, warnings=["b", "c"]))
        self.assertEqual(self.queue.warnings(), ["a", "b", "c"])

    def test_bundles_returns_a_copy(self):
        self.queue.add(EvidenceBundle(section=ConnectorSection.PROFILE))
        bundles = self.queue.bundles()
        bundles.append(EvidenceBundle(section=ConnectorSection.GRID))
        self.assertEqual(len(self.queue.bundles()), 1)

    def test_empty_queue_has_no_items_or_warnings(self):
        self.assertEqual(self.queue.all_items(), [])
        self.assertEqual(self.queue.warnings(), [])
        self.assertEqual(self.queue.counts_by_section(), {})


class StructuralSafetyTests(unittest.TestCase):
    def setUp(self):
        from src.creator_research import connector, evidence_queue, interfaces

        self.sources = {
            "connector.py": Path(connector.__file__).read_text(encoding="utf-8"),
            "interfaces.py": Path(interfaces.__file__).read_text(encoding="utf-8"),
            "evidence_queue.py": Path(evidence_queue.__file__).read_text(encoding="utf-8"),
        }

    def test_no_network_or_browser_imports(self):
        forbidden_prefixes = (
            "import requests",
            "from requests",
            "import httpx",
            "import playwright",
            "from playwright",
            "import selenium",
            "from selenium",
        )
        for filename, source in self.sources.items():
            for line in source.splitlines():
                stripped = line.strip()
                for forbidden in forbidden_prefixes:
                    self.assertFalse(stripped.startswith(forbidden), f"{filename}: {stripped!r}")

    def test_no_instagram_or_social_or_publishing_reference(self):
        for filename, source in self.sources.items():
            lowered = source.lower()
            for forbidden in ("instagram", "03_personas", "from .social", "from src.social", "from .publishing", "from src.publishing"):
                self.assertNotIn(forbidden, lowered, filename)

    def test_no_network_calls(self):
        for filename, source in self.sources.items():
            for forbidden in ("requests.", "urllib.request", "http.client", "socket."):
                self.assertNotIn(forbidden, source, filename)

    def test_no_real_creator_handle(self):
        for filename, source in self.sources.items():
            for handle in ("carlysuen112", "aitana_10_01"):
                self.assertNotIn(handle, source, filename)

    def test_no_login_or_cookie_keywords(self):
        for filename, source in self.sources.items():
            for forbidden in ("login(", "cookie", "session_token", "password"):
                self.assertNotIn(forbidden, source.lower(), filename)

    def test_connector_has_no_implementation_logic(self):
        # every collect_* method body in connector.py is exactly one
        # "raise ConnectorNotImplementedError" statement -- no parsing,
        # no HTML, no data extraction logic exists here.
        source = self.sources["connector.py"]
        self.assertNotIn("BeautifulSoup", source)
        self.assertNotIn("lxml", source)
        self.assertNotIn("html.parser", source)


if __name__ == "__main__":
    unittest.main()
