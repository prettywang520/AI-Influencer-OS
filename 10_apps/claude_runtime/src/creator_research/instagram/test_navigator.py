import unittest

from src.creator_research.instagram.navigator import FakeBrowserAdapter, FakeElement


class FakeBrowserAdapterNavigationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeBrowserAdapter()

    def test_open_profile_sets_current_url(self):
        self.adapter.open_profile("https://example.invalid/demo")
        self.assertEqual(self.adapter.get_current_url(), "https://example.invalid/demo")

    def test_open_profile_creates_page_if_missing(self):
        self.adapter.open_profile("https://example.invalid/demo")
        self.assertIn("https://example.invalid/demo", self.adapter.pages)

    def test_wait_for_page_ready_true_by_default(self):
        self.adapter.open_profile("https://example.invalid/demo")
        self.assertTrue(self.adapter.wait_for_page_ready(5.0))

    def test_close_sets_closed_flag(self):
        self.adapter.close()
        self.assertTrue(self.adapter.closed)


class SelectorCandidateFallbackTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeBrowserAdapter()
        self.adapter.open_profile("https://example.invalid/demo")
        self.page = self.adapter.pages["https://example.invalid/demo"]

    def test_query_returns_none_when_no_candidate_matches(self):
        self.assertIsNone(self.adapter.query(["a.missing", "b.missing"]))

    def test_query_tries_candidates_in_order(self):
        self.page.register("b.secondary", [FakeElement(text="found via fallback")])
        result = self.adapter.query(["a.primary", "b.secondary"])
        self.assertEqual(self.adapter.get_text(result), "found via fallback")

    def test_query_prefers_first_matching_candidate(self):
        self.page.register("a.primary", [FakeElement(text="primary")])
        self.page.register("b.secondary", [FakeElement(text="secondary")])
        result = self.adapter.query(["a.primary", "b.secondary"])
        self.assertEqual(self.adapter.get_text(result), "primary")

    def test_query_all_returns_empty_list_when_no_match(self):
        self.assertEqual(self.adapter.query_all(["a.missing"]), [])

    def test_query_all_falls_back_to_next_candidate(self):
        self.page.register("b.secondary", [FakeElement(text="x"), FakeElement(text="y")])
        result = self.adapter.query_all(["a.primary", "b.secondary"])
        self.assertEqual(len(result), 2)


class ScrollPaginationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeBrowserAdapter()
        self.adapter.open_profile("https://example.invalid/demo")
        self.page = self.adapter.pages["https://example.invalid/demo"]
        self.items = [FakeElement(text=f"item{i}") for i in range(10)]

    def test_reveal_batch_limits_initial_visibility(self):
        self.page.register("article a", self.items, reveal_batch=3)
        self.assertEqual(len(self.adapter.query_all(["article a"])), 3)

    def test_scroll_reveals_more_items(self):
        self.page.register("article a", self.items, reveal_batch=3)
        self.adapter.scroll()
        self.assertEqual(len(self.adapter.query_all(["article a"])), 6)

    def test_scroll_never_reveals_beyond_total(self):
        self.page.register("article a", self.items, reveal_batch=3)
        for _ in range(10):
            self.adapter.scroll()
        self.assertEqual(len(self.adapter.query_all(["article a"])), 10)

    def test_scroll_count_increments(self):
        self.adapter.scroll()
        self.adapter.scroll()
        self.assertEqual(self.adapter.scroll_count, 2)

    def test_no_reveal_batch_shows_all_immediately(self):
        self.page.register("article a", self.items)
        self.assertEqual(len(self.adapter.query_all(["article a"])), 10)


class ElementAccessTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeBrowserAdapter()
        self.adapter.open_profile("https://example.invalid/demo")
        self.page = self.adapter.pages["https://example.invalid/demo"]

    def test_get_text_returns_element_text(self):
        self.page.register("a", [FakeElement(text="hello")])
        element = self.adapter.query(["a"])
        self.assertEqual(self.adapter.get_text(element), "hello")

    def test_get_text_none_element_returns_none(self):
        self.assertIsNone(self.adapter.get_text(None))

    def test_get_attribute_returns_value(self):
        self.page.register("a", [FakeElement(attributes={"href": "/p/1/"})])
        element = self.adapter.query(["a"])
        self.assertEqual(self.adapter.get_attribute(element, "href"), "/p/1/")

    def test_get_attribute_missing_key_returns_none(self):
        self.page.register("a", [FakeElement(attributes={})])
        element = self.adapter.query(["a"])
        self.assertIsNone(self.adapter.get_attribute(element, "href"))

    def test_get_attribute_none_element_returns_none(self):
        self.assertIsNone(self.adapter.get_attribute(None, "href"))

    def test_click_is_logged(self):
        self.page.register("a", [FakeElement()])
        element = self.adapter.query(["a"])
        self.adapter.click(element)
        self.assertEqual(self.adapter.click_log, [element])

    def test_get_screenshot_reference_returns_a_reference_string(self):
        ref = self.adapter.get_screenshot_reference("profile")
        self.assertIsInstance(ref, str)
        self.assertIn("profile", ref)


class MultiplePagesTests(unittest.TestCase):
    def test_pages_are_independent(self):
        adapter = FakeBrowserAdapter()
        page_a = adapter.add_page("https://example.invalid/a")
        page_b = adapter.add_page("https://example.invalid/b")
        page_a.register("h1", [FakeElement(text="page a")])
        page_b.register("h1", [FakeElement(text="page b")])

        adapter.open_profile("https://example.invalid/a")
        self.assertEqual(adapter.get_text(adapter.query(["h1"])), "page a")

        adapter.open_profile("https://example.invalid/b")
        self.assertEqual(adapter.get_text(adapter.query(["h1"])), "page b")


if __name__ == "__main__":
    unittest.main()
