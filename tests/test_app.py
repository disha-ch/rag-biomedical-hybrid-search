import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.client = Mock()
        def respond(**kwargs):
            text = json.dumps({"expansion_1": "RET Hirschsprung association", "expansion_2": "Aganglionosis genes"}) \
                if "text" in kwargs else "RET is associated with Hirschsprung disease [77]."
            return SimpleNamespace(status="completed", output_text=text)
        self.client.responses.create.side_effect = respond
        raw = [{"id": 77, "passage": "RET is associated with Hirschsprung disease."},
               {"id": 42, "passage": "Insulin regulates glucose."}, {"id": 99, "passage": "nan"}]
        dense = Mock()
        dense.search.return_value = [{"passage_id": 77, "score": 0.9, "rank": 1, "text": raw[0]["passage"]}]
        for patcher in (
            patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-key"}),
            patch("dotenv.load_dotenv"),
            patch("datasets.load_dataset", return_value=raw),
            patch("src.retrieval.dense.DenseRetriever", return_value=dense),
            patch("src.generation.answer.OpenAI", return_value=self.client),
            patch("src.retrieval.query_expansion.OpenAI", return_value=self.client),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_all_modes_show_evidence_answer_citations_and_record(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        for mode in ("Lexical", "Dense", "Hybrid", "Hybrid + Query Expansion"):
            with self.subTest(mode=mode):
                app.text_input[0].set_value("Hirschsprung genes")
                app.selectbox[0].select(mode)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertIn("[77]", app.session_state["answer"]["answer"])
                self.assertEqual(app.session_state["record"]["retrieval_mode"], mode)
                self.assertTrue(any(c.value == "Source link: not available in dataset" for c in app.caption))
                self.assertTrue(any("Passage ID 77" in m.value for m in app.markdown))
                self.assertIn("RET is associated", " ".join(t.value for t in app.text))
                self.assertEqual(len(app.session_state["record"]["expanded_queries"]),
                                 2 if mode == "Hybrid + Query Expansion" else 0)
        app.text_input[0].set_value(" ")
        app.button[0].click().run()
        self.assertNotIn("record", app.session_state)
        self.assertTrue(app.warning)

    def test_insufficient_evidence_is_visible(self):
        self.client.responses.create.side_effect = None
        self.client.responses.create.return_value = SimpleNamespace(status="completed", output_text="INSUFFICIENT_EVIDENCE")
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.text_input[0].set_value("Unknown gene")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(w.value == "Insufficient evidence in the retrieved passages." for w in app.warning))
        self.assertTrue(any(c.value == "Citation validation: not applicable" for c in app.caption))

    def test_unsupported_citation_rejected_in_ui(self):
        self.client.responses.create.side_effect = None
        self.client.responses.create.return_value = SimpleNamespace(status="completed", output_text="Invented claim [999].")
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.text_input[0].set_value("Hirschsprung genes")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Answer rejected" in e.value for e in app.error))
        self.assertFalse(any("Invented claim" in m.value for m in app.markdown))

    def test_llm_error_keeps_evidence_and_missing_key_blocks_expansion(self):
        self.client.responses.create.side_effect = RuntimeError("simulated API failure")
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.text_input[0].set_value("Hirschsprung genes")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["record"]["results"])
        self.assertTrue(any("Answer generation unavailable" in w.value for w in app.warning))
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            app.selectbox[0].select("Hybrid + Query Expansion")
            app.button[0].click().run()
            self.assertTrue(any("Set OPENAI_API_KEY" in e.value for e in app.error))
            self.assertNotIn("record", app.session_state)
