"""Deterministic OFFLINE tests for the RAG pipeline (no model download, no API key, no network except localhost).
Run from project root:  python -m unittest discover -s Code/tests -v
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from autodoc.ingest import ingest_pdf
from autodoc.llm import LLMClient, LLMUnavailable
from autodoc.rag import SYSTEM_PROMPT, RagEngine, build_prompt, check_citations
from autodoc.retrieval import build_chunks
from autodoc.vectorstore import VectorStore, faiss
from fakes import FakeEmbedder, MockLLMServer

INPUT = Path(__file__).resolve().parents[2] / "Input_Data"
V10 = INPUT / "SYNTHETIC_Wiper_HLD_v1.0.pdf"


@unittest.skipUnless(V10.exists(), "run Code/generate_synthetic.py first")
class TestRag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = ingest_pdf(V10.name, V10.read_bytes())

    def engine(self, llm=None, embedder=FakeEmbedder()):
        return RagEngine([self.doc], embedder=embedder, llm=llm, use_disk_cache=False)

    # ---- chunking / metadata
    def test_chunks_have_metadata_and_stay_on_one_page(self):
        chunks = build_chunks([self.doc])
        self.assertTrue(chunks)
        for c in chunks:
            self.assertEqual(c["doc"], V10.name)
            self.assertIn(c["page"], range(1, 6))
            self.assertTrue(c["text"].strip())

    def test_chunk_overlap_repeats_sentences(self):
        small = build_chunks([self.doc], max_words=18, overlap_sentences=1)
        none = build_chunks([self.doc], max_words=18, overlap_sentences=0)
        self.assertGreaterEqual(len(small), len(none))
        same_seg = [(a, b) for a, b in zip(small, small[1:]) if a["page"] == b["page"] and a["section"] == b["section"]]
        self.assertTrue(any(a["text"].split(". ")[-1][:15] in b["text"] for a, b in same_seg))

    def test_chunking_makes_progress_with_huge_overlap(self):
        self.assertTrue(build_chunks([self.doc], max_words=5, overlap_sentences=50))  # must terminate

    # ---- vector store
    def test_store_roundtrip_and_search(self):
        emb = FakeEmbedder()
        chunks = build_chunks([self.doc])
        vs = VectorStore(emb.encode([c["text"] for c in chunks]), chunks)
        tmp = Path(tempfile.mkdtemp())
        try:
            vs.save(tmp)
            vs2 = VectorStore.load(tmp)
            q = emb.encode(["port provided by RainSensorSwc"])[0]
            a, b = vs.search(q, 3), vs2.search(q, 3)
            self.assertEqual([c["chunk_id"] for c, _ in a], [c["chunk_id"] for c, _ in b])
            self.assertEqual(a[0][0]["page"], 3)
        finally:
            shutil.rmtree(tmp)

    @unittest.skipIf(faiss is None, "faiss-cpu not installed (optional)")
    def test_faiss_matches_numpy(self):
        emb = FakeEmbedder()
        chunks = build_chunks([self.doc])
        v = emb.encode([c["text"] for c in chunks])
        q = emb.encode(["motor current limit"])[0]
        f, n = VectorStore(v, chunks, True), VectorStore(v, chunks, False)
        self.assertEqual(f.backend, "faiss")
        self.assertEqual(n.backend, "numpy")
        self.assertEqual([c["chunk_id"] for c, _ in f.search(q, 3)], [c["chunk_id"] for c, _ in n.search(q, 3)])

    # ---- prompt grounding
    def test_prompt_contains_rules_context_and_page_metadata(self):
        eng = self.engine()
        hits = eng.retrieve("Which port is provided by RainSensorSwc?")
        p = build_prompt("Which port is provided by RainSensorSwc?", hits)
        self.assertIn("[S1] (file: " + V10.name, p)
        self.assertIn("page ", p)
        for rule in ("ONLY using", "INSUFFICIENT_EVIDENCE", "untrusted", "Cite"):
            self.assertIn(rule, SYSTEM_PROMPT)

    def test_check_citations(self):
        self.assertEqual(check_citations("x [S1] y [S2]", 3)["cited"], [1, 2])
        self.assertEqual(check_citations("x [S9]", 3)["invalid"], [9])
        self.assertFalse(check_citations("no cites", 3)["has_valid_citation"])

    # ---- end to end with mock LLM
    def test_llm_rag_answer_has_citations_and_evidence(self):
        srv = MockLLMServer("The port is P_RainIntensity [S1].")
        try:
            eng = self.engine(LLMClient("openai_compatible", srv.url, "mock-model", "SECRET-KEY-123"))
            res = eng.ask("Which port is provided by RainSensorSwc?", min_score=0.05)
            self.assertEqual(res["mode"], "llm_rag")
            self.assertEqual(res["status"], "ok")
            self.assertTrue(res["citations"])
            self.assertTrue(res["evidence"])
            sent = srv.requests[0]
            self.assertIn("CONTEXT:", sent["messages"][1]["content"])
            self.assertEqual(sent["_auth"], "Bearer SECRET-KEY-123")  # key goes to header only
            self.assertNotIn("SECRET-KEY-123", str(res))             # and never into results
        finally:
            srv.close()

    def test_uncited_llm_answer_is_flagged(self):
        srv = MockLLMServer("The port is P_RainIntensity.")
        try:
            eng = self.engine(LLMClient("openai_compatible", srv.url, "mock-model"))
            res = eng.ask("Which port is provided by RainSensorSwc?", min_score=0.05)
            self.assertTrue(any("NO valid" in w for w in res["warnings"]))
        finally:
            srv.close()

    def test_insufficient_evidence_refuses_without_calling_llm(self):
        srv = MockLLMServer("should never be used [S1]")
        try:
            eng = self.engine(LLMClient("openai_compatible", srv.url, "mock-model"))
            res = eng.ask("What is the recipe for chocolate cake?", min_score=0.30)
            self.assertEqual(res["mode"], "refused")
            self.assertEqual(res["status"], "insufficient")
            self.assertEqual(len(srv.requests), 0)
            self.assertIn("Missing", res["message"])
        finally:
            srv.close()

    def test_model_can_refuse_itself(self):
        srv = MockLLMServer("INSUFFICIENT_EVIDENCE: the document does not state the wiper blade material.")
        try:
            eng = self.engine(LLMClient("openai_compatible", srv.url, "mock-model"))
            res = eng.ask("What is the wiper blade material?", min_score=0.0)
            self.assertEqual(res["status"], "insufficient")
        finally:
            srv.close()

    # ---- fallbacks are labelled honestly
    def test_no_llm_gives_retrieval_only_never_llm_label(self):
        res = self.engine(LLMClient("none")).ask("Which port is provided by RainSensorSwc?", min_score=0.05)
        self.assertEqual(res["mode"], "retrieval_only")
        self.assertNotIn("LLM +", res["mode_label"])
        self.assertEqual(res["answer"], "")

    def test_dead_llm_server_falls_back_with_warning(self):
        eng = self.engine(LLMClient("openai_compatible", "http://127.0.0.1:9/v1", "m"))
        res = eng.ask("Which port is provided by RainSensorSwc?", min_score=0.05)
        self.assertEqual(res["mode"], "retrieval_only")
        self.assertTrue(res["warnings"])

    def test_no_embedder_uses_labelled_tfidf_fallback(self):
        eng = RagEngine([self.doc], embedder=None, llm=LLMClient("none"), embed_error="not installed")
        res = eng.ask("Which port is provided by RainSensorSwc?")
        self.assertEqual(res["mode"], "lexical_fallback")
        self.assertIn("FALLBACK", res["mode_label"])
        self.assertTrue(res["points"] and "page" in res["points"][0])

    def test_llm_client_health_and_unconfigured(self):
        self.assertFalse(LLMClient("none").configured)
        with self.assertRaises(LLMUnavailable):
            LLMClient("none").chat("s", "u")
        srv = MockLLMServer()
        try:
            self.assertTrue(LLMClient("openai_compatible", srv.url, "mock-model").health()["ok"])
        finally:
            srv.close()


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(V10.exists(), "run Code/generate_synthetic.py first")
class TestRagEvalScript(unittest.TestCase):
    def test_eval_script_logic_with_test_doubles(self):
        """Exercises evaluate_rag.run() plumbing with fakes. Output goes to a temp dir and is NOT a real result."""
        import evaluate_rag
        srv = MockLLMServer("According to the document [S1].")
        tmp = Path(tempfile.mkdtemp())
        try:
            summary, rows = evaluate_rag.run(FakeEmbedder(), LLMClient("openai_compatible", srv.url, "mock"),
                                             tmp, min_score=0.30)
            self.assertEqual(summary["embedding_model"], "fake-hash-embedder")
            self.assertEqual(len(rows), 10)
            self.assertTrue((tmp / "rag_llm_results.csv").exists())
        finally:
            srv.close()
            shutil.rmtree(tmp)
