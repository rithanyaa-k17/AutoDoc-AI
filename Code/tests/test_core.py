"""Unit tests. Run from the project root:  python -m unittest discover -s Code/tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf

from autodoc.checks import run_checks
from autodoc.compare import compare_docs
from autodoc.extract import build_inventory, find_entities
from autodoc.ingest import ingest_pdf
from autodoc.retrieval import Index, answer

INPUT = Path(__file__).resolve().parents[2] / "Input_Data"
V10, V11 = INPUT / "SYNTHETIC_Wiper_HLD_v1.0.pdf", INPUT / "SYNTHETIC_Wiper_HLD_v1.1.pdf"


@unittest.skipUnless(V10.exists() and V11.exists(), "run Code/generate_synthetic.py first")
class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d10 = ingest_pdf(V10.name, V10.read_bytes())
        cls.d11 = ingest_pdf(V11.name, V11.read_bytes())
        cls.index = Index([cls.d10])

    def test_ingest_page_count_and_status(self):
        self.assertEqual(self.d10["status"], "ok")
        self.assertEqual(self.d10["page_count"], 5)
        self.assertEqual([p["page"] for p in self.d10["pages"]], [1, 2, 3, 4, 5])

    def test_answer_has_page_citation(self):
        res = answer(self.index, "Which port is provided by RainSensorSwc?")
        self.assertEqual(res["status"], "ok")
        self.assertIn(3, [h["page"] for h in res["evidence"]])
        self.assertTrue(all("page" in p for p in res["points"]))

    def test_insufficient_evidence(self):
        self.assertEqual(answer(self.index, "What is the recipe for chocolate cake?")["status"], "insufficient")

    def test_inventory_contains_known_components(self):
        inv, _ = build_inventory([self.d10])
        names = {r["entity"] for r in inv if r["type"] == "component"}
        self.assertIn("WiperControlSwc", names)
        self.assertNotIn("NotInDocumentSwc", names)  # nothing is invented

    def test_compare_detects_changes(self):
        rows, summ, _ = compare_docs(self.d10, self.d11)
        self.assertGreaterEqual(summ["added"], 1)
        self.assertGreaterEqual(summ["removed"], 1)
        self.assertGreaterEqual(summ["modified"], 1)

    def test_checks_find_duplicate_id(self):
        _, ps = build_inventory([self.d10])
        self.assertTrue(any(f["rule"] == "R1" and "HLD-FUN-002" in f["finding"] for f in run_checks(ps)))


class TestRobustness(unittest.TestCase):
    def test_invalid_pdf(self):
        self.assertEqual(ingest_pdf("bad.pdf", b"not a pdf")["status"], "invalid")

    def test_empty_bytes(self):
        self.assertIn(ingest_pdf("empty.pdf", b"")["status"], ("invalid", "empty"))

    def test_encrypted_pdf(self):
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "secret text here for the encrypted test document")
        data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
        self.assertEqual(ingest_pdf("enc.pdf", data)["status"], "encrypted")

    def test_scanned_like_pdf_flags_low_text(self):
        doc = pymupdf.open()
        doc.new_page()  # blank page, no text layer
        res = ingest_pdf("blank.pdf", doc.tobytes())
        self.assertEqual(res["status"], "no_text")
        self.assertEqual(res["low_text_pages"], [1])

    def test_entity_patterns(self):
        names = {e[2] for e in find_entities("The WiperSwc uses Sig_Speed and P_Out via FooInterface.")}
        self.assertTrue({"WiperSwc", "Sig_Speed", "P_Out", "FooInterface"} <= names)


if __name__ == "__main__":
    unittest.main()
