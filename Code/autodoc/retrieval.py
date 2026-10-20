"""Evidence-grounded retrieval: page-level chunks + TF-IDF + cosine similarity.

This is EXTRACTIVE question answering. There is no generative model: every sentence shown
is copied verbatim from the uploaded PDF together with its document name and page number.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .config import CFG
from .ingest import split_sentences

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:::[A-Za-z][A-Za-z0-9_]*)*")
PARTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def expand(text: str) -> str:
    """Keep each identifier AND its CamelCase / snake_case parts, so that the query
    'wiper control' can match the identifier 'WiperControlSwc'."""
    out = []
    for tok in TOKEN.findall(text):
        out.append(tok)
        parts = PARTS.findall(tok.replace("::", " ").replace("_", " "))
        if len(parts) > 1:
            out.extend(parts)
    return " ".join(out)


def content_terms(text: str):
    return {t.lower() for t in expand(text).split()
            if len(t) > 1 and t.lower() not in ENGLISH_STOP_WORDS}


def build_chunks(docs, max_words=None, overlap_sentences=None):
    """Overlapping, sentence-aligned chunks. Never cross a page or section-segment boundary, so each chunk
    has exactly one filename + page + section. `overlap_sentences` sentences are repeated at the start of the
    next chunk (context continuity)."""
    limit = max_words or CFG["retrieval"]["chunk_max_words"]
    overlap = CFG["retrieval"].get("chunk_overlap_sentences", 1) if overlap_sentences is None else overlap_sentences
    chunks = []
    for d in docs:
        for p in d["pages"]:
            for seg in p.get("segments", []):
                if seg["is_heading"] or re.search(r"(\.\s){6,}", seg["text"]):
                    continue  # skip headings and table-of-contents style text (dot leaders)
                sents = split_sentences(seg["text"])
                i = 0
                while i < len(sents):
                    cur, words, j = [], 0, i
                    while j < len(sents) and (not cur or words + len(sents[j].split()) <= limit):
                        cur.append(sents[j])
                        words += len(sents[j].split())
                        j += 1
                    chunks.append({"chunk_id": len(chunks), "doc": d["name"], "page": p["page"],
                                   "section": seg["section"], "text": " ".join(cur)})
                    if j >= len(sents):
                        break
                    i = max(j - overlap, i + 1)  # step forward, keeping `overlap` sentences; always progress
    return chunks


class Index:
    def __init__(self, docs):
        self.chunks = build_chunks(docs)
        self.vec = None
        if self.chunks:
            self.vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                       sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b")
            self.matrix = self.vec.fit_transform(
                [expand(c["section"] + " " + c["text"]) for c in self.chunks])

    def search(self, query: str, top_k=None, doc_filter=None):
        if not self.chunks or not query.strip():
            return []
        top_k = top_k or CFG["retrieval"]["top_k"]
        q = self.vec.transform([expand(query)])
        scores = linear_kernel(q, self.matrix).ravel()
        qterms = content_terms(query)
        order = scores.argsort()[::-1]
        hits = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= top_k:
                break
            c = self.chunks[i]
            if doc_filter and c["doc"] not in doc_filter:
                continue
            cterms = content_terms(c["text"] + " " + c["section"])
            cov = len(qterms & cterms) / len(qterms) if qterms else 0.0
            hits.append({**c, "score": float(scores[i]), "coverage": round(cov, 2)})
        return hits


def confidence_label(score: float) -> str:
    r = CFG["retrieval"]
    if score >= r["confidence_high"]:
        return "High (heuristic)"
    if score >= r["confidence_medium"]:
        return "Medium (heuristic)"
    return "Low (heuristic)"


def answer(index: Index, question: str, doc_filter=None) -> dict:
    """Extractive, evidence-grounded answer. Returns status 'ok' or 'insufficient'."""
    r = CFG["retrieval"]
    hits = index.search(question, doc_filter=doc_filter)
    good = [h for h in hits if h["score"] >= r["min_score"] and h["coverage"] >= r["min_query_coverage"]]
    if not good:
        return {"status": "insufficient", "question": question, "evidence": hits,
                "message": "The uploaded document(s) do not provide sufficient evidence to answer "
                           "this question. Weak matches (if any) are shown below for transparency only.",
                "confidence": "None", "top_score": hits[0]["score"] if hits else 0.0, "points": []}
    qterms = content_terms(question)
    points = []
    for h in good[:3]:
        sents = split_sentences(h["text"])
        scored = []
        for s in sents:
            st = content_terms(s)
            overlap = len(qterms & st)
            scored.append((overlap / (len(st) ** 0.5 + 1e-9), overlap, s))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        best = [x for x in scored[:2] if x[1] > 0]
        for _, _, s in best[: (2 if h is good[0] else 1)]:
            points.append({"text": s, "doc": h["doc"], "page": h["page"], "section": h["section"]})
    return {"status": "ok", "question": question, "evidence": good, "points": points,
            "confidence": confidence_label(good[0]["score"]), "top_score": good[0]["score"],
            "message": "Extractive answer: sentences copied verbatim from the retrieved passages."}
