"""Rule-based extraction of CANDIDATE architecture elements (no ML, no invention).

Every item is a pattern match on the document text and is returned with its page and the
sentence it came from. Items must be verified by a human engineer.
"""
import re
from collections import defaultdict

from .config import CFG
from .ingest import sentences_of
from .retrieval import content_terms

E = CFG["extraction"]
_cs = "|".join(map(re.escape, E["component_suffixes"]))
_is = "|".join(map(re.escape, E["interface_suffixes"]))
_tc = "|".join(map(re.escape, E["title_case_component_words"]))
_ti = "|".join(map(re.escape, E["title_case_interface_words"]))
LEADING = {"The", "A", "An", "Each", "If", "This", "These", "Those", "Some", "Other", "All", "For",
           "Of", "In", "On", "And", "Or", "Its", "Their", "Any", "Both", "Between", "With", "By"}

# (type, regex, rule description). Order matters for overlap resolution (earlier wins).
PATTERNS = [
    ("requirement_id", re.compile(r"(?<![A-Za-z0-9-])[A-Z]{2,6}-[A-Z]{2,6}-\d{2,4}\b|\b(?:RS|SWS|TPS|SRS)_[A-Za-z0-9]+_\d{3,6}\b"),
     "ID such as HLD-FUN-001 or RS_AP_00001"),
    ("api_namespace", re.compile(r"\bara::[A-Za-z_]+(?:::[A-Za-z_]+)*"), "namespace starting with 'ara::'"),
    ("port", re.compile(r"\b[PR]Port_[A-Za-z0-9_]+\b|\b[PR]_[A-Z][A-Za-z0-9]+\b"),
     "name starting with P_/R_/PPort_/RPort_"),
    ("signal", re.compile(r"\bSig_[A-Za-z0-9_]+\b|\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b"),
     "name starting with Sig_ or UPPER_SNAKE_CASE"),
    ("interface", re.compile(rf"\b(?:[A-Z][a-z]+\s){{1,4}}(?:{_ti})\b"), "Title Case phrase ending in Interface"),
    ("component", re.compile(rf"\b(?:[A-Z][a-z]+(?:\s|-)){{1,4}}(?:{_tc})\b"),
     "Title Case phrase ending in Management/Manager/Cluster/..."),
    ("component", re.compile(rf"\b[A-Z][A-Za-z0-9]*(?:{_cs})\b"), "CamelCase name ending in Swc/Controller/Manager/..."),
    ("interface", re.compile(rf"\b[A-Z][A-Za-z0-9]*(?:{_is})\b"), "CamelCase name ending in Interface/Service/If/..."),
]
RELATION_CUES = [
    (re.compile(r"\b(sends?|transmits?|publish(?:es)?|forwards?|reports?)\b", re.I), "sends/reports to"),
    (re.compile(r"\b(receives?|subscribes? to|consumes?|reads?)\b", re.I), "receives/reads"),
    (re.compile(r"\b(calls?|invokes?|requests?|uses?|queries|triggers?)\b", re.I), "calls/uses/requests"),
    (re.compile(r"\b(provides?|offers?|exposes?|supplies|supplied)\b", re.I), "provides/supplies"),
    (re.compile(r"\b(depends? on|requires?|relies on)\b", re.I), "depends on/requires"),
]
FLOW_CUES = re.compile(r"\b(first|then|next|after|before|finally|when|once|triggers?|sequence|flow|step|"
                       r"startup|shutdown|sends?|receives?|requests?|reports?|detects?|if)\b", re.I)


def _strip_leading(name: str) -> str:
    words = name.split()
    for i in range(len(words) - 1, -1, -1):  # keep only the part after the LAST filler word
        if words[i] in LEADING:
            words = words[i + 1:]
            break
    return " ".join(words)


def find_entities(text: str):
    """[(start, end, name, type)] with overlaps removed (earlier-listed pattern wins)."""
    found, taken = [], []
    for etype, rx, _ in PATTERNS:
        for m in rx.finditer(text):
            name = m.group(0)
            start = m.start()
            if etype in ("component", "interface") and " " in name:
                cleaned = _strip_leading(re.sub(r"\s+", " ", name))
                if len(cleaned.split()) < 2:
                    continue
                start += name.index(cleaned.split()[0])
                name = cleaned
            end = start + len(name)
            if any(not (end <= a or start >= b) for a, b in taken):
                continue
            taken.append((start, end))
            found.append((start, end, name.strip(), etype))
    return sorted(found)


def build_inventory(docs):
    """Returns (inventory_rows, entities_by_sentence). Rows are CANDIDATES."""
    agg = {}
    per_sentence = []
    for d in docs:
        for s in sentences_of(d):
            ents = find_entities(s["text"])
            per_sentence.append((s, ents))
            for _, _, name, etype in ents:
                key = (name, etype)
                row = agg.setdefault(key, {"entity": name, "type": etype, "mentions": 0, "docs": set(),
                                           "pages": set(), "first_doc": s["doc"], "first_page": s["page"],
                                           "first_evidence": s["text"][:300]})
                row["mentions"] += 1
                row["docs"].add(s["doc"])
                row["pages"].add(s["page"])
    rows = []
    for row in agg.values():
        rows.append({"entity": row["entity"], "type": row["type"], "mentions": row["mentions"],
                     "documents": "; ".join(sorted(row["docs"])),
                     "pages": ", ".join(str(p) for p in sorted(row["pages"])[:25]),
                     "first_doc": row["first_doc"], "first_page": row["first_page"],
                     "first_evidence": row["first_evidence"],
                     "status": "Candidate - needs human verification"})
    rows.sort(key=lambda r: (r["type"], -r["mentions"], r["entity"]))
    return rows, per_sentence


def build_relationships(per_sentence):
    """Candidate relations: two different entities in one sentence + a verb cue.
    Direction follows textual order and can be WRONG (e.g. passive voice)."""
    maxn = E["max_entities_per_sentence_for_relations"]
    kinds = {"component", "interface", "port", "signal", "api_namespace"}
    rows = []
    for s, ents in per_sentence:
        ents = [e for e in ents if e[3] in kinds]
        names = []
        for e in ents:
            if e[2] not in [n[2] for n in names]:
                names.append(e)
        if not 2 <= len(names) <= maxn:
            continue
        cue_label = None
        for rx, label in RELATION_CUES:
            if rx.search(s["text"]):
                cue_label = label
                break
        if not cue_label:
            continue
        for a, b in zip(names, names[1:]):
            rows.append({"from": a[2], "from_type": a[3], "relation": cue_label, "to": b[2],
                         "to_type": b[3], "doc": s["doc"], "page": s["page"], "evidence": s["text"][:400],
                         "status": "Candidate - sentence co-occurrence, direction unverified"})
    return rows


def flow_summary(index, per_sentence_docs, topic: str, max_sentences=8):
    """Evidence list (verbatim, cited) + a clearly-labelled rule-based 'inferred' ordering.
    No paraphrasing: the 'summary' is the flow-cue sentences ordered as they appear in the document."""
    hits = [h for h in index.search(topic, top_k=6) if h["score"] >= CFG["retrieval"]["min_score"]]
    if not hits:
        return {"status": "insufficient", "evidence": [], "summary": [], "sections": []}
    topic_terms = content_terms(topic)
    cand = []
    for h in hits:
        from .ingest import split_sentences
        for pos, s in enumerate(split_sentences(h["text"])):
            t = content_terms(s + " " + h["section"])  # section title counts as topic context
            if FLOW_CUES.search(s) and (topic_terms & t or len(topic_terms) == 0):
                cand.append({"doc": h["doc"], "page": h["page"], "section": h["section"], "pos": pos,
                             "text": s, "score": h["score"]})
    seen, uniq = set(), []
    for c in sorted(cand, key=lambda x: -x["score"]):
        if c["text"] not in seen:
            seen.add(c["text"])
            uniq.append(c)
    uniq = uniq[:max_sentences]
    ordered = sorted(uniq, key=lambda x: (x["doc"], x["page"], x["pos"]))
    sections = sorted({c["section"] for c in ordered if c["section"]})
    return {"status": "ok" if ordered else "insufficient", "evidence": hits, "summary": ordered,
            "sections": sections}
