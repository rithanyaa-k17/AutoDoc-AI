"""Change Impact Radar: sentence-level comparison of two document versions (difflib)."""
import difflib
import re

from .extract import find_entities
from .ingest import sentences_of


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _word_delta(a: str, b: str) -> str:
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, aw, bw, autojunk=False)
    removed, added = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            removed.append(" ".join(aw[i1:i2]))
        if tag in ("replace", "insert"):
            added.append(" ".join(bw[j1:j2]))
    return f"removed: [{' | '.join(removed)}]  added: [{' | '.join(added)}]"


def _affected(*texts):
    names = set()
    for t in texts:
        for _, _, name, etype in find_entities(t or ""):
            if etype in ("component", "interface", "port", "signal", "api_namespace"):
                names.add(name)
    return sorted(names)


def _row(kind, a, b):
    before = a["text"] if a else ""
    after = b["text"] if b else ""
    return {"change_type": kind,
            "before_page": a["page"] if a else "", "after_page": b["page"] if b else "",
            "section": (b or a)["section"],
            "before_text": before, "after_text": after,
            "word_level_change": _word_delta(before, after) if a and b else "",
            "potentially_affected_elements": ", ".join(_affected(before, after)),
            "review_note": "Textual change only. Not classified as a defect; engineer to assess impact."}


def compare_docs(old_doc: dict, new_doc: dict, match_threshold: float = 0.5):
    old = sentences_of(old_doc)
    new = sentences_of(new_doc)
    old_t = [_norm(s["text"]) for s in old]
    new_t = [_norm(s["text"]) for s in new]
    sm = difflib.SequenceMatcher(None, old_t, new_t, autojunk=len(old_t) > 2000)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            rows += [_row("removed", old[i], None) for i in range(i1, i2)]
        elif tag == "insert":
            rows += [_row("added", None, new[j]) for j in range(j1, j2)]
        else:  # replace: greedily pair similar sentences as "modified"
            free_new = list(range(j1, j2))
            for i in range(i1, i2):
                best, best_r = None, 0.0
                for j in free_new:
                    r = difflib.SequenceMatcher(None, old_t[i], new_t[j]).ratio()
                    if r > best_r:
                        best, best_r = j, r
                if best is not None and best_r >= match_threshold:
                    rows.append(_row("modified", old[i], new[best]))
                    free_new.remove(best)
                else:
                    rows.append(_row("removed", old[i], None))
            rows += [_row("added", None, new[j]) for j in free_new]
    summary = {k: sum(1 for r in rows if r["change_type"] == k) for k in ("added", "removed", "modified")}
    impact = {}
    for r in rows:
        for name in filter(None, r["potentially_affected_elements"].split(", ")):
            impact[name] = impact.get(name, 0) + 1
    impact_rows = [{"element": k, "changed_sentences": v,
                    "note": "Flagged for engineering review (mentioned in changed text)."}
                   for k, v in sorted(impact.items(), key=lambda kv: -kv[1])]
    return rows, summary, impact_rows
