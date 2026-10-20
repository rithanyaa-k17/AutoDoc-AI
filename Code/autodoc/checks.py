"""Potential inconsistency report. Four transparent rules; every finding is a CANDIDATE.

These checks are NOT AUTOSAR compliance verification and will produce false positives
and miss real problems.
"""
import difflib
import re
from collections import defaultdict

from .config import CFG
from .extract import find_entities

RULES = {
    "R1": "Duplicate identifier definition: the same requirement-style ID starts two or more sentences "
          "with different text.",
    "R2": "Possibly undefined element: a component/interface appears in only ONE sentence in the whole "
          "document set, and that sentence also names other elements (it is referenced, never described).",
    "R3": "Near-duplicate names: two names of the same type differ only slightly (case, underscores, or "
          "similarity >= threshold) - possible inconsistent naming of one element.",
    "R4": "Conflicting timing values: the same signal is given different 'every N ms' values.",
}
ID_DEF = re.compile(r"^([A-Z]{2,6}-[A-Z]{2,6}-\d{2,4}|(?:RS|SWS|TPS|SRS)_[A-Za-z0-9]+_\d{3,6})\s*[:\-]\s*(.+)$")
TIMING = re.compile(r"\b(?:every|period of|cycle time of|cycle of)\s+(\d+(?:\.\d+)?)\s*ms\b", re.I)


def run_checks(per_sentence):
    findings = []

    # R1
    defs = defaultdict(list)
    for s, _ in per_sentence:
        m = ID_DEF.match(s["text"])
        if m:
            defs[m.group(1)].append((m.group(2).strip().lower(), s))
    for ident, items in defs.items():
        texts = {t for t, _ in items}
        if len(items) > 1 and len(texts) > 1:
            findings.append({"rule": "R1", "finding": f"Identifier {ident} is defined {len(items)} times with different text",
                             "evidence": " || ".join(f"{s['doc']} p.{s['page']}: {s['text'][:160]}" for _, s in items),
                             "pages": ", ".join(str(s["page"]) for _, s in items)})

    # R2
    count = defaultdict(int)
    where = {}
    for s, ents in per_sentence:
        for _, _, name, etype in ents:
            if etype in ("component", "interface"):
                count[name] += 1
                where.setdefault(name, (s, len(ents)))
    for name, c in count.items():
        s, nents = where[name]
        if c == 1 and nents >= 2:
            findings.append({"rule": "R2", "finding": f"'{name}' is mentioned only once in the document set",
                             "evidence": f"{s['doc']} p.{s['page']}: {s['text'][:250]}", "pages": str(s["page"])})

    # R3
    by_type = defaultdict(set)
    first = {}
    for s, ents in per_sentence:
        for _, _, name, etype in ents:
            if etype in ("component", "interface", "signal", "port"):
                by_type[etype].add(name)
                first.setdefault(name, s)
    thr = CFG["checks"]["near_duplicate_ratio"]
    for etype, names in by_type.items():
        names = sorted(names)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if etype == "port":  # P_X / R_X is the normal provided/required pair, not an inconsistency
                    sa_, sb_ = (re.sub(r"^[PR](Port)?_", "", x) for x in (a, b))
                    if sa_ == sb_:
                        continue
                na, nb = re.sub(r"[\s_-]", "", a).lower(), re.sub(r"[\s_-]", "", b).lower()
                ratio = difflib.SequenceMatcher(None, na, nb).ratio()
                if (na == nb or ratio >= thr) and na != "":
                    sa, sb = first[a], first[b]
                    findings.append({"rule": "R3", "finding": f"Similar {etype} names: '{a}' vs '{b}' (similarity {ratio:.2f})",
                                     "evidence": f"p.{sa['page']}: {sa['text'][:120]} || p.{sb['page']}: {sb['text'][:120]}",
                                     "pages": f"{sa['page']}, {sb['page']}"})
    # near-duplicates by common stem + interface-style suffix (RainSensorIf vs RainSensorInterface)
    for etype in ("interface",):
        stems = defaultdict(set)
        for n in by_type[etype]:
            stems[re.sub(r"(Interface|If|Service)$", "", n)].add(n)
        for stem, group in stems.items():
            if stem and len(group) > 1:
                g = sorted(group)
                if not any(f["rule"] == "R3" and all(x in f["finding"] for x in g[:2]) for f in findings):
                    sa, sb = first[g[0]], first[g[1]]
                    findings.append({"rule": "R3", "finding": f"Interface names share stem '{stem}': {', '.join(g)}",
                                     "evidence": f"p.{sa['page']}: {sa['text'][:120]} || p.{sb['page']}: {sb['text'][:120]}",
                                     "pages": f"{sa['page']}, {sb['page']}"})

    # R4
    vals = defaultdict(list)
    for s, ents in per_sentence:
        m = TIMING.search(s["text"])
        if not m:
            continue
        for _, _, name, etype in ents:
            if etype == "signal":
                vals[name].append((m.group(1), s))
    for name, items in vals.items():
        if len({v for v, _ in items}) > 1:
            findings.append({"rule": "R4", "finding": f"Signal {name} has conflicting timing values: "
                                                      f"{sorted({v for v, _ in items})} ms",
                             "evidence": " || ".join(f"p.{s['page']}: {s['text'][:140]}" for _, s in items),
                             "pages": ", ".join(str(s["page"]) for _, s in items)})

    for f in findings:
        f["rule_explanation"] = RULES[f["rule"]]
        f["status"] = "Potential issue - needs human review (may be a false positive)"
    return findings
