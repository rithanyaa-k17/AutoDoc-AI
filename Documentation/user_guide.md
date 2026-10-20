# User guide
1. **Load documents** (sidebar): upload PDFs or pick from `Input_Data/`. The Documents tab shows file name, page count, status (`ok`, `partial`, `no_text`, `encrypted`, `invalid`, `empty`), pages with little text, content hash, and page-level character counts.
2. **System status panel** (top): provider/model + reachability, embedding model, chunk count + vector backend, index cache, and the active **answer mode**. Green = genuine LLM + RAG; amber = labelled fallback with the reason.
2b. **Ask (Q&A)**: choose *RAG* (semantic retrieval + LLM) or *Legacy TF-IDF*. RAG shows the answer mode, the generated answer with `[S#]` citations, a citation list (file/page) and the retrieved passages [S1..Sn] to verify. REFUSED = similarity below the sidebar threshold (LLM not called). Sidebar: provider, model, base URL, optional session-only API key, similarity slider.
3. **Architecture knowledge**: candidate inventory (component, interface, port, signal, api_namespace, requirement_id). Use the `review_status` and `reviewer_note` columns; they flow into the export. Below it: possible relationships (sentence co-occurrence + verb cue; direction unverified).
4. **Functional flows**: enter a topic (e.g. "automatic wiping flow"). Evidence sentences are verbatim and cited; the "inferred flow" is just those sentences in document order.
5. **Change Impact Radar**: load two versions, choose older/newer. See counts, before/after text, word-level change, and elements mentioned in changed text. A change is not a defect.
6. **Potential inconsistencies**: rules R1-R4 with evidence and pages. Findings are candidates.
7. **Exports & review**: CSV/JSON for inventory (with your review status), relationships, inconsistencies, retrieved evidence (last question), flow summary, comparison, change impact.
Tip: review edits live in the browser session; download the CSV before closing the tab.
