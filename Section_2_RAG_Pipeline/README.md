# RAG Pipeline over Food-Delivery Support Docs (LangChain)

This is my submission for Section 2 — Task 2.1. I went with my own domain
docs rather than the provided set, since it lets this tie into the same
food-delivery support theme as Section 1's voice agent.

## Document choice

Five short markdown files under `documents/`, written by me: refund
policy, delivery times, account security, driver tips, and a general FAQ.
Around 150-250 words each, which is enough to produce several chunks per
doc and actually exercise retrieval instead of just returning "the whole
document" every time.

## What's in here

- `documents/` — the 5 source docs.
- `embeddings.py` — the embedding backend. Default is a local TF-IDF + SVD
  embedding (via scikit-learn) that needs no API key and no model
  download, so the whole pipeline runs offline. Set `USE_HF_EMBEDDINGS=1`
  to switch to real neural embeddings
  (`sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`)
  once you have network access — nothing else in the code changes, same
  decoupling idea as the STT/TTS swap in Section 1.
- `rag_pipeline.py` — chunking (LangChain's `RecursiveCharacterTextSplitter`),
  FAISS indexing, retrieval, citation formatting, and the "no relevant
  context" refusal logic.
- `llm.py` — answer generation. Groq if `GROQ_API_KEY` is set, otherwise a
  small extractive fallback (pulls the most relevant sentence out of the
  retrieved chunks by keyword overlap) so the retrieval + citation logic
  is demonstrable without any API key.
- `demo.py` — runs the questions below and writes `sample_answers.txt`.

## Running it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key   # optional — console.groq.com, free tier
python demo.py
```

## The 3 example questions (plus a deliberate out-of-scope one)

Ran with the offline fallback (no `GROQ_API_KEY` set) so this is exactly
what came out of an actual run:

**Q: How long do I have to request a refund after delivery?**
A: Customers may request a refund within 48 hours of delivery for orders
that arrived incorrect, incomplete, or damaged. [1]
Sources: refund_policy.md (chunk 0), refund_policy.md (chunk 3),
delivery_times.md (chunk 2)

**Q: What should I do if I think someone accessed my account without permission?**
A: If a customer suspects unauthorized access to their account, such as
orders they did not place or a changed delivery address they don't
recognize, they should immediately change their password from the
Account Settings screen and contact support to flag the account for
review. [2]
Sources: account_security.md (chunk 0), account_security.md (chunk 1),
faq.md (chunk 1)

**Q: Can I still change a driver's tip after the delivery is done?**
A: Customers can add a tip for their courier either before placing the
order, as a percentage or fixed amount, or after delivery from the order
receipt screen. [1]
Sources: driver_tips.md (chunk 0), driver_tips.md (chunk 1),
driver_tips.md (chunk 3)

**Q: What's the best way to bake a chocolate lava cake? (out of scope, on purpose)**
A: I don't have information about that in the provided documents.
Sources: (none — top match scored below the relevance threshold)

That last one is the important one for the assignment: nothing in the
docs is about baking, the top similarity score came in under the
threshold, and the pipeline refuses instead of calling the LLM at all —
so there's no chance of it hallucinating a recipe out of delivery-app
policy text.

One honest note: with real Groq answers instead of the offline fallback,
the phrasing is naturally better (the model can synthesize across
sentences instead of just extracting one), but the citations and the
refusal behavior work identically either way, since both live in
`rag_pipeline.py`, not in the LLM.

---

## Write-up: what I'd change for longer documents

Everything here works because the docs are short — each one is basically
one topic, so even naive chunking rarely splits a fact away from its
context, and TF-IDF-level similarity is enough to find the right chunk.
That stops being true once documents get longer (say, a 40-page policy
manual instead of a 200-word markdown file), and a few things would start
breaking:

**Chunking.** Fixed-size character chunking (what I'm using here) starts
cutting sentences and even whole clauses in half once documents are long
and structurally complex, which is exactly how you get a chunk that
contains "the fee, unless" with the actual exception in the *next* chunk.
I'd switch to a structure-aware splitter that chunks along headings,
sections, or paragraphs first, falls back to sentence boundaries within
those, and keeps a small overlap so a fact split across two chunks isn't
lost entirely. I'd also probably chunk more granularly (smaller chunks)
but retrieve more of them per query, then let re-ranking (below) sort out
which ones actually matter — better than fewer, bigger chunks where half
the tokens sent to the LLM are irrelevant filler.

**Retrieval quality.** My TF-IDF + SVD embeddings only work here because
the corpus is tiny and each doc is topically distinct — I actually saw
this break down in testing, where a generic question like "do you sell
pizza on the app?" scored a higher similarity than a real, in-scope
question, purely because of shared common words like "app." On longer,
more overlapping documents that problem gets much worse. First fix is
swapping to real embeddings (the `USE_HF_EMBEDDINGS=1` path is already
there for this) since dense neural embeddings separate genuinely
different topics far better than a bag-of-words method ever will. Beyond
that:
- **Hybrid search** — combine dense vector similarity with a sparse
  method like BM25, then merge scores. Dense embeddings are good at
  semantic similarity but can miss exact keyword matches (an order ID, an
  exact policy term); BM25 catches those. Longer documents especially
  benefit from this since exact terminology matters more.
- **Re-ranking** — retrieve a wider candidate set (say top 10-20 chunks)
  with the cheap method, then run a cross-encoder re-ranker over just
  those candidates to actually score query-chunk relevance properly
  before picking the final top-k to send to the LLM. This is usually the
  single biggest lever for precision once retrieval starts pulling in
  plausible-but-wrong chunks, which happens a lot more on long,
  similar-sounding documents.
- **Better refusal calibration** — the fixed similarity threshold I'm
  using works for this small demo but isn't very robust (see the
  pizza-question example above). A re-ranker's relevance score is a much
  more reliable signal to threshold on than raw retrieval similarity, so
  I'd move the "no relevant context" decision to after re-ranking rather
  than before it.
