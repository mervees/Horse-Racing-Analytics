# LLM Features

The `src/llm/` package adds natural-language explanations, retrieval, and a
race assistant **on top of the numbers** — it never invents facts. Every LLM
feature has a deterministic fallback, so the whole package works with no API
key (it just produces template text and uses local TF-IDF retrieval).

## Design rule: narrate, don't predict

The system prompt (`explain._SYSTEM`) is strict: the model may only describe
numbers that were already computed (model probability, market-implied
probability, edge, top rule signals, risk flags). It must not invent form,
must not promise outcomes, must distinguish model probability from market
probability, and must not give staking advice. Facts are extracted from the
priced race (`_runner_facts`) and passed in, so the model is *grounding on
data*, not recalling training trivia.

## Explanation features (`src/llm/explain.py`)

- **`explain_prediction(row)`** — why the model rates one horse as it does:
  model vs. market probability, the edge, and the few signals that moved its
  score most.
- **`summarize_race(priced_race, race_meta)`** — a short race preview: the
  model favourite, the main value angles, and a variance caveat.
- **`compare_horses(priced_race, horse_ids)`** — a head-to-head on probability,
  price, edge, and key signals.

Each runs through the LLM if `ANTHROPIC_API_KEY` is set (model
`claude-sonnet-4-6`), otherwise returns a clean deterministic template built
from the same facts.

## Retrieval-augmented generation (`src/llm/rag.py`)

- **`build_race_documents(merged, max_races)`** turns each race into a compact
  text "card" (runners, draws, ratings, odds, result context).
- **Embedders**: `TfidfEmbedder` (local, sklearn, no API — the default) or
  `DenseEmbedder` (an adapter that wraps any real embedding function; the
  docstring shows a Voyage example).
- **`RetrievalIndex`** does cosine-similarity retrieval over the cards.
- **`rag_answer(question, index, client, k)`** retrieves the top-k cards and
  has the LLM answer **grounded strictly in that retrieved context**; offline,
  it returns the retrieved cards directly.

## Assistant (`src/llm/assistant.py`)

`RaceAssistant(priced, index, client).ask(question)` is a **rule-routed**
assistant (deterministic routing, not an open agent loop): it detects race and
horse IDs in the question and routes to `compare_horses`, `summarize_race`,
`value_picks`, or a RAG fallback. `value_picks(min_edge, top)` lists the
highest-edge runners. With no API key it still answers with structured,
data-grounded text.

## Enabling the real LLM

```bash
pip install anthropic          # optional
export ANTHROPIC_API_KEY=...    # your key
```

For dense embeddings, install your provider's SDK and pass an `embed_fn` to
`DenseEmbedder` (see its docstring). Nothing else changes — the rest of the
pipeline is identical whether the LLM is on or off.

## Why this is safe

Because the language layer only ever sees and repeats **computed** quantities,
it cannot fabricate a horse's record or a probability. The worst case if the
API is unavailable is plainer wording, not wrong numbers.
