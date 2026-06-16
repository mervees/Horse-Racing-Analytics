"""
A lightweight race assistant.

Combines the three LLM building blocks into one object you can ask questions of:

* the **priced predictions** table (model prob, edge, value, confidence per runner),
* the **retrieval index** (historical race cards) for RAG,
* the **LLM client** for fluent answers (optional).

``RaceAssistant.ask`` does very simple intent routing -- value picks, race
summary, horse comparison, or a general RAG lookup -- and always degrades to a
useful structured answer when no API key is configured. This is intentionally
rule-routed rather than an agent loop: predictable, cheap, and easy to audit.
"""
from __future__ import annotations

import re

import pandas as pd

from ..data import schema
from . import explain
from .rag import RetrievalIndex, rag_answer


class RaceAssistant:
    def __init__(
        self,
        priced: pd.DataFrame,
        index: RetrievalIndex | None = None,
        client: explain.LLMClient | None = None,
    ):
        self.priced = priced
        self.index = index
        self.client = client or explain.LLMClient()

    # ------------------------------------------------------------ helpers
    def _race(self, race_id: int) -> pd.DataFrame:
        return self.priced[self.priced[schema.RACE_KEY] == race_id]

    def value_picks(self, min_edge: float = 0.05, top: int = 10) -> pd.DataFrame:
        cols = [schema.RACE_KEY, "horse_id", "win_odds", "model_prob",
                "fair_market_prob", "edge", "confidence"]
        cols = [c for c in cols if c in self.priced.columns]
        return (
            self.priced[self.priced["edge"] >= min_edge]
            .sort_values("edge", ascending=False)[cols]
            .head(top)
        )

    # ------------------------------------------------------------ ask
    def ask(self, question: str) -> str:
        q = question.lower()
        race_ids = [int(x) for x in re.findall(r"race\s+(\d+)", q)]
        horse_ids = [int(x) for x in re.findall(r"horse\s+(\d+)", q)]

        # Compare specific horses in a race.
        if race_ids and len(horse_ids) >= 2:
            return explain.compare_horses(self._race(race_ids[0]), horse_ids, self.client)

        # Summarise a specific race.
        if race_ids and ("summary" in q or "summarise" in q or "summarize" in q
                         or "preview" in q or "brief" in q):
            r = self._race(race_ids[0])
            meta = {
                "race_id": race_ids[0],
                "venue": r["venue"].iloc[0] if "venue" in r else "?",
                "distance": r["distance"].iloc[0] if "distance" in r else "?",
                "going": r["going"].iloc[0] if "going" in r else "?",
                "field_size": len(r),
            }
            return explain.summarize_race(r, meta, self.client)

        # Value / overlay questions.
        if any(w in q for w in ("value", "overlay", "edge", "bet", "pick")):
            picks = self.value_picks()
            if picks.empty:
                return "No runners currently clear the value threshold."
            header = "Top value selections (model prob exceeds fair market prob):"
            return header + "\n" + picks.to_string(index=False)

        # Fall back to RAG over historical cards.
        if self.index is not None:
            return rag_answer(question, self.index, self.client)
        return ("I can answer questions about specific races (e.g. 'summarise race 12'), "
                "compare horses ('compare horse 3 and horse 7 in race 12'), or list "
                "value picks. Build a retrieval index for open-ended history questions.")
