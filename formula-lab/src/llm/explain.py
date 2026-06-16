"""
AI explanation features.

Turns the numbers (model probability, value/edge, rule-score contributions) into
plain-language race summaries, horse comparisons, and per-pick reasoning.

Two modes, chosen automatically:

* **LLM mode** -- if ``ANTHROPIC_API_KEY`` is set and the ``anthropic`` SDK is
  installed, the structured facts are sent to Claude with a strict
  "explain ONLY these numbers, invent nothing" system prompt. This is
  retrieval-grounded generation: the model never sees free text it could
  hallucinate from, only the computed fields.
* **Template mode** -- otherwise, a deterministic template produces the same
  information without any network call. Always available; great for tests/CI.

Grounding the model on computed fields (rather than asking it to "predict the
race") is the key safety property: the LLM is a *narrator*, not a tipster.
"""
from __future__ import annotations

import os
import textwrap

import pandas as pd

from ..models.scoring import top_signals_for_runner

DEFAULT_MODEL = "claude-sonnet-4-6"

_SIGNAL_LABELS = {
    "horse_win_rate": "career strike-rate",
    "horse_place_rate": "place consistency",
    "horse_form_win_3": "recent winning form",
    "horse_form_pos_3": "recent finishing positions",
    "rating_vs_field": "official rating vs the field",
    "jockey_win_rate": "jockey strike-rate",
    "jockey_form_win_20": "jockey recent form",
    "trainer_win_rate": "trainer strike-rate",
    "weight_vs_field": "weight carried vs rivals",
    "draw_norm": "barrier draw",
    "horse_days_since": "days since last run",
}


# --------------------------------------------------------------------------- #
# LLM client (graceful, optional)
# --------------------------------------------------------------------------- #
class LLMClient:
    """Thin wrapper over the Anthropic Messages API with an offline flag."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None
        self.available = False
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=key)
                self.available = True
            except Exception:
                self.available = False

    def generate(self, system: str, prompt: str, max_tokens: int = 700) -> str | None:
        if not self.available:
            return None
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        except Exception as exc:  # pragma: no cover
            return f"[LLM error: {exc}]"


_SYSTEM = textwrap.dedent("""
    You are a horse-racing analyst writing concise, neutral briefings. You will be
    given ONLY structured numbers computed by a model. Rules:
    - Explain strictly what the numbers say. Do NOT invent form, names, injuries,
      track bias, or any fact not present in the data.
    - Never imply a guaranteed outcome. Racing is high-variance.
    - Distinguish clearly between the MODEL's probability and the MARKET's price.
    - Flag value (model prob > market prob) and risk signals when present.
    - Keep it tight and readable. No betting advice or stake recommendations.
""").strip()


# --------------------------------------------------------------------------- #
# Fact extraction (the grounding payload)
# --------------------------------------------------------------------------- #
def _runner_facts(row: pd.Series) -> dict:
    facts = {
        "horse_id": int(row.get("horse_id", -1)),
        "draw": _maybe_int(row.get("draw")),
        "model_prob": _round(row.get("model_prob")),
        "market_implied_prob": _round(row.get("implied_prob")),
        "fair_market_prob": _round(row.get("fair_market_prob")),
        "win_odds": _round(row.get("win_odds")),
        "edge": _round(row.get("edge")),
        "confidence": _round(row.get("confidence"), 1),
        "is_value": bool(row.get("is_value", False)),
    }
    for flag in ("thin_edge", "longshot", "low_history",
                 "model_market_divergence", "large_field"):
        if flag in row and bool(row[flag]):
            facts.setdefault("risk_flags", []).append(flag)
    return facts


def _round(v, nd: int = 3):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _maybe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Public explanation functions
# --------------------------------------------------------------------------- #
def explain_prediction(row: pd.Series, client: LLMClient | None = None) -> str:
    """Reasoning for a single runner's prediction."""
    facts = _runner_facts(row)
    signals = top_signals_for_runner(row) if any(
        str(k).startswith("contrib_") for k in row.index
    ) else []
    sig_text = ", ".join(
        f"{_SIGNAL_LABELS.get(name, name)} ({'+' if val >= 0 else ''}{val:.2f})"
        for name, val in signals
    )

    if client and client.available:
        prompt = (
            f"Per-runner facts: {facts}\n"
            f"Top scoring signals (positive helps, negative hurts): {sig_text or 'n/a'}\n"
            "Write 2-3 sentences explaining this runner's prediction."
        )
        out = client.generate(_SYSTEM, prompt, max_tokens=250)
        if out:
            return out

    # ---- template fallback ----
    parts = [f"Horse {facts['horse_id']} (draw {facts['draw']})."]
    if facts["model_prob"] is not None:
        parts.append(
            f"Model gives a {facts['model_prob']*100:.1f}% win chance vs a market "
            f"implied {(facts['fair_market_prob'] or 0)*100:.1f}% (price {facts['win_odds']})."
        )
    if facts["is_value"]:
        parts.append(f"This reads as value: edge {facts['edge']:+.1%}.")
    else:
        parts.append(f"No value at this price (edge {facts['edge']:+.1%}).")
    if sig_text:
        parts.append(f"Main drivers: {sig_text}.")
    if facts.get("risk_flags"):
        parts.append("Risk flags: " + ", ".join(facts["risk_flags"]) + ".")
    return " ".join(parts)


def summarize_race(priced_race: pd.DataFrame, race_meta: dict,
                   client: LLMClient | None = None) -> str:
    """A short briefing for one race."""
    top = priced_race.sort_values("model_prob", ascending=False).head(4)
    runners = [_runner_facts(r) for _, r in top.iterrows()]
    value_picks = priced_race[priced_race["is_value"]].sort_values(
        "edge", ascending=False
    )
    value_facts = [_runner_facts(r) for _, r in value_picks.head(3).iterrows()]

    if client and client.available:
        prompt = (
            f"Race meta: {race_meta}\n"
            f"Top runners by model probability: {runners}\n"
            f"Value selections (edge>0): {value_facts}\n"
            "Write a 4-6 sentence race briefing."
        )
        out = client.generate(_SYSTEM, prompt, max_tokens=500)
        if out:
            return out

    # ---- template fallback ----
    lines = [
        f"Race {race_meta.get('race_id','?')} at {race_meta.get('venue','?')}, "
        f"{race_meta.get('distance','?')}m, going {race_meta.get('going','?')}, "
        f"{int(race_meta.get('field_size', len(priced_race)))} runners."
    ]
    lead = runners[0]
    lines.append(
        f"Model favourite is horse {lead['horse_id']} at {(lead['model_prob'] or 0)*100:.1f}% "
        f"(market price {lead['win_odds']})."
    )
    if value_facts:
        vs = ", ".join(
            f"#{v['horse_id']} (edge {v['edge']:+.1%}, {v['win_odds']})" for v in value_facts
        )
        lines.append(f"Value angles: {vs}.")
    else:
        lines.append("No clear overlays — the model broadly agrees with the market here.")
    lines.append("All figures are model estimates; outcomes are high-variance.")
    return " ".join(lines)


def compare_horses(priced_race: pd.DataFrame, horse_ids: list[int],
                   client: LLMClient | None = None) -> str:
    """Head-to-head comparison of specific horses in a race."""
    sub = priced_race[priced_race["horse_id"].isin(horse_ids)]
    facts = [_runner_facts(r) for _, r in sub.iterrows()]

    if client and client.available:
        prompt = (
            f"Compare these runners head-to-head using only these facts: {facts}\n"
            "Write 3-4 sentences contrasting their model chances, prices and value."
        )
        out = client.generate(_SYSTEM, prompt, max_tokens=400)
        if out:
            return out

    # ---- template fallback ----
    facts.sort(key=lambda f: (f["model_prob"] or 0), reverse=True)
    bits = [
        f"#{f['horse_id']}: model {(f['model_prob'] or 0)*100:.1f}%, price {f['win_odds']}, "
        f"edge {f['edge']:+.1%}{' (value)' if f['is_value'] else ''}"
        for f in facts
    ]
    return "Head-to-head — " + "; ".join(bits) + "."
