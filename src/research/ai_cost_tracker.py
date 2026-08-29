"""Tracks real Anthropic API usage for the dashboard's live AI-cost widget
(2026-08-26, owner request after a real $18/day cost shock: "also dont forget the
visual ai cost on each server... add a estimate daily amount too"). Records every
real messages.create() call's actual token usage -- both the direct sequential
path and the Batch API path -- and exposes a running today-so-far estimate.

Purely additive, observational instrumentation -- never influences any trading
decision. Every write is wrapped so a tracking failure can never break a real
Claude call or crash a caller; see AICostTracker.record()'s own try/except.

Pricing verified 2026-08-26 against Anthropic's own published API pricing:
Claude Haiku 4.5 is $1/M input tokens, $5/M output tokens at the standard
(non-batch) rate; the Batch API is a documented flat 50% discount on both. This
is an ESTIMATE, not Anthropic's real billing figure -- there is no API access to
actual billing from this codebase -- and does not account for prompt-cache
discounts. A model name not found in PRICING_PER_MILLION falls back to a
conservative mid-tier estimate rather than silently reporting $0, since
under-reporting is the worse failure mode for a cost-visibility feature."""
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PRICING_PER_MILLION = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
}
_DEFAULT_PRICING = (3.00, 15.00)  # conservative mid-tier fallback for an unrecognized model
BATCH_DISCOUNT = 0.5


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int,
                       is_batch: bool = False) -> float:
    """Pure cost-estimate function -- list-price $/M-token math, halved for the
    documented flat Batch API discount. Returns 0.0 for negative/malformed token
    counts rather than raising."""
    if input_tokens < 0 or output_tokens < 0:
        return 0.0
    in_price, out_price = PRICING_PER_MILLION.get(model, _DEFAULT_PRICING)
    cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    return cost * BATCH_DISCOUNT if is_batch else cost


class AICostTracker:
    """In-memory running total for TODAY (real calendar day in the configured
    timezone), persisted to a small JSON file so a restart mid-day doesn't reset
    the visible count to zero -- same "small cache file next to the DB"
    precedent this codebase already uses for report/on-deck caches.

    Also maintains a separate, append-only per-day history file (2026-08-27,
    owner request: "click AI-cost badge -> day-by-day cost history popup") --
    each day's totals are settled into it the moment that day ends (a real
    rollover while running, or a stale cached day discovered at startup after a
    restart landed on a new day before rollover ever fired). `summary()` remains
    the live, still-accumulating view of TODAY only; `history()` is every
    already-settled day, most-recent-first. No retention cap, matching this
    codebase's own precedent for Day/Week P/L history (`_daily_pnl_buckets`/
    `_weekly_pnl_buckets`): "reads the full, unfiltered list with no recency
    window." """

    def __init__(self, cache_path: str = "data/ai_cost_today.json",
                 history_path: str = "data/ai_cost_history.json",
                 tz: str = "America/New_York"):
        self._cache_path = Path(cache_path)
        self._history_path = Path(history_path)
        self._tz = ZoneInfo(tz)
        self._date_str = ""
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._batch_calls = 0
        self._batch_input_tokens = 0
        self._batch_output_tokens = 0
        self._load()

    def _today_str(self) -> str:
        return datetime.now(self._tz).strftime("%Y-%m-%d")

    def _roll_if_new_day(self) -> None:
        today = self._today_str()
        if today != self._date_str:
            self._append_history_row(
                self._date_str, self._calls, self._input_tokens, self._output_tokens,
                self._batch_calls, self._batch_input_tokens, self._batch_output_tokens,
            )
            self._date_str = today
            self._calls = 0
            self._input_tokens = 0
            self._output_tokens = 0
            self._batch_calls = 0
            self._batch_input_tokens = 0
            self._batch_output_tokens = 0

    def _append_history_row(self, date_str: str, calls: int, input_tokens: int,
                             output_tokens: int, batch_calls: int,
                             batch_input_tokens: int, batch_output_tokens: int) -> None:
        """Settles one finished day's raw totals into the history file. Raw
        token/call counts are stored, not a precomputed dollar estimate, so a
        later pricing correction in PRICING_PER_MILLION naturally applies to
        historical rows too when re-read via history() -- never re-derived from
        stale $ figures. Upserts by date (removes any existing row for the same
        date before appending) as an idempotency guard against a rare
        double-finalize race (e.g. two near-simultaneous restarts both finding
        the same stale cached day); skips entirely if nothing real happened that
        day, matching this codebase's own precedent of not padding a P/L history
        list with empty non-trading days."""
        if calls == 0 and batch_calls == 0:
            return
        try:
            rows = []
            if self._history_path.exists():
                rows = json.loads(self._history_path.read_text(encoding="utf-8"))
            rows = [r for r in rows if r.get("date") != date_str]
            rows.append({
                "date": date_str, "calls": calls, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "batch_calls": batch_calls,
                "batch_input_tokens": batch_input_tokens,
                "batch_output_tokens": batch_output_tokens,
            })
            rows.sort(key=lambda r: r["date"])
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            self._history_path.write_text(json.dumps(rows), encoding="utf-8")
        except Exception as e:
            logger.debug("AICostTracker._append_history_row failed (non-fatal): %s", e)

    def history(self, model_for_estimate: str = "claude-haiku-4-5") -> list[dict]:
        """Every already-settled day's totals plus an estimated dollar cost,
        most-recent-first. Today itself is never included here (it hasn't
        settled yet) -- summary() is the live view for today."""
        try:
            if not self._history_path.exists():
                return []
            rows = json.loads(self._history_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("AICostTracker.history failed (non-fatal): %s", e)
            return []
        result = []
        for r in rows:
            input_tokens = r.get("input_tokens", 0)
            output_tokens = r.get("output_tokens", 0)
            batch_input_tokens = r.get("batch_input_tokens", 0)
            batch_output_tokens = r.get("batch_output_tokens", 0)
            cost = (estimate_cost_usd(model_for_estimate, input_tokens, output_tokens,
                                       is_batch=False)
                    + estimate_cost_usd(model_for_estimate, batch_input_tokens,
                                         batch_output_tokens, is_batch=True))
            result.append({
                "date": r.get("date"),
                "calls": r.get("calls", 0) + r.get("batch_calls", 0),
                "sequential_calls": r.get("calls", 0),
                "batch_calls": r.get("batch_calls", 0),
                "input_tokens": input_tokens + batch_input_tokens,
                "output_tokens": output_tokens + batch_output_tokens,
                "estimated_cost_usd": round(cost, 4),
            })
        result.sort(key=lambda r: r["date"], reverse=True)
        return result

    def record(self, model: str, input_tokens: int, output_tokens: int,
               is_batch: bool = False) -> None:
        """Records one real call's actual usage. Never raises -- a tracking bug
        must never be able to interrupt or fail a real Claude call/response
        already in hand."""
        try:
            self._roll_if_new_day()
            input_tokens = max(0, int(input_tokens or 0))
            output_tokens = max(0, int(output_tokens or 0))
            if is_batch:
                self._batch_calls += 1
                self._batch_input_tokens += input_tokens
                self._batch_output_tokens += output_tokens
            else:
                self._calls += 1
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens
            self._save()
        except Exception as e:
            logger.debug("AICostTracker.record failed (non-fatal): %s", e)

    def summary(self, model_for_estimate: str = "claude-haiku-4-5") -> dict:
        """Today's running totals plus an estimated dollar cost. model_for_estimate
        is used for BOTH the sequential and batch token pools -- accurate for this
        codebase's real configuration (every research.model_* dial confirmed set to
        the same claude-haiku-4-5 as of 2026-08-26), and a reasonable single-number
        estimate even if a caller later mixes models, since per-model tracking
        would require carrying a model breakdown through record() that nothing
        currently needs."""
        self._roll_if_new_day()
        cost = (estimate_cost_usd(model_for_estimate, self._input_tokens,
                                   self._output_tokens, is_batch=False)
                + estimate_cost_usd(model_for_estimate, self._batch_input_tokens,
                                     self._batch_output_tokens, is_batch=True))
        return {
            "date": self._date_str,
            "calls": self._calls + self._batch_calls,
            "sequential_calls": self._calls,
            "batch_calls": self._batch_calls,
            "input_tokens": self._input_tokens + self._batch_input_tokens,
            "output_tokens": self._output_tokens + self._batch_output_tokens,
            "estimated_cost_usd": round(cost, 4),
        }

    def _save(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps({
                "date": self._date_str,
                "calls": self._calls,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "batch_calls": self._batch_calls,
                "batch_input_tokens": self._batch_input_tokens,
                "batch_output_tokens": self._batch_output_tokens,
            }), encoding="utf-8")
        except Exception as e:
            logger.debug("AICostTracker._save failed (non-fatal): %s", e)

    def _load(self) -> None:
        self._date_str = self._today_str()
        try:
            if self._cache_path.exists():
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                cached_date = data.get("date")
                if cached_date == self._date_str:
                    self._calls = data.get("calls", 0)
                    self._input_tokens = data.get("input_tokens", 0)
                    self._output_tokens = data.get("output_tokens", 0)
                    self._batch_calls = data.get("batch_calls", 0)
                    self._batch_input_tokens = data.get("batch_input_tokens", 0)
                    self._batch_output_tokens = data.get("batch_output_tokens", 0)
                elif cached_date:
                    # A restart landed on a new day before the prior day's totals
                    # were ever rolled over (_roll_if_new_day never got a chance to
                    # fire) -- settle that stale day into history now rather than
                    # silently discarding real recorded usage.
                    self._append_history_row(
                        cached_date, data.get("calls", 0), data.get("input_tokens", 0),
                        data.get("output_tokens", 0), data.get("batch_calls", 0),
                        data.get("batch_input_tokens", 0), data.get("batch_output_tokens", 0),
                    )
        except Exception as e:
            logger.debug("AICostTracker._load failed (non-fatal): %s", e)


class _CostTrackingMessages:
    """Thin proxy around the real Anthropic client's .messages so every actual
    messages.create() call (the direct/sequential path) is recorded for the
    dashboard's AI-cost widget, without touching any of engine.py's ~10
    individual call sites. Delegates every other attribute straight through to
    the real .messages object unchanged."""

    def __init__(self, real_messages, tracker: AICostTracker):
        self._real = real_messages
        self._tracker = tracker

    def create(self, *args, **kwargs):
        response = self._real.create(*args, **kwargs)
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._tracker.record(
                    kwargs.get("model", "unknown"),
                    getattr(usage, "input_tokens", 0),
                    getattr(usage, "output_tokens", 0),
                    is_batch=False,
                )
        except Exception as e:
            logger.debug("AI cost tracking failed for a messages.create() call "
                         "(non-fatal, response already returned): %s", e)
        return response

    def __getattr__(self, name):
        return getattr(self._real, name)


class CostTrackingClient:
    """Wraps a real anthropic.Anthropic client so every real messages.create()
    call is tracked, with zero changes needed at any of this codebase's many
    call sites. Only .messages.create is intercepted -- .beta (the Batch API:
    .beta.messages.batches.create/.retrieve/.cancel/.results) and every other
    attribute pass straight through untouched via __getattr__, since Batch API
    usage is tracked separately (see fetch_batch_results in engine.py, where
    each succeeded result's own Message object already carries real usage)."""

    def __init__(self, real_client, tracker: AICostTracker):
        self._real = real_client
        self.messages = _CostTrackingMessages(real_client.messages, tracker)

    def __getattr__(self, name):
        return getattr(self._real, name)
