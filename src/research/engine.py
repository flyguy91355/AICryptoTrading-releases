"""Core crypto research engine powered by Claude AI.

Adapted from AITrading's ResearchEngine. The Claude-call machinery
(_call_claude_with_retry, the temperature-deprecation retry, the AI Data
Integrity fallback rules) carries over unchanged -- none of that is
asset-class-specific. ANALYSIS_PROMPT itself is written fresh for crypto:
no earnings/SEC-filings/insider-activity sections (none of that exists for
crypto), leaning instead on price action/technicals and news/sentiment.
Deep Dive/Deeper Dive tiers, the Anthropic Batch API path, and the
recommend_dip_entry-style AI judgment calls are all deferred past Phase 1
per the design spec -- this engine has exactly one research tier.
"""

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic

from src.data.market_data import MarketDataFetcher, format_long_term_trend_summary
from src.data.news_feed import NewsFeed
from src.research.asset_profile import build_asset_profile_section
from src.research.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


class Signal(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"
    NO_ACTION = "NO ACTION"


class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


def _fmt_price(price: float) -> str:
    """Format a price for prompt/display regardless of magnitude.
    Normal prices (>= $0.01): 2 decimal places. Sub-cent prices (SHIB, PEPE, etc.):
    enough decimals to show 4 significant figures — prevents $0.000004470 from
    being displayed as '$0.00' to Claude, which caused 'insufficient data' responses."""
    if price == 0.0:
        return "0.00"
    if abs(price) >= 0.01:
        return f"{price:.2f}"
    decimals = max(2, -math.floor(math.log10(abs(price))) + 3)
    return f"{price:.{decimals}f}"


def _ticker_to_custom_id(ticker: str) -> str:
    """Anthropic's Batch API requires custom_id to match ^[a-zA-Z0-9_-]{1,64}$ --
    confirmed live 2026-08-16 as a real 400 (AITrading's own stock-ticker version of
    this code never hits this, since stock tickers have no slash). Every crypto
    ticker in this project's universe is "<COIN>/USD", so replacing the one "/" is
    sufficient and collision-free; fetch_batch_results reverses this by re-deriving
    it from each candidate real ticker, not by trying to parse the id backwards."""
    return ticker.replace("/", "-")


def _clamp_ai_stop_loss(raw_stop: float, entry_price: float, min_pct: float, max_pct: float) -> float:
    """Bounds an AI-chosen stop-loss into [entry*(1 - max_pct/100), entry*(1 - min_pct/100)].
    Only applied when research.ai_chosen_stop_tp_enabled is true."""
    if entry_price <= 0:
        return raw_stop
    floor = entry_price * (1 - max_pct / 100)
    ceiling = entry_price * (1 - min_pct / 100)
    return max(floor, min(raw_stop, ceiling))


def _clamp_ai_final_trail_pct(raw_pct: float, min_pct: float, max_pct: float) -> float:
    """Bounds an AI-chosen final-tranche trailing-stop width (the tightest the
    graduated trail ever gets, once price has run deep into profit) into [min_pct,
    max_pct]. Only applied when research.ai_chosen_stop_tp_enabled is true -- same
    guardrail role _clamp_ai_stop_loss plays for the initial stop."""
    return max(min_pct, min(raw_pct, max_pct))


@dataclass
class ResearchReport:
    ticker: str
    asset_name: str
    generated_at: datetime
    conviction_score: float
    signal: Signal
    risk_level: RiskLevel
    thesis: str
    technical_summary: str
    news_summary: str
    risk_factors: str
    recommended_action: str
    entry_price: float
    position_size_pct: float
    stop_loss: float
    take_profit_targets: list[float] = field(default_factory=list)
    time_horizon: str = ""
    reasoning: str = ""
    is_fallback: bool = False
    fair_value_estimate: float = 0.0
    margin_of_safety_pct: float = 0.0
    asset_summary: str = ""
    final_trail_pct: float | None = None


ANALYSIS_PROMPT = """\
You are a senior crypto research analyst. Analyze the following crypto asset data and produce a structured trading recommendation.

TODAY'S DATE: {current_date} — any event or timeframe you reference must be realistic relative to this date. Never reference a date that has already passed as if it were upcoming.

IMPORTANT RULES:
- Be conservative. The cardinal rule is NEVER LOSE MONEY.
- Only recommend BUY or STRONG BUY if conviction is {min_conviction:.1f}/10 or higher.
- Every recommendation MUST include a stop loss.
- Risk/reward ratio must be at least {rr_base:.1f}:1 for an asset right at the {min_conviction:.1f}/10 conviction minimum, scaling down modestly for higher conviction — never below {rr_floor:.1f}:1 regardless of how high conviction is.
- Crypto trades 24/7 and is meaningfully more volatile than equities — weigh that directly: a technically-driven move here can be pure volatility rather than a genuine trend change, and a stop/target that would be reasonable for a stock may be too tight here.
- If the data is insufficient or unclear, recommend NO ACTION.
- Every numeric field you output (entry_price, stop_loss, take_profit_targets) must be used consistently everywhere else in your response, including inside reasoning — if you cite risk/reward math or a specific target price in reasoning, it must match the actual entry_price/stop_loss/take_profit_targets values you output, and "first target" must mean take_profit_targets[0], not a later one.
- There are no earnings reports, SEC filings, or insider-trading data for crypto — do not invent or assume any. Base your analysis only on the price action, technicals, and news/sentiment given below.
- If a LONG-TERM TREND section is present below, weigh it explicitly: judge whether the current setup looks like a genuine breakout versus a bounce back toward a level this asset has already failed at before.
- If a HISTORICAL CHARACTER section is present below, weigh it too: it's a real backtest of THIS asset's own past breakout reliability and drawdown behavior, not generic crypto advice — use it to calibrate how much weight a similar-looking setup deserves for this specific asset, not as a mechanical rule.

ASSET: {ticker} — {asset_name}
CURRENT PRICE: ${current_price}

── TECHNICAL CONTEXT ──
{technical_summary}

── NEWS & SENTIMENT ──
{news_summary}
{market_context_section}{volatility_section}{long_term_trend_section}{asset_profile_section}{trade_history_section}
Based on all of the above, provide your analysis as JSON with these exact fields:
{{
    "conviction_score": <1-10, one decimal place, e.g. 7.3>,
    "signal": "<STRONG BUY|BUY|HOLD|SELL|STRONG SELL|NO ACTION>",
    "risk_level": "<LOW|MODERATE|HIGH>",
    "asset_summary": "<1-2 factual sentences on what this asset/protocol actually is or does — not an investment opinion, just what it is>",
    "thesis": "<2-3 sentence trading thesis>",
    "risk_factors": "<key risks that could invalidate the thesis>",
    "recommended_action": "<specific action to take>",
    "entry_price": <recommended entry price as number>,
    {stop_tp_instructions}
    "position_size_pct": <recommended position size as % of portfolio, 1-10>,
    "time_horizon": "<hours|days|weeks>",
    "reasoning": "<detailed reasoning connecting technicals and news/sentiment>",
    "fair_value_estimate": <quick estimate of a fair near-term price as a number>,
    "margin_of_safety_pct": <percentage below fair value the current price represents — negative if overvalued>
}}

Respond with ONLY the JSON object, no other text.
"""


def _build_trade_history_section(trade_history_summary: str) -> str:
    if not trade_history_summary:
        return ""
    return (
        "\n── PRIOR TRADING HISTORY (for reference only — weigh current technicals "
        "and news as the primary basis for your decision) ──\n"
        f"{trade_history_summary}\n"
    )


def _build_long_term_trend_section(long_term_trend_summary: str) -> str:
    if not long_term_trend_summary:
        return ""
    return f"\n── LONG-TERM TREND ──\n{long_term_trend_summary}\n"


def _volatility_tier_label(realized_vol_30d: float) -> tuple[str, str]:
    """Maps annualized 30-day realized volatility to a human-readable tier + description.
    Thresholds calibrated for crypto: BTC/ETH sit ~50-80% annualized; mid-caps 80-150%;
    small-caps/meme coins 150%+. Returns (tier_name, one-line description)."""
    if realized_vol_30d <= 0:
        return ("UNKNOWN", "volatility data unavailable")
    if realized_vol_30d < 0.60:
        return ("LOW", f"{realized_vol_30d*100:.0f}% annualized — behaves like a large-cap stablecoin-adjacent asset; tighter stops are reasonable")
    if realized_vol_30d < 1.00:
        return ("MODERATE", f"{realized_vol_30d*100:.0f}% annualized — typical for BTC/ETH class; standard crypto stop widths apply")
    if realized_vol_30d < 1.60:
        return ("HIGH", f"{realized_vol_30d*100:.0f}% annualized — mid-cap or sector coin; stops need meaningful room to avoid routine noise shakeouts")
    return ("EXTREME", f"{realized_vol_30d*100:.0f}% annualized — small-cap or meme coin territory; a tight stop WILL get hit by normal daily swings, not a real thesis break")


def _build_volatility_section(realized_vol_30d: float) -> str:
    """Prompt section telling Claude this coin's volatility tier so it calibrates
    stop width and trailing-stop width accordingly — not a mechanical rule, just
    real data + framing for the model to reason from."""
    tier, desc = _volatility_tier_label(realized_vol_30d)
    if tier == "UNKNOWN":
        return ""
    return (
        f"\n── VOLATILITY TIER ──\n"
        f"{tier} — {desc}.\n"
        f"Calibrate stop-loss distance and trailing-stop width to this tier: a {tier.lower()}-volatility "
        f"asset needs proportionally {'more' if tier in ('HIGH','EXTREME') else 'standard'} room "
        f"between entry and stop so that normal price noise doesn't trigger a stop that was "
        f"placed for a real thesis break, not just routine chop.\n"
    )


def _build_market_context_section(market_change_pct: float | None) -> str:
    """Today's broad crypto-market direction (BTC as bellwether). Returns "" when the
    fetch failed, same graceful-omission pattern every other optional section here uses."""
    if market_change_pct is None:
        return ""
    direction = "up" if market_change_pct >= 0 else "down"
    return (
        f"\n── TODAY'S CRYPTO MARKET ──\n"
        f"BTC/USD: {market_change_pct:+.2f}% today — the broad crypto market is {direction} "
        f"today. Weigh this as a real factor: an asset's own price strength on a day BTC is "
        f"also strongly up may partly reflect market-wide momentum rather than something "
        f"unique to this asset — but a broadly up market is generally a favorable "
        f"environment for taking new positions, so don't be reflexively conservative if "
        f"the technicals already support a buy. On a broadly down market, be more "
        f"selective and cautious, since risk is elevated market-wide.\n"
    )


class ResearchEngine:
    def __init__(self, config: dict, market_data: MarketDataFetcher, news_feed: NewsFeed):
        self.config = config
        self.market_data = market_data
        self.news_feed = news_feed
        self.reports: dict[str, ResearchReport] = {}

        self.sentiment_analyzer = SentimentAnalyzer(config)

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None

        self.reports_dir = Path("data/research_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        research_cfg = config.get("research", {})
        self._market_tz = ZoneInfo(research_cfg.get("market_timezone", "America/New_York"))

        self.asset_profiles_path = Path("data/asset_profiles.json")
        self.asset_profiles: dict[str, dict] = {}
        self.reload_asset_profiles()

    def reload_asset_profiles(self):
        """Re-reads data/asset_profiles.json from disk -- called at startup and after
        scripts/build_asset_profiles.py regenerates it (either a manual run or the
        in-app weekly scheduled refresh in web/app.py), so a running process picks up
        fresh profiles without needing a restart. Missing/malformed file degrades to
        "no profiles yet" rather than crashing -- same AI Data Integrity precedent as
        every other optional context section in this file (a missing profile just
        omits that prompt section, it never blocks or fakes analysis)."""
        try:
            self.asset_profiles = json.loads(self.asset_profiles_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.info("No usable asset_profiles.json yet (%s) -- proceeding without it", e)
            self.asset_profiles = {}

    def _now_et(self) -> datetime:
        return datetime.now(self._market_tz)

    async def gather_analysis_inputs(self, ticker: str, asset_name: str) -> dict | None:
        """Fetches every real input ANALYSIS_PROMPT needs for one asset -- quote,
        technicals, long-term trend, news/sentiment -- and folds them into the same
        technical_summary/news_summary strings analyze_asset has always built. Shared
        by the sequential path (analyze_asset) and the batch path
        (submit_analysis_batch, added 2026-08-16 for the daily-cost reduction) so both
        build prompts from identical data -- one source of truth, same precedent as
        AITrading's own submit_analysis_batch/analyze_stock split. Returns None on any
        fetch failure rather than raising, so one bad ticker can't take down a whole
        batch submission; the caller (submit_analysis_batch) just excludes it."""
        try:
            quote = await self.market_data.get_quote(ticker)
            technicals = await self.market_data.get_technicals(ticker)
            long_term_trend = await self.market_data.get_long_term_trend(
                ticker, quote.price,
                years=self.config.get("research", {}).get("long_term_trend_years", 3))
            news_items = await self.news_feed.get_asset_news(ticker, asset_name, days=3)
            sentiment_analysis = await self.sentiment_analyzer.analyze(news_items)
        except Exception as e:
            logger.warning("gather_analysis_inputs failed for %s: %s", ticker, e)
            return None

        technical_summary = (
            f"Price: ${_fmt_price(quote.price)} | 24h change: {quote.change_pct:+.2f}% | "
            f"SMA50: ${_fmt_price(technicals.sma_50)} | SMA200: ${_fmt_price(technicals.sma_200)} | "
            f"RSI: {technicals.rsi:.1f} | Support: ${_fmt_price(technicals.support_level)} | "
            f"Resistance: ${_fmt_price(technicals.resistance_level)} | "
            f"Avg Volume 30d: {technicals.avg_volume_30d:,} | "
            f"30d Realized Vol: {technicals.realized_vol_30d*100:.0f}% annualized"
        )
        return {
            "current_price": quote.price,
            "technical_summary": technical_summary,
            "news_summary": sentiment_analysis.summary,
            "long_term_trend_summary": format_long_term_trend_summary(long_term_trend),
            "realized_vol_30d": technicals.realized_vol_30d,
            "technicals": technicals,
            "sentiment_analysis": sentiment_analysis,
        }

    async def analyze_asset(
        self, ticker: str, asset_name: str, trade_history_summary: str = "",
        model: str | None = None,
    ) -> ResearchReport:
        """ticker is this project's canonical symbol (e.g. "BTC/USD"); asset_name is the
        plain display/search name (e.g. "Bitcoin"), sourced from the asset universe
        config -- both come from the same config entry, never derived from one another."""
        logger.info("Starting research analysis for %s (%s)", ticker, asset_name)

        inp = await self.gather_analysis_inputs(ticker, asset_name)
        if inp is None:
            # Data-gathering itself failed (bad quote/technicals fetch, not a Claude
            # error) -- there's no current_price to even build a sensible fallback
            # report from, so surface this the same way a Claude failure would.
            report = self._fallback_report(
                ticker, asset_name, 0.0, "", "",
                RuntimeError("market data / news gathering failed"),
            )
            self.reports[ticker] = report
            await asyncio.to_thread(self._save_report, report)
            return report

        if self.client:
            report = await self._claude_analysis(
                ticker, asset_name, inp["current_price"], inp["technical_summary"],
                inp["news_summary"], trade_history_summary,
                inp["long_term_trend_summary"], model,
                realized_vol_30d=inp.get("realized_vol_30d", 0.0),
            )
        else:
            logger.warning("No ANTHROPIC_API_KEY — generating rule-based report for %s", ticker)
            report = self._rule_based_analysis(
                ticker, asset_name, inp["current_price"], inp["technicals"], inp["sentiment_analysis"],
            )

        self.reports[ticker] = report
        await asyncio.to_thread(self._save_report, report)
        logger.info("Research complete for %s — Signal: %s, Conviction: %d/10",
                     ticker, report.signal.value, report.conviction_score)
        return report

    async def submit_analysis_batch(self, assets: list[dict]) -> tuple[str | None, dict[str, dict]]:
        """Gathers real inputs for every asset concurrently (capped at 5 at once, same
        throttle AITrading's own submit_analysis_batch uses to avoid hammering
        yfinance/NewsAPI), builds one Batch API request per asset from the identical
        ANALYSIS_PROMPT the sequential path uses, and submits them as a single
        Anthropic Batch API call -- 50% off list price. `assets` is a list of
        {"ticker":..., "name":...} entries (the config universe shape). Returns
        (batch_id, inputs_by_ticker); inputs_by_ticker is needed by fetch_batch_results
        to reconstruct each ResearchReport, since the batch response only echoes back
        Claude's own output, not the summaries the prompt was built from. An asset
        whose data-gathering fails is silently excluded -- skipped this cycle, not
        fatal to the whole batch."""
        if not self.client:
            return None, {}

        _gather_semaphore = asyncio.Semaphore(5)

        async def _gather_safe(asset: dict) -> tuple[str, str, dict | None]:
            ticker, name = asset["ticker"], asset["name"]
            async with _gather_semaphore:
                return ticker, name, await self.gather_analysis_inputs(ticker, name)

        gathered = await asyncio.gather(*(_gather_safe(a) for a in assets))
        # Keyed by the real ticker (e.g. "BTC/USD") for the caller's convenience --
        # the Batch API's own custom_id can't carry the "/" a crypto ticker needs
        # (must match ^[a-zA-Z0-9_-]{1,64}$; confirmed live 2026-08-16, a real 400
        # AITrading's own stock-ticker version of this code never hits), so each
        # request below uses a sanitized custom_id and fetch_batch_results maps it
        # back via this same ticker_by_custom_id() function.
        inputs_by_ticker = {
            ticker: {**inp, "asset_name": name}
            for ticker, name, inp in gathered if inp is not None
        }
        if not inputs_by_ticker:
            return None, {}

        model = self.config.get("research", {}).get("model_quick_scan", "claude-haiku-4-5")
        market_change_pct = await self.market_data.get_market_change_pct()
        requests = []
        for ticker, inp in inputs_by_ticker.items():
            prompt = self._build_analysis_prompt(
                ticker, inp["asset_name"], inp["current_price"], inp["technical_summary"],
                inp["news_summary"], "", inp["long_term_trend_summary"], market_change_pct,
                realized_vol_30d=inp.get("realized_vol_30d", 0.0),
            )
            requests.append({
                "custom_id": _ticker_to_custom_id(ticker),
                "params": {
                    "model": model,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            })

        try:
            batch = await asyncio.to_thread(self.client.beta.messages.batches.create, requests=requests)
        except Exception as e:
            logger.error("Batch submission failed for %d assets: %s", len(requests), e)
            return None, {}

        return batch.id, inputs_by_ticker

    async def poll_batch_status(self, batch_id: str):
        """Returns the raw Anthropic batch status object (.processing_status,
        .request_counts.succeeded/.processing/.errored)."""
        return await asyncio.to_thread(self.client.beta.messages.batches.retrieve, batch_id)

    async def cancel_batch(self, batch_id: str) -> None:
        """Best-effort cancel for a batch the caller has given up on. Failures are
        logged and swallowed -- the caller has already moved on to the sequential
        fallback and doesn't need this to succeed to make progress."""
        try:
            await asyncio.to_thread(self.client.beta.messages.batches.cancel, batch_id)
        except Exception as e:
            logger.warning("Failed to cancel stuck batch %s: %s", batch_id, e)

    async def fetch_batch_results(
        self, batch_id: str, inputs_by_ticker: dict[str, dict],
    ) -> dict[str, ResearchReport]:
        """Retrieves and parses a completed batch's results into ResearchReport
        objects, keyed by ticker, using the same parsing/fallback construction the
        sequential path uses so results are structurally identical regardless of
        submission mechanism. Also stores into self.reports and persists to disk,
        exactly as analyze_asset() already does."""
        try:
            results = await asyncio.to_thread(
                lambda: list(self.client.beta.messages.batches.results(batch_id)))
        except Exception as e:
            logger.error("Batch result retrieval failed for batch %s: %s", batch_id, e)
            return {}

        custom_id_to_ticker = {_ticker_to_custom_id(t): t for t in inputs_by_ticker}
        reports: dict[str, ResearchReport] = {}
        for result in results:
            ticker = custom_id_to_ticker.get(result.custom_id)
            inp = inputs_by_ticker.get(ticker) if ticker else None
            if inp is None:
                continue

            if result.result.type == "succeeded":
                try:
                    text_block = next(
                        (b for b in result.result.message.content if hasattr(b, "text")), None)
                    if not text_block:
                        raise ValueError("No text block in batch response")
                    report = self._parse_analysis_response(
                        ticker, inp["asset_name"], inp["current_price"],
                        inp["technical_summary"], inp["news_summary"], text_block.text,
                    )
                except Exception as e:
                    report = self._fallback_report(
                        ticker, inp["asset_name"], inp["current_price"],
                        inp["technical_summary"], inp["news_summary"], e,
                    )
            else:
                report = self._fallback_report(
                    ticker, inp["asset_name"], inp["current_price"],
                    inp["technical_summary"], inp["news_summary"],
                    RuntimeError(f"Batch result type: {result.result.type}"),
                )

            self.reports[ticker] = report
            await asyncio.to_thread(self._save_report, report)
            logger.info("Research complete for %s — Signal: %s, Conviction: %d/10",
                         ticker, report.signal.value, report.conviction_score)
            reports[ticker] = report

        return reports

    async def _call_claude_with_retry(
        self, model: str, max_tokens: int, messages: list,
        max_retries: int = 3, base_delay: float = 2.0,
    ) -> str:
        """Call Claude API with exponential backoff retry. Ported verbatim from
        AITrading's own version -- the temperature-deprecation handling isn't
        asset-class-specific."""
        last_error: Exception | None = None
        for attempt in range(max_retries):
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning("Claude API retry %d/%d in %.0fs...", attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
            try:
                try:
                    message = await asyncio.to_thread(
                        lambda: self.client.messages.create(
                            model=model, max_tokens=max_tokens, messages=messages,
                            temperature=0.0,
                        )
                    )
                except Exception as temp_err:
                    msg = str(temp_err).lower()
                    if "temperature" in msg and "deprecat" in msg:
                        logger.info("Model %s rejects temperature param — retrying without it", model)
                        message = await asyncio.to_thread(
                            lambda: self.client.messages.create(
                                model=model, max_tokens=max_tokens, messages=messages,
                            )
                        )
                    else:
                        raise
                text_block = next((b for b in message.content if hasattr(b, "text")), None)
                if not text_block:
                    raise ValueError("No text block in Claude response")
                return text_block.text
            except Exception as e:
                last_error = e
                logger.warning("Claude API attempt %d/%d failed: %s", attempt + 1, max_retries, e)
        raise last_error  # type: ignore[misc]

    def _build_analysis_prompt(
        self, ticker: str, asset_name: str, current_price: float,
        technical_summary: str, news_summary: str,
        trade_history_summary: str = "", long_term_trend_summary: str = "",
        market_change_pct: float | None = None,
        realized_vol_30d: float = 0.0,
    ) -> str:
        asset_profile = self.asset_profiles.get(ticker, {}).get("profile")
        tp_cfg = self.config.get("take_profit", {})
        research_cfg = self.config.get("research", {})
        min_conviction = research_cfg.get("min_conviction_score", 7)
        rr_base = research_cfg.get("min_risk_reward_ratio", 2.0)
        rr_floor = research_cfg.get("rr_floor", 1.5)
        if research_cfg.get("ai_chosen_stop_tp_enabled", False):
            stop_tp_instructions = (
                '"stop_loss": <stop loss price as number — judge a specific level from '
                'the real support/resistance/SMA/RSI data above; place it just below a '
                'genuine support level or recent swing low, not an arbitrary percentage '
                'off entry>,\n'
                '    "take_profit_targets": [<T1, T2, T3 — three ascending price levels '
                'judged from real resistance levels and the fair value estimate, not a '
                'fixed percentage above entry>],\n'
                '    "final_trail_pct": <a trailing-stop width, as a percentage of price '
                '(e.g. 1.5 means 1.5%), for once this position has run deep into profit '
                'near T3 — judge this from the asset\'s own real volatility above, not '
                'one generic number for every asset: a calmer, more liquid asset can '
                'support a tighter trail without getting stopped out on routine noise, '
                'while a more volatile one needs more room>,'
            )
        else:
            sl_pct = tp_cfg.get("stop_loss_pct", 12.0)
            t1_pct = tp_cfg.get("t1_pct", 8.0)
            t2_pct = tp_cfg.get("t2_pct", 16.0)
            t3_pct = tp_cfg.get("t3_pct", 28.0)
            stop_tp_instructions = (
                f'"stop_loss": <stop loss price as number — target {sl_pct:.0f}% below entry>,\n'
                f'    "take_profit_targets": [<T1 — ~{t1_pct:.0f}% above entry>, '
                f'<T2 — ~{t2_pct:.0f}% above entry>, <T3 — ~{t3_pct:.0f}% above entry>],'
            )
        return ANALYSIS_PROMPT.format(
            current_date=self._now_et().strftime("%Y-%m-%d"),
            min_conviction=min_conviction,
            rr_base=rr_base,
            rr_floor=rr_floor,
            ticker=ticker,
            asset_name=asset_name,
            current_price=_fmt_price(current_price),
            technical_summary=technical_summary,
            news_summary=news_summary,
            trade_history_section=_build_trade_history_section(trade_history_summary),
            long_term_trend_section=_build_long_term_trend_section(long_term_trend_summary),
            asset_profile_section=build_asset_profile_section(asset_profile),
            market_context_section=_build_market_context_section(market_change_pct),
            volatility_section=_build_volatility_section(realized_vol_30d),
            stop_tp_instructions=stop_tp_instructions,
        )

    async def _claude_analysis(
        self, ticker: str, asset_name: str, current_price: float,
        technical_summary: str, news_summary: str,
        trade_history_summary: str = "", long_term_trend_summary: str = "",
        model: str | None = None, realized_vol_30d: float = 0.0,
    ) -> ResearchReport:
        market_change_pct = await self.market_data.get_market_change_pct()
        prompt = self._build_analysis_prompt(
            ticker, asset_name, current_price, technical_summary, news_summary,
            trade_history_summary, long_term_trend_summary, market_change_pct,
            realized_vol_30d=realized_vol_30d,
        )
        try:
            response_text = await self._call_claude_with_retry(
                model=model or self.config.get("research", {}).get("model_quick_scan", "claude-haiku-4-5"),
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_analysis_response(
                ticker, asset_name, current_price, technical_summary, news_summary, response_text,
            )
        except Exception as e:
            return self._fallback_report(ticker, asset_name, current_price, technical_summary, news_summary, e)

    def _parse_analysis_response(
        self, ticker: str, asset_name: str, current_price: float,
        technical_summary: str, news_summary: str, response_text: str,
    ) -> ResearchReport:
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
        response_text = response_text.rstrip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].rstrip()

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = json.loads(response_text.replace('\r\n', ' ').replace('\n', ' '))

        entry_price = float(data["entry_price"]) if data.get("entry_price") is not None else current_price
        raw_stop_loss = float(data["stop_loss"]) if data.get("stop_loss") is not None else current_price * 0.9
        research_cfg = self.config.get("research", {})
        final_trail_pct: float | None = None
        if research_cfg.get("ai_chosen_stop_tp_enabled", False):
            stop_loss = _clamp_ai_stop_loss(
                raw_stop_loss, entry_price,
                research_cfg.get("ai_stop_loss_min_pct", 4.0),
                research_cfg.get("ai_stop_loss_max_pct", 18.0),
            )
            raw_final_trail_pct = data.get("final_trail_pct")
            if raw_final_trail_pct is not None:
                final_trail_pct = _clamp_ai_final_trail_pct(
                    float(raw_final_trail_pct),
                    research_cfg.get("ai_final_trail_min_pct", 0.5),
                    research_cfg.get("ai_final_trail_max_pct", 5.0),
                )
        else:
            stop_loss = raw_stop_loss

        # NaN guard -- json.loads accepts bare NaN by Python's own non-standard extension,
        # and NaN < anything is always False, which would silently bypass the conviction
        # gate downstream (same real bug class AITrading's own #65 fix closed).
        conviction_score = round(float(data.get("conviction_score", 0)), 1)
        if math.isnan(conviction_score):
            raise ValueError(f"conviction_score parsed as NaN for {ticker} — malformed Claude response")

        return ResearchReport(
            ticker=ticker,
            asset_name=asset_name,
            generated_at=self._now_et(),
            conviction_score=conviction_score,
            signal=Signal(data.get("signal", "NO ACTION").strip().upper()),
            risk_level=RiskLevel(data.get("risk_level", "HIGH").strip().upper()),
            thesis=data.get("thesis", ""),
            asset_summary=data.get("asset_summary", ""),
            technical_summary=technical_summary,
            news_summary=news_summary,
            risk_factors=data.get("risk_factors", ""),
            recommended_action=data.get("recommended_action", ""),
            entry_price=entry_price,
            position_size_pct=float(data["position_size_pct"]) if data.get("position_size_pct") is not None else 3,
            stop_loss=stop_loss,
            take_profit_targets=[float(t) for t in (data.get("take_profit_targets") or []) if t is not None and float(t) > 0],
            time_horizon=data.get("time_horizon", ""),
            reasoning=data.get("reasoning", ""),
            fair_value_estimate=float(data["fair_value_estimate"]) if data.get("fair_value_estimate") is not None else 0.0,
            margin_of_safety_pct=float(data["margin_of_safety_pct"]) if data.get("margin_of_safety_pct") is not None else 0.0,
            final_trail_pct=final_trail_pct,
        )

    def _fallback_report(
        self, ticker: str, asset_name: str, current_price: float,
        technical_summary: str, news_summary: str, error: Exception,
    ) -> ResearchReport:
        logger.error("Claude analysis failed for %s: %s", ticker, error)
        return ResearchReport(
            ticker=ticker,
            asset_name=asset_name,
            generated_at=self._now_et(),
            conviction_score=0.0,
            signal=Signal.NO_ACTION,
            risk_level=RiskLevel.HIGH,
            thesis=f"⚠ Claude API error — analysis could not be completed: {error}",
            technical_summary=technical_summary,
            news_summary=news_summary,
            risk_factors="AI analysis failed — do not trade on this asset until re-scanned successfully.",
            recommended_action="NO ACTION — AI error",
            entry_price=current_price,
            position_size_pct=0,
            stop_loss=current_price * 0.9,
            is_fallback=True,
        )

    def _rule_based_analysis(
        self, ticker, asset_name, price, technicals, sentiment,
    ) -> ResearchReport:
        """AI Data Integrity: never generates BUY/STRONG BUY signals -- same rule as
        AITrading's own rule-based fallback. Only fires when ANTHROPIC_API_KEY is
        unset."""
        score = 5

        if sentiment.overall_sentiment >= 0.3:
            score += 1
        elif sentiment.overall_sentiment <= -0.3:
            score -= 1

        if technicals.rsi < 30:
            score += 1
        elif technicals.rsi > 70:
            score -= 1

        score = max(1, min(10, score))

        if score >= 5:
            signal = Signal.HOLD
        elif score >= 3:
            signal = Signal.SELL
        else:
            signal = Signal.STRONG_SELL

        tp_cfg = self.config.get("take_profit", {})
        sl_pct = tp_cfg.get("stop_loss_pct", 12.0) / 100
        t1_pct = tp_cfg.get("t1_pct", 8.0) / 100
        t2_pct = tp_cfg.get("t2_pct", 16.0) / 100
        t3_pct = tp_cfg.get("t3_pct", 28.0) / 100
        stop_loss = round(price * (1 - sl_pct), 2)
        t1 = round(price * (1 + t1_pct), 2)
        t2 = round(price * (1 + t2_pct), 2)
        t3 = round(price * (1 + t3_pct), 2)

        logger.warning("Rule-based fallback analysis for %s — AI unavailable, buy signals blocked", ticker)
        technical_summary = (
            f"Price: ${_fmt_price(price)} | RSI: {technicals.rsi:.1f} | "
            f"Support: ${_fmt_price(technicals.support_level)} | Resistance: ${_fmt_price(technicals.resistance_level)}"
        )
        return ResearchReport(
            ticker=ticker,
            asset_name=asset_name,
            generated_at=self._now_et(),
            conviction_score=score,
            signal=signal,
            risk_level=RiskLevel.HIGH,
            thesis=f"⚠ Rule-based estimate only (AI unavailable). Score: {score}/10 — do not trade without AI confirmation.",
            technical_summary=technical_summary,
            news_summary=sentiment.summary,
            risk_factors="AI analysis unavailable — rule-based estimates only. Do not trade on this signal.",
            recommended_action="HOLD — await AI analysis",
            entry_price=price,
            position_size_pct=0.0,
            stop_loss=stop_loss,
            take_profit_targets=[t1, t2, t3],
            time_horizon="unknown",
            reasoning=(
                f"⚠ No AI analysis available. Rule-based score {score}/10 from: "
                f"Sentiment {sentiment.overall_sentiment:.2f}, RSI {technicals.rsi:.1f}."
            ),
            is_fallback=True,
        )

    def _save_report(self, report: ResearchReport):
        filename = f"{report.ticker.replace('/', '')}_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.reports_dir / filename
        data = {
            "ticker": report.ticker,
            "asset_name": report.asset_name,
            "generated_at": report.generated_at.isoformat(),
            "conviction_score": report.conviction_score,
            "signal": report.signal.value,
            "risk_level": report.risk_level.value,
            "thesis": report.thesis,
            "technical_summary": report.technical_summary,
            "news_summary": report.news_summary,
            "risk_factors": report.risk_factors,
            "recommended_action": report.recommended_action,
            "entry_price": report.entry_price,
            "position_size_pct": report.position_size_pct,
            "stop_loss": report.stop_loss,
            "take_profit_targets": report.take_profit_targets,
            "time_horizon": report.time_horizon,
            "reasoning": report.reasoning,
            "is_fallback": report.is_fallback,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    async def explain_buy_decision(
        self, ticker: str, asset_name: str, entry_price: float, stop_loss: float,
        take_profit_targets: list[float], fair_value_estimate: float | None,
        conviction: int | None, rr: float | None, required_rr: float | None,
        opened_at_days_ago: float | None,
    ) -> dict | None:
        """Retroactive backfill for "Why AI Bought This" (2026-08-21, same feature as
        AITrading/AIShortTrading's own -- see AITrading's CLAUDE_HISTORY.md 2026-08-21
        entry for the full design). Only used for a position bought BEFORE Position.
        buy_thesis existed to capture the real decision automatically -- see
        scripts/backfill_buy_rationale.py, the sole caller. A genuinely new buy never
        calls this; it gets the real, original thesis/reasoning/R/R for free from the
        exact analysis that triggered it (OrderManager.execute_buy).

        This is necessarily a RECONSTRUCTION, not the original reasoning verbatim (that
        moment is gone) -- the prompt is explicit about this and grounds Claude in the
        real, immutable trade parameters (entry, stop, targets, fair value, conviction,
        the exact R/R math) rather than asking it to invent a narrative from nothing.
        Returns None on any failure -- the backfill script simply skips that position
        and leaves it blank rather than fabricating something. Uses model_quick_scan
        (this project's only per-decision model dial -- no separate dip-entry-style
        setting exists here the way it does in AITrading/AIShortTrading)."""
        if not self.client:
            return None
        risk = entry_price - stop_loss
        rr_line = ""
        if rr is not None and required_rr is not None:
            rr_line = (f"\nRisk/reward at purchase: {rr:.2f} against a required {required_rr:.2f} "
                       f"for this conviction level (this is what actually allowed the purchase -- "
                       f"reward is the distance from entry ${entry_price:.4f} to the fair value "
                       f"estimate below; risk is the distance from entry to the stop ${stop_loss:.4f}, "
                       f"${risk:.4f} away).")
        fv_line = f"\nFair value estimate at purchase: ${fair_value_estimate:.4f}" if fair_value_estimate else ""
        conv_line = f"\nConviction score: {conviction}/10" if conviction is not None else ""
        age_line = (f"\nThis position was opened approximately {opened_at_days_ago:.0f} day(s) ago."
                    if opened_at_days_ago is not None else "")
        targets_str = ", ".join(f"${t:.4f}" for t in take_profit_targets) if take_profit_targets else "none recorded"

        prompt = f"""\
You are reconstructing the investment case for a real crypto trade this system already made,
for a "Why AI Bought This" explanation shown to the account owner. The original real-time
reasoning text from the moment of purchase was not captured for this specific position (it
predates that tracking), so you're rebuilding a faithful, well-reasoned explanation from the
real, immutable trade parameters below -- not inventing a story disconnected from them.

ASSET: {ticker} — {asset_name}
Entry price: ${entry_price:.4f}
Stop loss: ${stop_loss:.4f}
Take-profit targets: {targets_str}{fv_line}{conv_line}{rr_line}{age_line}

Write a genuine, comprehensive walk-through of the likely investment thesis: what about this
crypto asset's fundamentals and technical setup would make it a buy candidate, how the
risk/reward math above specifically cleared this system's gate, and what the conviction level
implies about how strong the case was. Be specific and grounded in the real numbers given --
don't write generic boilerplate that could apply to any asset. This should read like a real
analyst's reasoning, several sentences of substance, not a one-line summary.

Respond with ONLY a JSON object:
{{"thesis": "<2-3 sentence core investment thesis>",
  "reasoning": "<4-6 sentences of detailed reasoning covering fundamentals/technicals, why the
  R/R math worked, and what the conviction level reflects>"}}
"""
        model = self.config.get("research", {}).get("model_quick_scan", "claude-haiku-4-5")
        try:
            response_text = await self._call_claude_with_retry(
                model=model, max_tokens=500, messages=[{"role": "user", "content": prompt}],
                max_retries=2,
            )
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            text = text.rstrip()
            if text.endswith("```"):
                text = text[:-3].rstrip()
            data = json.loads(text)
            thesis = str(data.get("thesis", "")).strip()
            reasoning = str(data.get("reasoning", "")).strip()
            if not thesis and not reasoning:
                return None
            return {"thesis": thesis, "reasoning": reasoning}
        except Exception as e:
            logger.warning("%s: explain_buy_decision failed: %s", ticker, e)
            return None

    async def analyze_sell_decision(
        self, ticker: str, asset_name: str, buy_thesis: str, buy_reasoning: str,
        buy_conviction: int | None, buy_rr: float | None, buy_required_rr: float | None,
        entry_price: float, opened_at_days_ago: float | None,
        tranches: list[dict], price_history: list[dict],
    ) -> dict | None:
        """"Recent Sell" post-mortem, immediate half (2026-08-21, same feature as
        AITrading/AIShortTrading's own). Fired once per fully-closed trade by a periodic
        job draining Portfolio.get_pending_sell_analyses() (this file has no DB access
        of its own). Judges the hold itself -- entry to final exit -- using the real
        original buy thesis, every real sell tranche (date/price/reason/pnl), and real
        price history across the window. Returns None on any failure -- the caller
        leaves the row pending and retries on a later pass."""
        if not self.client:
            return None
        conv_line = f"\nConviction at purchase: {buy_conviction}/10" if buy_conviction is not None else ""
        rr_line = ""
        if buy_rr is not None and buy_required_rr is not None:
            rr_line = f"\nR/R at purchase: {buy_rr:.2f} against a required {buy_required_rr:.2f}."
        age_line = (f"\nHeld for approximately {opened_at_days_ago:.0f} day(s)."
                    if opened_at_days_ago is not None else "")
        tranches_str = "\n".join(
            f"- {t['date']}: sold at ${t['price']:.4f} ({t['reason']})"
            + (f", P&L {'+' if t['pnl'] >= 0 else ''}${t['pnl']:.2f}" if t.get("pnl") is not None else "")
            for t in tranches
        ) or "none recorded"
        history_str = "\n".join(
            f"- {p['date']}: ${p['close']:.4f}" for p in price_history
        ) or "not available"

        prompt = f"""\
You are writing a post-mortem for a real, already-completed crypto trade this system made, to
help improve future sell decisions on this same asset. Judge whether a genuinely better exit
was realistically available given what was known and visible at the time -- not simple
hindsight ("it went up later" alone isn't evidence of a mistake if nothing signaled that at
the time).

ASSET: {ticker} — {asset_name}
Original buy thesis: {buy_thesis}
Original buy reasoning: {buy_reasoning}
Entry price: ${entry_price:.4f}{conv_line}{rr_line}{age_line}

Sell tranches (chronological):
{tranches_str}

Price history across the hold (chronological daily closes):
{history_str}

Write a genuine, specific assessment: did the exit(s) capture a reasonable outcome given the
setup, or does the price history show a clearly better exit was available (e.g. price
recovered sharply right after a stop fired, or kept climbing well past where a take-profit
closed the position)? Be concrete and grounded in the real numbers above -- don't write
generic boilerplate. This is for future reference on this same asset, not a performance score.

Respond with ONLY a JSON object:
{{"thesis": "<1-2 sentence verdict>",
  "reasoning": "<3-5 sentences of specific supporting detail from the real price history and tranches above>"}}
"""
        model = self.config.get("research", {}).get("model_quick_scan", "claude-haiku-4-5")
        try:
            response_text = await self._call_claude_with_retry(
                model=model, max_tokens=500, messages=[{"role": "user", "content": prompt}],
                max_retries=2,
            )
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            text = text.rstrip()
            if text.endswith("```"):
                text = text[:-3].rstrip()
            data = json.loads(text)
            thesis = str(data.get("thesis", "")).strip()
            reasoning = str(data.get("reasoning", "")).strip()
            if not thesis and not reasoning:
                return None
            return {"thesis": thesis, "reasoning": reasoning}
        except Exception as e:
            logger.warning("%s: analyze_sell_decision failed: %s", ticker, e)
            return None

    async def analyze_sell_followup(
        self, ticker: str, asset_name: str, post_mortem_thesis: str,
        post_mortem_reasoning: str, exit_price: float, closed_at_days_ago: float | None,
        price_history_since: list[dict],
    ) -> str | None:
        """"Recent Sell" post-mortem, delayed half (2026-08-21, same feature as
        AITrading/AIShortTrading's own). Fired once, some days after the immediate
        post-mortem, revisiting the original verdict using real price action that has
        happened SINCE the sale. Returns a single reasoning string or None on any
        failure, same fail-open-to-nothing principle as analyze_sell_decision."""
        if not self.client:
            return None
        age_line = (f"\nThat was approximately {closed_at_days_ago:.0f} day(s) ago."
                    if closed_at_days_ago is not None else "")
        history_str = "\n".join(
            f"- {p['date']}: ${p['close']:.4f}" for p in price_history_since
        ) or "not available"

        prompt = f"""\
You previously wrote this post-mortem for a real, completed crypto trade on {ticker} — {asset_name}:

Verdict: {post_mortem_thesis}
Reasoning: {post_mortem_reasoning}

The position was closed at ${exit_price:.4f}.{age_line} Real price history has accumulated
since then:
{history_str}

With this additional real price action now visible, does your original verdict still hold, or
does it change? Be concrete about what the price did after the sale and what that confirms or
revises about the original exit's timing.

Respond with ONLY a JSON object:
{{"reasoning": "<3-5 sentences, referencing the real post-sale price action above>"}}
"""
        model = self.config.get("research", {}).get("model_quick_scan", "claude-haiku-4-5")
        try:
            response_text = await self._call_claude_with_retry(
                model=model, max_tokens=400, messages=[{"role": "user", "content": prompt}],
                max_retries=2,
            )
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            text = text.rstrip()
            if text.endswith("```"):
                text = text[:-3].rstrip()
            data = json.loads(text)
            reasoning = str(data.get("reasoning", "")).strip()
            return reasoning or None
        except Exception as e:
            logger.warning("%s: analyze_sell_followup failed: %s", ticker, e)
            return None
