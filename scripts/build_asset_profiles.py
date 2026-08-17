#!/usr/bin/env python3
"""Generates/refreshes data/asset_profiles.json -- the per-asset historical
character profiles injected into ANALYSIS_PROMPT (see
src/research/asset_profile.py for the full design rationale).

For each asset in the configured universe: fetches ~3 years of real daily
OHLCV, computes real volatility/breakout/drawdown statistics
(compute_asset_stats -- pure, no Claude call), then makes ONE Claude call
per asset to synthesize those numbers into a short character profile.

Idempotent by default: a ticker whose profile is younger than
research.asset_profile_refresh_days (config/settings.yaml, default 7) is
skipped, so re-running this doesn't re-spend real Claude calls on profiles
that are still fresh. Run manually any time, or via the in-app weekly
scheduled check in web/app.py (_maybe_refresh_asset_profiles) -- both call
the same generate_all_profiles() function so there's exactly one real
"how does a profile get (re)built" code path.

Usage:
    python scripts/build_asset_profiles.py             # refresh anything stale
    python scripts/build_asset_profiles.py --force      # regenerate everything
    python scripts/build_asset_profiles.py --ticker BTC/USD   # just one, forced
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic

from src.data.market_data import MarketDataFetcher
from src.research.asset_profile import compute_asset_stats, build_asset_profile_prompt, build_sma_period_prompt
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROFILES_PATH = Path("data/asset_profiles.json")


def _load_existing() -> dict:
    try:
        return json.loads(PROFILES_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _is_fresh(entry: dict, refresh_days: int) -> bool:
    generated_at = entry.get("generated_at")
    if not generated_at:
        return False
    try:
        ts = datetime.fromisoformat(generated_at)
    except ValueError:
        return False
    return datetime.now(timezone.utc) - ts < timedelta(days=refresh_days)


async def _call_claude(client, model: str, prompt: str, max_tokens: int) -> str | None:
    """Shared thin wrapper: makes one Claude call, returns stripped text or None on failure."""
    try:
        message = await asyncio.to_thread(
            lambda: client.messages.create(
                model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
            )
        )
        block = next((b for b in message.content if hasattr(b, "text")), None)
        return block.text.strip() if block and block.text.strip() else None
    except Exception as e:
        logger.warning("Claude call failed: %s", e)
        return None


async def generate_all_profiles(config: dict, force: bool = False, only_ticker: str | None = None) -> dict:
    """Returns {ticker: {"profile": str, "generated_at": iso, "stats_summary": {...},
    "sma_fast_period": int, "sma_slow_period": int}}.
    Real function used by both the CLI script and the in-app scheduled refresh --
    kept here (not duplicated in web/app.py) so there's one source of truth."""
    market_data = MarketDataFetcher(config)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("No ANTHROPIC_API_KEY -- cannot generate real profiles, aborting")
        return _load_existing()
    client = anthropic.Anthropic(api_key=api_key)
    model = config.get("research", {}).get("model_quick_scan", "claude-haiku-4-5")
    refresh_days = config.get("research", {}).get("asset_profile_refresh_days", 7)
    sma_mode = config.get("research", {}).get("sma_mode", "auto")

    profiles = _load_existing()
    universe = config.get("universe", [])
    if only_ticker:
        universe = [a for a in universe if a["ticker"] == only_ticker]
        if not universe:
            logger.error("Ticker %s not found in configured universe", only_ticker)
            return profiles

    for asset in universe:
        ticker, name = asset["ticker"], asset["name"]
        existing = profiles.get(ticker)
        if not force and existing and _is_fresh(existing, refresh_days):
            logger.info("%s profile is still fresh (< %dd old) -- skipping", ticker, refresh_days)
            continue

        try:
            bars = await market_data.get_historical(ticker, period="3y", interval="1d")
            quote = await market_data.get_quote(ticker)
        except Exception as e:
            logger.warning("%s: history/quote fetch failed (%s) -- leaving prior profile in place", ticker, e)
            continue

        stats = compute_asset_stats(ticker, bars, current_price=quote.price)
        if stats is None:
            logger.info("%s: not enough history yet for a real profile (%d bars) -- skipping", ticker, len(bars))
            continue

        # Profile call (character summary for ANALYSIS_PROMPT)
        profile_text = await _call_claude(
            client, model, build_asset_profile_prompt(ticker, name, stats), max_tokens=400,
        )
        if not profile_text:
            logger.warning("%s: Claude profile generation failed -- leaving prior profile in place", ticker)
            continue

        # SMA period call (only for auto mode — flat/manual don't need per-asset AI choices)
        sma_fast, sma_slow = 50, 200  # safe defaults if call skipped or fails
        if sma_mode == "auto":
            await asyncio.sleep(0.5)  # brief pause between the two calls for this ticker
            sma_text = await _call_claude(
                client, model, build_sma_period_prompt(ticker, name, stats), max_tokens=50,
            )
            if sma_text:
                import re, json as _json
                try:
                    # Accept raw JSON or a JSON object embedded in light prose
                    m = re.search(r'\{[^}]+\}', sma_text)
                    if m:
                        parsed = _json.loads(m.group())
                        fast_raw = int(parsed.get("sma_fast", 50))
                        slow_raw = int(parsed.get("sma_slow", 200))
                        # Enforce the guard: fast < slow, both in sane ranges
                        fast_raw = max(15, min(100, fast_raw))
                        slow_raw = max(80, min(300, slow_raw))
                        if slow_raw >= fast_raw + 40:
                            sma_fast, sma_slow = fast_raw, slow_raw
                            logger.info("%s: AI chose SMA%d / SMA%d", ticker, sma_fast, sma_slow)
                        else:
                            logger.warning("%s: AI SMA periods too close (%d/%d) -- keeping defaults", ticker, fast_raw, slow_raw)
                except Exception as e:
                    logger.warning("%s: SMA period parse failed (%s) -- keeping defaults", ticker, e)

        profiles[ticker] = {
            "profile": profile_text,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sma_fast_period": sma_fast,
            "sma_slow_period": sma_slow,
            "stats_summary": {
                "years": stats.years,
                "high": stats.high, "low": stats.low, "pct_off_high": stats.pct_off_high,
                "daily_volatility_pct": stats.daily_volatility_pct,
                "breakout_count": stats.breakout_count,
                "breakout_success_rate": stats.breakout_success_rate,
                "max_drawdown_pct": stats.max_drawdown_pct,
            },
        }
        logger.info("%s: profile generated (%d chars), SMA%d/SMA%d", ticker, len(profile_text), sma_fast, sma_slow)
        await asyncio.sleep(1)  # small pacing gap, same courtesy the main scan loop uses

    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_PATH.write_text(json.dumps(profiles, indent=2))
    logger.info("Wrote %d profiles to %s", len(profiles), PROFILES_PATH)
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate every profile regardless of freshness")
    parser.add_argument("--ticker", default=None, help="Only (re)generate one ticker, e.g. BTC/USD")
    args = parser.parse_args()

    config = load_config()
    asyncio.run(generate_all_profiles(config, force=args.force or bool(args.ticker), only_ticker=args.ticker))


if __name__ == "__main__":
    main()
