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
from src.research.asset_profile import compute_asset_stats, build_asset_profile_prompt
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


async def generate_all_profiles(config: dict, force: bool = False, only_ticker: str | None = None) -> dict:
    """Returns {ticker: {"profile": str, "generated_at": iso, "stats_summary": {...}}}.
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

        prompt = build_asset_profile_prompt(ticker, name, stats)
        try:
            message = await asyncio.to_thread(
                lambda: client.messages.create(
                    model=model, max_tokens=400, messages=[{"role": "user", "content": prompt}],
                )
            )
            text_block = next((b for b in message.content if hasattr(b, "text")), None)
            if not text_block or not text_block.text.strip():
                raise ValueError("Empty profile response")
            profile_text = text_block.text.strip()
        except Exception as e:
            logger.warning("%s: Claude profile generation failed (%s) -- leaving prior profile in place", ticker, e)
            continue

        profiles[ticker] = {
            "profile": profile_text,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stats_summary": {
                "years": stats.years,
                "high": stats.high, "low": stats.low, "pct_off_high": stats.pct_off_high,
                "daily_volatility_pct": stats.daily_volatility_pct,
                "breakout_count": stats.breakout_count,
                "breakout_success_rate": stats.breakout_success_rate,
                "max_drawdown_pct": stats.max_drawdown_pct,
            },
        }
        logger.info("%s: profile generated (%d chars)", ticker, len(profile_text))
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
