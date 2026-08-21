"""One-time retroactive backfill for "Why AI Bought This" (2026-08-21, same feature as
AITrading/AIShortTrading's own -- see AITrading's CLAUDE_HISTORY.md 2026-08-21 entry
for the full design and owner request).

Position.buy_thesis/buy_reasoning/etc. are captured automatically for every NEW buy
going forward (see OrderManager.execute_buy) at zero extra cost -- this script exists
purely to fill in the same fields for positions that were already open before that
capture existed, since there's no original decision data left to pull from for those.
Each currently-held position missing buy_thesis gets ONE real Claude call
(ResearchEngine.explain_buy_decision, model_quick_scan -- this project's only
per-decision model dial) that reconstructs a grounded explanation from the position's
real, immutable trade parameters (entry, stop, targets, fair value if known, conviction
if known, the exact R/R math) -- see that function's own docstring for why this is a
faithful reconstruction, not an invented narrative.

Idempotent: a position that already has buy_thesis is skipped, so re-running this only
ever costs real money for positions still missing it.

Run manually: python scripts/backfill_buy_rationale.py
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.decision.portfolio import Portfolio
from src.research.engine import ResearchEngine
from src.data.market_data import MarketDataFetcher
from src.data.news_feed import NewsFeed


def load_config() -> dict:
    with open(Path(__file__).resolve().parent.parent / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def load_asset_names() -> dict[str, str]:
    """Best-effort real asset names from the cached research reports -- purely
    cosmetic prompt quality; falls back to the ticker itself if this file is missing
    or a given ticker was never cached."""
    path = Path(__file__).resolve().parent.parent / "data" / "latest_reports_cache.json"
    try:
        data = json.loads(path.read_text())
        return {t: r.get("asset_name", t) for t, r in data.items() if isinstance(r, dict)}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


async def main():
    config = load_config()
    portfolio = Portfolio(config)
    await portfolio.initialize()

    market_data = MarketDataFetcher(config)
    news_feed = NewsFeed(config)
    engine = ResearchEngine(config, market_data, news_feed)
    asset_names = load_asset_names()

    candidates = [
        pos for pos in portfolio.positions.values()
        if not pos.buy_thesis and not pos.buy_reasoning
    ]
    print(f"{len(portfolio.positions)} open position(s), "
          f"{len(candidates)} missing a buy rationale.")

    filled = 0
    skipped = 0
    for pos in candidates:
        days_ago = (datetime.now() - pos.opened_at).total_seconds() / 86400 if pos.opened_at else None
        result = await engine.explain_buy_decision(
            ticker=pos.ticker,
            asset_name=asset_names.get(pos.ticker, pos.ticker),
            entry_price=pos.entry_price,
            stop_loss=pos.stop_loss,
            take_profit_targets=pos.take_profit_targets,
            fair_value_estimate=pos.buy_fair_value,
            conviction=pos.buy_conviction,
            rr=pos.buy_rr,
            required_rr=pos.buy_required_rr,
            opened_at_days_ago=days_ago,
        )
        if result is None:
            print(f"  {pos.ticker}: FAILED -- left blank, can retry by re-running this script")
            skipped += 1
            continue
        pos.buy_thesis = result["thesis"]
        pos.buy_reasoning = result["reasoning"]
        await portfolio._save_position(pos)
        print(f"  {pos.ticker}: backfilled")
        filled += 1
        await asyncio.sleep(1)  # same pacing courtesy the sibling projects' scripts use

    print(f"Done. {filled} backfilled, {skipped} failed (re-run to retry those).")


if __name__ == "__main__":
    asyncio.run(main())
