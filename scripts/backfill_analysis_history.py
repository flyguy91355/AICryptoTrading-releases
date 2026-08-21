"""One-time seed for the "Analysis History" feed-forward feature (2026-08-21, same
feature as AITrading/AIShortTrading's own -- see AITrading's CLAUDE_HISTORY.md
2026-08-21 entry for the full design and owner request: "backfeed it with the data too
whatever you can").

analysis_history only starts accumulating real rows going forward (see
DashboardState._persist_analysis_history, web/app.py) -- this script seeds ONE starting
row per ticker already in data/latest_reports_cache.json, using whatever's already
cached (the most recent real analysis for that ticker). This is not a reconstruction of
multiple past calls -- the cache only ever kept the LATEST report per ticker, so
there's only ever one data point available per ticker to seed from. watch_condition is
left empty for every seeded row (that field didn't exist before this feature, so no
cached report has a real value for it).

Idempotent-ish: re-running adds a SECOND row for any ticker that already has one (this
script doesn't check analysis_history for existing rows), so only run this once, right
after deploying this feature.

Run manually: python scripts/backfill_analysis_history.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.decision.portfolio import Portfolio


def load_config() -> dict:
    with open(Path(__file__).resolve().parent.parent / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def load_reports_cache() -> dict:
    path = Path(__file__).resolve().parent.parent / "data" / "latest_reports_cache.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


async def main():
    config = load_config()
    portfolio = Portfolio(config)
    await portfolio.initialize()

    reports = load_reports_cache()
    print(f"{len(reports)} cached report(s) found in latest_reports_cache.json.")

    seeded = 0
    for ticker, r in reports.items():
        if not isinstance(r, dict):
            continue
        conviction = r.get("conviction_score")
        if conviction is None:
            continue
        await portfolio.save_analysis_history(
            ticker,
            r.get("generated_at", ""),
            float(conviction),
            r.get("signal", "NO ACTION"),
            r.get("entry_price"),
            r.get("fair_value_estimate"),
            "",  # watch_condition didn't exist before this feature -- nothing to seed
        )
        seeded += 1

    print(f"Done. Seeded {seeded} starting row(s) into analysis_history.")


if __name__ == "__main__":
    asyncio.run(main())
