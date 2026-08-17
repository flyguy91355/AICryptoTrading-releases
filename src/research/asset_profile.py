"""Per-asset historical character profiles.

With only ~22 assets in the crypto universe (vs. AITrading's 1,500+ stock
universe), a per-asset supplement to the shared ANALYSIS_PROMPT is affordable
and worth doing -- but as ONE new prompt SECTION built from real computed
history, not 22 separately-maintained full prompt strings (that would
duplicate the shared JSON-schema/gate rules 22 times, and a future rule
change would need editing 22 places instead of one -- the exact class of
drift this codebase has had real bugs from elsewhere).

Two pure, testable pieces:
- compute_asset_stats(): turns raw daily OHLCV into real numeric descriptors
  of an asset's own historical behavior (volatility, distance from its
  high/low, and a simple mechanical breakout backtest -- how often a new
  N-day-high has actually kept running vs. reversed for THIS asset).
- format_asset_stats_summary(): turns those numbers into a labeled prompt
  section, fed to a dedicated Claude call (ASSET_PROFILE_PROMPT) that
  synthesizes a short, asset-specific character profile in plain prose.

This intentionally does NOT hardcode a mechanical "breakout = buy" rule --
same "give the model real data + framing, trust its judgment" pattern this
whole codebase already uses (AITrading's long-term-trend/market-context
sections, the AI-chosen stop-loss work). The mechanical stats are real and
computed once from real history; the JUDGMENT about what they mean for a
current setup is still Claude's, both here (writing the profile) and later
(reading it inside ANALYSIS_PROMPT).

Refreshed periodically (see scripts/build_asset_profiles.py and the
in-app scheduled check in web/app.py) rather than hand-written once, so
"keep the prompts up to date as time goes on" doesn't depend on anyone
remembering to re-run anything.
"""

from dataclasses import dataclass
from statistics import mean, median, pstdev


@dataclass
class AssetStats:
    ticker: str
    years: float
    bars_used: int
    high: float
    high_date: str
    low: float
    low_date: str
    current_price: float
    pct_off_high: float
    avg_daily_range_pct: float
    daily_volatility_pct: float
    breakout_count: int
    breakout_success_rate: float | None  # fraction of breakouts that kept running; None if too few to matter
    breakout_median_follow_through_pct: float | None
    max_drawdown_pct: float
    major_drawdown_count: int  # count of distinct drawdowns past drawdown_threshold_pct


def compute_asset_stats(
    ticker: str,
    bars: list[dict],  # each: {"date": "YYYY-MM-DD", "open","high","low","close","volume"}
    current_price: float,
    breakout_lookback_days: int = 20,
    follow_through_days: int = 10,
    follow_through_threshold_pct: float = 3.0,
    drawdown_threshold_pct: float = 30.0,
) -> AssetStats | None:
    """Pure: no network/API calls, just arithmetic over already-fetched daily bars.
    Returns None when there isn't enough history to say anything real (rather than
    fabricating stats from a handful of points) -- callers should skip profile
    generation entirely for that ticker until more history exists."""
    if len(bars) < breakout_lookback_days + follow_through_days + 5:
        return None

    closes = [b["close"] for b in bars]
    dates = [b["date"] for b in bars]

    high = max(b["high"] for b in bars)
    high_idx = max(range(len(bars)), key=lambda i: bars[i]["high"])
    low = min(b["low"] for b in bars)
    low_idx = min(range(len(bars)), key=lambda i: bars[i]["low"])

    daily_ranges_pct = [(b["high"] - b["low"]) / b["close"] * 100 for b in bars if b["close"] > 0]
    daily_returns_pct = [
        (closes[i] - closes[i - 1]) / closes[i - 1] * 100
        for i in range(1, len(closes)) if closes[i - 1] > 0
    ]

    # Mechanical breakout backtest: a "breakout" is the first day price closes above
    # the prior `breakout_lookback_days` window's high, with a cooldown so one genuine
    # move doesn't get counted as several overlapping breakouts.
    breakout_returns: list[float] = []
    last_breakout_idx = -breakout_lookback_days
    for i in range(breakout_lookback_days, len(closes) - follow_through_days):
        window_high = max(closes[i - breakout_lookback_days:i])
        if closes[i] > window_high and (i - last_breakout_idx) >= breakout_lookback_days:
            last_breakout_idx = i
            forward_close = closes[i + follow_through_days]
            breakout_returns.append((forward_close - closes[i]) / closes[i] * 100)

    breakout_count = len(breakout_returns)
    if breakout_count >= 3:
        successes = sum(1 for r in breakout_returns if r >= follow_through_threshold_pct)
        breakout_success_rate = successes / breakout_count
        breakout_median_follow_through = median(breakout_returns)
    else:
        breakout_success_rate = None
        breakout_median_follow_through = None

    # Drawdown scan: running peak, count distinct excursions past the threshold
    # (must recover back near the prior peak before a new one can be counted, so
    # one long decline isn't double-counted as it merely fluctuates near its low).
    peak = closes[0]
    max_dd = 0.0
    major_dd_count = 0
    in_major_dd = False
    for c in closes:
        if c > peak:
            peak = c
        dd = (c - peak) / peak * 100 if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
        if dd <= -drawdown_threshold_pct and not in_major_dd:
            major_dd_count += 1
            in_major_dd = True
        elif dd >= -drawdown_threshold_pct / 3 and in_major_dd:
            in_major_dd = False  # recovered enough to count a future drawdown as a new one

    return AssetStats(
        ticker=ticker,
        years=round(len(bars) / 365, 1),
        bars_used=len(bars),
        high=high, high_date=dates[high_idx],
        low=low, low_date=dates[low_idx],
        current_price=current_price,
        pct_off_high=(current_price - high) / high * 100 if high > 0 else 0.0,
        avg_daily_range_pct=mean(daily_ranges_pct) if daily_ranges_pct else 0.0,
        daily_volatility_pct=pstdev(daily_returns_pct) if len(daily_returns_pct) >= 2 else 0.0,
        breakout_count=breakout_count,
        breakout_success_rate=breakout_success_rate,
        breakout_median_follow_through_pct=breakout_median_follow_through,
        max_drawdown_pct=max_dd,
        major_drawdown_count=major_dd_count,
    )


def format_asset_stats_summary(stats: AssetStats) -> str:
    """Pure formatter -- turns AssetStats into the labeled numeric section fed to
    ASSET_PROFILE_PROMPT. Deliberately plain numbers with no interpretation of its
    own; Claude does the judgment when synthesizing the profile."""
    lines = [
        f"History available: {stats.years:.1f} years ({stats.bars_used} daily bars)",
        f"Range: Low ${stats.low:.4g} ({stats.low_date}) | High ${stats.high:.4g} ({stats.high_date}) | "
        f"Current ${stats.current_price:.4g} ({stats.pct_off_high:+.1f}% vs. the high)",
        f"Average daily trading range: {stats.avg_daily_range_pct:.2f}% | "
        f"Daily volatility (stdev of daily % change): {stats.daily_volatility_pct:.2f}%",
        f"Largest peak-to-trough drawdown in this window: {stats.max_drawdown_pct:.1f}% | "
        f"Number of distinct drawdowns past 30%: {stats.major_drawdown_count}",
    ]
    if stats.breakout_count >= 3:
        lines.append(
            f"Mechanical breakout backtest ({stats.breakout_count} instances of price closing "
            f"above its prior 20-day high): {stats.breakout_success_rate * 100:.0f}% went on to gain "
            f"3%+ over the following 10 trading days; median follow-through was "
            f"{stats.breakout_median_follow_through_pct:+.1f}%."
        )
    else:
        lines.append(
            f"Mechanical breakout backtest: only {stats.breakout_count} qualifying instance(s) in "
            f"this window -- too few to draw a real success-rate conclusion from."
        )
    return "\n".join(lines)


ASSET_PROFILE_PROMPT = """\
You are a crypto market historian. Based ONLY on the real historical statistics below \
for {ticker} ({asset_name}), write a concise character profile (3-5 sentences) that a \
trader should keep in mind when judging whether a CURRENT technical setup for this \
specific asset is a genuine opportunity or a likely false signal.

Focus on what makes THIS asset's own historical behavior distinctive -- its typical \
volatility relative to other crypto assets, how reliable a fresh breakout has \
historically been for it specifically, and how it tends to behave during drawdowns \
(sharp V-shaped recoveries vs. long grinding declines). Do not give generic crypto \
advice that would apply equally to any asset, and do not recommend a current action -- \
this profile is background context for a separate, later analysis, not a trade call.

── {ticker} HISTORICAL STATISTICS ──
{stats_summary}

Respond with ONLY the profile text, no preamble, no JSON, no markdown formatting.
"""


def build_asset_profile_prompt(ticker: str, asset_name: str, stats: AssetStats) -> str:
    return ASSET_PROFILE_PROMPT.format(
        ticker=ticker, asset_name=asset_name,
        stats_summary=format_asset_stats_summary(stats),
    )


SMA_PERIOD_PROMPT = """\
Choose two chart moving-average lookback periods for {ticker} ({asset_name}).

Based ONLY on the real historical statistics below, pick:
- A "fast" SMA (short-term trend, 15–100 days)
- A "slow" SMA (medium/long-term trend, 80–300 days)

The goal is chart legibility — which periods make THIS asset's own trend direction \
and major reversals clearly visible on a daily candlestick chart? A highly volatile \
asset with sharp swings may need longer SMAs so noise doesn't produce constant false \
crosses; a smoother asset can use shorter periods for faster signal. The slow period \
must be at least 40 days longer than the fast period.

── {ticker} HISTORICAL STATISTICS ──
{stats_summary}

Respond with ONLY a JSON object, no explanation, no markdown:
{{"sma_fast": <integer>, "sma_slow": <integer>}}
"""


def build_sma_period_prompt(ticker: str, asset_name: str, stats: AssetStats) -> str:
    return SMA_PERIOD_PROMPT.format(
        ticker=ticker, asset_name=asset_name,
        stats_summary=format_asset_stats_summary(stats),
    )


def build_asset_profile_section(profile: str | None) -> str:
    """Pure formatter -- builds the ANALYSIS_PROMPT section text from a stored profile.
    Returns "" when there's no profile yet, so the section is omitted cleanly rather
    than showing garbage (same pattern as _build_long_term_trend_section)."""
    if not profile:
        return ""
    return f"\n── THIS ASSET'S HISTORICAL CHARACTER (from a real multi-year backtest) ──\n{profile}\n"
