"""Pure, testable logic for the event-triggered scan loop.

No I/O, no state — just price math and threshold checks. The loop in
web/app.py owns all the I/O (quote fetches, Claude calls, cooldown tracking).
"""


def compute_rsi_from_buffer(prices: list[float], period: int = 14) -> float | None:
    """Wilder's RSI from a plain price list. Returns None when there aren't
    enough prices to produce a meaningful result (needs period+1 minimum).
    Mirrors MarketDataFetcher._compute_rsi's algorithm so results are
    consistent — uses the same SMA seed + Wilder smoothing approach."""
    if len(prices) < period + 1:
        return None

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def check_event_triggers(
    ticker: str,
    current_price: float,
    prev_price: float | None,
    rsi: float | None,
    cfg: dict,
) -> list[str]:
    """Returns a list of human-readable trigger descriptions.
    Empty list = no trigger this tick.

    cfg keys used (all optional, defaults match config/settings.yaml):
      rsi_oversold          (default 30)   -- RSI below this fires
      price_dip_pct         (default 4.0)  -- % drop since last tick fires
      price_surge_pct       (default 3.0)  -- % rise since last tick fires
    """
    triggers = []
    oversold = cfg.get("rsi_oversold", 30)
    dip_pct = cfg.get("price_dip_pct", 4.0)
    surge_pct = cfg.get("price_surge_pct", 3.0)

    if rsi is not None and rsi <= oversold:
        triggers.append(f"RSI oversold ({rsi:.1f} ≤ {oversold})")

    if prev_price is not None and prev_price > 0:
        change_pct = (current_price - prev_price) / prev_price * 100
        if change_pct <= -dip_pct:
            triggers.append(f"sharp dip ({change_pct:.1f}% in last interval)")
        elif change_pct >= surge_pct:
            triggers.append(f"momentum surge ({change_pct:+.1f}% in last interval)")

    return triggers
