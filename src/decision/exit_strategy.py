"""Pure exit-strategy math: graduated trailing stop width.

Ported from AITrading's web/app.py (where these live inline rather than in
their own module, since that file builds a live DashboardState at import
time and can't be unit-imported the same way). Kept as their own module here
specifically so they're directly testable without any app-level machinery.
"""


def derive_stop_pct(entry_price: float, stop_loss: float, default_pct: float) -> float:
    """A position's own stop-loss %, derived from Claude's real stop_loss
    recommendation at analysis time rather than one flat config value shared by every
    asset. Returns default_pct if entry_price/stop_loss aren't usable."""
    if entry_price <= 0 or stop_loss <= 0 or stop_loss >= entry_price:
        return default_pct
    return (entry_price - stop_loss) / entry_price * 100


def graduated_trail_pct(
    entry_price: float, current_price: float, start_pct: float, final_pct: float,
    t1_price: float | None, t2_price: float | None, t3_price: float | None,
    follow_tp_targets: bool,
) -> float:
    """Trail width (as a whole percentage, e.g. 2.0 = 2%) for the graduated trailing
    stop, interpolated along a single curve from entry_price (start_pct) to t3_price
    (final_pct) -- one continuous curve covering the whole life of the trade, so a fast
    mover that reaches T1 quickly gets a tighter trail sooner than one that ground
    there slowly.

    follow_tp_targets=False draws a straight line with just two anchors (entry, T3).
    follow_tp_targets=True additionally bends the curve through T1/T2 as checkpoints
    when both are available and sit strictly between entry and T3 -- falls back to the
    straight line otherwise. Returns start_pct (the widest, safest end of the range) if
    t3_price is missing or the entry->T3 range itself isn't valid."""
    if t3_price is None or entry_price <= 0 or t3_price <= entry_price:
        return start_pct

    if (follow_tp_targets and t1_price is not None and t2_price is not None
            and entry_price < t1_price < t2_price < t3_price):
        step = (start_pct - final_pct) / 3
        anchors = [
            (entry_price, start_pct),
            (t1_price, start_pct - step),
            (t2_price, start_pct - 2 * step),
            (t3_price, final_pct),
        ]
    else:
        anchors = [(entry_price, start_pct), (t3_price, final_pct)]

    if current_price <= anchors[0][0]:
        return anchors[0][1]
    if current_price >= anchors[-1][0]:
        return anchors[-1][1]

    for (p_lo, pct_lo), (p_hi, pct_hi) in zip(anchors, anchors[1:]):
        if p_lo <= current_price <= p_hi:
            progress = (current_price - p_lo) / (p_hi - p_lo)
            return pct_lo - progress * (pct_lo - pct_hi)

    return final_pct  # unreachable given the bounds checks above -- safe fallback regardless
