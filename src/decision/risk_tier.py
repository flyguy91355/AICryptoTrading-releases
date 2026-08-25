"""Risk-tier slider (2026-08-21) -- a single 0-100 dial per program that computes and
applies conviction gate, R/R gate, position sizing, stop-loss/take-profit widths, and
portfolio circuit breakers together, instead of leaving each as an independent Settings-
page field with no shared concept tying them together. See
docs/superpowers/specs/2026-08-21-risk-tier-design.md for the full design and the
owner's real request behind it.

tier_value=50 ("Medium") is defined as exactly reproducing a program's own real,
already-live settings at the moment this feature's anchors were captured -- not a
generic industry-standard "medium." Low(0)/High(100) are offsets/multipliers off that
anchor. Interpolation is piecewise-linear across two segments (0->50, 50->100) so that
tier=50 always reproduces the anchor exactly regardless of how asymmetric the Low/High
endpoints are (e.g. cash reserve's x3.0 low-side vs. x0.5 high-side)."""

CONVICTION_FLOOR = 3.0
RR_FLOOR = 1.0


def _interp(t: float, low: float, anchor: float, high: float) -> float:
    if t <= 50.0:
        return low + (anchor - low) * (t / 50.0)
    return anchor + (high - anchor) * ((t - 50.0) / 50.0)


def compute_risk_tier_settings(tier_value: float, anchors: dict) -> dict:
    """Returns the 11 real settings values (8 factors + the 3 take-profit ladder
    prices, which scale by the same multiplier as stop_loss_pct to preserve each
    program's existing stop:target ratio) at the given tier. tier_value is clamped to
    [0, 100] first."""
    t = max(0.0, min(100.0, tier_value))

    conviction_high = max(CONVICTION_FLOOR, anchors["min_conviction_score"] - 1.5)
    rr_high = max(RR_FLOOR, anchors["min_risk_reward_ratio"] - 0.5)

    stop_loss_pct = _interp(
        t, anchors["stop_loss_pct"] * 0.6, anchors["stop_loss_pct"],
        anchors["stop_loss_pct"] * 1.6,
    )
    stop_loss_multiplier = stop_loss_pct / anchors["stop_loss_pct"]

    result = {
        "min_conviction_score": _interp(
            t, anchors["min_conviction_score"] + 1.5, anchors["min_conviction_score"],
            conviction_high,
        ),
        "min_risk_reward_ratio": _interp(
            t, anchors["min_risk_reward_ratio"] + 0.5, anchors["min_risk_reward_ratio"],
            rr_high,
        ),
        "starting_position_pct": _interp(
            t, anchors["starting_position_pct"] * 0.5, anchors["starting_position_pct"],
            anchors["starting_position_pct"] * 2.0,
        ),
        "max_loss_per_trade_pct": _interp(
            t, anchors["max_loss_per_trade_pct"] * 0.6, anchors["max_loss_per_trade_pct"],
            anchors["max_loss_per_trade_pct"] * 1.6,
        ),
        "stop_loss_pct": stop_loss_pct,
        "t1_pct": anchors["t1_pct"] * stop_loss_multiplier,
        "t2_pct": anchors["t2_pct"] * stop_loss_multiplier,
        "t3_pct": anchors["t3_pct"] * stop_loss_multiplier,
        "min_cash_reserve_pct": _interp(
            t, anchors["min_cash_reserve_pct"] * 3.0, anchors["min_cash_reserve_pct"],
            anchors["min_cash_reserve_pct"] * 0.5,
        ),
        "drawdown_halt_pct": _interp(
            t, anchors["drawdown_halt_pct"] * 0.6, anchors["drawdown_halt_pct"],
            anchors["drawdown_halt_pct"] * 1.6,
        ),
        "daily_loss_limit_pct": _interp(
            t, anchors["daily_loss_limit_pct"] * 0.6, anchors["daily_loss_limit_pct"],
            anchors["daily_loss_limit_pct"] * 1.6,
        ),
    }
    # Rounded before returning (fixed 2026-08-25, owner report: a live t1_pct/t2_pct/
    # t3_pct in config/settings.yaml showed ~15 digits after the decimal, e.g.
    # 11.840000000000002) -- every one of these values is either an _interp() result
    # or a chain of division-then-multiplication (stop_loss_multiplier = stop_loss_pct
    # / anchors["stop_loss_pct"], then anchors["tN_pct"] * stop_loss_multiplier), both
    # classic sources of IEEE-754 residue. Nothing downstream needs more precision than
    # what's already shown everywhere these values are displayed (Settings page
    # inputs, the risk-tier preview table's own .toFixed(1)/.toFixed(2) formatting) --
    # min_risk_reward_ratio rounds to 2 decimals (matches its own .toFixed(2) display
    # as an R/R ratio, e.g. "2.10:1"); every other field is a plain percentage,
    # rounded to 1 decimal (matches their .toFixed(1) display). Rounding here, once,
    # means every caller (the live apply-to-settings path, the read-only preview
    # endpoint, and whatever ends up written to config/settings.yaml) gets a clean
    # value for free, instead of each needing its own defensive rounding. Ported to
    # AITrading/AIShortTrading the same day.
    for key in result:
        result[key] = round(result[key], 2 if key == "min_risk_reward_ratio" else 1)
    return result


def risk_tier_label(tier_value: float) -> str:
    """Buckets a 0-100 tier value into a human-readable label for both the Settings-page
    display and the AI prompt section. Input is clamped to [0, 100] first."""
    t = max(0.0, min(100.0, tier_value))
    if t < 20.0:
        return "Low"
    if t < 40.0:
        return "Medium-Low"
    if t < 60.0:
        return "Medium"
    if t < 80.0:
        return "Medium-High"
    return "High"


_RISK_TIER_POSTURES = {
    "Low": (
        "Operate with a LOW risk tolerance right now. Prioritize capital "
        "preservation over upside -- decline a marginal setup rather than stretch "
        "for it, and favor an earlier, more conservative exit over riding a "
        "position further."
    ),
    "Medium-Low": (
        "Operate with a MEDIUM-LOW risk tolerance right now. Lean toward "
        "selectivity and caution, but a genuinely strong setup is still worth "
        "taking."
    ),
    "Medium": (
        "Operate with a balanced, MEDIUM risk tolerance right now -- weigh upside "
        "and downside evenly, neither reaching for marginal setups nor declining "
        "solid ones out of excess caution."
    ),
    "Medium-High": (
        "Operate with a MEDIUM-HIGH risk tolerance right now. Lean toward "
        "capturing upside -- a solid (not just a perfect) setup is worth taking, "
        "and a bit more room before cutting a loss is acceptable."
    ),
    "High": (
        "Operate with a HIGH risk tolerance right now. Greater risk tolerance is "
        "acceptable -- don't let a merely-good setup pass waiting for a perfect "
        "one, and give a position more room to work before treating a pullback as "
        "a reason to exit."
    ),
}


def build_risk_tier_prompt_section(tier_value: float) -> str:
    """The AI-facing framing half of the risk-tier feature -- tells Claude directly
    what risk posture this portfolio is operating under, alongside the mechanical
    gate/sizing numbers compute_risk_tier_settings already changes. See
    _RISK_TIER_POSTURES for the owner-directed interpretive lean at each bucket
    (mirrors _build_market_context_section's own "give real data + explicit framing,
    trust Claude's judgment" pattern in src/research/engine.py)."""
    label = risk_tier_label(tier_value)
    t = max(0.0, min(100.0, tier_value))
    return (
        f"\n── RISK TIER ──\n"
        f"This portfolio's current risk tier is {label} ({t:.0f}/100).\n"
        f"{_RISK_TIER_POSTURES[label]}\n"
    )


RISK_TIER_DOTKEYS = {
    # No "starting_position_pct" entry -- AICryptoTrading has no flat position-size
    # config field (position size is the AI's own per-report 1-10% judgment). The
    # computed position-size ceiling at a given tier is still meaningful -- see
    # build_risk_tier_position_note below, which feeds it into the AI's prompt as
    # guidance instead of a config write. compute_risk_tier_settings still computes
    # this factor internally (anchors must include a "starting_position_pct" value),
    # it's just never one of the dotkeys apply_risk_tier_to_settings writes out.
    "min_conviction_score": "research.min_conviction_score",
    "min_risk_reward_ratio": "research.min_risk_reward_ratio",
    "max_loss_per_trade_pct": "risk_management.max_loss_per_trade_pct",
    "stop_loss_pct": "take_profit.stop_loss_pct",
    "t1_pct": "take_profit.t1_pct",
    "t2_pct": "take_profit.t2_pct",
    "t3_pct": "take_profit.t3_pct",
    "min_cash_reserve_pct": "risk_management.min_cash_reserve_pct",
    "drawdown_halt_pct": "risk_management.drawdown_halt_pct",
    "daily_loss_limit_pct": "risk_management.daily_loss_limit_pct",
}


def apply_risk_tier_to_settings(coerced_values: dict, anchors: dict) -> dict:
    """If a /api/settings payload's already-coerced values include "risk_tier.value",
    computes the 11 real factor values at that tier and merges them into a NEW dict
    (the input is never mutated), overwriting any of the same dotkeys the payload
    already had -- the slider move is the explicit action being applied, so it wins
    over whatever the browser's individual fields happened to hold at submit time.
    Returns the input dict unchanged (same contents) when "risk_tier.value" isn't
    present, so a plain settings save that never touches the slider is a no-op here."""
    if "risk_tier.value" not in coerced_values:
        return coerced_values
    computed = compute_risk_tier_settings(coerced_values["risk_tier.value"], anchors)
    result = dict(coerced_values)
    for factor_key, dotkey in RISK_TIER_DOTKEYS.items():
        result[dotkey] = computed[factor_key]
    return result


def build_risk_tier_position_note(tier_value: float, anchors: dict) -> str:
    """AICryptoTrading-specific (2026-08-21) -- this program has no flat
    starting_position_pct config field (position size is the AI's own per-report
    1-10% judgment, bounded only by max_loss_per_trade_pct), so the position-sizing
    row of the shared formula table is fed into the AI's prompt as guidance instead
    of a config write. Appended after build_risk_tier_prompt_section's own text, not
    a replacement for it."""
    ceiling = compute_risk_tier_settings(tier_value, anchors)["starting_position_pct"]
    return (
        f"Position sizes up to roughly {ceiling:.1f}% of the portfolio are reasonable "
        f"at this risk tier (still bounded by the max-loss-per-trade rule below).\n"
    )
