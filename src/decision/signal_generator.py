"""Generates buy/sell/hold signals from research reports.

Ported from AITrading's SignalGenerator near-verbatim -- this is pure
conviction/risk-reward gate math with nothing stock-specific in it, so it
carries over unchanged.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from src.research.engine import ResearchEngine, ResearchReport, Signal
from src.decision.risk_manager import RiskManager
from src.decision.portfolio import Portfolio

logger = logging.getLogger(__name__)


def _required_rr(conviction: float, min_conviction: float, base_rr: float, step: float, floor: float) -> float:
    """Conviction-scaled R/R threshold: a flat min_risk_reward_ratio applied
    identically to every asset doesn't account for how much more confident
    Claude's thesis is; each conviction point above the minimum earns a small
    reduction in the required bar, floored so it never gets too lenient."""
    extra_conviction = max(0, conviction - min_conviction)
    return max(floor, base_rr - extra_conviction * step)


def compute_rr(entry_price: float, stop_loss: float, fair_value_estimate: float | None) -> float | None:
    """Raw reward/risk ratio from a report's own entry/stop/fair-value numbers, or
    None when it isn't meaningfully computable (no real risk, no usable fair value,
    or no positive reward) -- the single shared source both the real buy gate
    (_evaluate_report) and the dashboard's display-only R/R figure read from, so
    they can't drift apart the way AITrading's own R/R math once did across
    multiple call sites."""
    risk = entry_price - stop_loss
    if risk <= 0:
        return None
    if not fair_value_estimate or fair_value_estimate <= 0:
        return None
    reward = fair_value_estimate - entry_price
    if reward <= 0:
        return None
    return reward / risk


@dataclass
class TradeSignal:
    ticker: str
    signal: Signal
    conviction: int
    entry_price: float
    stop_loss: float
    take_profit_targets: list[float]
    position_size_pct: float
    position_size_dollars: float
    shares: float
    reasoning: str
    research_report: ResearchReport
    generated_at: datetime
    sector: str = ""
    should_execute: bool = False
    trade_id: str | None = None
    final_trail_pct: float | None = None
    # The exact R/R and its required gate at the moment THIS signal cleared to buy
    # (2026-08-21) -- not recomputed later, so Position.buy_rr/buy_required_rr (see
    # that field's own docstring) reflect the real numbers that allowed the purchase,
    # not a reconstruction from possibly-since-changed data.
    rr: float | None = None
    required_rr: float | None = None


class SignalGenerator:
    def __init__(
        self,
        config: dict,
        research_engine: ResearchEngine,
        risk_manager: RiskManager,
        portfolio: Portfolio,
    ):
        self.config = config
        self.research_engine = research_engine
        self.risk_manager = risk_manager
        self.portfolio = portfolio

    def _evaluate_report(self, report: ResearchReport) -> TradeSignal | None:
        min_conviction = self.config["research"]["min_conviction_score"]
        if report.conviction_score < min_conviction:
            logger.info("  %s REJECTED: conviction %d < %d", report.ticker, report.conviction_score, min_conviction)
            return None

        rr = compute_rr(report.entry_price, report.stop_loss, report.fair_value_estimate)
        if rr is None:
            logger.info(
                "  %s REJECTED: risk/reward not computable (entry $%.2f, stop $%.2f, fair_value $%.2f)",
                report.ticker, report.entry_price, report.stop_loss, report.fair_value_estimate or 0.0,
            )
            return None
        research_cfg = self.config["research"]
        required_rr = _required_rr(
            report.conviction_score, min_conviction,
            research_cfg["min_risk_reward_ratio"],
            research_cfg.get("rr_conviction_step", 0.1),
            research_cfg.get("rr_floor", 1.5),
        )
        if rr < required_rr:
            logger.info("  %s REJECTED: R/R %.2f < %.2f (fair_value=$%.2f, entry=$%.2f, stop=$%.2f)", report.ticker, rr, required_rr, report.fair_value_estimate, report.entry_price, report.stop_loss)
            return None

        position_size = self.risk_manager.calculate_position_size(
            report.entry_price, report.stop_loss, self.portfolio.total_value
        )

        if not self.risk_manager.check_all_rules(report, self.portfolio):
            logger.info("  %s REJECTED: failed risk_manager.check_all_rules", report.ticker)
            return None

        # Use fractional units — position_size_dollars is the notional amount
        shares = position_size / report.entry_price if report.entry_price > 0 else 0
        if shares < 0.0001:
            logger.info("  %s REJECTED: position size too small ($%.2f)", report.ticker, position_size)
            return None

        return TradeSignal(
            ticker=report.ticker,
            signal=report.signal,
            conviction=report.conviction_score,
            entry_price=report.entry_price,
            stop_loss=report.stop_loss,
            take_profit_targets=report.take_profit_targets,
            position_size_pct=report.position_size_pct,
            position_size_dollars=position_size,
            shares=shares,
            reasoning=report.reasoning,
            research_report=report,
            generated_at=datetime.now(),
            sector=getattr(report, 'sector', ''),
            should_execute=report.signal in (Signal.STRONG_BUY, Signal.BUY),
            final_trail_pct=report.final_trail_pct,
            rr=rr, required_rr=required_rr,
        )
