"""Risk management and position sizing.

Ported from AITrading's RiskManager -- cash reserve, daily loss limit, and
drawdown checks are fully generic and carry over unchanged. Two real
differences for Phase 1, per the design spec:

- **No wash-sale cooldown.** Under current IRS guidance, IRC §1091 (the wash
  sale rule) applies to "securities," and crypto has historically not been
  treated as a security for this purpose -- so the rebuy-block mechanism
  AITrading relies on doesn't apply the same way here. This is based on
  current law, not tax advice; confirm with a real CPA before this system
  ever touches real money.
- **No sector concentration check.** Crypto doesn't have a clean equivalent
  to GICS sectors, and a 20-25-asset universe with per-position sizing caps
  already provides reasonable diversification without one. Can be added back
  later (e.g. a Layer-1/DeFi/stablecoin-adjacent categorization) if real
  trading shows concentration risk worth gating on.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.research.engine import ResearchReport
    from src.decision.portfolio import Portfolio

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: dict):
        rm = config["risk_management"]
        self.max_position_pct = rm["max_position_pct"] / 100
        self.max_loss_per_trade_pct = rm["max_loss_per_trade_pct"] / 100
        self.min_cash_reserve_pct = rm["min_cash_reserve_pct"] / 100
        self.daily_loss_limit_pct = rm["daily_loss_limit_pct"] / 100
        self.drawdown_halt_pct = rm["drawdown_halt_pct"] / 100
        self.drawdown_defensive_pct = rm["drawdown_defensive_pct"] / 100
        self.drawdown_exit_review_pct = rm["drawdown_exit_review_pct"] / 100
        self.max_positions = max(1, config.get("portfolio", {}).get("max_positions", 10))

    def calculate_position_size(self, entry_price: float, stop_loss: float, portfolio_value: float) -> float:
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return 0.0

        max_risk_dollars = portfolio_value * self.max_loss_per_trade_pct
        shares_by_risk = max_risk_dollars / risk_per_share
        size_by_risk = shares_by_risk * entry_price

        # Cap per-position so filling all slots still respects cash reserve
        derived_pct = (1.0 - self.min_cash_reserve_pct) / self.max_positions
        max_position = portfolio_value * min(self.max_position_pct, derived_pct)
        return min(size_by_risk, max_position)

    def check_cash_reserve(self, portfolio: Portfolio, order_cost: float) -> bool:
        remaining_cash = portfolio.cash - order_cost
        return remaining_cash >= portfolio.total_value * self.min_cash_reserve_pct

    def check_drawdown(self, portfolio: Portfolio) -> str:
        if portfolio.peak_value == 0:
            return "normal"
        drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
        if drawdown >= self.drawdown_exit_review_pct:
            return "exit_review"
        if drawdown >= self.drawdown_defensive_pct:
            return "defensive"
        if drawdown >= self.drawdown_halt_pct:
            return "halt"
        return "normal"

    def check_daily_loss(self, portfolio: Portfolio) -> bool:
        if portfolio.total_value == 0:
            return False
        if portfolio.day_start_value == 0:
            return True  # no measurable loss yet; allow trading
        daily_loss = (portfolio.day_start_value - portfolio.total_value) / portfolio.day_start_value
        return daily_loss < self.daily_loss_limit_pct

    def check_all_rules(self, report: ResearchReport, portfolio: Portfolio) -> bool:
        order_cost = self.calculate_position_size(report.entry_price, report.stop_loss, portfolio.total_value)

        if not self.check_cash_reserve(portfolio, order_cost):
            remaining = portfolio.cash - order_cost
            required = portfolio.total_value * self.min_cash_reserve_pct
            logger.info("  %s RULE FAIL: cash reserve — need $%.0f, would have $%.0f after $%.0f order (cash: $%.0f)",
                        report.ticker, required, remaining, order_cost, portfolio.cash)
            return False
        if not self.check_daily_loss(portfolio):
            logger.info("  %s RULE FAIL: daily loss limit exceeded", report.ticker)
            return False
        drawdown_status = self.check_drawdown(portfolio)
        if drawdown_status != "normal":
            logger.info("  %s RULE FAIL: drawdown status = %s", report.ticker, drawdown_status)
            return False
        return True
