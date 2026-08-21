"""Portfolio state and tracking with SQLite persistence.

Ported from AITrading's Portfolio/Position. Two real differences:

- **No wash-sale tracking** (`recent_losses`, `_load_recent_losses`) -- dropped
  per the design spec, since the wash-sale rule doesn't apply to crypto under
  current IRS guidance.
- **No column-migration cruft.** AITrading's version carries a long history of
  `ALTER TABLE ... ADD COLUMN` migrations because its schema evolved under a
  live, already-running database over many months. This project starts fresh,
  so the schema below just includes every column from day one -- nothing to
  migrate.

Everything else (Position's P&L properties, the graduated take-profit/
trailing-stop fields, trade_id lot-linkage, day P/L snapshotting) is
asset-class-agnostic and carries over unchanged.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


def _format_trade_reason(reason: str | None) -> str:
    """Python port of formatSellReason() in dashboard.html -- kept in sync manually.
    Falls back to the raw string unchanged for anything not recognized."""
    if not reason:
        return "Not recorded"

    def frac(n: str) -> str:
        return "1/3" if n == "1" else "2/3"

    m = re.match(r"^Take-Profit T(\d)$", reason)
    if m:
        return f"{frac(m.group(1))} — Take-Profit (T{m.group(1)})"

    m = re.match(r"^Take-Profit T(\d) \(estimated\)$", reason)
    if m:
        return f"{frac(m.group(1))} — Take-Profit (T{m.group(1)}, estimated)"

    if reason == "Take-Profit (final tranche)":
        return "Final 1/3 — Take-Profit"

    if reason == "UNRECONCILED FILL (real order not found — needs manual review)":
        return "Unreconciled Fill — real order not found, needs manual review"

    if reason == "Stop Loss (gap-through market sell, full close)":
        return "Full Close — Stop Loss (price gapped through, sold at market)"
    if reason == "Stop Loss (gap-through market sell)":
        return "Partial — Stop Loss (price gapped through, sold at market)"

    if reason == "Stop loss hit":
        return "Full Close — Stop Loss"
    if reason == "Trailing stop hit":
        return "Full Close — Trailing Stop"
    if reason == "Manual sell":
        return "Full Close — Manual Sell"

    return reason


def _format_trade_history_summary(ticker: str, rows: list[tuple]) -> str:
    """Formats raw trade_history rows for one asset into a compact, chronological
    summary for inclusion in a Claude prompt as minor supplementary context. Each
    row is (ticker, action, shares, price, pnl, timestamp, reason). Returns an
    empty string for an asset with no trade history at all -- callers must treat
    that as "omit this section entirely", not display filler text."""
    if not rows:
        return ""
    lines = [f"Prior trading history on {ticker}:"]
    for _ticker, action, shares, price, pnl, timestamp, reason in rows:
        date = timestamp.split("T")[0] if timestamp else "unknown date"
        if action == "BUY":
            lines.append(f"- {date}: Bought at ${price:.2f}")
            continue
        reason_str = _format_trade_reason(reason)
        pnl_str = ""
        if pnl is not None:
            sign = "+" if pnl >= 0 else "-"
            pnl_str = f", P&L {sign}${abs(pnl):.2f}"
            if shares:
                implied_entry = price - (pnl / shares)
                if implied_entry > 0:
                    pnl_pct = (pnl / (shares * implied_entry)) * 100
                    pct_sign = "+" if pnl_pct >= 0 else ""
                    pnl_str += f" ({pct_sign}{pnl_pct:.1f}%)"
        lines.append(f"- {date}: Sold ({reason_str}) at ${price:.2f}{pnl_str}")
    return "\n".join(lines)


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit_targets: list[float]
    sector: str
    opened_at: datetime
    trailing_stop: float | None = None
    day_open_price: float | None = None
    final_tranche_start_price: float | None = None
    realized_pnl: float = 0.0
    shares_sold: float = 0.0
    t1_target_price: float | None = None
    t2_target_price: float | None = None
    profit_target_hit: bool = False
    trade_id: str | None = None
    final_trail_pct: float | None = None

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

    @property
    def lifetime_pnl(self) -> float:
        """Total gain/loss since this position was first opened: unrealized gain on
        whatever shares remain, PLUS every partial exit's real realized gain along
        the way."""
        return self.unrealized_pnl + self.realized_pnl

    @property
    def lifetime_pnl_pct(self) -> float:
        original_cost = (self.shares + self.shares_sold) * self.entry_price
        if original_cost == 0:
            return 0.0
        return (self.lifetime_pnl / original_cost) * 100

    @property
    def day_pnl(self) -> float:
        ref = self.day_open_price if self.day_open_price is not None else self.entry_price
        return (self.current_price - ref) * self.shares

    @property
    def day_pnl_pct(self) -> float:
        ref = self.day_open_price if self.day_open_price is not None else self.entry_price
        if not ref:
            return 0.0
        return ((self.current_price - ref) / ref) * 100


class Portfolio:
    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config["portfolio"]["initial_capital"]
        self.cash = self.initial_capital
        self.positions: dict[str, Position] = {}
        self.peak_value = self.initial_capital
        self.day_start_value = self.initial_capital
        self.day_start_date: str | None = None
        # Set while a rotation swap's sell is in flight (submitted but not yet filled) —
        # during this window the sold position and its replacement can be briefly
        # double-counted in total_value, so update_peak() must not ratchet on it.
        self._rotation_in_progress = False
        self.db_path = config.get("database", {}).get("path", "data/aicryptotrading.db")
        self._db: aiosqlite.Connection | None = None

    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def cash_pct(self) -> float:
        if self.total_value == 0:
            return 0.0
        return (self.cash / self.total_value) * 100

    @property
    def day_pnl(self) -> float:
        return self.total_value - self.day_start_value

    async def initialize(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                ticker TEXT PRIMARY KEY,
                shares REAL,
                entry_price REAL,
                current_price REAL,
                stop_loss REAL,
                take_profit_targets TEXT,
                sector TEXT,
                opened_at TEXT,
                trailing_stop REAL,
                day_open_price REAL,
                final_tranche_start_price REAL,
                realized_pnl REAL,
                shares_sold REAL,
                t1_target_price REAL,
                t2_target_price REAL,
                profit_target_hit INTEGER,
                trade_id TEXT,
                final_trail_pct REAL
            )
        """)
        # final_trail_pct was added after this project's initial launch (2026-08-15,
        # AI-judged trailing-stop feature) -- CREATE TABLE IF NOT EXISTS above only
        # covers a brand-new DB, so an already-live one (real paper positions since
        # launch day) needs this one-time ALTER. try/except is the standard SQLite
        # idiom for "add column if missing" -- no IF NOT EXISTS support for ADD COLUMN.
        try:
            await self._db.execute("ALTER TABLE positions ADD COLUMN final_trail_pct REAL")
            await self._db.commit()
        except Exception:
            pass  # column already exists
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cash REAL,
                peak_value REAL,
                day_start_value REAL,
                day_start_date TEXT
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                action TEXT,
                shares REAL,
                price REAL,
                pnl REAL,
                timestamp TEXT,
                reason TEXT,
                trade_id TEXT
            )
        """)
        await self._db.commit()

        await self._load_state()

    async def _load_state(self):
        async with self._db.execute(
            "SELECT cash, peak_value, day_start_value, day_start_date FROM portfolio_state WHERE id = 1"
        ) as cur:
            row = await cur.fetchone()
            if row:
                self.cash = row[0]
                self.peak_value = row[1]
                self.day_start_value = row[2]
                self.day_start_date = row[3] if len(row) > 3 else None

        async with self._db.execute("SELECT * FROM positions") as cur:
            async for row in cur:
                targets = json.loads(row[5]) if row[5] else []
                self.positions[row[0]] = Position(
                    ticker=row[0],
                    shares=row[1],
                    entry_price=row[2],
                    current_price=row[3],
                    stop_loss=row[4],
                    take_profit_targets=targets,
                    sector=row[6],
                    opened_at=datetime.fromisoformat(row[7]),
                    trailing_stop=row[8],
                    day_open_price=row[9] if len(row) > 9 else None,
                    final_tranche_start_price=row[10] if len(row) > 10 else None,
                    realized_pnl=row[11] if len(row) > 11 and row[11] is not None else 0.0,
                    shares_sold=row[12] if len(row) > 12 and row[12] is not None else 0.0,
                    t1_target_price=row[13] if len(row) > 13 else None,
                    t2_target_price=row[14] if len(row) > 14 else None,
                    profit_target_hit=bool(row[15]) if len(row) > 15 and row[15] is not None else False,
                    trade_id=row[16] if len(row) > 16 else None,
                    final_trail_pct=row[17] if len(row) > 17 else None,
                )

    async def _save_state(self):
        if not self._db:
            return
        await self._db.execute(
            "INSERT OR REPLACE INTO portfolio_state (id, cash, peak_value, day_start_value, day_start_date) VALUES (1, ?, ?, ?, ?)",
            (self.cash, self.peak_value, self.day_start_value, self.day_start_date),
        )
        await self._db.commit()

    async def _save_position(self, position: Position):
        if not self._db:
            return
        await self._db.execute(
            "INSERT OR REPLACE INTO positions (ticker, shares, entry_price, current_price, stop_loss, take_profit_targets, sector, opened_at, trailing_stop, day_open_price, final_tranche_start_price, realized_pnl, shares_sold, t1_target_price, t2_target_price, profit_target_hit, trade_id, final_trail_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                position.ticker, position.shares, position.entry_price,
                position.current_price, position.stop_loss,
                json.dumps(position.take_profit_targets), position.sector,
                position.opened_at.isoformat(), position.trailing_stop,
                position.day_open_price, position.final_tranche_start_price,
                position.realized_pnl, position.shares_sold,
                position.t1_target_price, position.t2_target_price,
                int(position.profit_target_hit), position.trade_id,
                position.final_trail_pct,
            ),
        )
        await self._db.commit()

    async def _remove_position_db(self, ticker: str):
        if not self._db:
            return
        await self._db.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
        await self._db.commit()

    async def get_trade_history_summary(self, ticker: str) -> str:
        """Every trade_history row for this asset, formatted as compact context for a
        Claude prompt when reconsidering it as a buy candidate. Fails open (returns
        "") on any DB error -- must never delay or break a real buy decision."""
        if not self._db:
            return ""
        try:
            async with self._db.execute(
                "SELECT ticker, action, shares, price, pnl, timestamp, reason "
                "FROM trade_history WHERE ticker = ? ORDER BY timestamp",
                (ticker,),
            ) as cur:
                rows = await cur.fetchall()
        except Exception as e:
            logger.warning("get_trade_history_summary failed for %s: %s", ticker, e)
            return ""
        return _format_trade_history_summary(ticker, rows)

    async def get_recent_sells(self, limit: int = 30) -> list[dict]:
        """Most-recent-first SELL rows from trade_history, for the dashboard's Recent
        Sells panel. Fails open (returns []) on any DB error -- this is display-only,
        must never block anything else."""
        if not self._db:
            return []
        try:
            async with self._db.execute(
                "SELECT ticker, shares, price, pnl, timestamp, reason "
                "FROM trade_history WHERE action = 'SELL' ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        except Exception as e:
            logger.warning("get_recent_sells failed: %s", e)
            return []
        return [
            {
                "ticker": r[0], "shares": r[1], "price": r[2],
                "pnl": r[3], "timestamp": r[4], "reason": r[5],
            }
            for r in rows
        ]

    def update_peak(self):
        if self._rotation_in_progress:
            return
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value

    def new_trading_day(self, today_str: str | None = None):
        """Snapshots the "start of today" baseline both portfolio-wide (day_start_value)
        and per-position (day_open_price). Crypto trades continuously, so "today" is
        just a plain calendar-day boundary (ET, for consistency with AITrading's own
        convention) -- not tied to any market session."""
        self.day_start_value = self.total_value
        if today_str is not None:
            self.day_start_date = today_str
        for pos in self.positions.values():
            pos.day_open_price = pos.current_price

    def add_position(self, position: Position):
        self.positions[position.ticker] = position
        self.cash -= position.cost_basis

    def close_position(self, ticker: str) -> float:
        position = self.positions.pop(ticker, None)
        if position is None:
            return 0.0
        self.cash += position.market_value
        self.update_peak()
        return position.unrealized_pnl

    async def add_position_async(self, position: Position):
        self.add_position(position)
        await self._save_position(position)
        await self._save_state()

    async def close_position_async(
        self, ticker: str, *, exit_shares: float = None, exit_price: float = None,
        reason: str = "",
    ) -> float:
        """Real, confirmed bug fixed 2026-08-21: this branch's own comment claimed
        "cash was already credited per-fill in check_take_profits," but neither of
        this function's two REAL callers with both exit_shares/exit_price set
        (OrderManager._handle_apparent_close, for a stop-loss/apparent close;
        OrderManager.execute_sell, for a manual dashboard sell) ever credits cash
        beforehand -- check_take_profits doesn't even call this function at all; it
        credits cash and closes the position entirely inline, itself. Every position
        closed via either real caller silently never had its sale proceeds added to
        cash. Confirmed live: LINK/USD's final tranche closed via
        _handle_apparent_close (an unreconciled fill, ~$390.44 in real proceeds) with
        the PNL correctly recorded in trade_history for display, but the cash itself
        never credited -- local cash read ~$391 below Alpaca's real cash, the root
        cause behind a real "-261"-magnitude Day P/L distortion the owner reported
        twice (see docs/CLAUDE_HISTORY.md's 2026-08-21 entry for the full incident,
        including the separate, also-real day_start_value/day_start_date bug found
        alongside this one)."""
        pos = self.positions.get(ticker)
        _exit_shares = exit_shares if exit_shares is not None else (pos.shares if pos else 0)
        _exit_price = exit_price if exit_price is not None else (pos.current_price if pos else 0)
        _trade_id = pos.trade_id if pos else None
        if exit_shares is not None and exit_price is not None:
            position = self.positions.pop(ticker, None)
            if position:
                self.cash += _exit_price * _exit_shares
                self.update_peak()
            pnl = (_exit_price - position.entry_price) * _exit_shares if position else 0
        else:
            pnl = self.close_position(ticker)
        await self._remove_position_db(ticker)
        await self._save_state()
        if self._db:
            await self._db.execute(
                "INSERT INTO trade_history (ticker, action, shares, price, pnl, timestamp, reason, trade_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, "SELL", _exit_shares, _exit_price, pnl, datetime.now().isoformat(), reason, _trade_id),
            )
            await self._db.commit()
        return pnl
