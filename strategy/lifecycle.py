# strategy/lifecycle.py

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from execution.notifier import TelegramNotifier
from strategy.account_state import account_state

def _tg_debug(msg: str) -> None:
    """Fire-and-forget debug message to Telegram. Never raises."""
    try:
        TelegramNotifier().debug(f"[LIFECYCLE] {msg}")
    except Exception:
        print(f"[TG DEBUG FALLBACK] {msg}")

def _update_atr(prev_atr, prev_close, high, low, close, period=14):
    tr = max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )
    alpha = 2 / (period + 1)
    return tr if prev_atr is None else prev_atr + alpha * (tr - prev_atr)

POSITIONS_DIR = "data/positions"
POSITIONS_FILE   = os.path.join(POSITIONS_DIR, "open_positions.json")
BAR_HISTORY_FILE = os.path.join(POSITIONS_DIR, "bar_history.json")
ENTRY_TS_FILE    = os.path.join(POSITIONS_DIR, "last_entry_ts.json")
EXECUTED_SIGNALS_FILE   = os.path.join(POSITIONS_DIR, "executed_signals.json")
REENTRY_LOCK_FILE = os.path.join(POSITIONS_DIR, "reentry_lock.json")

class PositionManager:

    # ==========================================================
    # TRADE LIFECYCLE SETTINGS (mirrors SignalBacktester exactly)
    # ==========================================================
    POSITION_VALUE_USDT = 10
    INCUBATION_BARS = 6        # 30 minutes (6×5m)
    VALIDATION_BARS = 18       # 90 minutes total
    PRESSURE_BARS   = 6        # stop proximity exit

    NO_FOLLOW_MFE  = 0.3       # 0.3R required after validation
    STOP_PROXIMITY = 0.2       # within 0.2R of stop = danger

    ATR_MULT         = 1.5
    ATR_AFTER_HALF_R = 2.0   # middle ground — protects without choking pullbacks
    ATR_AFTER_ONE_R  = 1.5   # tighter once 1R+ secured

    USE_ACCOUNT_GATES = False

    SIGNAL_EXPIRY_BARS      = 6    # replay: signal dies after 6×5m = 30 minutes
    SIGNAL_EXPIRY_BARS_LIVE = 12  # live: 60 minutes from when entry becomes valid

    def __init__(self, persist=True, notify=True):
        self.persist  = persist
        self.notify   = notify
        # proxy for live vs replay: replay uses persist=False, live uses persist=True
        self._is_live = persist
        self.positions = {}
        os.makedirs(POSITIONS_DIR, exist_ok=True)

        self.notifier = TelegramNotifier()

        if self.persist:
            self._load()
            if self.USE_ACCOUNT_GATES:
                account_state.open_positions = len(self.positions)
                account_state._save()

        # Prevent duplicate entries on same candle
        self._last_entry_ts = {}

        # 5m bar cache per symbol
        self._bar_history: dict[str, list] = {}

        # Streaming ATR state per symbol (FAST)
        self._atr_state = {}
        self._executed_signals = set()
        self._reentry_lock: dict[str, int] = {}
        self._reentry_lock_ts: dict[str, pd.Timestamp] = {}
        self._just_unlocked: set[str] = set()
        self._dirty = False

    # --------------------------------------------------
    # MAIN UPDATE  (called once per new 5M candle close)
    # --------------------------------------------------
    def update(
        self,
        df: pd.DataFrame,
        symbol: str,
        lltf_df: pd.DataFrame,
        external_signal: int,
        external_row: pd.Series,
        current_5m_row: pd.Series,
    ):

        signal = 0 if pd.isna(external_signal) else int(external_signal)
        position = self.positions.get(symbol)

        o = float(current_5m_row["open"])
        h = float(current_5m_row["high"])
        l = float(current_5m_row["low"])
        c = float(current_5m_row["close"])

        # =====================================================
        # ROLLING ATR (precomputed — O(1) lookup)
        # =====================================================
        atr = float(current_5m_row["ATR"]) if "ATR" in current_5m_row.index and not pd.isna(current_5m_row["ATR"]) else None

        current_ts = current_5m_row.name

        signal = 0 if pd.isna(external_signal) else int(external_signal)

        # -------------------------------------------------------
        # APPEND current 5M bar to history (for window checks)
        # -------------------------------------------------------
        if position:

            # APPEND BAR FIRST
            atr_5m = float(current_5m_row["ATR_5M"]) if "ATR_5M" in current_5m_row.index and not pd.isna(current_5m_row["ATR_5M"]) else None

            new_bar = {
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "ATR": atr,
                "ATR_5M": atr_5m,
                "ts": str(current_ts),
            }

            # Guard: if bar_history for this symbol is missing or empty after
            # _load(), rebuild it from lltf_df so OIE always has a real window.
            # This is the stateless-worker fix — each cron tick gets a fresh PM
            # instance, and if the bar history file failed to persist or load,
            # we reconstruct it from the already-available 5m data rather than
            # starting from scratch and condemning OIE to window-too-short forever.
            if symbol not in self._bar_history or len(self._bar_history[symbol]) == 0:
                if lltf_df is not None and not lltf_df.empty:
                    entry_ts = pd.Timestamp(position["entry_5m_ts"])
                    if entry_ts.tzinfo is None:
                        entry_ts = entry_ts.tz_localize("UTC")
                    history_df = lltf_df[lltf_df.index >= entry_ts].copy()
                    # exclude the current bar — we append it fresh below
                    history_df = history_df[history_df.index < current_ts]
                    rebuilt = []
                    for ts_h, row_h in history_df.iterrows():
                        rebuilt.append({
                            "open":   float(row_h["open"]),
                            "high":   float(row_h["high"]),
                            "low":    float(row_h["low"]),
                            "close":  float(row_h["close"]),
                            "ATR":    float(row_h["ATR"]) if "ATR" in row_h.index and not pd.isna(row_h["ATR"]) else atr,
                            "ATR_5M": float(row_h["ATR_5M"]) if "ATR_5M" in row_h.index and not pd.isna(row_h["ATR_5M"]) else None,
                            "ts":     str(ts_h),
                        })
                    self._bar_history[symbol] = rebuilt
                    print(f"[BAR HISTORY REBUILT] {symbol} — {len(rebuilt)} bars from lltf_df")

            self._bar_history.setdefault(symbol, []).append(new_bar)

            # KEEP WINDOW SIZE
            if len(self._bar_history[symbol]) > 200:
                self._bar_history[symbol] = self._bar_history[symbol][-200:]

            # DEFINE THIS EARLY (BEFORE USING IT)
            _entry_ts = pd.Timestamp(position["entry_5m_ts"])
            if _entry_ts.tzinfo is None:
                _entry_ts = _entry_ts.tz_localize("UTC")
            _current_ts_norm = current_ts if current_ts.tzinfo is not None else pd.Timestamp(current_ts).tz_localize("UTC")
            is_entry_candle = _entry_ts == _current_ts_norm

            if is_entry_candle:
                skip_exit_checks = True
            else:
                skip_exit_checks = False

            entry_price = position["entry_price"]
            stop        = position["stop_loss"]
            side        = position["direction"]
            R           = abs(entry_price - position["initial_stop"])  # always original risk

            if R == 0:
                R = 1e-9   # safety

            # Current price for R calculation (best price this bar)
            current_price = h if side == 1 else l

            # Trail anchored to MFE price (best price ever seen)
            if side == 1:
                if h - entry_price >= position["MFE"]:
                    position["mfe_price"] = h
                    position["last_mfe_bar"] = position["bars_in_trade"]
            else:
                if entry_price - l >= position["MFE"]:
                    position["mfe_price"] = l
                    position["last_mfe_bar"] = position["bars_in_trade"]

            trail_price = position.get("mfe_price", c)

            # =========================
            # TRUE MFE / MAE TRACKING
            # =========================

            if "MFE" not in position:
                position["MFE"] = 0.0
            if "MAE" not in position:
                position["MAE"] = 0.0

            # per-bar movement (NOT candle-close derived pnl)
            if side == 1:
                move_high = h - entry_price
                move_low  = l - entry_price

                position["MFE"] = max(position["MFE"], move_high)
                position["MAE"] = min(position["MAE"], move_low)

            else:
                move_high = entry_price - l
                move_low  = entry_price - h

                position["MFE"] = max(position["MFE"], move_high)
                position["MAE"] = min(position["MAE"], move_low)

            # -------------------------------------------------------
            # BUILD 5M WINDOW (REAL ATR HISTORY — BACKTEST PARITY)
            # -------------------------------------------------------
            window_5m = pd.DataFrame(self._bar_history[symbol])

            # convert to R
            position["mfe_r"] = position["MFE"] / R

            # ONLY for reporting (not exit logic anymore)
            if side == 1:
                position["pnl_r"] = (c - entry_price) / R
            else:
                position["pnl_r"] = (entry_price - c) / R

            exit_reason = None
            fill_price  = None

            def try_exit(reason, price=None):
                nonlocal exit_reason, fill_price
                if exit_reason is None:
                    exit_reason = reason
                    if price is not None:
                        fill_price = price

            # ===================================================
            # STAGE 2 — EXPANSION (>90m)
            # Backtest: opposite_impulse again (labeled momentum_decay)
            # ===================================================
            if not skip_exit_checks:
                atr_for_trail = atr_5m if atr_5m and not pd.isna(atr_5m) and atr_5m > 0 else (atr * 0.20 if atr else None)
                self._update_dynamic_stop(position, trail_price, atr_for_trail, side)

                # if self._stealth_distribution_exit(window_5m, position, side):
                #     try_exit("stealth_distrib", current_price)

                # if self._momentum_decay_exit(position):
                #     try_exit("momentum_decay", current_price)

                if self._opposite_impulse_exit(window_5m, side, position):
                    try_exit("opposite_impulse", current_price)

                # ===================================================
                # HARD STOP — always last
                # ===================================================
                if side == 1 and l <= position["stop_loss"]:
                    try_exit("stop_loss", position["stop_loss"])
                elif side == -1 and h >= position["stop_loss"]:
                    try_exit("stop_loss", position["stop_loss"])

                if exit_reason:
                    if fill_price is None:
                        fill_price = float(current_5m_row["close"])

                    closed = self._close(symbol, fill_price, current_ts, exit_reason)

                    return {"state": "CLOSED", "exit": closed}

            # increment real trade age AFTER exit checks (backtest parity)
            position["bars_in_trade"] += 1
            self._dirty = True
        
        # =====================================================
        # 🔓 REENTRY UNLOCK LOGIC
        # =====================================================
        # Clear grace flag from the previous bar before checking locks.
        self._just_unlocked.discard(symbol)

        if symbol in self._reentry_lock:
            locked_dir = self._reentry_lock[symbol]
            locked_at = self._reentry_lock_ts.get(symbol)

            # unlock only when a new 1H candle has formed since the stop
            current_1h = current_ts.floor("h") if hasattr(current_ts, "floor") else pd.Timestamp(current_ts).floor("h")
            locked_1h = locked_at.floor("h") if locked_at is not None else None

            new_candle_formed = locked_1h is not None and current_1h > locked_1h

            if new_candle_formed:
                self._reentry_lock.pop(symbol, None)
                self._just_unlocked.add(symbol)  # block entry this bar only
                self._dirty = True


        # ===================================================
        # SIGNAL EXPIRY CHECK
        # ===================================================
        if signal != 0 and not position:
            # expiry measured from the first 5m bar AFTER the signal 1H bar closes
            # signal_bar_end = signal 1H bar open + 1 hour = first valid execution bar
            signal_bar_end = external_row.name + pd.Timedelta(hours=1)
            signal_age_bars = len(
                lltf_df[(lltf_df.index >= signal_bar_end) & (lltf_df.index <= current_ts)]
            )
            expiry_limit = self.SIGNAL_EXPIRY_BARS_LIVE if self._is_live else self.SIGNAL_EXPIRY_BARS
            if signal_age_bars > expiry_limit:
                signal = 0
            
        # =====================================================
        # ENTRY LOGIC (MUST BE LAST STEP PER BAR)
        # This guarantees: EXIT and ENTRY cannot happen on same candle
        # =====================================================
        position = self.positions.get(symbol)

        if not position and signal != 0:

            if symbol in self._just_unlocked:
                _tg_debug(
                    f"[ENTRY BLOCKED — JUST UNLOCKED] {symbol} "
                    f"dir={signal} bar={current_ts}"
                )
                return {"state": "FLAT"}

            if symbol in self._just_unlocked:
                _tg_debug(
                    f"[ENTRY BLOCKED — JUST UNLOCKED] {symbol} "
                    f"dir={signal} bar={current_ts}"
                )
                return {"state": "FLAT"}

            if symbol in self._reentry_lock:
                locked_dir = self._reentry_lock[symbol]
                locked_at = self._reentry_lock_ts.get(symbol, "unknown")
                _tg_debug(
                    f"[REENTRY LOCK STATE] {symbol}\n"
                    f"locked_dir={locked_dir} current_signal={signal}\n"
                    f"locked_at={locked_at}\n"
                    f"would_block={signal == locked_dir}"
                )
                if signal == locked_dir:
                    _tg_debug(f"[ENTRY BLOCKED — REENTRY LOCK] {symbol} dir={signal} locked_at={locked_at}")
                    return {"state": "FLAT"}
                    # NOTE: no return here intentionally until we confirm fix is needed

            signal_ts = current_5m_row.name

            # latest_bar_ts = lltf_df.index[-1] -- LIVE ONLY
            # if signal_ts != latest_bar_ts:
            #     print(f"[ENTRY BLOCKED — NOT LATEST BAR] {symbol} signal_ts={signal_ts} latest={latest_bar_ts}")
            #     return {"state": "FLAT"}

            signal_id = (
                symbol + "|" +
                str(external_row.name) + "|" +
                str(signal)
            )

            if signal_id in self._executed_signals:
                _tg_debug(
                    f"[EXECUTED SIGNAL BLOCK] {symbol}\n"
                    f"signal_id={signal_id}\n"
                    f"all_executed_for_symbol={[s for s in self._executed_signals if symbol in s]}"
                )
                return {"state": "FLAT"}

            self._executed_signals.add(signal_id)
            _tg_debug(
                f"[EXECUTED SIGNAL REGISTERED] {symbol}\n"
                f"signal_id={signal_id}\n"
                f"total_executed={len(self._executed_signals)}"
            )

            # Use current 5m bar open as fill price.
            # external_row["open"] is the signal bar open — a price that
            # existed before the signal condition (close > resistance) was true.
            # current_5m_row["open"] is the first price available after confirmation.
            entry_price = float(current_5m_row["open"])

            if atr is None or atr <= 0 or np.isnan(atr):
                _tg_debug(
                    f"[WARN ATR INVALID] {symbol}\n"
                    f"ts={current_ts} atr={atr}\n"
                    f"ENTRY WILL BE BLOCKED"
                )
                atr = 0.000001

            atr = float(atr)

            _tg_debug(
                f"[PRE-OPEN] {symbol}\n"
                f"signal={signal} atr={atr}\n"
                f"current_ts={current_ts}\n"
                f"entry_price_raw={float(current_5m_row['open'])}\n"
                f"positions_before_open={list(self.positions.keys())}"
            )

            if atr <= 0:
                _tg_debug(f"[ENTRY BLOCKED — ATR] {symbol} @ {current_ts} atr={atr}")
                return {"state": "FLAT"}

            new_pos = self._open(symbol, signal, entry_price, current_ts, atr)

            if new_pos:
                # initialize bar history with entry candle
                self._bar_history[symbol] = [{
                    "open": o,
                    "high": h,
                    "low":  l,
                    "close": c,
                    "ATR": atr,
                    "ts": current_ts,
                }]

                return {"state": "OPEN", "position": new_pos}

        return {"state": "FLAT"}

    # --------------------------------------------------
    # LIFECYCLE HELPERS  (exact mirrors of backtest)
    # --------------------------------------------------
    def _momentum_decay_exit(self, position: dict) -> bool:

        mfe_r = position.get("mfe_r", 0.0)
        if mfe_r < 2.0:
            return False  # only protect meaningful winners

        bars = position.get("bars_in_trade", 0)

        # -------------------------------------------------
        # Track bars since last new MFE
        # -------------------------------------------------
        last_mfe_bar = position.get("last_mfe_bar")

        # If we never stored it yet, initialise it now
        if last_mfe_bar is None:
            position["last_mfe_bar"] = bars
            return False

        bars_since_mfe = bars - last_mfe_bar

        # -------------------------------------------------
        # Stall detection thresholds
        # -------------------------------------------------
        # After 2R → allow ~1 hour of stall
        # After 3R → allow less stall (trend should accelerate)
        if mfe_r > 3.0:
            stall_limit = 4    # 20 minutes
        elif mfe_r > 2.0:
            stall_limit = 6    # 30 minutes
        else:
            stall_limit = 8    # 40 minutes (rarely used)

        if bars_since_mfe >= stall_limit:
            return True

        return False

    def _stealth_distribution_exit(self, window: pd.DataFrame, position: dict, side: int) -> bool:
        # Only look for distribution if we are in solid profit
        if position.get("mfe_r", 0.0) < 2.0:
            return False

        LOOKBACK = 6 # 30 mins of quiet consolidation
        if len(window) < LOOKBACK:
            return False

        recent = window.iloc[-LOOKBACK:]
        mfe_price = position.get("mfe_price")
        
        if not mfe_price:
            return False

        # Use the most recent valid ATR
        atr = recent["ATR"].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            atr = window.iloc[0]["ATR"]

        # Stealth requires tight, quiet candles (compressing volatility)
        bodies = abs(recent["close"] - recent["open"])
        avg_body = bodies.mean()
        
        if avg_body > (atr * 0.6):
            return False # Too volatile to be "stealthy"

        if side == 1: # LONG
            # 1. Proximity: Hovering near the top (all recent closes within 1 ATR of MFE)
            near_high = (mfe_price - recent["close"]).max() < atr 
            # 2. Exhaustion: Failing to break the peak in the recent window
            failing_to_push = recent["high"].max() < mfe_price
            
            return near_high and failing_to_push

        else: # SHORT (Stealth Accumulation near lows)
            # 1. Proximity: Hovering near the bottom
            near_low = (recent["close"] - mfe_price).max() < atr
            # 2. Exhaustion: Failing to break the low
            failing_to_push = recent["low"].min() > mfe_price

            return near_low and failing_to_push

    def _opposite_impulse_exit(self, window: pd.DataFrame, side: int, position: dict) -> bool:
        if len(window) < 3:
            _tg_debug(f"[OIE] SKIP — window too short ({len(window)})")
            return False

        last = window.iloc[-1]

        # ══════════════════════════════════════════
        # 1. ATR
        # ══════════════════════════════════════════
        if "ATR_5M" in window.columns:
            atr = window["ATR_5M"].iloc[-3:].mean()
            if pd.isna(atr) or atr <= 0:
                atr = window["ATR_5M"].iloc[0]
        else:
            atr = None

        if atr is None or pd.isna(atr) or atr <= 0:
            atr_1h = window["ATR"].iloc[-3:].mean()
            if pd.isna(atr_1h) or atr_1h <= 0:
                atr_1h = window["ATR"].iloc[0]
            if pd.isna(atr_1h) or atr_1h <= 0:
                _tg_debug(f"[OIE] SKIP — no valid ATR (window_len={len(window)})")
                return False
            atr = atr_1h * 0.20

        # ══════════════════════════════════════════
        # 2. BODY SIZE
        # ══════════════════════════════════════════
        body = abs(last["close"] - last["open"])
        big_candle = body > atr * 1.2

        # ══════════════════════════════════════════
        # 3. DIRECTION CHECK
        # ══════════════════════════════════════════
        if side == 1:
            wrong_direction = last["close"] < last["open"]
        else:
            wrong_direction = last["close"] > last["open"]

        # ══════════════════════════════════════════
        # 4. CLOSE LOCATION
        # ══════════════════════════════════════════
        entry = position["entry_price"]
        stop  = position["stop_loss"]
        R     = abs(entry - position["initial_stop"])

        if R == 0:
            close_to_stop_r = 0.0
            location_blocked = False
        else:
            if side == 1:
                close_to_stop_r = (last["close"] - stop) / R
            else:
                close_to_stop_r = (stop - last["close"]) / R
            mfe_r = position.get("mfe_r", 0.0)
            location_blocked = close_to_stop_r > 1.5 and mfe_r < 1.0

        # ══════════════════════════════════════════
        # 5. VOLUME CONFIRMATION
        # ══════════════════════════════════════════
        vol_blocked = False
        avg_vol = float("nan")
        last_vol = last.get("volume", float("nan")) if hasattr(last, "get") else last["volume"]
        if "volume" in window.columns:
            avg_vol = window["volume"].iloc[-10:].mean()
            last_vol = last["volume"]
            if len(window) >= 10 and not pd.isna(avg_vol) and avg_vol > 0:
                if last_vol < avg_vol * 0.8:
                    vol_blocked = True

        # ══════════════════════════════════════════
        # LOG EVERY BAR
        # ══════════════════════════════════════════
        fired = big_candle and wrong_direction and not location_blocked and not vol_blocked
        symbol = position.get("symbol", "?")
        bars   = position.get("bars_in_trade", "?")
        mfe_r  = position.get("mfe_r", 0.0)

        _tg_debug(
            f"[OIE] {symbol} bar={bars} | {'🔥FIRED' if fired else 'miss'}\n"
            f"o={last['open']:.6f} c={last['close']:.6f} | "
            f"body={body:.6f} atr={atr:.6f} thr={atr*1.2:.6f} big={big_candle}\n"
            f"wrong_dir={wrong_direction} | "
            f"csr={close_to_stop_r:.3f} loc_block={location_blocked} mfe_r={mfe_r:.3f}\n"
            f"vol: last={last_vol:.0f} avg={avg_vol:.0f} wlen={len(window)} vol_block={vol_blocked}"
        )

        return fired

    def _stop_pressure_exit(
        self, window: pd.DataFrame, stop_price: float, side: int, R: float
    ) -> bool:
        if len(window) < self.PRESSURE_BARS:
            return False
        recent = window.iloc[-self.PRESSURE_BARS:]
        if side == 1:
            dist = recent["close"] - stop_price
        else:
            dist = stop_price - recent["close"]
        return (dist <= R * self.STOP_PROXIMITY).all()

    def _no_follow_through_exit(self, mfe_r: float, bars_5m: int) -> bool:
        if bars_5m < self.VALIDATION_BARS:
            return False
        return mfe_r < self.NO_FOLLOW_MFE

    def _update_dynamic_stop(self, position, current_price, atr, side):
        mfe_r = position["mfe_r"]

        if mfe_r <= 0.5:
            return

        bars = position.get("bars_in_trade", 0)
        last_trail_bar = position.get("last_trail_bar", 0)
        if bars - last_trail_bar < 3:
            return
        position["last_trail_bar"] = bars

        entry = position["entry_price"]
        current_stop = position["stop_loss"]

        # Use tighter multiplier only after 2R secured
        atr_mult = self.ATR_AFTER_ONE_R if mfe_r > 2.0 else self.ATR_AFTER_HALF_R

        if side == 1:
            trail_candidate = current_price - atr * atr_mult
            if mfe_r > 2.0:
                trail_candidate = max(trail_candidate, entry)
            # only move stop if trail candidate is profitable
            if trail_candidate <= entry:
                return
            new_stop = max(current_stop, trail_candidate)
        else:
            trail_candidate = current_price + atr * atr_mult
            if mfe_r > 2.0:
                trail_candidate = min(trail_candidate, entry)
            # only move stop if trail candidate is profitable
            if trail_candidate >= entry:
                return
            new_stop = min(current_stop, trail_candidate)

        if new_stop != current_stop:
            position["stop_loss"] = new_stop
            position["trailing_activated"] = True

    # --------------------------------------------------
    # OPEN / CLOSE
    # --------------------------------------------------
    def _open(self, symbol, direction, price, ts, atr):
        stop = (
            price - self.ATR_MULT * atr if direction == 1
            else price + self.ATR_MULT * atr
        )

        position_value = self.POSITION_VALUE_USDT
        stop_dist = abs(price - stop)
        stop_pct = stop_dist / price if price > 0 else 0
        risk_usd = round(position_value * stop_pct, 4)
        quantity = round(position_value / price, 4) if price > 0 else 0

        position = {
            "symbol":        symbol,
            "trade_id":      TelegramNotifier.make_trade_id(symbol),
            "direction":     direction,
            "entry_price":   price,
            "entry_time":    ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "entry_5m_ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "stop_loss":     stop,
            "initial_stop":  stop,
            "R": abs(price - stop),
            "state":         "OPEN",
            "mfe_r": 0.0,
            "pnl_r": 0.0,
            "MAE": 0.0,
            "MFE": 0.0,
            # ── new ──
            "risk_usd":      risk_usd,
            "quantity":      quantity,
            "position_value": position_value,
            "bars_in_trade": 0,
            "last_mfe_bar": 0,
            "last_trail_bar": 0,
        }

        self.positions[symbol] = position
        if self.persist:
            self._save()

        if self.USE_ACCOUNT_GATES:
            account_state.on_position_open()

        if self.notify:
            self.notifier.notify_open(
                symbol=symbol,
                direction=direction,
                entry_price=price,
                stop_loss=stop,
                timestamp=ts,
                trade_id=position["trade_id"],
                risk_usd=risk_usd,
                quantity=quantity,
                position_value=position_value
            )

        print(f"[OPEN EXECUTED] {symbol} dir={direction} price={price}")
        _tg_debug(
            f"[OPEN EXECUTED] {symbol}\n"
            f"dir={direction} price={price}\n"
            f"stop={stop} atr={atr}\n"
            f"risk_usd={risk_usd} qty={quantity}\n"
            f"ts={ts}"
        )

        self._dirty = True

        return position.copy()

    def _close(self, symbol, price, ts, reason):
        if price is None:
            raise ValueError(f"[_close] fill_price is None for {symbol}, reason={reason}")
        
        pos = self.positions.pop(symbol)
        duration_bars = pos.get("bars_in_trade", 0)
        
        self._reentry_lock[symbol] = pos["direction"]
        # Use the bar's timestamp, not wall clock — critical for replay correctness
        self._reentry_lock_ts[symbol] = pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
        if self._reentry_lock_ts[symbol].tzinfo is None:
            self._reentry_lock_ts[symbol] = self._reentry_lock_ts[symbol].tz_localize("UTC")
        self._bar_history.pop(symbol, None)
        self._last_entry_ts.pop(symbol, None)
        
        direction = pos["direction"]
        entry     = pos["entry_price"]
        qty       = pos.get("quantity", 0.0)  # Grab the quantity

        if reason == "stop_loss":
            theoretical_stop = pos["stop_loss"]
            if price is not None:
                if pos["direction"] == 1:
                    fill_price = min(price, theoretical_stop)
                else:
                    fill_price = max(price, theoretical_stop)
            else:
                fill_price = theoretical_stop
        else:
            fill_price = price

        # 1. Price Difference
        price_diff = (fill_price - entry) if direction == 1 else (entry - fill_price)
        
        # 2. Actual Dollar PnL (Price Diff * Quantity)
        pnl_usd = price_diff * qty

        # 3. R-Multiple
        R = pos.get("R", None)
        pnl_r = price_diff / R if R and R > 0 else 0.0

        pos["state"] = "CLOSED"
        pos["exit"]  = {
            "price":   fill_price,
            "time":    ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "reason":  reason,
            "pnl":     pnl_usd,  # <-- Pass the Dollar amount!
            "pnl_r":   round(pnl_r, 3),
            "duration": duration_bars, 
        }

        # Update account balance with actual dollars, not price cents
        if self.USE_ACCOUNT_GATES:
            account_state.on_position_close(pnl_usd)

        if self.notify:
            self.notifier.notify_close(
                symbol=symbol,
                direction=direction,
                exit_price=price,
                timestamp=ts,
                reason=reason,
                pnl_r=pnl_r,
                trade_id=pos["trade_id"],
                trailing_activated=pos.get("trailing_activated", False),
                risk_usd=pos.get("risk_usd", 0),
                entry_time=pos.get("entry_time"),
            )

        if self.persist:
            self._save()

        self._dirty = True

        print(
            f"[CLOSE EXECUTED] {symbol} | "
            f"Reason: {reason} | "
            f"Price: {fill_price} | "
            f"PnL $: {round(pnl_usd, 3)} | "
            f"PnL R: {round(pnl_r, 3)} | "
            f"Duration: {duration_bars} bars"
        )

        return pos.copy()

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self.positions
    
    def flush(self):
        """Persist state to disk only if something changed."""
        if not self.persist:
            return
        if not self._dirty:
            return

        self._save()
        self._dirty = False

    # --------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------
    def _load(self):
        # POSITIONS
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r") as f:
                    content = f.read().strip()
                self.positions = json.loads(content) if content else {}
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted positions file — starting fresh")
                self.positions = {}

        # BAR HISTORY
        if os.path.exists(BAR_HISTORY_FILE):
            try:
                with open(BAR_HISTORY_FILE, "r") as f:
                    content = f.read().strip()
                raw = json.loads(content) if content else {}
                # re-cast ts strings back to pd.Timestamp (lost on JSON round-trip)
                for sym, bars in raw.items():
                    for bar in bars:
                        if "ts" in bar and isinstance(bar["ts"], str):
                            try:
                                bar["ts"] = pd.Timestamp(bar["ts"])
                            except Exception:
                                pass
                self._bar_history = raw
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted bar history file — starting fresh")
                self._bar_history = {}

        # ENTRY TIMESTAMPS
        if os.path.exists(ENTRY_TS_FILE):
            try:
                with open(ENTRY_TS_FILE, "r") as f:
                    content = f.read().strip()
                raw = json.loads(content) if content else {}
                self._last_entry_ts = {k: pd.Timestamp(v) for k, v in raw.items()}
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted entry timestamps — starting fresh")
                self._last_entry_ts = {}

        # EXECUTED SIGNALS
        if os.path.exists(EXECUTED_SIGNALS_FILE):
            try:
                with open(EXECUTED_SIGNALS_FILE, "r") as f:
                    content = f.read().strip()
                loaded = set(json.loads(content)) if content else set()
                # FIX: 2-hour window is enough to prevent duplicates within a session
                # 48 hours was creating phantom blocks that persisted across restarts
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
                kept = set()
                for s in loaded:
                    try:
                        parts = s.split("|")
                        if len(parts) < 2:
                            continue
                        ts = pd.Timestamp(parts[1])
                        if ts.tzinfo is None:
                            ts = ts.tz_localize("UTC")
                        if ts >= cutoff:
                            kept.add(s)
                    except Exception:
                        pass  # drop malformed entries silently
                self._executed_signals = kept
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted executed signals — starting fresh")
                self._executed_signals = set()

        # REENTRY LOCK
        if os.path.exists(REENTRY_LOCK_FILE):
            try:
                with open(REENTRY_LOCK_FILE, "r") as f:
                    content = f.read().strip()
                raw = json.loads(content) if content else {}
                # convert values to int, drop entries older than 48h
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
                self._reentry_lock = {}
                self._reentry_lock_ts = {}
                for k, v in raw.items():
                    if isinstance(v, dict):
                        locked_at = pd.Timestamp(v.get("locked_at", "2000-01-01"), tz="UTC")
                        if locked_at >= cutoff:
                            self._reentry_lock[k] = int(v["direction"])
                            self._reentry_lock_ts[k] = locked_at
                    else:
                        # legacy format — drop it, no timestamp to validate
                        pass
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted reentry lock — starting fresh")
                self._reentry_lock = {}
                self._reentry_lock_ts = {}

    def _save(self):
        if not self.persist:
            return
        with open(POSITIONS_FILE + ".tmp", "w") as f:
            json.dump(self.positions, f, indent=2, default=str)
        os.replace(POSITIONS_FILE + ".tmp", POSITIONS_FILE)

        with open(BAR_HISTORY_FILE + ".tmp", "w") as f:
            json.dump(self._bar_history, f, indent=2, default=str)
        os.replace(BAR_HISTORY_FILE + ".tmp", BAR_HISTORY_FILE)

        with open(ENTRY_TS_FILE + ".tmp", "w") as f:
            json.dump(
                {k: v.isoformat() for k, v in self._last_entry_ts.items()},
                f, indent=2
            )
        os.replace(ENTRY_TS_FILE + ".tmp", ENTRY_TS_FILE)

        # REENTRY LOCK
        lock_payload = {
            k: {
                "direction": v,
                "locked_at": self._reentry_lock_ts.get(k, pd.Timestamp.now(tz="UTC")).isoformat()
            }
            for k, v in self._reentry_lock.items()
        }
        with open(REENTRY_LOCK_FILE + ".tmp", "w") as f:
            json.dump(lock_payload, f, indent=2)
        os.replace(REENTRY_LOCK_FILE + ".tmp", REENTRY_LOCK_FILE)

        # EXECUTED SIGNALS
        with open(EXECUTED_SIGNALS_FILE + ".tmp", "w") as f:
            json.dump(list(self._executed_signals), f, indent=2)
        os.replace(EXECUTED_SIGNALS_FILE + ".tmp", EXECUTED_SIGNALS_FILE)