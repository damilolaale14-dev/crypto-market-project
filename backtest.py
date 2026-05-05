import pandas as pd
import numpy as np

class SignalBacktester:
    def __init__(
        self,
        df,
        htf_df=None,
        lltf_df=None,
        initial_balance=1000,
        fixed_risk_per_trade=10.0,
        fee=0.0005,
        atr_period=14,
        atr_mult=1.5,
        be_trigger_r=1.2,
        trailing=False,
        leverage=1,
    ):
        self.df = df.copy()
        self.htf_df = htf_df.copy() if htf_df is not None else None

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fixed_risk = fixed_risk_per_trade
        self.fee = fee

        self.position = 0
        self.entry_price = None
        self.units = 0
        self.stop_loss = None
        self.trailing_stop = None

        self.be_activated = False
        self.be_trigger_r = be_trigger_r

        self.trades = []

        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.trailing = trailing

        self.leverage = max(1, leverage)
        self.max_bars_in_trade = 6          # ~6 hours max edge lifespan
        self.expansion_lookback = 3         # detect shrinking expansion
        self.trap_wick_ratio = 0.6          # wick dominance threshold
        self.trap_close_ratio = 0.3         # weak close threshold

        # ==========================================
        # 1H SIGNAL LOCK (prevents revenge entries)
        # ==========================================
        self.current_ltf_index = None      # tracks active 1H candle
        self.trade_taken_this_ltf = False  # did we already trade this 1H idea?

        # -------------------------------------------------
        # Align datasets to common start date
        # -------------------------------------------------
        if lltf_df is not None:
            self.lltf_df = lltf_df.copy()

            # Find common start timestamp across TFs
            start_time = max(self.df.index[0], self.lltf_df.index[0])

            # Trim BOTH datasets so they start together
            self.df = self.df[self.df.index >= start_time]
            self.lltf_df = self.lltf_df[self.lltf_df.index >= start_time]

            # Reset indices after trimming
            self.df = self.df.copy()
            self.lltf_df = self.lltf_df.copy()

            # -------------------------------------------------
            # Map every 5m candle to its parent 1h candle index
            # -------------------------------------------------
            self.lltf_df['final_signal'] = np.nan
            self.lltf_df['ltf_index'] = np.nan

            ltf_times = self.df.index

            for i in range(len(ltf_times)):
                start = ltf_times[i]
                end = ltf_times[i+1] if i+1 < len(ltf_times) else self.lltf_df.index[-1] + pd.Timedelta(seconds=1)

                mask = (self.lltf_df.index >= start) & (self.lltf_df.index < end)

                self.lltf_df.loc[mask, 'final_signal'] = self.df['final_signal'].iloc[i]
                self.lltf_df.loc[mask, 'ltf_index'] = i

            # Drop any candles that STILL didn't get mapped (safety)
            self.lltf_df = self.lltf_df.dropna(subset=['ltf_index'])

            # Now conversion is safe
            self.lltf_df['ltf_index'] = self.lltf_df['ltf_index'].astype(int)

        self._prepare_indicators()

    # ------------------------
    # Indicators
    # ------------------------
    def _prepare_indicators(self):
        df = self.df
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(self.atr_period).mean()
        self.df = df

        # compute 5m ATR on lltf_df for opposite impulse exit
        if hasattr(self, 'lltf_df') and self.lltf_df is not None:
            lltf_tr = pd.concat([
                self.lltf_df['high'] - self.lltf_df['low'],
                (self.lltf_df['high'] - self.lltf_df['close'].shift()).abs(),
                (self.lltf_df['low'] - self.lltf_df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            self.lltf_df['ATR_5M'] = lltf_tr.ewm(span=self.atr_period, adjust=False).mean()

    # ==========================================================
    # TRADE LIFECYCLE SETTINGS
    # ==========================================================

    INCUBATION_BARS = 6        # 30 minutes (6×5m)
    VALIDATION_BARS = 18       # 90 minutes total
    PRESSURE_BARS = 6          # stop proximity exit

    NO_FOLLOW_MFE = 0.3        # 0.3R required after validation
    STOP_PROXIMITY = 0.2       # within 0.2R of stop = danger

    ATR_INIT_MULT = 1.5
    ATR_AFTER_HALF_R = 1.0

    def get_5m_window(self, entry_time, current_time):
        df = self.lltf_df if hasattr(self, 'lltf_df') else self.df
        return df.loc[entry_time:current_time]
    
    def opposite_impulse_exit(self, window, side, trade=None):
        if len(window) < 3:
            return False

        last = window.iloc[-1]

        # ══════════════════════════════════════════
        # 1. ATR — use 5m ATR for candle body comparison
        # ══════════════════════════════════════════
        if "ATR_5M" in window.columns:
            atr = window["ATR_5M"].iloc[-3:].mean()
            if pd.isna(atr) or atr <= 0:
                atr = window["ATR_5M"].iloc[0]
        else:
            atr = None

        if atr is None or pd.isna(atr) or atr <= 0:
            atr_1h = window["ATR"].iloc[-3:].mean() if "ATR" in window.columns else (window['high'] - window['low']).mean()
            atr = atr_1h * 0.20

        if pd.isna(atr) or atr <= 0:
            return False

        # ══════════════════════════════════════════
        # 2. BODY SIZE
        # ══════════════════════════════════════════
        body = abs(last.close - last.open)
        big_candle = body > atr * 1.2

        if not big_candle:
            return False

        # ══════════════════════════════════════════
        # 3. DIRECTION CHECK
        # ══════════════════════════════════════════
        if side == 1:
            wrong_direction = last.close < last.open
        else:
            wrong_direction = last.close > last.open

        if not wrong_direction:
            return False

        # ══════════════════════════════════════════
        # 4. CLOSE LOCATION
        # ══════════════════════════════════════════
        if trade is not None:
            entry = trade["entry_price"]
            stop  = trade["stop_loss"]
            R     = abs(entry - trade.get("ATR", abs(entry - stop)))

            if R > 0:
                if side == 1:
                    close_to_stop_r = (last.close - stop) / R
                else:
                    close_to_stop_r = (stop - last.close) / R

                if close_to_stop_r > 1.5:
                    return False

        # ══════════════════════════════════════════
        # 5. VOLUME CONFIRMATION
        # ══════════════════════════════════════════
        if "volume" in window.columns:
            avg_vol = window["volume"].iloc[-10:].mean()
            last_vol = last.volume
            if len(window) >= 10 and not pd.isna(avg_vol) and avg_vol > 0:
                if last_vol < avg_vol * 0.8:
                    return False

        return True
        
    def stop_pressure_exit(self, window, stop_price, side):
        if len(window) < self.PRESSURE_BARS:
            return False

        recent = window.iloc[-self.PRESSURE_BARS:]
        if side == 1:
            dist = (recent.close - stop_price)
        else:
            dist = (stop_price - recent.close)

        return (dist <= self.R * self.STOP_PROXIMITY).all()
    
    def no_follow_through_exit(self, mfe_r, bars_in_trade):
        if bars_in_trade < self.VALIDATION_BARS:
            return False
        return mfe_r < self.NO_FOLLOW_MFE
    
    def update_dynamic_stop(self, trade, current_price, atr):
        r_multiple = trade['pnl_r']

        # tighten after 0.5R
        if r_multiple > 0.5:
            if trade['side'] == 1:
                trade['stop_loss'] = max(trade['stop_loss'], current_price - atr * self.ATR_AFTER_HALF_R)
            else:
                trade['stop_loss'] = min(trade['stop_loss'], current_price + atr * self.ATR_AFTER_HALF_R)

        # breakeven after 1R
        if r_multiple > 1.0:
            if trade['side'] == 1:
                trade['stop_loss'] = max(trade['stop_loss'], trade['entry_price'])
            else:
                trade['stop_loss'] = min(trade['stop_loss'], trade['entry_price'])

    # ------------------------
    # Position sizing
    # ------------------------
    def _calc_units(self, entry, stop):
        risk_per_unit = abs(entry - stop)
        if risk_per_unit == 0:
            return 0
        return self.fixed_risk / risk_per_unit

    # ------------------------
    # Entry
    # ------------------------
    def _enter(self, side, price, idx, align_to_ltf_open=False):
        ltf_idx = self.lltf_df['ltf_index'].iloc[idx] if hasattr(self, 'lltf_df') else idx
        atr = self.df['ATR'].iloc[ltf_idx]
        if np.isnan(atr):
            return

        # Align entry price to parent 1h candle open
        if align_to_ltf_open and hasattr(self, 'df'):
            price = self.df['open'].iloc[ltf_idx]

        if side == 1:
            stop = price - self.atr_mult * atr
        else:
            stop = price + self.atr_mult * atr

        units = self._calc_units(price, stop)
        if units <= 0:
            return

        self.position = side
        self.entry_price = price
        self.stop_loss = stop
        self.trailing_stop = stop
        self.units = units
        self.be_activated = False

        self.current_trade = {
            "side": side,
            "entry_idx": idx,
            "entry_time": self.lltf_df.index[idx] if hasattr(self, 'lltf_df') else self.df.index[idx],
            "entry_price": price,
            "units": units,
            "stop_loss": stop,
            "ATR": atr,
            "MAE": 0.0,
            "MFE": 0.0
        }

        # Fee applies to notional, not margin — so leverage increases fee cost
        self.balance -= abs(units * price) * self.fee
        
        # Liquidation price tracking
        margin_per_unit = price / self.leverage
        if side == 1:
            self.liquidation_price = price - margin_per_unit * 0.9
        else:
            self.liquidation_price = price + margin_per_unit * 0.9

    # ------------------------
    # Exit
    # ------------------------
    def _exit(self, price, idx, reason):
        raw_pnl = (
            (price - self.entry_price) * self.units
            if self.position == 1 else
            (self.entry_price - price) * self.units
        )

        # Stop loss and liquidation are capped at -1R by definition
        # All other exits (winners, early exits) are amplified by leverage
        if reason in ("stop_loss", "break_even", "liquidated"):
            pnl = raw_pnl
        else:
            pnl = raw_pnl * self.leverage

        if reason == "stop_loss" and self.be_activated:
            reason = "break_even"

        self.balance += pnl
        self.balance -= abs(self.units * price) * self.fee * self.leverage
        self.liquidation_price = None

        entry_i = self.current_trade["entry_idx"]
        bars_held = idx - entry_i

        exit_time = self.lltf_df.index[idx] if hasattr(self, 'lltf_df') else self.df.index[idx]
        entry_time = self.current_trade["entry_time"]

        hours_held = (exit_time - entry_time).total_seconds() / 3600

        self.current_trade.update({
            "exit_price": price,
            "exit_idx": idx,
            "exit_time": exit_time,
            "bars_held": bars_held,
            "hours_held": hours_held,
            "pnl": pnl,
            "exit_reason": reason
        })
        self.trades.append(self.current_trade)

        self.position = 0
        self.entry_price = None
        self.units = 0
        self.be_activated = False
        self.trade_taken_this_ltf = True

    # ------------------------
    # Excursion tracking
    # ------------------------
    def _update_excursions(self, high, low):
        if self.position == 0:
            return

        if self.position == 1:
            self.current_trade["MAE"] = min(self.current_trade["MAE"], (low  - self.entry_price) * self.units)
            self.current_trade["MFE"] = max(self.current_trade["MFE"], (high - self.entry_price) * self.units)
        else:
            self.current_trade["MAE"] = min(self.current_trade["MAE"], (self.entry_price - high) * self.units)
            self.current_trade["MFE"] = max(self.current_trade["MFE"], (self.entry_price - low)  * self.units)
    
    def _exec_df(self):
        return self.lltf_df if hasattr(self, 'lltf_df') else self.df
        
    # ------------------------
    # Candle anatomy
    # ------------------------
    def _upper_wick(self, i):
        df = self._exec_df()
        row = df.iloc[i]
        return row['high'] - max(row['open'], row['close'])

    def _lower_wick(self, i):
        df = self._exec_df()
        row = df.iloc[i]
        return min(row['open'], row['close']) - row['low']

    def _body_size(self, i):
        df = self._exec_df()
        row = df.iloc[i]
        return abs(row['close'] - row['open'])
    
    # ------------------------
    # Expansion Failure Exit
    # ------------------------
    def _momentum_decay_exit(self, i):
        if self.position == 0:
            return False

        ltf_idx = self.lltf_df['ltf_index'].iloc[i] if hasattr(self, 'lltf_df') else i
        row = self.df.iloc[ltf_idx]

        # trend energy collapsing
        continuation = row['CONTINUATION_STRENGTH']
        velocity     = row['CONTINUATION_VELOCITY']
        stability    = row['STATE_STABILITY']

        if np.isnan(continuation):
            return False

        # core idea:
        # expansion strength is fading + regime stability dropping
        energy_decay = (
            (continuation < 0) or
            (velocity < -0.15)
        )

        regime_breakdown = stability < 0.35

        return energy_decay and regime_breakdown
        
    # ------------------------
    # Trap / Absorption Exit
    # ------------------------
    def _liquidity_reversal_exit(self, i):
        if self.position == 0:
            return False

        ltf_idx = self.lltf_df['ltf_index'].iloc[i] if hasattr(self, 'lltf_df') else i
        row = self.df.iloc[ltf_idx]

        flow = row['FLOW_STRENGTH']
        pressure = row['COMPOSITE_PRESSURE']

        if self.position == 1:
            return (flow < -0.5) and (pressure < 0)

        else:
            return (flow > 0.5) and (pressure > 0)
        
    # ------------------------
    # Time Decay Exit
    # ------------------------
    def _structural_exhaustion_exit(self, i):
        if self.position == 0:
            return False

        ltf_idx = self.lltf_df['ltf_index'].iloc[i] if hasattr(self, 'lltf_df') else i
        row = self.df.iloc[ltf_idx]

        trend_quality = row['TREND_QUALITY']
        transition    = row['TRANSITION_FORCE']

        if np.isnan(trend_quality):
            return False

        # strong trend suddenly enters transition regime
        return (trend_quality > 0.8) and (transition > 1.5)
    
    def _is_new_ltf_candle(self, i):
        if not hasattr(self, 'lltf_df'):
            return True
        if i == 0:
            return False
        return self.lltf_df['ltf_index'].iloc[i] != self.lltf_df['ltf_index'].iloc[i-1]

    # ------------------------
    # Intrabar management
    # ------------------------
    def _check_intrabar(self, high, low, idx):
        if self.position == 0:
            return

        # -------------------------------------------------
        # 1️⃣ Update excursions (5m MFE/MAE tracking)
        # -------------------------------------------------
        if self.position == 1:
            self.current_trade["MAE"] = min(self.current_trade["MAE"], (low  - self.entry_price) * self.units)
            self.current_trade["MFE"] = max(self.current_trade["MFE"], (high - self.entry_price) * self.units)
            current_price = high
        else:
            self.current_trade["MAE"] = min(self.current_trade["MAE"], (self.entry_price - high) * self.units)
            self.current_trade["MFE"] = max(self.current_trade["MFE"], (self.entry_price - low)  * self.units)
            current_price = low

        R_multiple = self.current_trade["MFE"] / self.fixed_risk
        bars_in_trade = idx - self.current_trade["entry_idx"]

        # ==========================================================
        # REAL TRADE LIFECYCLE ENGINE (5m)
        # ==========================================================

        trade = self.current_trade
        exec_df = self.lltf_df if hasattr(self, 'lltf_df') else self.df
        current_time = exec_df.index[idx]
        current_price = high if self.position == 1 else low

        # R is the original risk in price terms
        self.R = abs(trade["entry_price"] - trade["stop_loss"])

        # Get 5m window since entry
        window_5m = self.get_5m_window(trade['entry_time'], current_time)
        bars_5m = len(window_5m)

        # Calculate pnl in R
        if trade['side'] == 1:
            pnl_r = (current_price - trade['entry_price']) / self.R
        else:
            pnl_r = (trade['entry_price'] - current_price) / self.R

        # store CURRENT R
        trade['pnl_r'] = pnl_r

        # store BEST R
        trade['mfe_r'] = max(trade.get('mfe_r', -999), pnl_r)

        # =============================
        # STAGE 0 — INCUBATION (0–30m)
        # =============================
        if bars_5m <= self.INCUBATION_BARS:
            if self.opposite_impulse_exit(window_5m, trade['side'], trade=self.current_trade):
                self._exit(current_price, idx, "momentum_decay")
                return

        # =============================
        # STAGE 1 — VALIDATION (30–90m)
        # =============================
        elif bars_5m <= self.VALIDATION_BARS:

            if self.stop_pressure_exit(window_5m, trade['stop_loss'], trade['side']):
                self._exit(current_price, idx, "stop_pressure")
                return

            if self.no_follow_through_exit(trade['mfe_r'], bars_5m):
                self._exit(current_price, idx, "no_follow_through")
                return

        # =============================
        # STAGE 2 — EXPANSION (>90m)
        # =============================
        else:
            atr = window_5m['high'].sub(window_5m['low']).rolling(14).mean().iloc[-1]
            self.update_dynamic_stop(trade, current_price, atr)

            if self.opposite_impulse_exit(window_5m, trade['side']):
                self._exit(current_price, idx, "momentum_decay")
                return

        # =============================
        # LIQUIDATION CHECK (leverage)
        # must come before hard stop —
        # liquidation can fire closer to
        # entry than the SL at high leverage
        # =============================
        if self.leverage > 1 and hasattr(self, 'liquidation_price') and self.liquidation_price is not None:
            if self.position == 1 and low <= self.liquidation_price:
                self._exit(self.liquidation_price, idx, "liquidated")
                return
            elif self.position == -1 and high >= self.liquidation_price:
                self._exit(self.liquidation_price, idx, "liquidated")
                return

        # =============================
        # HARD STOP ALWAYS LAST
        # =============================
        if self.position == 1:
            if low <= self.stop_loss:
                self._exit(self.stop_loss, idx, "stop_loss")
        else:
            if high >= self.stop_loss:
                self._exit(self.stop_loss, idx, "stop_loss")

    # ------------------------
    # Run backtest
    # ------------------------
    def run(self):
        df_5m = self.lltf_df if hasattr(self, 'lltf_df') else self.df
        equity = []
        timestamps = []

        for i in range(len(df_5m) - 1):
            ltf_idx = df_5m['ltf_index'].iloc[i] if hasattr(self, 'lltf_df') else i

            if self.current_ltf_index is None:
                self.current_ltf_index = ltf_idx

            if ltf_idx != self.current_ltf_index:
                # New 1H candle = new trading opportunity
                self.current_ltf_index = ltf_idx
                self.trade_taken_this_ltf = self.position != 0  # ← fix

            signal = df_5m['final_signal'].iloc[i]

            o = df_5m['open'].iloc[i]
            h = df_5m['high'].iloc[i]
            l = df_5m['low'].iloc[i]

            # 2. ENTER (only one trade allowed per 1H candle)
            if self.position == 0 and not self.trade_taken_this_ltf:
                if signal == 1:
                    self._enter(1, o, i, align_to_ltf_open=True)
                elif signal == -1:
                    self._enter(-1, o, i, align_to_ltf_open=True)

            # 3. Update excursions
            if self.position != 0:
                self._update_excursions(h, l)

            # 4. Intrabar exits (trap, expansion failure, time decay, stops)
            if self.position != 0 and (i - self.current_trade["entry_idx"]) >= 1:
                self._check_intrabar(h, l, i)

            equity.append(self.balance)
            timestamps.append(df_5m.index[i])

        if self.position != 0:
            self._exit(df_5m['close'].iloc[-1], len(df_5m) - 1, "end_of_data")

        equity_df = pd.DataFrame({
            "timestamp": timestamps,
            "equity":    equity
        }).set_index("timestamp")

        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df["direction"] = trades_df["side"].map({1: "LONG", -1: "SHORT"})
            trades_df["pnl_pct"]   = trades_df["pnl"] / self.initial_balance * 100

        liquidations = len(trades_df[trades_df["exit_reason"] == "liquidated"]) if not trades_df.empty else 0

        summary = {
            "initial_balance": self.initial_balance,
            "final_balance":   round(self.balance, 2),
            "net_profit":      round(self.balance - self.initial_balance, 2),
            "return_pct":      round((self.balance / self.initial_balance - 1) * 100, 2),
            "total_trades":    len(trades_df),
            "win_rate":        round((trades_df["pnl"] > 0).mean() * 100, 2) if not trades_df.empty else 0.0,
            "avg_win":         trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() if not trades_df.empty else 0.0,
            "avg_loss":        trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean() if not trades_df.empty else 0.0,
            "leverage":        self.leverage,
            "liquidations":    liquidations,
        }

        return {
            "summary":      summary,
            "equity_curve": equity_df,
            "trades":       trades_df
        }