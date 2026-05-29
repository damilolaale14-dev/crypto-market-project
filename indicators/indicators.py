import pandas as pd
import numpy as np

# ==========================================================
# CORE UTILITIES
# ==========================================================
def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def atr_ema(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    return 100 - (100 / (1 + rs))

# ==========================================================
# TREND CONTEXT
# ==========================================================
def positioning_pressure(df, fast=10, slow=50):
    """
    Replaces trend_bias() / TREND_QUALITY.

    Philosophy: stop asking "is this trend smooth?"
    Start asking "is positioning becoming unstable?"

    POSITIONING_PRESSURE measures whether one side of the
    market is accumulating stress faster than price resolves it.
    This is durable across regimes because it is tied to
    human forced-behavior mechanics, not trend elegance.

    Outputs:
        TREND_QUALITY       — drop-in replacement, same name,
                              now means positioning pressure
                              rather than slope * R2
        POSITION_STRESS     — raw unsigned instability magnitude
        POSITION_DIRECTION  — signed bias (+1 bull / -1 bear)
        POSITION_DIVERGENCE — flow vs price disagreement
                              (precursor to reversals)
    """

    # --------------------------------------------------
    # 1. SIGNED DELTA FLOW (who is getting trapped)
    # Price move per unit of volume participation.
    # Negative = volume not confirming price = trapped longs.
    # Positive = volume confirming = pressure building.
    # --------------------------------------------------
    price_move   = df['close'].diff()
    dollar_flow  = df['volume'] * (df['close'] - df['open'])

    # Normalize flow by its own slow EWMA to make it regime-independent
    flow_baseline = dollar_flow.ewm(span=200, adjust=False).mean()
    flow_norm     = dollar_flow / (flow_baseline.abs() + 1e-9)

    # --------------------------------------------------
    # 2. PRICE RESPONSE ELASTICITY
    # How much price moves per unit of flow.
    # Falling elasticity = absorption = trapped side building.
    # --------------------------------------------------
    flow_ewm_fast = dollar_flow.ewm(span=fast, adjust=False).mean()
    flow_ewm_slow = dollar_flow.ewm(span=slow, adjust=False).mean()

    price_ewm_fast = price_move.ewm(span=fast, adjust=False).mean()
    price_ewm_slow = price_move.ewm(span=slow, adjust=False).mean()

    # elasticity: price response per unit flow, fast vs slow
    elast_fast = price_ewm_fast / (flow_ewm_fast.abs() + 1e-9)
    elast_slow = price_ewm_slow / (flow_ewm_slow.abs() + 1e-9)

    # Elasticity collapse = one side absorbing without price movement
    # = positioning stress building
    df['ELAST_RATIO'] = elast_fast / (elast_slow.abs() + 1e-9)

    # --------------------------------------------------
    # 3. FLOW DIVERGENCE FROM PRICE
    # Flow trending one way while price moves another
    # = inventory imbalance building = positioning instability
    # --------------------------------------------------
    flow_direction  = np.sign(flow_ewm_fast)
    price_direction = np.sign(price_ewm_fast)

    df['POSITION_DIVERGENCE'] = (flow_direction != price_direction).astype(float)

    # Smooth divergence signal
    df['POSITION_DIVERGENCE'] = df['POSITION_DIVERGENCE'].ewm(span=5, adjust=False).mean()

    # --------------------------------------------------
    # 4. POSITIONING STRESS (unsigned instability magnitude)
    # Combines:
    #   - flow imbalance magnitude
    #   - elasticity collapse signal
    #   - flow/price divergence
    # --------------------------------------------------
    flow_imbalance = (flow_ewm_fast - flow_ewm_slow).abs()
    flow_imbalance_norm = hybrid_zscore(flow_imbalance).clip(0, 3) / 3

    elast_collapse = (1 - df['ELAST_RATIO'].clip(0, 1))  # 1 = fully collapsed

    df['POSITION_STRESS'] = (
        0.5 * flow_imbalance_norm +
        0.3 * elast_collapse +
        0.2 * df['POSITION_DIVERGENCE']
    ).ewm(span=3, adjust=False).mean()

    # --------------------------------------------------
    # 5. DIRECTION OF STRESS
    # Which side is under pressure?
    # +1 = bears trapped (bullish pressure)
    # -1 = bulls trapped (bearish pressure)
    # --------------------------------------------------
    df['POSITION_DIRECTION'] = np.sign(flow_ewm_fast) * df['POSITION_STRESS']

    # --------------------------------------------------
    # 6. TREND_QUALITY — drop-in replacement
    # Same column name so nothing downstream breaks.
    # Now means: signed positioning pressure
    # rather than slope * R2.
    # Range stays roughly -1 to +1 after normalization.
    # --------------------------------------------------
    raw = df['POSITION_DIRECTION']
    df['TREND_QUALITY'] = hybrid_zscore(raw).clip(-2, 2) / 2

    return df

# ==========================================================
# WICK ANALYSIS
# ==========================================================
def wick_rejection(df):
    body = (df['close'] - df['open']).abs()
    upper = df['high'] - df[['close', 'open']].max(axis=1)
    lower = df[['close', 'open']].min(axis=1) - df['low']

    df['UPPER_WICK_RATIO'] = upper / (body + 1e-9)
    df['LOWER_WICK_RATIO'] = lower / (body + 1e-9)

    return df

# ==========================================================
# VOLUME CONFIRMATION
# ==========================================================
def volume_confirmation(df, lookback=20):
    df['VOL_MA'] = df['volume'].rolling(lookback).mean()
    df['VOL_RATIO'] = df['volume'] / (df['VOL_MA'] + 1e-9)
    return df

# ==========================================================
# SUPPORT / RESISTANCE
# ==========================================================
def support_resistance(df, lookback=20):
    df['RESISTANCE'] = df['high'].rolling(lookback).max()
    df['SUPPORT'] = df['low'].rolling(lookback).min()
    return df

# ==========================================================
# BREAKOUT LOGIC
# ==========================================================
def liquidity_displacement(df, vol_lookback=20, accel_threshold=1.4):
    """
    Replaces breakout_logic() / BREAK_RESISTANCE / BREAK_SUPPORT.

    Philosophy: stop asking "did price break structure?"
    Start asking "did price displace liquidity violently enough
    to imply inventory imbalance?"

    A real displacement has three simultaneous properties:
        1. Price cleared structure (still required, not removed)
        2. The move ACCELERATED through the level (not crept)
        3. Volume spiked asymmetrically AT the displacement bar
           (not before, not after — at)

    Creeping through resistance = absorption = trap.
    Accelerating through resistance = displacement = real.

    Outputs:
        BREAK_RESISTANCE    — drop-in, same name
        BREAK_SUPPORT       — drop-in, same name
        DISPLACEMENT_SCORE  — continuous 0-1 quality of the event
        ABSORBED_LONG       — price cleared but was absorbed (bearish)
        ABSORBED_SHORT      — price cleared but was absorbed (bullish)
    """

    if 'ATR' not in df.columns:
        df['ATR'] = atr_ema(df)

    resistance = df['RESISTANCE'].shift(1)
    support    = df['SUPPORT'].shift(1)

    # --------------------------------------------------
    # 1. PRICE CLEARED STRUCTURE (baseline, same as before)
    # Kept intentionally — displacement requires structure
    # breach. ATR buffer unchanged from original.
    # --------------------------------------------------
    cleared_resistance = df['close'] > (resistance + 0.5 * df['ATR'])
    cleared_support    = df['close'] < (support    - 0.5 * df['ATR'])

    # --------------------------------------------------
    # 2. MOVE ACCELERATION
    # Measured over a 3-bar window around the break.
    # Real displacements don't always peak on the exact
    # structure bar — the energy can lead or trail by 1-2 bars.
    # We look at the peak candle size in a [-1, 0, +0] window
    # (causal: current and prior bar only, no lookahead).
    # --------------------------------------------------
    candle_size     = (df['high'] - df['low'])
    avg_candle_size = candle_size.ewm(span=vol_lookback, adjust=False).mean()
    candle_accel    = candle_size / (avg_candle_size + 1e-9)

    # Peak acceleration within a 2-bar causal window
    # This catches the bar before the break AND the break bar itself
    accel_window    = candle_accel.rolling(2).max()
    acceleration_ok = accel_window > accel_threshold

    # --------------------------------------------------
    # 3. VOLUME ASYMMETRY
    # Volume spike within a 3-bar causal window.
    # Institutional accumulation often builds across
    # multiple bars before structure gives way.
    # --------------------------------------------------
    vol_baseline       = df['volume'].ewm(span=vol_lookback, adjust=False).mean()
    vol_spike          = df['volume'] / (vol_baseline + 1e-9)
    vol_spike_baseline = vol_spike.ewm(span=500, adjust=False).mean()

    # Peak volume within a 3-bar causal window
    vol_window    = vol_spike.rolling(3).max()
    vol_displaced = vol_window > (vol_spike_baseline * 1.1)

    # --------------------------------------------------
    # 4. DISPLACEMENT SCORE (continuous quality 0→1)
    # Uses windowed values so score reflects the event
    # quality, not just the exact bar's raw numbers.
    # --------------------------------------------------
    accel_norm = ((accel_window - 1).clip(0, 3) / 3)
    vol_norm   = ((vol_window - 1).clip(0, 3) / 3)

    df['DISPLACEMENT_SCORE'] = (
        0.5 * accel_norm +
        0.5 * vol_norm
    ).ewm(span=2, adjust=False).mean()

    # --------------------------------------------------
    # 5. ABSORPTION DETECTION
    # Structure cleared but BOTH acceleration AND volume
    # were weak across the entire window = absorbed.
    # --------------------------------------------------
    df['ABSORBED_LONG']  = cleared_resistance & ~acceleration_ok & ~vol_displaced
    df['ABSORBED_SHORT'] = cleared_support    & ~acceleration_ok & ~vol_displaced

    # --------------------------------------------------
    # 6. FINAL OUTPUTS
    # Structure + at least ONE of the two energy conditions
    # within the window. Requiring both simultaneously was
    # the source of the over-restriction — real displacement
    # events rarely peak all three signals on the same bar.
    # Using OR here preserves the philosophy (energy required)
    # without demanding simultaneity that markets don't honor.
    # --------------------------------------------------
    df['BREAK_RESISTANCE'] = cleared_resistance & (acceleration_ok | vol_displaced)
    df['BREAK_SUPPORT']    = cleared_support    & (acceleration_ok | vol_displaced)

    return df

# ==========================================================
# VOLATILITY EXPANSION PHYSICS (REPLACES ATR PERCENTILE)
# ==========================================================
def volatility_expansion(df, fast=14, slow=50):
    """
    Volatility Expansion Ratio (VER)
    Measures if volatility is expanding or contracting RIGHT NOW.

    fast ATR reacts quickly
    slow ATR defines background regime
    """

    # Fast and slow volatility
    df['ATR_FAST'] = atr_ema(df, fast)
    df['ATR_SLOW'] = atr_ema(df, slow)

    # Volatility Expansion Ratio
    df['VER'] = df['ATR_FAST'] / (df['ATR_SLOW'] + 1e-9)

    # Smooth slightly to remove noise
    df['VER'] = df['VER'].ewm(span=3).mean()

    return df

# ==========================================================
# VOLATILITY STATE (PHYSICS VERSION)
# ==========================================================
def volatility_state(df):
    """
    Adaptive anchor volatility state.
    Threshold evolves slowly with the market's own volatility baseline
    rather than being fixed at 2026 calibration values.
    span=1000 ≈ 6 weeks at 1H — slow enough to be structural,
    fast enough to adapt across market eras.
    """

    if 'VER' not in df.columns:
        df = volatility_expansion(df)

    # Slow-moving structural baseline
    ver_baseline = df['VER'].ewm(span=1000, adjust=False).mean()

    # Thresholds adapt with the baseline, not against a fixed number
    df['VOL_COMPRESS_TH'] = ver_baseline * 0.90
    df['VOL_EXPAND_TH']   = ver_baseline * 1.10

    df['VOL_STATE'] = np.select(
        [
            df['VER'] < df['VOL_COMPRESS_TH'],
            df['VER'] > df['VOL_EXPAND_TH']
        ],
        [-1, 1],
        default=0
    )

    return df

def trend_efficiency_state(df, lookback=50, er_window=20):
    """
    Adaptive anchor trend efficiency state.
    ER baseline adapts slowly over ~800 bars (~33 days at 1H)
    so the trend/compression threshold evolves with the market
    rather than being frozen at a universal constant.
    """

    df['ER'] = efficiency_ratio(df['close'], er_window)

    # Slow structural baseline — longer than ER's own lookback
    er_baseline = df['ER'].ewm(span=800, adjust=False).mean()

    # Thresholds shift with the baseline
    df['ER_COMPRESS_TH'] = er_baseline - 0.10
    df['ER_TREND_TH']    = er_baseline + 0.12

    df['STRUCT_STATE'] = np.select(
        [
            df['ER'] < df['ER_COMPRESS_TH'],
            df['ER'] > df['ER_TREND_TH']
        ],
        [-1, 1],
        default=0
    )

    return df

def pressure_state(df):

    close_loc = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    df['PRESSURE'] = close_loc - 0.5

    return df

# ==========================================================
# INSTITUTIONAL PARTICIPATION (SIGNED DOLLAR FLOW MODEL)
# ==========================================================
def participation_state(df, lookback=20, threshold=0.5):

    # ── 1. Signed institutional flow ─────────────────────────────
    df['FLOW'] = df['volume'] * (df['close'] - df['open'])

    # ── 2. Recursive flow normalization ──────────────────────────
    df['FLOW_Z'] = _ewma_zscore_series(df['FLOW'], alpha=0.05, min_periods=20)

    # ── 3. Capital accumulation — EWM instead of rolling mean ────
    df['FLOW_ROLL'] = df['FLOW_Z'].ewm(span=lookback, adjust=False).mean()

    # ── 4. Stealth accumulation ───────────────────────────────────
    # ewm(span=10).mean() * 10 is a weighted sum proxy, no hard window
    df['ACCUMULATION'] = df['FLOW_Z'].ewm(span=10, adjust=False).mean() * 10

    price_drift = df['close'].pct_change(10)
    vol = df['close'].pct_change().ewm(span=50, adjust=False).std()
    df['PRICE_DRIFT_NORM'] = price_drift / (vol + 1e-9)

    df['STEALTH_ACCUM'] = (
        (df['ACCUMULATION'] > 1.5) &
        (df['PRICE_DRIFT_NORM'].abs() < 0.5)
    )
    df['STEALTH_DISTRIB'] = (
        (df['ACCUMULATION'] < -1.5) &
        (df['PRICE_DRIFT_NORM'].abs() < 0.5)
    )

    # ── 5. Flow strength ──────────────────────────────────────────
    df['FLOW_STRENGTH'] = df['FLOW_ROLL']
    df.loc[df['STEALTH_ACCUM'],   'FLOW_STRENGTH'] += 0.5
    df.loc[df['STEALTH_DISTRIB'], 'FLOW_STRENGTH'] -= 0.5

    # ── 6. Classification ─────────────────────────────────────────
    df['PARTICIPATION'] = np.select(
        [
            df['FLOW_STRENGTH'] > threshold,
            df['FLOW_STRENGTH'] < -threshold
        ],
        [1, -1],
        default=0
    )

    return df

def classify_phase(df):

    df['PHASE'] = 0

    pre_breakout = (
        (df['VOL_STATE'] == -1) &
        (df['STRUCT_STATE'] == -1) &
        (
            (df['PARTICIPATION'] == 1) |
            (df['STEALTH_ACCUM'])
        )
    )

    trend = (
        (df['VOL_STATE'] == 1) &
        (df['STRUCT_STATE'] == 1) &
        (df['PARTICIPATION'] == 1)
    )

    exhaustion = (
        (df['VOL_STATE'] == 1) &
        (df['PARTICIPATION'] == -1)
    )

    df.loc[pre_breakout, 'PHASE'] = 1
    df.loc[trend, 'PHASE'] = 2
    df.loc[exhaustion, 'PHASE'] = 3

    return df

def vol_compression_slope(df, lookback=50, rv_period=20, alpha=0.2):
    # Compute realized volatility
    df['REALIZED_VOL'] = ewma_realized_vol(df, period=rv_period, alpha=alpha)

    # Compute slope of realized volatility
    df['RV_SLOPE'] = df['REALIZED_VOL'].diff(1)

    # Rolling mean slope -> compression signal
    df['VOL_COMPRESS'] = df['RV_SLOPE'].rolling(lookback).mean() < 0

    return df

def transition_detector(df):
    df['TRANSITION_LONG'] = (
        (df['VOL_COMPRESS'])
    )
    df['TRANSITION_SHORT'] = (
        (df['VOL_COMPRESS'])
    )
    
    df['TRANSITION_SIGNAL'] = 0
    df.loc[df['TRANSITION_LONG'], 'TRANSITION_SIGNAL'] = 1
    df.loc[df['TRANSITION_SHORT'], 'TRANSITION_SIGNAL'] = -1

    return df

# ==========================================================
# CANDLESTICK PATTERNS
# ==========================================================
def candle_body(df):
    df['body'] = df['close'] - df['open']
    df['body_dir'] = np.where(df['body'] > 0, 1, np.where(df['body'] < 0, -1, 0))
    df['body_size'] = df['body'].abs()
    return df

def composite_pressure(df):

    # Ensure VOL_RATIO exists
    if 'VOL_RATIO' not in df.columns:
        df = volume_confirmation(df)

    # Normalize VOL_RATIO to roughly -1..1 around 1
    vol_norm = df['VOL_RATIO'] - 1.0

    # Composite: pressure * normalized volume
    df['COMPOSITE_PRESSURE'] = df['PRESSURE'] * vol_norm

    return df

def rolling_zscore(series, window):
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)

# ==========================================================
# EXPANSION IGNITION ENGINE (replaces contextual_displacement)
# ==========================================================
def expansion_ignition(df):

    # ------------------------------------------------------
    # 1️⃣ Expansion pressure (continuous)
    # ------------------------------------------------------
    expansion_pressure = (
        0.5 * df['ATR_ACCEL_NORM'] +
        0.3 * df['TRANSITION_FORCE'] +
        0.2 * df['PRESSURE_VOL_NORM']
    )

    df['EXPANSION_PRESSURE'] = expansion_pressure.ewm(span=3).mean()

    # ------------------------------------------------------
    # 2️⃣ Expansion inflection (birth of move)
    # ------------------------------------------------------
    df['EXPANSION_INFLECT'] = (
        df['EXPANSION_PRESSURE'].diff() > 0
    )

    # ------------------------------------------------------
    # 3️⃣ Compression release bonus
    # ------------------------------------------------------
    compression = (
        (df['VOL_STATE'] == -1) &
        (df['STRUCT_STATE'] == -1)
    )

    df.loc[compression, 'EXPANSION_PRESSURE'] += 0.3

    # ------------------------------------------------------
    # 4️⃣ Final ignition score (continuous)
    # ------------------------------------------------------
    df['IGNITION_SCORE'] = (
        0.7 * df['EXPANSION_PRESSURE'] +
        0.3 * df['STATE_STABILITY']
    )

    df['IGNITION_OK'] = df['IGNITION_SCORE'] > 0.4

    return df

# ==========================================================
# EXPANSION CONTINUATION MODEL (replaces follow_through)
# ==========================================================
def expansion_continuation(df):

    # Growth of core drivers
    vol_growth   = df['ATR_ACCEL_NORM'].ewm(span=3).mean()
    flow_growth  = df['FLOW_STRENGTH'].ewm(span=3).mean()
    trend_growth = df['TREND_QUALITY'].ewm(span=3).mean()

    # Composite continuation strength
    df['CONTINUATION_STRENGTH'] = (
        0.4 * vol_growth +
        0.3 * flow_growth +
        0.3 * trend_growth
    )

    # Continuation velocity (important!)
    df['CONTINUATION_VELOCITY'] = df['CONTINUATION_STRENGTH'].diff()

    # Stable continuation regime
    df['CONTINUATION_OK'] = (
        (df['CONTINUATION_STRENGTH'] > 0) &
        (df['CONTINUATION_VELOCITY'] > -0.1) &
        (df['STATE_STABILITY'] > 0.4)
    )

    return df

def validated_breakouts(df, body_ratio=0.6, atr_mult=1.2):
    body = (df['close'] - df['open']).abs()
    range_ = df['high'] - df['low']

    # Only keep dynamic_state_engine — it feeds VOL_STATE/STRUCT_STATE
    # which expansion_maturity now reads directly.
    # expansion_ignition and expansion_continuation are removed —
    # they were the deepest part of the abstraction chain and
    # their outputs (IGNITION_OK, CONTINUATION_OK) are commented
    # out in VALID_BREAK anyway, so they were dead weight.
    df = dynamic_state_engine(df)
    df = expansion_maturity(df)      # now shallow — 2 direct reads
    df = compression_detector(df)

    # --- ATR expansion (kept — it's a direct primitive read)
    df['ATR_EXPAND'] = df['ATR'] > df['ATR'].rolling(20).mean() * atr_mult

    pressure_z = hybrid_zscore(df['COMPOSITE_PRESSURE'])
    recent_avg = pressure_z.rolling(20).mean()
    recent_std = pressure_z.rolling(20).std()
    df['PRESSURE_ELEVATED_LONG']  = pressure_z > (recent_avg + recent_std)
    df['PRESSURE_ELEVATED_SHORT'] = pressure_z < (recent_avg - recent_std)

    compression_ok = df['COMPRESSION_BARS'] >= 3

    vol_baseline     = df['VOL_RATIO'].ewm(span=500, adjust=False).mean()
    volume_confirmed = df['VOL_RATIO'] > vol_baseline * 1.15
    
    displacement_ok = df['DISPLACEMENT_SCORE'] > 0.15
    displacement_long_ok  = (
        df['BREAK_RESISTANCE'] | (df['DISPLACEMENT_SCORE'] > 0.45)
    )
    displacement_short_ok = (
        df['BREAK_SUPPORT'] | (df['DISPLACEMENT_SCORE'] > 0.45)
    )
    # Close location bias during compression
    # Where is price closing within the local compression range?
    # Consistently high closes = buyers winning inside the box = long bias
    # Consistently low closes = sellers winning inside the box = short bias

    comp_lookback = 10  # same order as compression detection

    comp_high = df['high'].rolling(comp_lookback).max()
    comp_low  = df['low'].rolling(comp_lookback).min()
    comp_range = comp_high - comp_low

    # 0 = closed at bottom of range, 1 = closed at top
    close_location = (df['close'] - comp_low) / (comp_range + 1e-9)

    # Smooth it — we want the drift over the compression, not a single bar
    close_location_bias = close_location.rolling(comp_lookback).mean()

    # Above 0.55 = consistently closing in upper half = long bias
    # Below 0.45 = consistently closing in lower half = short bias
    # Between 0.45-0.55 = genuinely ambiguous, no trade
    flow_bias_long  = close_location_bias > 0.6
    flow_bias_short = close_location_bias < 0.4

    df['VALID_BREAK_LONG'] = (
        df['EARLY_EXPANSION'] &
        volume_confirmed &
        flow_bias_long &
        df['MICRO_BREAK_LONG']
    )

    df['VALID_BREAK_SHORT'] = (
        df['EARLY_EXPANSION'] &
        volume_confirmed &
        flow_bias_short &
        df['MICRO_BREAK_SHORT']
    )

    _l = df.iloc[-1]
    print(
        f"[SIGNAL GATE] "
        f"EARLY_EXPANSION={int(_l['EARLY_EXPANSION'])} "
        f"(FLOW_STRENGTH={_l['FLOW_STRENGTH']:.4f}) | "
        f"volume_confirmed={int(_l['VOL_RATIO'] > vol_baseline.iloc[-1] * 1.15)} "
        f"(VOL_RATIO={_l['VOL_RATIO']:.4f} baseline={vol_baseline.iloc[-1]:.4f}) | "
        f"MICRO_BREAK_LONG={int(_l['MICRO_BREAK_LONG'])} "
        f"MICRO_BREAK_SHORT={int(_l['MICRO_BREAK_SHORT'])} | "
        f"close_location_bias={close_location_bias.iloc[-1]:.3f} "
        f"(flow_bias_long={int(flow_bias_long.iloc[-1])} flow_bias_short={int(flow_bias_short.iloc[-1])}) | "
        f"VALID_BREAK_LONG={int(_l['VALID_BREAK_LONG'])} "
        f"VALID_BREAK_SHORT={int(_l['VALID_BREAK_SHORT'])}"
    )

    df['BARS_SINCE_LONG_BREAK']  = bars_since_event(df['VALID_BREAK_LONG'])
    df['BARS_SINCE_SHORT_BREAK'] = bars_since_event(df['VALID_BREAK_SHORT'])

    return df

# ==========================================================
# COMPRESSION DETECTOR (Replaces Resistance Age)
# ==========================================================
def compression_detector(df, er_window=20):
    """
    Detects how long price has been coiling before breakout.

    Compression = low volatility + low directional efficiency
    """

    # ------------------------------------------------------
    # 1️⃣ Ensure required inputs exist
    # ------------------------------------------------------
    if 'VER' not in df.columns:
        df = volatility_expansion(df)

    # Efficiency Ratio (trend efficiency)
    df['ER'] = efficiency_ratio(df['close'], er_window)

    # ------------------------------------------------------
    # 2️⃣ Compression definition (the IMPORTANT part)
    # ------------------------------------------------------
    df['IS_COMPRESSION'] = (
        (df['VER'] < 0.95) &     # volatility contracting
        (df['ER']  < 0.45)       # price moving sideways
    )

    # ------------------------------------------------------
    # 3️⃣ Count consecutive compression bars
    # ------------------------------------------------------
    comp = df['IS_COMPRESSION'].astype(int)

    compression_bars = pd.Series(0, index=df.index)
    for idx in range(1, len(df)):
        if comp.iloc[idx] == 1:
            compression_bars.iloc[idx] = compression_bars.iloc[idx - 1] + 1
        else:
            compression_bars.iloc[idx] = 0
    df['COMPRESSION_BARS'] = compression_bars

    return df

# ==========================================================
# MICRO CONSOLIDATION DETECTOR (INSIDE TRENDS)
# ==========================================================
def micro_consolidation(df, lookback=12, tightness=0.6):

    # local range
    local_high = df['high'].rolling(lookback).max()
    local_low  = df['low'].rolling(lookback).min()
    width = local_high - local_low

    # normalize by ATR so it's regime-independent
    norm_width = width / (df['ATR'] + 1e-9)

    # tight box condition
    df['MICRO_BOX'] = norm_width < tightness

    # breakout levels (shifted so breakout is real)
    df['MICRO_HIGH'] = local_high.shift(1)
    df['MICRO_LOW']  = local_low.shift(1)

    # breakout detection
    df['MICRO_BREAK_LONG'] = df['close'] > df['MICRO_HIGH']
    df['MICRO_BREAK_SHORT'] = df['close'] < df['MICRO_LOW']

    # strength score (normalized)
    expansion_strength = hybrid_zscore(width).clip(0, 2)

    df['MICRO_BREAK_SCORE'] = np.select(
        [df['MICRO_BREAK_LONG'], df['MICRO_BREAK_SHORT']],
        [expansion_strength, -expansion_strength],
        default=0
    )

    return df

def supertrend(df, period=10, multiplier=3, eps=1e-6,
               flip_margin_atr=0.10, min_flip_bars=2):
    """
    SuperTrend with two real performance improvements:

    1. FLIP MARGIN
       A trend flip requires close to clear the band by
       flip_margin_atr * ATR, not just epsilon.
       Eliminates whipsaw flips from marginal closes in
       choppy regimes — exactly when stability matters most.

    2. MINIMUM FLIP BARS
       A flip is only accepted if the prior trend has held
       for at least min_flip_bars. Prevents single-bar
       flip/reflip pairs that generate entry+immediate reversal
       signals before any real move develops.

    Both parameters are intentionally small defaults —
    flip_margin_atr=0.10 means 10% of ATR, min_flip_bars=2
    means at least 2 bars in the prior trend. These filter
    noise without meaningfully lagging real trend changes.

    For HTF use (4H), consider min_flip_bars=3 to account
    for the lower bar frequency.
    """

    atr_series = atr_ema(df, period).round(6)
    atr_vals   = atr_series.values

    hl2       = (df['high'] + df['low']) / 2
    upper_raw = (hl2 + multiplier * atr_series).values
    lower_raw = (hl2 - multiplier * atr_series).values
    close     = df['close'].values
    n         = len(df)

    final_upper = upper_raw.copy()
    final_lower = lower_raw.copy()
    trend       = np.ones(n, dtype=np.int8)

    bars_since_flip = 0

    for i in range(1, n):

        # ── Ratchet bands (unchanged — this part is correct) ──
        final_upper[i] = (
            min(upper_raw[i], final_upper[i-1])
            if close[i-1] <= final_upper[i-1] + eps
            else upper_raw[i]
        )
        final_lower[i] = (
            max(lower_raw[i], final_lower[i-1])
            if close[i-1] >= final_lower[i-1] - eps
            else lower_raw[i]
        )

        # ── Flip logic with margin + minimum holding period ──
        flip_margin = atr_vals[i] * flip_margin_atr

        bull_flip = close[i] > final_upper[i-1] + flip_margin
        bear_flip = close[i] < final_lower[i-1] - flip_margin

        if bull_flip and trend[i-1] == -1 and bars_since_flip >= min_flip_bars:
            trend[i]        = 1
            bars_since_flip = 0
        elif bear_flip and trend[i-1] == 1 and bars_since_flip >= min_flip_bars:
            trend[i]        = -1
            bars_since_flip = 0
        else:
            trend[i]        = trend[i-1]
            bars_since_flip += 1

    df['SUPERTREND'] = trend
    df['ST_UPPER']   = final_upper
    df['ST_LOWER']   = final_lower

    return df

def supertrend_htf(df, htf_df, period=10, multiplier=3):
    """
    Computes SuperTrend on HTF and aligns it to LTF df.
    Returns a series of 1 (bull) / -1 (bear)
    """
    htf_df = htf_df.copy()
    htf_df = supertrend(htf_df, period=period, multiplier=multiplier)
    
    # Align to LTF
    return htf_df['SUPERTREND'].reindex(df.index, method='ffill').fillna(0)

# ==========================================================
# RSI RISK FILTER (NON-GATING)
# ==========================================================
def rsi_risk_filter(df, period=14, overbought=70, oversold=30):
    rsi = RSI(df['close'], period)

    long_ok = rsi < overbought
    short_ok = rsi > oversold

    return long_ok.fillna(True), short_ok.fillna(True)

# ==========================================================
# ANCHORED VWAP RISK FILTER (NON-GATING)
# ==========================================================
def anchored_vwap_risk(df, anchor_period=50):

    typical = (df['high'] + df['low'] + df['close']) / 3
    vol_price = typical * df['volume']

    rolling_vol_price = vol_price.rolling(anchor_period).sum()
    rolling_vol = df['volume'].rolling(anchor_period).sum()

    avwap = rolling_vol_price / (rolling_vol + 1e-9)

    long_ok = df['close'] >= avwap
    short_ok = df['close'] <= avwap

    return long_ok.fillna(True), short_ok.fillna(True)

def momentum_continuity(df, window=20, min_move=0.001):

    ret = df['close'].pct_change()

    # ignore tiny moves (noise)
    ret = ret.where(ret.abs() >= min_move, 0)

    sign_ret = ret.apply(np.sign)

    persistence = (
        (sign_ret * sign_ret.shift(1)) > 0
    ).astype(int)

    df['MOMENTUM_CONTINUITY'] = persistence.rolling(window).mean()

    return df

# ==========================================================
# DYNAMIC STATE ENGINE (INSTITUTIONAL GRADE)
# ==========================================================
def dynamic_state_engine(df, window=10):

    # -----------------------------------
    # BASE STATE (composite environment)
    # -----------------------------------
    df['STATE_SCORE'] = (
        0.25 * df['VOL_STATE'] +
        0.25 * df['STRUCT_STATE'] +
        0.25 * df['PARTICIPATION'] +
        0.25 * np.sign(df['COMPOSITE_PRESSURE'])
    )

    # normalize to -1 → 1
    df['STATE_SCORE'] = df['STATE_SCORE'].clip(-1,1)

    # -----------------------------------
    # VELOCITY → first derivative
    # -----------------------------------
    df['STATE_VELOCITY'] = df['STATE_SCORE'].diff()

    # -----------------------------------
    # ACCELERATION → second derivative
    # -----------------------------------
    df['STATE_ACCEL'] = df['STATE_VELOCITY'].diff()

    # -----------------------------------
    # INFLECTION POINTS
    # sign flip of velocity
    # -----------------------------------
    df['STATE_INFLECT'] = (
        np.sign(df['STATE_VELOCITY']) !=
        np.sign(df['STATE_VELOCITY'].shift(1))
    )

    df['PRESSURE_VOL'] = pressure_volatility(df, period=20, alpha=0.2)
    df['PRESSURE_VOL_NORM'] = (hybrid_zscore(df['PRESSURE_VOL']).clip(0, 3) / 3)

    # -----------------------------------
    # STABILITY
    # low variance = stable regime
    # -----------------------------------
    state_vol = df['STATE_SCORE'].rolling(window).std()

    df['STATE_STABILITY'] = (
        1 / (state_vol + 1e-9)
    )

    df['STATE_STABILITY'] = hybrid_zscore(df['STATE_STABILITY'])
    df['STATE_STABILITY'] = df['STATE_STABILITY'].clip(-2,2)

    # convert to 0–1 confidence score
    df['STATE_STABILITY'] = (df['STATE_STABILITY'] + 2) / 4

    df['STATE_STABILITY'] *= (1 - 0.5 * df['PRESSURE_VOL_NORM'])  # volatile regimes are less stable

    # -----------------------------------
    # STABILITY DECAY
    # detects regime breakdown
    # -----------------------------------
    df['STABILITY_DECAY'] = df['STATE_STABILITY'].diff()

    # -----------------------------------
    # TRANSITION INTENSITY
    # combines velocity + accel + decay
    # -----------------------------------
    df['TRANSITION_FORCE'] = (
        df['STATE_VELOCITY'].abs() +
        df['STATE_ACCEL'].abs() +
        df['STABILITY_DECAY'].abs()
    )

    # Damp TRANSITION_FORCE based on volatility instability
    df['TRANSITION_FORCE'] *= (1 - 0.5 * df['PRESSURE_VOL_NORM'])  # max 50% damp

    # -----------------------------------
    # VOLATILITY SHOCK REGIME INSTABILITY
    # -----------------------------------

    if 'VOL_SHOCK' in df.columns:
        df['TRANSITION_FORCE'] += 0.5 * df['VOL_SHOCK_INTENSITY']

    return df

# ==========================================================
# TREND QUALITY UTILITIES (Slope + R²)
# ==========================================================

def rolling_slope(series, window=50):
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y):
        if np.any(np.isnan(y)):
            return np.nan
        return ((x - x_mean) * (y - y.mean())).sum() / x_var

    return series.rolling(window).apply(_slope, raw=True)


def rolling_r2(series, window=50):
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _r2(y):
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        ss_tot = ((y - y_mean) ** 2).sum()
        if ss_tot < 1e-12:
            return 0.0
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        y_hat = slope * x + intercept
        ss_res = ((y - y_hat) ** 2).sum()
        return 1.0 - ss_res / ss_tot

    return series.rolling(window).apply(_r2, raw=True)

def efficiency_ratio(series, window=50):

    direction = (series - series.shift(window)).abs()
    volatility = series.diff().abs().rolling(window).sum()

    er = direction / (volatility + 1e-9)

    return er.clip(0,1)

# ==========================================================
# VOLATILITY UTILITIES
# ==========================================================
def ewma_realized_vol(df, period=20, alpha=0.2):
    log_ret = np.log(df['close']).diff()
    rv = log_ret.pow(2).ewm(alpha=alpha, adjust=False).mean().pow(0.5)
    return rv

def pressure_volatility(df, period=20, alpha=0.2):
    """
    EWMA volatility of COMPOSITE_PRESSURE
    Captures magnitude jitter and institutional activity instability
    """
    if 'COMPOSITE_PRESSURE' not in df.columns:
        df = composite_pressure(df)  # generate if missing
    pv = df['COMPOSITE_PRESSURE'].ewm(alpha=alpha, adjust=False).std()
    return pv

def compute_htf_scores(htf_df,
                       part_lookback=50,
                       regime_window=10,
                       er_window=20):
    """
    Fully recursive HTF quality scorer.
    All components use EWM or causal online estimators.
    No rolling windows → identical backtest/live output.
    """
    htf = htf_df.copy()

    # ── 1. DIRECTION ─────────────────────────────────────────────
    htf = supertrend(htf, period=20, multiplier=3, flip_margin_atr=0.15, min_flip_bars=3)
    htf['HTF_DIRECTION'] = htf['SUPERTREND']

    # ── 2. VOL SCORE ─────────────────────────────────────────────
    htf['HTF_ATR']      = atr_ema(htf)
    htf['HTF_ATR_FAST'] = htf['HTF_ATR'].ewm(span=20, adjust=False, min_periods=5).mean()
    htf['HTF_ATR_SLOW'] = htf['HTF_ATR'].ewm(span=50, adjust=False, min_periods=5).mean()
    ver = htf['HTF_ATR_FAST'] / (htf['HTF_ATR_SLOW'] + 1e-9)
    htf['VOL_SCORE'] = ((ver - 0.8) / 0.4).clip(0, 1)

    # ── 3. PARTICIPATION SCORE ───────────────────────────────────
    htf['HTF_VOL_EWM']   = htf['volume'].ewm(span=part_lookback, adjust=False, min_periods=5).mean()
    htf['HTF_VOL_RATIO'] = htf['volume'] / (htf['HTF_VOL_EWM'] + 1e-9)
    htf['PART_SCORE']    = ((htf['HTF_VOL_RATIO'] - 1) / 1).clip(0, 1)

    # ── 4. REGIME PERSISTENCE — recursive directional memory ─────
    # Replaces rolling(regime_window).apply(abs mean) — no hard window cliff
    htf['REGIME_SCORE'] = (
        htf['HTF_DIRECTION']
        .ewm(span=regime_window, adjust=False, min_periods=3)
        .mean()
        .abs()
        .clip(0, 1)
    )

    # ── 5. STRUCTURE QUALITY — recursive efficiency ratio ─────────
    direction_move = (htf['close'] - htf['close'].shift(er_window)).abs()
    path_length    = htf['close'].diff().abs().ewm(
        span=er_window, adjust=False, min_periods=3
    ).mean() * er_window
    htf['HTF_ER']          = (direction_move / (path_length + 1e-9)).clip(0, 1)
    htf['STRUCTURE_SCORE'] = htf['HTF_ER']

    # ── 6. MOMENTUM SCORE (already EWM — unchanged) ───────────────
    window = 12
    price_slope = htf['close'].diff(window)
    htf['HTF_TREND_MOMENTUM'] = (
        price_slope / (htf['HTF_ATR'] * window + 1e-9)
    ).ewm(span=3, min_periods=3).mean()
    htf['HTF_TREND_MOMENTUM_NORM'] = np.tanh(htf['HTF_TREND_MOMENTUM'] * 3.0)
    htf['MOMENTUM_SCORE'] = ((htf['HTF_TREND_MOMENTUM_NORM'] + 1) / 2).clip(0, 1)

    # ── 7. COMPOSITE ─────────────────────────────────────────────
    htf['HTF_QUALITY'] = (
        0.25 * htf['VOL_SCORE']       +
        0.20 * htf['PART_SCORE']      +
        0.20 * htf['REGIME_SCORE']    +
        0.20 * htf['STRUCTURE_SCORE'] +
        0.15 * htf['MOMENTUM_SCORE']
    )

    return htf[['HTF_DIRECTION', 'HTF_QUALITY']]


def align_htf_scores(htf_scores, df, is_live=False):
    # htf_scores is indexed on bar OPEN times (e.g. 16:00 UTC)
    # but the score is only valid after the bar CLOSES (e.g. 20:00 UTC)
    # so we reindex the scores to their close times before aligning
    htf_scores_copy = htf_scores.copy()
    htf_scores_copy.index = htf_scores_copy.index + pd.Timedelta(hours=4)
    aligned = htf_scores_copy.reindex(df.index, method='ffill')
    return aligned.fillna(0)


def htf_structural_stack(df, htf_df,
                         vol_lookback=200,
                         part_lookback=50,
                         regime_window=10,
                         er_window=20,
                         is_live=False):
    """
    Backward-compatible wrapper. Used in backtest and anywhere a precomputed
    cache isn't available. Internally calls the split functions.

    htf_df must already exclude the open 4H bar before being passed here —
    that is the lookahead guard, not the shift inside align_htf_scores.
    """
    htf_scores = compute_htf_scores(
        htf_df,
        part_lookback=part_lookback,
        regime_window=regime_window,
        er_window=er_window,
    )
    return align_htf_scores(htf_scores, df)

# ==========================================================
# VOLATILITY SHOCK DETECTOR
# ==========================================================
def volatility_shock(df, lookback=20, shock_mult=1.8):

    # baseline volatility
    atr_mean = df['ATR'].rolling(lookback).mean()

    # shock ratio
    shock_ratio = df['ATR'] / (atr_mean + 1e-9)

    df['VOL_SHOCK'] = (shock_ratio > shock_mult).astype(int)

    # intensity (continuous)
    df['VOL_SHOCK_INTENSITY'] = (shock_ratio - 1).clip(0, 3)

    # ======================================================
    # DECAY SPEED (REGIME ADAPTIVE)
    # ======================================================
    # normalize ATR → regime detector
    df['ATR_Z'] = hybrid_zscore(df['ATR']).clip(-2, 2)

    # high vol → faster signal expiration
    df['DECAY_SPEED'] = np.exp(df['ATR_Z'] * 0.35)

    return df

# ==========================================================
# PRESSURE–ELASTICITY DIVERGENCE
# ==========================================================
def pressure_elasticity_divergence(df, window=5):

    # -----------------------------------------
    # Price response (volatility normalized)
    # -----------------------------------------
    ret = df['close'].pct_change()

    vol = ret.rolling(50).std()

    response = ret / (vol + 1e-9)

    # -----------------------------------------
    # Pressure impulse
    # -----------------------------------------
    pressure_change = df['COMPOSITE_PRESSURE'].diff()

    # -----------------------------------------
    # Elasticity (response per unit pressure)
    # -----------------------------------------
    elasticity = response / (df['COMPOSITE_PRESSURE'].abs() + 1e-9)

    elasticity_change = elasticity.diff()

    # -----------------------------------------
    # Divergence: force vs response mismatch
    # -----------------------------------------
    df['PRESS_ELAST_DIV'] = (
        pressure_change - elasticity_change
    ).rolling(window).mean()

    # normalize to stable range
    df['PRESS_ELAST_DIV_NORM'] = hybrid_zscore(df['PRESS_ELAST_DIV']).clip(-3,3)

    return df

# ==========================================================
# TEMPORAL PHASE ASYMMETRY (LIQUIDITY SWEEP DETECTOR)
# ==========================================================
def temporal_phase_asymmetry(df, compress_window=20, expand_window=5):

    # ---------------------------------------
    # Compression duration
    # how long volatility stayed compressed
    # ---------------------------------------
    compression_time = (
        df['VOL_COMPRESS']
        .rolling(compress_window)
        .sum()
    )

    # ---------------------------------------
    # Expansion duration
    # how long volatility expanded
    # ---------------------------------------
    expansion_time = (
        df['ATR_EXPAND']
        .rolling(expand_window)
        .sum()
    )

    # ---------------------------------------
    # Time asymmetry ratio
    # ---------------------------------------
    df['TIME_ASYMM'] = expansion_time / (compression_time + 1e-9)

    # Normalize for stability
    df['TIME_ASYMM_NORM'] = hybrid_zscore(df['TIME_ASYMM']).clip(0,5)

    return df

# ==========================================================
# POST BREAKOUT PULLBACK ENTRY (PBPE)
# ==========================================================
def post_breakout_event_window(signal, window=3):
    """
    Creates a forward event window after a breakout signal.
    Marks the next N candles where entry is allowed.
    """
    future_window = signal.shift(1).rolling(window).max().fillna(0).astype(bool)
    return future_window.fillna(False)

def breakout_pullback_metrics(df):
    """
    Measures retracement after breakout using ATR-normalized distance.
    """

    # distance from recent high/low after breakout
    recent_high = df['high'].rolling(5).max()
    recent_low  = df['low'].rolling(5).min()

    df['PULLBACK_LONG'] = (recent_high - df['low']) / (df['ATR'] + 1e-9)
    df['PULLBACK_SHORT'] = (df['high'] - recent_low) / (df['ATR'] + 1e-9)

    return df

def continuation_candle(df):
    body = df['close'] - df['open']

    df['BULL_CONT'] = (
        (body > 0) &
        (df['close'] > df['high'].shift(1))
    )

    df['BEAR_CONT'] = (
        (body < 0) &
        (df['close'] < df['low'].shift(1))
    )

    return df

def pullback_entry(df):
    # Pullback = price retraced into the range but closed with momentum
    # resuming in the breakout direction.
    # BULL_CONT (close above prior high) contradicts a pullback — removed.
    # Instead require: body is positive (bull close) for longs,
    # negative (bear close) for shorts — momentum resuming after retrace.

    ideal_pullback_long = df['PULLBACK_LONG'].between(0.3, 1.5)
    ideal_pullback_short = df['PULLBACK_SHORT'].between(0.3, 1.5)

    bull_close = df['close'] > df['open']
    bear_close = df['close'] < df['open']

    df['PBPE_PULLBACK_LONG'] = (
        df['BREAKOUT_WINDOW_LONG'] 
        # ideal_pullback_long 
        # bull_close
    )

    df['PBPE_PULLBACK_SHORT'] = (
        df['BREAKOUT_WINDOW_SHORT'] 
        # ideal_pullback_short 
        # bear_close
    )

    return df

def micro_break_entry(df):

    df['PBPE_MICRO_LONG'] = (
        df['BREAKOUT_WINDOW_LONG'] &
        df['MICRO_BREAK_LONG']
    )

    df['PBPE_MICRO_SHORT'] = (
        df['BREAKOUT_WINDOW_SHORT'] &
        df['MICRO_BREAK_SHORT']
    )

    return df

def delayed_continuation(df):

    strong_momentum = df['MOMENTUM_CONTINUITY'] > 0.6

    df['PBPE_DELAY_LONG'] = (
        df['VALID_BREAK_LONG'].shift(2) &
        strong_momentum &
        (df['close'] > df['close'].shift(1))
    )

    df['PBPE_DELAY_SHORT'] = (
        df['VALID_BREAK_SHORT'].shift(2) &
        strong_momentum &
        (df['close'] < df['close'].shift(1))
    )

    return df

def post_breakout_entry(df):

    # 1) breakout event windows
    df['BREAKOUT_WINDOW_LONG']  = post_breakout_event_window(df['VALID_BREAK_LONG'], window=6)
    df['BREAKOUT_WINDOW_SHORT'] = post_breakout_event_window(df['VALID_BREAK_SHORT'], window=6)

    # 2) compute metrics
    df = breakout_pullback_metrics(df)
    df = continuation_candle(df)

    # 3) entry types
    df = pullback_entry(df)
    df = micro_break_entry(df)
    df = delayed_continuation(df)

    # 4) final execution signal
    df['ENTRY_LONG'] = (
        df['BULL_CONT'] 
        # df['PBPE_MICRO_LONG']
    )

    df['ENTRY_SHORT'] = (
        df['BEAR_CONT'] 
        # df['PBPE_MICRO_SHORT']
    )

    return df

def breakout_tracking_window(signal, window=5):
    """
    Tracks candles immediately AFTER breakout.
    """
    return (
        signal.shift(1)
        .rolling(window)
        .max()
        .fillna(0)
        .astype(bool)
    )

def compression_context(df, lookback=7, memory=6):
    if 'FRESHNESS_SHORT' not in df.columns:
        raise RuntimeError("compression_context requires entry_freshness() to be called first")
    
    # 1️⃣ Recent compression existed
    recent_compression = (
        df['VOL_COMPRESS']
        .rolling(lookback)
        .max()
    )

    # 2️⃣ How long since last compression?
    # Causal forward counter — same fix as bars_since_event
    bars_since_compression = pd.Series(999, index=df.index, dtype=int)
    counter = 999
    for idx in range(len(df)):
        if df['VOL_COMPRESS'].iloc[idx]:
            counter = 0
        else:
            if counter < 999:
                counter += 1
        bars_since_compression.iloc[idx] = counter

    # Normalize time since compression
    freshness = 1 - (bars_since_compression / memory).clip(0,1)

    # 3️⃣ Expansion hasn't already happened too long
    expansion_decay = (
        df['ATR_EXPAND']
        .rolling(memory)
        .sum() / memory
    )

    expansion_ok = expansion_decay < 0.6

    # 4️⃣ Final compression score (continuous)
    df['COMPRESSION_SCORE'] = (
        0.5 * recent_compression.astype(float) +
        0.5 * freshness
    ) * expansion_ok.astype(float)
    df['COMPRESSION_SCORE'] *= df[['FRESHNESS_LONG', 'FRESHNESS_SHORT']].max(axis=1)

    # 5️⃣ Convert to permission (like HTF_OK)
    df['COMPRESSION_OK'] = df['COMPRESSION_SCORE'] > 0.50

    return df

# ==========================================================
# ANCHORED NORMALIZATION ENGINE (Long-term memory)
# ==========================================================
import threading
_EWMA_STATE: dict = {}
_EWMA_LOCK = threading.Lock()

def _ewma_zscore_series(series: pd.Series,
                        alpha: float = 0.05,
                        min_periods: int = 30,
                        adaptive: bool = True) -> pd.Series:
    """
    Adaptive recursive online z-score.

    When adaptive=True, alpha scales with recent volatility of the series
    itself — faster adaptation in volatile regimes, slower in quiet ones.
    This prevents long-history baseline compression where 6 years of data
    causes recent signals to normalize against an overly broad baseline.

    alpha range: [alpha * 0.5, alpha * 3.0]
    — floor prevents memory from becoming infinite
    — ceiling prevents alpha from exploding in shock regimes
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.empty(n)
    out[:] = np.nan

    mu  = np.nan
    var = np.nan

    # short-term variance tracker for adaptive alpha
    recent_var = np.nan
    alpha_fast = alpha * 4.0  # faster EWM for recent vol estimate

    for i, x in enumerate(values):
        if np.isnan(x):
            continue

        if np.isnan(mu):
            mu  = x
            var = 0.0
            recent_var = 0.0
            continue

        # ── adaptive alpha ────────────────────────────────────
        if adaptive and not np.isnan(recent_var) and var > 1e-12:
            # how much does recent volatility differ from long-run vol?
            vol_ratio = np.sqrt(recent_var / var) if var > 1e-12 else 1.0
            vol_ratio = np.clip(vol_ratio, 0.5, 3.0)
            effective_alpha = np.clip(alpha * vol_ratio, alpha * 0.5, alpha * 3.0)
        else:
            effective_alpha = alpha

        # ── update long-run state ─────────────────────────────
        mu  = mu  + effective_alpha * (x - mu)
        var = (1.0 - effective_alpha) * var + effective_alpha * (x - mu) ** 2

        # ── update short-run variance (for next bar's alpha) ──
        recent_var = (1.0 - alpha_fast) * recent_var + alpha_fast * (x - mu) ** 2

        if i >= min_periods:
            std = np.sqrt(var) if var > 1e-12 else 1e-6
            out[i] = (x - mu) / std

    return pd.Series(out, index=series.index)


def anchored_zscore(series, min_periods=200):
    """Backward-compat shim — now delegates to recursive estimator."""
    return _ewma_zscore_series(series, alpha=0.02, min_periods=min_periods)


def hybrid_zscore(series, roll_window=200, anchor_weight=0.6, min_periods=200):
    """
    Drop-in replacement — now fully recursive.
    alpha=0.05 ≈ 39-bar half-life. Contextual but stable.
    Identical in backtest and live. No rolling windows.
    """
    return _ewma_zscore_series(series, alpha=0.05, min_periods=30)

def sanitize_features_for_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final safety airlock before signal generation.

    Guarantees:
    - No NaN / inf values reach entry or exit logic
    - Rolling indicators remain untouched during feature engineering
    - Live incremental updates cannot break exits
    """

    # Work on a copy to avoid side effects
    df = df.copy()

    # 1️⃣ Replace infinities from divisions / std / zscores
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2️⃣ Forward fill ONLY to preserve indicator continuity
    # (critical for rolling indicators in live pipelines)
    df.ffill(inplace=True)

    # 3️⃣ Zero-fill anything still missing
    # Remaining NaNs are from warmup periods or new columns
    df.fillna(0, inplace=True)

    return df

# ==========================================================
# EVENT AGE TRACKER (NEW)
# ==========================================================
def bars_since_event(event_series: pd.Series) -> pd.Series:
    age = pd.Series(999, index=event_series.index, dtype=int)
    counter = 999
    for idx in range(len(event_series)):
        if event_series.iloc[idx]:
            counter = 0
        else:
            if counter < 999:
                counter += 1
        age.iloc[idx] = counter
    return age

# ==========================================================
# ENTRY FRESHNESS ENGINE (NEW)
# ==========================================================
def entry_freshness(df, half_life=3):
    # Half-life of 3 bars (3 hours at 1H). Signal is mostly dead by bar 6.
    # No floor — stale signals die completely.
    # DECAY_SPEED still modulates: high-vol regimes expire faster.

    df['FRESHNESS_LONG'] = np.exp(
        -df['BARS_SINCE_LONG_BREAK'] /
        (half_life * df['DECAY_SPEED'])
    )

    df['FRESHNESS_SHORT'] = np.exp(
        -df['BARS_SINCE_SHORT_BREAK'] /
        (half_life * df['DECAY_SPEED'])
    )

    # No floor — let signals die. A 0.15 floor on a stale signal
    # keeps it alive through the compression_context score.

    return df

# ==========================================================
# HTF TREND MATURITY ENGINE
# Detects EARLY / MID / LATE trend lifecycle
# ==========================================================
def compute_htf_trend_maturity(df, htf_df):

    htf = htf_structural_stack(df, htf_df)
    htf_dir = htf['HTF_DIRECTION']

    # Detect when HTF trend flips
    trend_flip = htf_dir != htf_dir.shift(1)

    # Count bars since last flip
    trend_age = trend_flip.cumsum()
    trend_age = trend_age.groupby(trend_age).cumcount()

    # Normalize age (robust scaling)
    age_norm = trend_age / (trend_age.rolling(200).max() + 1e-9)

    # Classify lifecycle phases
    df['HTF_TREND_EARLY'] = age_norm < 0.33
    df['HTF_TREND_MID']   = (age_norm >= 0.33) & (age_norm < 0.66)
    df['HTF_TREND_LATE']  = age_norm >= 0.66

    return df

# ==========================================================
# LONG-TERM VOLATILITY REGIME INDEX (GLOBAL ANCHOR)
# ==========================================================
def volatility_regime_index(df, fast=200, slow=2000):
    """
    Long-term volatility anchor that prevents normalization drift.
    
    fast  = local volatility memory
    slow  = multi-month / multi-year baseline
    
    Output:
        VOL_REGIME_INDEX in range 0 → 1
        0 = structurally quiet market
        1 = structurally volatile market
    """

    # Ensure ATR exists
    if 'ATR' not in df.columns:
        df['ATR'] = atr_ema(df)

    # Fast and slow volatility memory
    fast_vol = df['ATR'].ewm(span=fast, adjust=False).mean()
    slow_vol = df['ATR'].ewm(span=slow, adjust=False).mean()

    # Volatility regime ratio
    vol_ratio = fast_vol / (slow_vol + 1e-9)

    # Smooth + squash to stable 0-1 range
    vri = np.tanh((vol_ratio - 1) * 2.5)

    df['VOL_REGIME_INDEX'] = (vri + 1) / 2

    return df

# ==========================================================
# EXPANSION MATURITY MODEL (replaces impulse_age)
# ==========================================================
def expansion_maturity(df, lookback=20):
    """
    Replaces the 6-layer abstraction chain with two direct
    primitive reads that are closer to actual market observables.

    OLD: tanh(EWM(composite(ATR_ACCEL_NORM, FLOW_STRENGTH, TREND_QUALITY)))
         — 4+ layers deep, measures its own smoothed history

    NEW: asks two questions directly from raw market data
         1. Is volatility actually expanding right now? (VER vs baseline)
         2. Is participation confirming the move? (FLOW_STRENGTH directly)

    EARLY_EXPANSION = True when expansion is nascent, not mature.
    We want to enter early — before the move is obvious.
    A mature expansion (high EXPANSION_MATURITY) was the wrong
    signal to gate on anyway; that's entering late.
    """

    # --------------------------------------------------
    # 1. DIRECT VOLATILITY EXPANSION CHECK
    # VER is already computed upstream — use it raw.
    # Is the current bar's volatility above its own slow baseline?
    # This is a 2-layer read: raw ATR → VER ratio.
    # Far shallower than the old chain.
    # --------------------------------------------------
    ver_expanding = df['VER'] > df['VOL_EXPAND_TH']

    # How long has expansion been running?
    # Count consecutive bars of expansion — simple forward counter.
    # Same pattern as COMPRESSION_BARS, which already works.
    expanding = ver_expanding.astype(int)
    expansion_bars = pd.Series(0, index=df.index)
    for idx in range(1, len(df)):
        if expanding.iloc[idx] == 1:
            expansion_bars.iloc[idx] = expansion_bars.iloc[idx - 1] + 1
        else:
            expansion_bars.iloc[idx] = 0
    df['EXPANSION_BARS'] = expansion_bars

    # --------------------------------------------------
    # 2. DIRECT FLOW CONFIRMATION
    # FLOW_STRENGTH is already computed in participation_state().
    # Read it directly — no re-smoothing, no re-compositing.
    # --------------------------------------------------
    flow_confirming = df['FLOW_STRENGTH'].abs() > 0.6

    # --------------------------------------------------
    # 3. EARLY_EXPANSION DEFINITION
    # Early = expansion just started (few bars in)
    #       AND flow is actually confirming
    # Late  = expansion has been running a long time
    #       OR flow has already faded
    #
    # Threshold of 8 bars = 8 hours at 1H.
    # After 8 bars of sustained expansion, the move is
    # no longer early — you're chasing.
    # --------------------------------------------------
    df['EARLY_EXPANSION'] = (
        # (df['EXPANSION_BARS'] <= 8) &
        # (df['EXPANSION_BARS'] >= 1) |
        flow_confirming
    )

    # Keep EXPANSION_STATE for anything downstream that reads it
    # but make it a simple normalized bar count rather than
    # a deep composite — honest about what it is.
    df['EXPANSION_STATE'] = (df['EXPANSION_BARS'] / 8).clip(0, 1)
    df['EXPANSION_MATURITY'] = df['EXPANSION_STATE']  # backward compat

    return df

# ==========================================================
# VOLATILITY ACCELERATION ENGINE (feeds expansion ignition)
# ==========================================================
def atr_acceleration(df, fast=5, slow=20):
    """
    Measures acceleration of volatility expansion.
    This is the missing input for the new expansion engine.
    Completely causal. No lookahead.
    """

    # Ensure ATR exists
    if 'ATR' not in df.columns:
        df['ATR'] = atr_ema(df)

    # ------------------------------------------------------
    # 1️⃣ Fast vs slow ATR (volatility impulse)
    # ------------------------------------------------------
    df['ATR_FAST'] = df['ATR'].ewm(span=fast).mean()
    df['ATR_SLOW'] = df['ATR'].ewm(span=slow).mean()

    # ------------------------------------------------------
    # 2️⃣ Volatility acceleration (rate of change)
    # ------------------------------------------------------
    df['ATR_ACCEL'] = df['ATR_FAST'] - df['ATR_SLOW']

    # ------------------------------------------------------
    # 3️⃣ Normalize to stable regime-independent scale
    # ------------------------------------------------------
    df['ATR_ACCEL_NORM'] = hybrid_zscore(df['ATR_ACCEL']).clip(-3, 3) / 3

    return df

def entry_location_filter(df, lookback=20):
    """
    Computes where current close sits within the N-bar range as a percentile.
    Longs require entry below the 60th percentile (not chasing).
    Shorts require entry above the 40th percentile (not chasing).
    This is orthogonal to compression/expansion/ignition — it measures
    entry location relative to recent structure, not breakout quality.
    """
    rolling_high = df['high'].rolling(lookback).max()
    rolling_low  = df['low'].rolling(lookback).min()
    rolling_range = rolling_high - rolling_low

    # 0 = bottom of range, 1 = top of range
    df['ENTRY_PERCENTILE'] = (df['close'] - rolling_low) / (rolling_range + 1e-9)

    # Longs: not too extended to the upside
    df['LOCATION_LONG_OK']  = df['ENTRY_PERCENTILE'] < 0.70

    # Shorts: not too extended to the downside
    df['LOCATION_SHORT_OK'] = df['ENTRY_PERCENTILE'] > 0.30

    return df

# ==========================================================
# INTEGRATE INTO SIGNAL GENERATION
# ==========================================================
def generate_signal(df, htf_df, atr_mult=1.5, live=False, as_of=None, symbol="?", htf_stack_cache=None):
    if df.empty:
        return df

    if as_of is not None:
        cutoff = pd.Timestamp(as_of).tz_convert("UTC") if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of).tz_localize("UTC")
        htf_df = htf_df[htf_df.index < cutoff].copy()
    # else: trust the caller — htf_df is already correctly clipped

    print(f"[DEBUG] generate_signal htf_df last={htf_df.index[-1] if not htf_df.empty else 'EMPTY'} len={len(htf_df)}")

    if df.empty or htf_df.empty:
        return df

    # =========================
    # Core processing
    # =========================
    df = positioning_pressure(df)
    df = wick_rejection(df)
    df = volume_confirmation(df)
    df = support_resistance(df)
    df = liquidity_displacement(df)

    df['ATR'] = atr_ema(df, period=14)

    df = atr_acceleration(df)
    df = volatility_shock(df)

    # 1H SuperTrend for LTF direction agreement filter
    df = supertrend(df, period=10, multiplier=3)
    df['LTF_DIRECTION'] = df['SUPERTREND']

    # =========================
    # STATE ENGINE
    # =========================
    df = volatility_expansion(df)
    df = volatility_state(df)
    df = trend_efficiency_state(df)
    df = pressure_state(df)
    df = participation_state(df)
    df = classify_phase(df)
    df = composite_pressure(df)  # 🔹 generate COMPOSITE_PRESSURE metric
    df = pressure_elasticity_divergence(df)
    df = vol_compression_slope(df, lookback=50, rv_period=20)
    df = micro_consolidation(df)
    df = validated_breakouts(df)
    df = entry_freshness(df)
    df = compression_context(df)
    df = temporal_phase_asymmetry(df)
    # --- Dynamic state analytics
    df = dynamic_state_engine(df)
    df = entry_location_filter(df, lookback=20)

    # =========================
    # NEW HTF STRUCTURAL STACK
    # =========================

    if htf_stack_cache is not None:
        # Use precomputed 4H scores, just reindex onto current LTF df
        htf_stack = align_htf_scores(htf_stack_cache, df, is_live=live)
    else:
        # Fallback: full recompute (backtest path, or cache unavailable)
        htf_stack = htf_structural_stack(df, htf_df, is_live=live)

    df = pd.concat([df, htf_stack], axis=1)

    htf_quality_baseline = df['HTF_QUALITY'].ewm(span=2000, adjust=False).mean()
    htf_quality_th       = (htf_quality_baseline * 1.05).clip(lower=0.30)

    HTF_LONG_OK = (
        # (df['HTF_DIRECTION'] == 1) |
        (df['LTF_DIRECTION'] == 1)
    )

    HTF_SHORT_OK = (
        # (df['HTF_DIRECTION'] == -1) |
        (df['LTF_DIRECTION'] == -1)
    )

    # =========================
    # PREDICTIVE MODULES
    # =========================
    df = transition_detector(df)
    df = momentum_continuity(df)
    df = post_breakout_entry(df)
    
    LONG_CONDITION = (df['VALID_BREAK_LONG'])
    SHORT_CONDITION = (df['VALID_BREAK_SHORT'])

    df['ENTRY_LONG'] = (
        # df['ENTRY_LONG'] 
        df['COMPRESSION_OK'] 
    )

    df['ENTRY_SHORT'] = (
        # df['ENTRY_SHORT'] 
        df['COMPRESSION_OK'] 
    )

    # LONG_CONDITION &= df['ENTRY_LONG']
    # SHORT_CONDITION &= df['ENTRY_SHORT']

    # LONG_CONDITION &= HTF_LONG_OK
    # SHORT_CONDITION &= HTF_SHORT_OK

    # LONG_CONDITION  &= df['LOCATION_LONG_OK']
    # SHORT_CONDITION &= df['LOCATION_SHORT_OK']

    # df['signal'] = 0
    # df.loc[LONG_CONDITION, 'signal'] = 1
    # df.loc[SHORT_CONDITION, 'signal'] = -1
    df['signal'] = 1

    # ── FILTER AUDIT ──────────────────────────────────────────────
    # try:
    #     from execution.notifier import TelegramNotifier
    #     _last = df.iloc[-1]
    #     _b = lambda val: "✅" if bool(val) else "❌"
    #     TelegramNotifier().debug(
    #         f"🔬 *FILTER AUDIT* `{symbol}`\n"
    #         f"HTF quality={float(_last['HTF_QUALITY']):.4f} dir={int(_last['HTF_DIRECTION'])}\n"
    #         f"HTF_LONG={_b(HTF_LONG_OK.iloc[-1])}  HTF_SHORT={_b(HTF_SHORT_OK.iloc[-1])}\n"
    #         f"EARLY_EXPANSION={_b(_last['EARLY_EXPANSION'])}\n"
    #         f"PRESSURE_LONG={_b(_last['PRESSURE_ELEVATED_LONG'])}  PRESSURE_SHORT={_b(_last['PRESSURE_ELEVATED_SHORT'])}\n"
    #         f"ENTRY_LONG={_b(_last['ENTRY_LONG'])}  ENTRY_SHORT={_b(_last['ENTRY_SHORT'])}"
    #     )
    # except Exception:
    #     pass
    # ── END FILTER AUDIT ──────────────────────────────────────────

    if live:
        df['final_signal'] = df['signal'].fillna(0).astype(int)
    else:
        df['final_signal'] = df['signal'].shift(1).fillna(0).astype(int)

    # =========================
    # DIAGNOSTICS
    # =========================

    # =========================
    # DIAGNOSTICS
    # =========================
    # print("\n=== STATE DIAGNOSTICS ===")
    # print("Phase counts:\n", df['PHASE'].value_counts())
    # print("Breakouts: Long =", df['BREAK_RESISTANCE'].sum(), "Short =", df['BREAK_SUPPORT'].sum())
    # print("Transition signals:", df['TRANSITION_SIGNAL'].value_counts())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    # ================= DEBUG SIGNAL SUMMARY =================
    signal_count = (df["final_signal"] != 0).sum()

    # print(
    #     f"[DBG-GEN] candles={len(df)} | signals={signal_count} | "
    #     f"first={df.index[0]} | last={df.index[-1]}"
    # )

    return df