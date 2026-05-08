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
def trend_bias(df, window=50):

    df['TREND_SLOPE'] = rolling_slope(df['close'], window)
    df['TREND_R2'] = rolling_r2(df['close'], window)

    # Combine direction + velocity + reliability
    df['TREND_QUALITY'] = df['TREND_SLOPE'] * df['TREND_R2']

    # Normalize to stable range (important)
    df['TREND_QUALITY'] = hybrid_zscore(df['TREND_QUALITY'])

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
def breakout_logic(df, atr_k=0.5):
    """
    Volatility-adjusted breakout.
    Break must clear structure by ATR fraction.
    """

    # Ensure ATR exists
    if 'ATR' not in df.columns:
        df['ATR'] = atr_ema(df)

    resistance = df['RESISTANCE'].shift(1)
    support = df['SUPPORT'].shift(1)

    df['BREAK_RESISTANCE'] = df['close'] > (resistance + atr_k * df['ATR'])
    df['BREAK_SUPPORT'] = df['close'] < (support - atr_k * df['ATR'])

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
    Uses Volatility Expansion Ratio instead of ATR percentile.

    This makes the system asset-agnostic and regime aware.
    """

    # Ensure VER exists
    if 'VER' not in df.columns:
        df = volatility_expansion(df)

    # Regime classification
    df['VOL_STATE'] = np.select(
        [
            df['VER'] < 0.9,   # compression
            df['VER'] > 1.1    # expansion
        ],
        [-1, 1],
        default=0
    )

    return df

def trend_efficiency_state(df, lookback=50, er_window=20):
    """
    Replaces raw range width with Trend Efficiency (Directional Persistence).
    Measures how efficiently price is moving in a direction over the lookback.
    """

    # ------------------------------------------------------
    # 1️⃣ Compute rolling efficiency ratio
    # ------------------------------------------------------
    df['ER'] = efficiency_ratio(df['close'], er_window)

    # ------------------------------------------------------
    # 2️⃣ Define compression / expansion state
    # Use ER instead of range width to detect sideways vs trending
    # ------------------------------------------------------
    df['STRUCT_STATE'] = np.select(
        [
            df['ER'] < 0.45,  # low efficiency → sideways / compression
            df['ER'] > 0.7    # high efficiency → trending / expansion
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
# INSTITUTIONAL PARTICIPATION (Magnitude-Aware VWM)
# ==========================================================
# ==========================================================
# INSTITUTIONAL PARTICIPATION (SIGNED DOLLAR FLOW MODEL)
# ==========================================================
def participation_state(df, lookback=20, threshold=0.5):

    # ------------------------------------------------------
    # 1️⃣ Signed institutional flow
    # ------------------------------------------------------
    df['FLOW'] = df['volume'] * (df['close'] - df['open'])

    # ------------------------------------------------------
    # 2️⃣ Normalize by rolling volatility (z-score style)
    # ------------------------------------------------------
    df['FLOW_Z'] = hybrid_zscore(df['FLOW'])

    # ------------------------------------------------------
    # 3️⃣ Capital accumulation (rolling flow)
    # ------------------------------------------------------
    df['FLOW_ROLL'] = df['FLOW_Z'].rolling(lookback).mean()

    # ------------------------------------------------------
    # 4️⃣ Institutional accumulation detector
    # ------------------------------------------------------

    # Cumulative capital inflow
    df['ACCUMULATION'] = df['FLOW_Z'].rolling(10).sum()

    # Price drift over same window
    price_drift = df['close'].pct_change(10)

    # Normalize drift by volatility so regime independent
    vol = df['close'].pct_change().rolling(50).std()

    df['PRICE_DRIFT_NORM'] = price_drift / (vol + 1e-9)

    # Stealth accumulation condition
    df['STEALTH_ACCUM'] = (
        (df['ACCUMULATION'] > 1.5) &      # sustained positive flow
        (df['PRICE_DRIFT_NORM'].abs() < 0.5)  # price not moving much
    )

    # Distribution version
    df['STEALTH_DISTRIB'] = (
        (df['ACCUMULATION'] < -1.5) &
        (df['PRICE_DRIFT_NORM'].abs() < 0.5)
    )

    # ------------------------------------------------------
    # 4️⃣ Final flow strength metric
    # ------------------------------------------------------
    df['FLOW_STRENGTH'] = df['FLOW_ROLL']

    # Boost if stealth accumulation detected
    df.loc[df['STEALTH_ACCUM'], 'FLOW_STRENGTH'] += 0.5
    df.loc[df['STEALTH_DISTRIB'], 'FLOW_STRENGTH'] -= 0.5

    # ------------------------------------------------------
    # 5️⃣ Participation classification
    # ------------------------------------------------------
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

    # Ensure contextual displacement exists
    df = dynamic_state_engine(df)
    df = expansion_ignition(df)
    df = expansion_continuation(df)

    # --- Strong body relative to candle range
    body_ratio_series = body / (range_ + 1e-9)
    df['STRONG_BODY'] = hybrid_zscore(body_ratio_series) > 0.5

    # --- ATR expansion confirms real move
    df['ATR_EXPAND'] = df['ATR'] > df['ATR'].rolling(20).mean() * atr_mult

    pressure_z = hybrid_zscore(df['COMPOSITE_PRESSURE'])
    recent_avg = pressure_z.rolling(20).mean()
    recent_std = pressure_z.rolling(20).std()
    df['PRESSURE_ELEVATED_LONG']  = pressure_z > (recent_avg + recent_std)
    df['PRESSURE_ELEVATED_SHORT'] = pressure_z < (recent_avg - recent_std)

    df = expansion_maturity(df)
    df = compression_detector(df)
    compression_ok = df['COMPRESSION_BARS'] >= 3

    df['VALID_BREAK_LONG'] = (
        compression_ok &
        df['EARLY_EXPANSION'] 
        # (df['IGNITION_OK'] |
        # df['CONTINUATION_OK']) 
        # df['PRESSURE_ELEVATED_LONG'] &
        # (df['COMPOSITE_PRESSURE'] > 0) 
    )

    df['VALID_BREAK_SHORT'] = (
        compression_ok &
        df['EARLY_EXPANSION'] 
        # (df['IGNITION_OK'] |
        # df['CONTINUATION_OK']) 
        # df['PRESSURE_ELEVATED_SHORT'] &
        # (df['COMPOSITE_PRESSURE'] < 0) 
    )

    # ======================================================
    # BREAKOUT AGE (ENTRY DECAY CORE)
    # ======================================================
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

def supertrend(df, period=10, multiplier=3, eps=1e-6):
    atr = atr_ema(df, period).round(6)

    hl2 = ((df['high'] + df['low']) / 2).round(6)

    upper_band = (hl2 + multiplier * atr).round(6)
    lower_band = (hl2 - multiplier * atr).round(6)

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    trend = pd.Series(1, index=df.index)

    close = df['close'].round(6)

    for i in range(1, len(df)):

        # stable band logic
        if close.iat[i-1] <= final_upper.iat[i-1] + eps:
            final_upper.iat[i] = min(upper_band.iat[i], final_upper.iat[i-1])
        else:
            final_upper.iat[i] = upper_band.iat[i]

        if close.iat[i-1] >= final_lower.iat[i-1] - eps:
            final_lower.iat[i] = max(lower_band.iat[i], final_lower.iat[i-1])
        else:
            final_lower.iat[i] = lower_band.iat[i]

        # stable trend flip detection
        if close.iat[i] > final_upper.iat[i-1] + eps:
            trend.iat[i] = 1
        elif close.iat[i] < final_lower.iat[i-1] - eps:
            trend.iat[i] = -1
        else:
            trend.iat[i] = trend.iat[i-1]

    df['SUPERTREND'] = trend.astype(int)
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

def htf_structural_stack(df, htf_df,
                         vol_lookback=200,
                         part_lookback=50,
                         regime_window=10,
                         er_window=20):

    htf = htf_df.copy()

    # ======================================================
    # 1️⃣ DIRECTION (Hard Anchor)
    # ======================================================

    htf = supertrend(htf, period=10, multiplier=3)
    htf['HTF_DIRECTION'] = htf['SUPERTREND']  # 1 / -1

    # ======================================================
    # 2️⃣ VOLATILITY STATE (Continuous)
    # ======================================================

    htf['HTF_ATR'] = atr_ema(htf)
    htf['HTF_VOL_PCTL'] = (
        htf['HTF_ATR']
        .rolling(vol_lookback)
        .rank(pct=True)
    )

    # Normalize to 0–1 range around expansion bias
    htf['VOL_SCORE'] = (htf['HTF_VOL_PCTL'] - 0.5).clip(0, 1)

    # ======================================================
    # 3️⃣ PARTICIPATION (Continuous)
    # ======================================================

    htf['HTF_VOL_MA'] = htf['volume'].rolling(part_lookback).mean()
    htf['HTF_VOL_RATIO'] = htf['volume'] / (htf['HTF_VOL_MA'] + 1e-9)

    htf['PART_SCORE'] = ((htf['HTF_VOL_RATIO'] - 1) / 1).clip(0, 1)

    # ======================================================
    # 4️⃣ REGIME PERSISTENCE (Continuous)
    # ======================================================

    direction_series = htf['HTF_DIRECTION']
    htf['HTF_REGIME_PERSIST'] = (
        direction_series
        .rolling(regime_window)
        .apply(lambda x: abs(x.mean()), raw=False)
    )

    htf['REGIME_SCORE'] = htf['HTF_REGIME_PERSIST'].clip(0, 1)

    # ======================================================
    # 5️⃣ STRUCTURE QUALITY (Continuous)
    # ======================================================

    htf['HTF_ER'] = efficiency_ratio(htf['close'], er_window)
    htf['STRUCTURE_SCORE'] = htf['HTF_ER'].clip(0, 1)

    # ======================================================
    # HTF MOMENTUM (Volatility Adjusted Price Slope)
    # ======================================================

    window = 12

    # price slope
    price_slope = (
        htf['close']
        .diff(window)
    )

    # normalize by volatility
    htf['HTF_TREND_MOMENTUM'] = (
        price_slope / (htf['HTF_ATR'] * window + 1e-9)
    )

    # smooth slightly
    htf['HTF_TREND_MOMENTUM'] = (
        htf['HTF_TREND_MOMENTUM']
        .ewm(span=3)
        .mean()
    )

    # normalize
    htf['HTF_TREND_MOMENTUM_NORM'] = hybrid_zscore(
        htf['HTF_TREND_MOMENTUM']
    ).clip(-2,2)

    # convert to 0-1 score
    htf['MOMENTUM_SCORE'] = (
        (htf['HTF_TREND_MOMENTUM_NORM'] + 2) / 4
    ).clip(0,1)

    # ======================================================
    # 6️⃣ COMPOSITE QUALITY SCORE
    # ======================================================

    htf['HTF_QUALITY'] = (
        0.25 * htf['VOL_SCORE'] +
        0.20 * htf['PART_SCORE'] +
        0.20 * htf['REGIME_SCORE'] +
        0.20 * htf['STRUCTURE_SCORE'] +
        0.15 * htf['MOMENTUM_SCORE']
    )

    # ======================================================
    # ALIGN TO LTF
    # ======================================================

    aligned = htf[[
        'HTF_DIRECTION',
        'HTF_QUALITY'
    ]].reindex(df.index, method='ffill')

    return aligned.fillna(0)

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

    ideal_pullback_long = df['PULLBACK_LONG'].between(0.2, 1.2)
    ideal_pullback_short = df['PULLBACK_SHORT'].between(0.2, 1.2)

    df['PBPE_PULLBACK_LONG'] = (
        df['BREAKOUT_WINDOW_LONG'] &
        ideal_pullback_long &
        df['BULL_CONT']
    )

    df['PBPE_PULLBACK_SHORT'] = (
        df['BREAKOUT_WINDOW_SHORT'] &
        ideal_pullback_short &
        df['BEAR_CONT']
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
    df['BREAKOUT_WINDOW_LONG']  = post_breakout_event_window(df['VALID_BREAK_LONG'])
    df['BREAKOUT_WINDOW_SHORT'] = post_breakout_event_window(df['VALID_BREAK_SHORT'])

    # 2) compute metrics
    df = breakout_pullback_metrics(df)
    df = continuation_candle(df)

    # 3) entry types
    df = pullback_entry(df)
    df = micro_break_entry(df)
    df = delayed_continuation(df)

    # 4) final execution signal
    df['ENTRY_LONG'] = (
        df['PBPE_PULLBACK_LONG'] |
        df['PBPE_MICRO_LONG'] |
        df['PBPE_DELAY_LONG']
    )

    df['ENTRY_SHORT'] = (
        df['PBPE_PULLBACK_SHORT'] |
        df['PBPE_MICRO_SHORT'] |
        df['PBPE_DELAY_SHORT']
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
    df['COMPRESSION_SCORE'] *= df['FRESHNESS_SHORT']

    # 5️⃣ Convert to permission (like HTF_OK)
    df['COMPRESSION_OK'] = df['COMPRESSION_SCORE'] > 0.35

    return df

# ==========================================================
# ANCHORED NORMALIZATION ENGINE (Long-term memory)
# ==========================================================
def anchored_zscore(series, min_periods=200):
    """
    Expanding (lifetime) z-score.
    Gives the model permanent memory of what 'normal' is.
    """
    mean = series.expanding(min_periods=min_periods).mean()
    std  = series.expanding(min_periods=min_periods).std()

    return (series - mean) / (std + 1e-9)


def hybrid_zscore(series, roll_window=200, anchor_weight=0.6):
    """
    Combines short-term regime awareness (rolling)
    with long-term memory (expanding).

    This is the institutional normalization pattern.
    """

    # short-term regime normalization
    roll_mean = series.rolling(roll_window).mean()
    roll_std  = series.rolling(roll_window).std()
    rolling_z = (series - roll_mean) / (roll_std + 1e-9)

    # long-term anchored normalization
    anchor_z = anchored_zscore(series)

    # blend them (prevents restrictiveness)
    return anchor_weight * anchor_z + (1 - anchor_weight) * rolling_z

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
def entry_freshness(df, half_life=6):

    # Exponential decay of breakout relevance
    df['FRESHNESS_LONG'] = np.exp(
        -df['BARS_SINCE_LONG_BREAK'] /
        (half_life * df['DECAY_SPEED'])
    )

    df['FRESHNESS_SHORT'] = np.exp(
        -df['BARS_SINCE_SHORT_BREAK'] /
        (half_life * df['DECAY_SPEED'])
    )

    # Late-entry soft floor (prevents total death)
    df['FRESHNESS_LONG']  = df['FRESHNESS_LONG'].clip(lower=0.15)
    df['FRESHNESS_SHORT'] = df['FRESHNESS_SHORT'].clip(lower=0.15)

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

    # Expansion proxy = volatility + participation + trend quality
    expansion_raw = (
        0.4 * df['ATR_ACCEL_NORM'] +
        0.3 * df['FLOW_STRENGTH'] +
        0.3 * df['TREND_QUALITY']
    )

    # Smooth expansion state
    df['EXPANSION_STATE'] = expansion_raw.ewm(span=5).mean()

    # Expansion velocity (growth vs decay)
    df['EXPANSION_VELOCITY'] = df['EXPANSION_STATE'].diff()

    # Expansion persistence (trend maturity)
    df['EXPANSION_PERSISTENCE'] = (
        df['EXPANSION_STATE']
        .rolling(lookback)
        .mean()
    )

    # Normalized maturity score 0 → 1
    exp_min = df['EXPANSION_PERSISTENCE'].expanding(min_periods=lookback).min()
    exp_max = df['EXPANSION_PERSISTENCE'].expanding(min_periods=lookback).max()

    maturity = (
        df['EXPANSION_PERSISTENCE'] - exp_min
    ) / (exp_max - exp_min + 1e-9)

    df['EXPANSION_MATURITY'] = maturity.clip(0, 1)

    # Entry window = early-to-mid expansion only
    df['EARLY_EXPANSION'] = df['EXPANSION_MATURITY'] < 0.6

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

# ==========================================================
# INTEGRATE INTO SIGNAL GENERATION
# ==========================================================
def generate_signal(df, htf_df, atr_mult=1.5):
    if df.empty:
        return df

    now_hour = pd.Timestamp.now(tz="UTC").floor("h")
    df     = df[df.index < now_hour].copy()
    htf_df = htf_df[htf_df.index < now_hour].copy()

    if df.empty or htf_df.empty:
        return df

    # =========================
    # Core processing
    # =========================
    df = trend_bias(df)
    df = wick_rejection(df)
    df = volume_confirmation(df)
    df = support_resistance(df)
    df = breakout_logic(df)

    df['ATR'] = atr_ema(df, period=14)

    df = atr_acceleration(df)
    df = volatility_shock(df)

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
    df = validated_breakouts(df)
    df = entry_freshness(df)
    df = compression_context(df)
    df = temporal_phase_asymmetry(df)
    # --- Dynamic state analytics
    df = dynamic_state_engine(df)

    # =========================
    # NEW HTF STRUCTURAL STACK
    # =========================

    htf_stack = htf_structural_stack(df, htf_df)

    df = pd.concat([df, htf_stack], axis=1)

    HTF_QUALITY_TH = 0.45  # tune 0.40–0.60

    HTF_LONG_OK = (
        (df['HTF_DIRECTION'] == 1) &
        (df['HTF_QUALITY'] > HTF_QUALITY_TH)
    )

    HTF_SHORT_OK = (
        (df['HTF_DIRECTION'] == -1) &
        (df['HTF_QUALITY'] > HTF_QUALITY_TH)
    )

    # =========================
    # PREDICTIVE MODULES
    # =========================
    df = transition_detector(df)
    df = micro_consolidation(df)
    df = momentum_continuity(df)
    df = post_breakout_entry(df)

    # ==========================================================
    # 🧠 PREDICTIVE-WEIGHTED ASYMMETRY ENGINE
    # ==========================================================

    # --- Directional pressure sign
    df['DIR'] = np.sign(df['COMPOSITE_PRESSURE']).fillna(0)

    # ======================================================
    # 1️⃣ DIRECTIONAL LEADING SCORES
    # ======================================================

    phase_long = ((df['PHASE'] == 1) | (df['PHASE'] == 2)).astype(int)
    phase_short = (df['PHASE'] == 3).astype(int)

    lead_long = (
        df['VOL_COMPRESS'].astype(int) +
        phase_long +
        df['STEALTH_ACCUM'].astype(int)
    )

    lead_short = (
        df['VOL_COMPRESS'].astype(int) +
        phase_short +
        df['STEALTH_DISTRIB'].astype(int)
    )

    lead_total = lead_long + lead_short + 1e-9

    df['LEAD_BIAS'] = (lead_long - lead_short) / lead_total
    lead_norm = df['LEAD_BIAS']

    # ======================================================
    # 2️⃣ CONFIRMATION SCORE (reactive signals)
    # ======================================================

    confirm_score = (
        (df['VALID_BREAK_LONG'] | df['VALID_BREAK_SHORT']).astype(int) +
        df['STRONG_BODY'].astype(int) +
        df['ATR_EXPAND'].astype(int) +
        (df['COMPOSITE_PRESSURE'] > 0).astype(int)
    )

    mom = df['MOMENTUM_CONTINUITY']

    df['MOMENTUM_SCORE'] = (
        (mom - 0.5) / 0.2
    ).clip(-1, 1)

    confirm_norm = (confirm_score / 4) + (0.20 * df['MOMENTUM_SCORE'])

    # ===============================
    # RSI + VWAP RISK ADJUSTMENTS
    # ===============================
    rsi_long_ok, rsi_short_ok = rsi_risk_filter(df)
    vwap_long_ok, vwap_short_ok = anchored_vwap_risk(df)

    risk_penalty = (
        (~rsi_long_ok).astype(int) * 0.2 +
        (~rsi_short_ok).astype(int) * 0.2 +
        (~vwap_long_ok).astype(int) * 0.25 +
        (~vwap_short_ok).astype(int) * 0.25
    )

    fail_score = (
        (df['PARTICIPATION'] == -1).astype(int)
    )

    fail_norm = (fail_score) + risk_penalty

    # ======================================================
    # 4️⃣ STRUCTURAL CONTEXT WEIGHT
    # ======================================================

    trend_weight = df['TREND_QUALITY'].abs().clip(0, 1)

    # ======================================================
    # 5️⃣ FINAL ASYMMETRY CALCULATION
    # ======================================================

    micro_weight = 1 + df['MICRO_BREAK_SCORE'].clip(-0.6, 0.6)
    state_weight = (
        df['STATE_STABILITY'] *
        (1 - df['TRANSITION_FORCE'].clip(0,1)) *
        (1 + df['STATE_ACCEL'].clip(-0.5,0.5))
    )
    lead_dynamic_weight = 0.3 + 0.7 * df['TRANSITION_FORCE'].clip(0,1)

    df['ASYM_RAW'] = state_weight * micro_weight * trend_weight * (
        (lead_dynamic_weight * lead_norm) +
        (0.7 * confirm_norm) -
        fail_norm
    )

    # -----------------------------------------
    # Directional Asymmetry Injection
    # -----------------------------------------
    df['DIR'] = np.sign(
        df['COMPOSITE_PRESSURE'].ewm(span=3).mean()
    ).fillna(0)

    df['ASYM_RAW'] *= df['DIR']

    # ======================================================
    # 6️⃣ SMOOTHING (REGIME STABILITY)
    # ======================================================

    df['ASYM_SCORE'] = df['ASYM_RAW'].ewm(span=3, adjust=False).mean() # tune here <-
    
    LONG_CONDITION = (df['VALID_BREAK_LONG'])
    SHORT_CONDITION = (df['VALID_BREAK_SHORT'])

    df['ENTRY_LONG'] = (
        df['ENTRY_LONG'] &
        df['COMPRESSION_OK'] 
        # df['BREAKOUT_HEALTH_LONG']
    )

    df['ENTRY_SHORT'] = (
        df['ENTRY_SHORT'] &
        df['COMPRESSION_OK'] 
        # df['BREAKOUT_HEALTH_SHORT']
    )

    LONG_CONDITION &= df['ENTRY_LONG']
    SHORT_CONDITION &= df['ENTRY_SHORT']

    LONG_CONDITION &= HTF_LONG_OK
    SHORT_CONDITION &= HTF_SHORT_OK

    df['signal'] = 0
    df.loc[LONG_CONDITION, 'signal'] = 1
    df.loc[SHORT_CONDITION, 'signal'] = -1
    # df['signal'] = -1

    df['final_signal'] = df['signal'].shift(1).fillna(0).astype(int)

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

    print(
        f"[DBG-GEN] candles={len(df)} | signals={signal_count} | "
        f"first={df.index[0]} | last={df.index[-1]}"
    )

    return df