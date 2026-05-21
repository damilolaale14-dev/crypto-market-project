import pandas as pd
from indicators.indicators import compute_htf_scores

htf = pd.read_parquet("data/cache/BTCUSDT_4h.parquet")
htf.index = pd.to_datetime(htf.index, utc=True)
scores = compute_htf_scores(htf)

print(scores['HTF_QUALITY'].describe())
print("% above 0.45:", (scores['HTF_QUALITY'] > 0.45).mean())