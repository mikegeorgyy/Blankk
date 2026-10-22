import os
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

st.set_page_config(page_title="TradeSignal Free", page_icon="📈", layout="wide")
st.title("📈 TradeSignal Free")
st.caption("US stocks · Day trading · Signal-only demo · Alpaca IEX feed")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Alpaca API Key", type="password", help="Use a data/API key from your Alpaca account. Never share it.")
    api_secret = st.text_input("Alpaca API Secret", type="password")
    symbols_text = st.text_input("Symbols (comma-separated)", "AAPL,NVDA,AMD,TSLA")
    lookback = st.selectbox("Historical lookback", ["1 day", "5 days", "10 days"], index=1)
    st.caption("Free feed: IEX only, not the entire US market.")
    refresh = st.button("Refresh data", type="primary", use_container_width=True)

st.warning("Educational prototype. Not financial advice. Signals are rule-based, can be wrong, and are not verified profitable. No orders are placed.")
st.info("This prototype uses completed 1-minute candles when available. Market data can be delayed, incomplete, or unavailable. Check the displayed timestamp and feed before acting.")

if not api_key or not api_secret:
    st.subheader("Get started")
    st.markdown("""
1. Create a free account at [Alpaca](https://alpaca.markets/).
2. In the Alpaca dashboard, create API keys for market-data access.
3. Enter the key and secret in the sidebar. Keys are used only in this running app session; do not publish this app with keys embedded.
4. Deploy/run this app privately. The feed is IEX, so prices/volume can differ from consolidated market data.
""")
    st.stop()

symbols = list(dict.fromkeys(s.strip().upper() for s in symbols_text.split(",") if s.strip()))
if not symbols:
    st.error("Enter at least one ticker symbol.")
    st.stop()

days = {"1 day": 1, "5 days": 5, "10 days": 10}[lookback]
try:
    client = StockHistoricalDataClient(api_key, api_secret)
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=days)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
        limit=10000,
    )
    raw = client.get_stock_bars(request).df
except Exception as e:
    st.error("Could not load data. Verify your Alpaca API keys, data access, network, ticker symbols, and subscription limits.")
    st.code(str(e))
    st.stop()

if raw is None or raw.empty:
    st.warning("No bars returned for these symbols and time range. Try during US market hours or check your symbols/data access.")
    st.stop()

def build_signal(frame):
    d = frame.copy().sort_index()
    # Only completed candles are considered by this prototype; newest bar is excluded to reduce partial-candle signals.
    if len(d) > 1:
        d = d.iloc[:-1].copy()
    if len(d) < 25:
        return d, {"signal": "WAIT", "reason": "Not enough completed 1-minute candles (need at least 25)."}
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    d["rsi14"] = 100 - (100 / (1 + rs))
    typical = (d["high"] + d["low"] + d["close"]) / 3
    # VWAP resets by calendar date in the data's exchange-local index.
    dates = pd.Index(d.index.date)
    d["pv"] = typical * d["volume"]
    d["vwap"] = d["pv"].groupby(dates).cumsum() / d["volume"].groupby(dates).cumsum().replace(0, float("nan"))
    d["vol_avg20"] = d["volume"].rolling(20).mean().shift(1)
    last = d.iloc[-1]
    conditions = {
        "Close above VWAP": last["close"] > last["vwap"],
        "EMA 9 above EMA 21": last["ema9"] > last["ema21"],
        "RSI 14 between 50 and 70": pd.notna(last["rsi14"]) and 50 < last["rsi14"] < 70,
        "Volume above prior 20-bar average": pd.notna(last["vol_avg20"]) and last["volume"] > last["vol_avg20"],
    }
    passed = sum(bool(x) for x in conditions.values())
    signal = "BUY" if all(conditions.values()) else "WAIT"
    reasons = "; ".join(k for k, v in conditions.items() if v) or "No entry conditions passed"
    # Illustrative levels only: stop 0.5% below entry, target 1% above entry (2:1 nominal ratio).
    entry = float(last["close"])
    stop = entry * 0.995
    target = entry * 1.01
    shares = int(2.50 / max(entry - stop, 0.01))
    shares = min(shares, int(250 / entry)) if entry > 0 else 0
    return d, {
        "signal": signal, "passed": passed, "reason": reasons,
        "price": entry, "time": str(d.index[-1]),
        "vwap": float(last["vwap"]), "ema9": float(last["ema9"]),
        "ema21": float(last["ema21"]), "rsi": float(last["rsi14"]) if pd.notna(last["rsi14"]) else None,
        "stop": stop, "target": target, "shares": shares,
        "risk": max(entry-stop, 0) * shares,
        "conditions": conditions,
    }

panels = st.columns(2)
all_results = {}
for idx, symbol in enumerate(symbols):
    try:
        df = raw.xs(symbol, level=0).copy()
    except Exception:
        continue
    d, info = build_signal(df)
    all_results[symbol] = (d, info)
    with panels[idx % 2]:
        with st.container(border=True):
            c1, c2 = st.columns([2, 1])
            c1.subheader(symbol)
            c2.metric("Signal", info["signal"])
            if "price" not in info:
                st.write(info["reason"])
                continue
            c1.metric("Last completed bar", f"${info['price']:.2f}")
            c2.metric("RSI (14)", f"{info['rsi']:.1f}" if info["rsi"] is not None else "N/A")
            st.caption(f"Bar timestamp: {info['time']} · Feed: IEX · Completed bar only")
            st.write(f"**Conditions passed:** {info['passed']}/4")
            for label, ok in info["conditions"].items():
                st.write(("✅ " if ok else "⬜ ") + label)
            st.write(f"**Illustrative entry:** ${info['price']:.2f} · **Stop:** ${info['stop']:.2f} · **Target:** ${info['target']:.2f}")
            st.write(f"**Position-size ceiling:** {info['shares']} shares (up to $250 notional; estimated risk ${info['risk']:.2f})")
            st.caption("Levels are examples, not recommendations. Stop orders do not guarantee execution at the stop price.")
            chart = go.Figure(data=[go.Candlestick(
                x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="1m"
            )])
            chart.add_trace(go.Scatter(x=d.index, y=d["vwap"], name="VWAP"))
            chart.add_trace(go.Scatter(x=d.index, y=d["ema9"], name="EMA 9"))
            chart.add_trace(go.Scatter(x=d.index, y=d["ema21"], name="EMA 21"))
            chart.update_layout(height=320, margin=dict(l=5, r=5, t=25, b=5), xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
            st.plotly_chart(chart, use_container_width=True)

st.divider()
st.subheader("Limitations before real-money use")
st.markdown("""
- IEX is one exchange's feed, not consolidated US market data; signals may differ from broader-market prices/volume.
- This version is a prototype, not a validated strategy. No historical backtest or performance claim is implied.
- The daily $5 loss limit is a manual policy in this signal-only build; it cannot observe positions or enforce stops unless trades are logged.
- Use completed candles; the latest returned bar may still be forming and is excluded.
- Do not paste API keys into a public deployment. For a shared deployment, use server-side secrets and authentication.
""")
