import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="FRVP Price Action", layout="wide")

st.title("FRVP Price Action — XAUUSD")
st.caption("Fixed Range Volume Profile + price action only")

with st.sidebar:
    st.header("Settings")
    rows = st.number_input("FRVP Row Size", min_value=10, max_value=200, value=60, step=5)
    value_area_pct = st.number_input("Value Area Volume %", min_value=50, max_value=90, value=70, step=1)
    lookback = st.number_input("Fixed Range Bars", min_value=20, max_value=1000, value=150, step=10)
    setup = st.selectbox("Setup", ["All", "POC Bounce", "POC Reversal", "VAH/VAL Breakout"])

    st.divider()
    st.info("Upload OHLCV CSV. Required columns: Open, High, Low, Close, Volume. A Date/Datetime column is optional.")

uploaded = st.file_uploader("Upload XAUUSD OHLCV CSV", type=["csv"])

def normalize_columns(df):
    rename = {}
    for c in df.columns:
        k = c.strip().lower().replace(" ", "").replace("_", "")
        if k in ("date", "datetime", "time", "timestamp"):
            rename[c] = "Date"
        elif k == "open":
            rename[c] = "Open"
        elif k == "high":
            rename[c] = "High"
        elif k == "low":
            rename[c] = "Low"
        elif k == "close":
            rename[c] = "Close"
        elif k in ("volume", "tickvolume"):
            rename[c] = "Volume"
    return df.rename(columns=rename)

def frvp(df, rows=60, va_pct=70):
    d = df.copy()
    lo = float(d["Low"].min())
    hi = float(d["High"].max())
    if hi <= lo:
        raise ValueError("Price range is invalid.")
    edges = np.linspace(lo, hi, rows + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    profile = np.zeros(rows)

    # Distribute each candle's volume across the price bins touched by its range.
    for _, r in d.iterrows():
        low, high, vol = float(r.Low), float(r.High), float(r.Volume)
        if high <= low or vol <= 0:
            idx = np.clip(np.searchsorted(edges, float(r.Close), side="right") - 1, 0, rows - 1)
            profile[idx] += max(vol, 0)
            continue
        first = np.clip(np.searchsorted(edges, low, side="right") - 1, 0, rows - 1)
        last = np.clip(np.searchsorted(edges, high, side="left"), 0, rows - 1)
        touched = np.arange(min(first, last), max(first, last) + 1)
        overlaps = np.maximum(
            0,
            np.minimum(high, edges[touched + 1]) - np.maximum(low, edges[touched])
        )
        if overlaps.sum() > 0:
            profile[touched] += vol * overlaps / overlaps.sum()

    poc_i = int(np.argmax(profile))
    total = profile.sum()
    target = total * va_pct / 100.0

    included = {poc_i}
    cum = profile[poc_i]
    left = poc_i - 1
    right = poc_i + 1
    while cum < target and (left >= 0 or right < rows):
        lv = profile[left] if left >= 0 else -1
        rv = profile[right] if right < rows else -1
        if rv >= lv:
            if right < rows:
                included.add(right); cum += profile[right]; right += 1
            else:
                included.add(left); cum += profile[left]; left -= 1
        else:
            if left >= 0:
                included.add(left); cum += profile[left]; left -= 1
            else:
                included.add(right); cum += profile[right]; right += 1

    va_low_i, va_high_i = min(included), max(included)
    return {
        "edges": edges, "centers": centers, "profile": profile,
        "POC": centers[poc_i],
        "VAL": edges[va_low_i],
        "VAH": edges[va_high_i + 1],
    }

def candle_signal(d, i, level, direction):
    if i < 1:
        return False
    c, p = d.iloc[i], d.iloc[i-1]
    body = abs(c.Close - c.Open)
    rng = max(c.High - c.Low, 1e-9)
    if direction == "buy":
        rejection = c.Low <= level and c.Close > level
        bullish = c.Close > c.Open and body / rng >= 0.35
        engulf = c.Close > p.High and c.Open <= p.Close
        return rejection and (bullish or engulf)
    rejection = c.High >= level and c.Close < level
    bearish = c.Close < c.Open and body / rng >= 0.35
    engulf = c.Close < p.Low and c.Open >= p.Close
    return rejection and (bearish or engulf)

def signal_for(d, profile):
    i = len(d) - 1
    c = d.iloc[i]
    poc, val, vah = profile["POC"], profile["VAL"], profile["VAH"]
    eps = max((vah-val) * 0.08, 1e-9)

    # POC bounce
    if setup in ("All", "POC Bounce"):
        if candle_signal(d, i, poc, "buy"):
            return "BUY", "POC Bounce", c.Close, c.Low, vah
        if candle_signal(d, i, poc, "sell"):
            return "SELL", "POC Bounce", c.Close, c.High, val

    # POC reversal: previous candle is on the other side; current candle reclaims/loses POC.
    if setup in ("All", "POC Reversal") and i >= 1:
        p = d.iloc[i-1]
        if p.Close < poc and c.Close > poc and c.Close > c.Open:
            return "BUY", "POC Reversal", c.Close, min(c.Low, p.Low), vah
        if p.Close > poc and c.Close < poc and c.Close < c.Open:
            return "SELL", "POC Reversal", c.Close, max(c.High, p.High), val

    # Value-area breakout + retest confirmation
    if setup in ("All", "VAH/VAL Breakout") and i >= 1:
        p = d.iloc[i-1]
        if p.Close > vah and c.Low <= vah + eps and c.Close > vah and c.Close > c.Open:
            return "BUY", "VAH Breakout", c.Close, c.Low, None
        if p.Close < val and c.High >= val - eps and c.Close < val and c.Close < c.Open:
            return "SELL", "VAL Breakout", c.Close, c.High, None

    return "WAIT", "No confirmed setup", np.nan, np.nan, np.nan

def dynamic_target(direction, entry, profile, d):
    poc, val, vah = profile["POC"], profile["VAL"], profile["VAH"]
    prev_high = float(d["High"].iloc[:-1].max()) if len(d) > 1 else entry
    prev_low = float(d["Low"].iloc[:-1].min()) if len(d) > 1 else entry

    if direction == "BUY":
        candidates = [x for x in [vah, prev_high] if x > entry]
        return min(candidates) if candidates else entry
    candidates = [x for x in [val, prev_low] if x < entry]
    return max(candidates) if candidates else entry

if uploaded:
    try:
        df = normalize_columns(pd.read_csv(uploaded))
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
            st.stop()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date")
        else:
            df["Date"] = np.arange(len(df))

        df = df.dropna(subset=["Open","High","Low","Close","Volume"]).reset_index(drop=True)
        work = df.tail(int(lookback)).copy()
        profile = frvp(work, int(rows), int(value_area_pct))
        direction, setup_name, entry, sl, first_target = signal_for(work, profile)

        if direction != "WAIT":
            if setup_name.endswith("Breakout"):
                tp = dynamic_target(direction, entry, profile, work)
            else:
                tp = first_target
            risk = abs(entry - sl)
            rr = abs(tp - entry) / risk if risk > 0 else np.nan
        else:
            tp, risk, rr = np.nan, np.nan, np.nan

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("POC", f"{profile['POC']:.2f}")
        col2.metric("VAL", f"{profile['VAL']:.2f}")
        col3.metric("VAH", f"{profile['VAH']:.2f}")
        col4.metric("Signal", direction)

        fig = go.Figure(data=[go.Candlestick(
            x=work["Date"], open=work["Open"], high=work["High"],
            low=work["Low"], close=work["Close"], name="XAUUSD"
        )])
        for y, name, dash in [
            (profile["POC"], "POC", "solid"),
            (profile["VAH"], "VAH", "dash"),
            (profile["VAL"], "VAL", "dash"),
        ]:
            fig.add_hline(y=y, line_dash=dash, annotation_text=name)
        if direction != "WAIT":
            fig.add_hline(y=entry, line_dash="dot", annotation_text="ENTRY")
            fig.add_hline(y=sl, line_dash="dot", annotation_text="SL")
            if not np.isnan(tp):
                fig.add_hline(y=tp, line_dash="dot", annotation_text="TP")
        fig.update_layout(height=650, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trade Plan")
        if direction == "WAIT":
            st.warning("WAIT — no confirmed FRVP price-action setup on the latest candle.")
        else:
            st.success(f"{direction} — {setup_name}")
            out = pd.DataFrame({
                "Item": ["Entry", "Stop Loss", "Target", "Risk", "R:R"],
                "Price": [entry, sl, tp, risk, rr]
            })
            st.dataframe(out, hide_index=True, use_container_width=True)

            if setup_name == "POC Bounce":
                st.write("Exit rule: reject the target → exit. Break and hold beyond the target → adjust target to the next relevant price/volume zone.")
            elif setup_name == "POC Reversal":
                st.write("Exit rule: if price closes back through POC against the position, exit. If the target breaks and holds, adjust to the next zone.")
            else:
                st.write("Exit rule: if price closes back inside value, exit. If breakout acceptance continues, adjust the target upward/downward.")

        st.subheader("Profile")
        prof_df = pd.DataFrame({"Price": profile["centers"], "Volume": profile["profile"]})
        st.bar_chart(prof_df.set_index("Price"), horizontal=True)

    except Exception as e:
        st.error(f"Could not process the file: {e}")
else:
    st.markdown("""
### How to use
1. Export XAUUSD OHLCV data from your charting platform as CSV.
2. Upload it above.
3. Select the fixed range.
4. The app calculates POC, VAH and VAL.
5. It checks only the three FRVP price-action setups.
6. Use the displayed Entry, SL, Target and Exit rule.

**Important:** This is a signal/backtesting aid, not an automatic broker execution system.
""")
