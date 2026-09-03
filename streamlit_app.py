#!/usr/bin/env python3
"""
Momentum Exhaustion Scanner — Streamlit Web App
Mobile-friendly version.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange

st.set_page_config(
    page_title="Momentum Exhaustion Scanner",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

DEFAULT_TICKERS = "AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,AMD,NFLX,SPY,QQQ"
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ADX_PERIOD = 14
VOLUME_SMA_PERIOD = 20
SHORT_MA_PERIOD = 5
MEDIUM_MA_PERIOD = 21
RUN_LOOKBACK = 10

def fetch_social_leaderboard(limit: int = 5) -> List[Dict[str, Any]]:
    url = "https://apewisdom.io/api/v1.0/filter/all-stocks"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])[:limit]
        leaderboard = []
        for item in results:
            leaderboard.append({
                "rank": item.get("rank"),
                "ticker": item.get("ticker"),
                "name": str(item.get("name", "")).replace("&amp;", "&"),
                "mentions": item.get("mentions"),
                "mentions_24h_ago": item.get("mentions_24h_ago"),
                "upvotes": item.get("upvotes"),
            })
        return leaderboard
    except Exception:
        return []

def fetch_data(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df.empty or len(df) < 40:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["RSI"] = RSIIndicator(close=close, window=RSI_PERIOD).rsi()
    macd = MACD(close=close, window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    adx = ADXIndicator(high=high, low=low, close=close, window=ADX_PERIOD)
    df["ADX"] = adx.adx()

    df["EMA5"] = EMAIndicator(close=close, window=SHORT_MA_PERIOD).ema_indicator()
    df["EMA21"] = EMAIndicator(close=close, window=MEDIUM_MA_PERIOD).ema_indicator()

    df["Vol_SMA20"] = volume.rolling(VOLUME_SMA_PERIOD).mean()
    df["Vol_Ratio"] = volume / df["Vol_SMA20"]

    df["Body"] = close - df["Open"]
    df["Upper_Wick"] = high - df[["Open", "Close"]].max(axis=1)
    df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - low
    df["Range"] = high - low
    return df

def detect_recent_run(df: pd.DataFrame, lookback: int = RUN_LOOKBACK) -> Dict[str, Any]:
    recent = df.tail(lookback)
    if len(recent) < lookback:
        return {"in_run": False, "pct_change": 0.0, "up_days": 0}
    pct_change = (recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1) * 100
    up_days = int((recent["Close"] > recent["Close"].shift(1)).sum())
    higher_highs = recent["High"].iloc[-1] > recent["High"].iloc[:-1].max() * 0.98
    in_run = pct_change > 5.0 and up_days >= lookback // 2 and higher_highs
    return {"in_run": bool(in_run), "pct_change": round(pct_change, 2), "up_days": up_days}

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 15) -> Dict[str, Any]:
    if len(df) < lookback + 5:
        return {"divergence": False, "details": ""}
    window = df.tail(lookback).copy()
    price_high_idx = window["Close"].idxmax()
    first_half = window.iloc[: lookback // 2]
    if first_half.empty:
        return {"divergence": False, "details": ""}
    prev_price_high_idx = first_half["Close"].idxmax()
    price_hh = window.loc[price_high_idx, "Close"] > window.loc[prev_price_high_idx, "Close"]
    rsi_lh = window.loc[price_high_idx, "RSI"] < window.loc[prev_price_high_idx, "RSI"]
    divergence = bool(price_hh and rsi_lh and window.loc[price_high_idx, "RSI"] > 55)
    details = ""
    if divergence:
        details = (
            f"Price HH {window.loc[prev_price_high_idx, 'Close']:.2f} -> "
            f"{window.loc[price_high_idx, 'Close']:.2f} | "
            f"RSI LH {window.loc[prev_price_high_idx, 'RSI']:.1f} -> "
            f"{window.loc[price_high_idx, 'RSI']:.1f}"
        )
    return {"divergence": divergence, "details": details}

def detect_volume_climax(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 25:
        return {"climax": False, "details": ""}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    recent_high = df["High"].tail(10).max()
    high_vol = last["Vol_Ratio"] > 1.8
    near_high = last["Close"] > recent_high * 0.97
    prev_high_vol_up = prev["Vol_Ratio"] > 1.7 and prev["Close"] > prev["Open"]
    today_down = last["Close"] < last["Open"]
    climax = bool(
        (high_vol and near_high and last["Upper_Wick"] > abs(last["Body"]) * 0.6)
        or (prev_high_vol_up and today_down and last["Vol_Ratio"] > 1.2)
    )
    details = f"Vol ratio {last['Vol_Ratio']:.1f}x" if climax else ""
    return {"climax": climax, "details": details}

def detect_ma_break(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 10:
        return {"break": False, "details": ""}
    last = df.iloc[-1]
    prev_few = df.tail(6)
    was_above = (prev_few["Close"] > prev_few["EMA5"]).sum() >= 4
    now_below = last["Close"] < last["EMA5"]
    break_signal = bool(was_above and now_below)
    details = f"Close {last['Close']:.2f} < EMA5 {last['EMA5']:.2f}" if break_signal else ""
    return {"break": break_signal, "details": details}

def detect_overbought_weakening(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 20:
        return {"signal": False, "details": ""}
    last = df.iloc[-1]
    prev = df.iloc[-5]
    overbought = last["RSI"] > 70
    adx_falling = last["ADX"] < prev["ADX"] and last["ADX"] > 20
    macd_contracting = last["MACD_hist"] < prev["MACD_hist"] and last["MACD"] > 0
    signal = bool(overbought and (adx_falling or macd_contracting))
    details = f"RSI {last['RSI']:.1f}, ADX {last['ADX']:.1f}" if signal else ""
    return {"signal": signal, "details": details}

def detect_candlestick_warning(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 5:
        return {"warning": False, "pattern": ""}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last["Body"])
    range_ = last["Range"] if last["Range"] > 0 else 1e-6
    shooting_star = (
        last["Upper_Wick"] > body * 1.5
        and last["Lower_Wick"] < body * 0.5
        and last["Close"] < last["Open"]
        and last["Close"] > df["Close"].iloc[-10:-1].mean()
    )
    engulfing = (
        prev["Close"] > prev["Open"]
        and last["Close"] < last["Open"]
        and last["Open"] >= prev["Close"]
        and last["Close"] <= prev["Open"]
    )
    doji = body / range_ < 0.15 and last["Close"] > df["Close"].iloc[-15:].quantile(0.7)
    pattern = ""
    if shooting_star:
        pattern = "Shooting star / long upper wick"
    elif engulfing:
        pattern = "Bearish engulfing"
    elif doji:
        pattern = "Doji / indecision near highs"
    return {"warning": bool(pattern), "pattern": pattern}

def score_exhaustion(signals: Dict[str, Any]) -> int:
    score = 0
    if signals["run"]["in_run"]:
        score += 1
    if signals["rsi_div"]["divergence"]:
        score += 2
    if signals["volume"]["climax"]:
        score += 2
    if signals["ma_break"]["break"]:
        score += 2
    if signals["overbought"]["signal"]:
        score += 1
    if signals["candle"]["warning"]:
        score += 1
    return score

def analyze_ticker(ticker: str, period: str = "6mo") -> Optional[Dict[str, Any]]:
    df = fetch_data(ticker, period=period)
    if df is None:
        return None
    df = add_indicators(df)
    df = df.dropna()
    if len(df) < 40:
        return None
    signals = {
        "run": detect_recent_run(df),
        "rsi_div": detect_rsi_divergence(df),
        "volume": detect_volume_climax(df),
        "ma_break": detect_ma_break(df),
        "overbought": detect_overbought_weakening(df),
        "candle": detect_candlestick_warning(df),
    }
    score = score_exhaustion(signals)
    last = df.iloc[-1]
    return {
        "ticker": ticker,
        "date": str(df.index[-1].date()),
        "close": round(float(last["Close"]), 2),
        "rsi": round(float(last["RSI"]), 1),
        "adx": round(float(last["ADX"]), 1),
        "vol_ratio": round(float(last["Vol_Ratio"]), 2),
        "score": score,
        "signals": signals,
    }

st.title("Momentum Exhaustion Scanner")
st.caption("Find stocks whose strong runs may be ending")

st.subheader("Live Top Talked-About Stocks (Reddit)")
with st.spinner("Loading social leaderboard..."):
    leaderboard = fetch_social_leaderboard(5)

if leaderboard:
    lb_rows = []
    for item in leaderboard:
        mentions = item.get("mentions") or 0
        prev = item.get("mentions_24h_ago") or 0
        delta = ((mentions - prev) / prev * 100) if prev > 0 else 0
        lb_rows.append({
            "Rank": item["rank"],
            "Ticker": item["ticker"],
            "Mentions": mentions,
            "24h change": f"{delta:+.0f}%",
            "Upvotes": item.get("upvotes", 0),
            "Name": item.get("name", "")[:25],
        })
    st.dataframe(pd.DataFrame(lb_rows), use_container_width=True, hide_index=True)
    st.caption("Source: ApeWisdom (Reddit investing communities)")
else:
    st.info("Social leaderboard temporarily unavailable.")

st.divider()
st.subheader("Scan Settings")
tickers_input = st.text_input("Tickers (comma-separated)", value=DEFAULT_TICKERS)
period = st.selectbox("Data period", ["3mo", "6mo", "1y"], index=1)
min_score = st.slider("Minimum alert score", 1, 6, 3)
include_social = st.checkbox("Also scan top Reddit tickers", value=True)
run_button = st.button("Run Scan", type="primary", use_container_width=True)

if run_button:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if include_social and leaderboard:
        social_tickers = [item["ticker"] for item in leaderboard if item.get("ticker")]
        combined = []
        seen = set()
        for t in social_tickers + tickers:
            if t not in seen:
                combined.append(t)
                seen.add(t)
        tickers = combined
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        progress = st.progress(0)
        status = st.empty()
        results = []
        alerts = []
        for i, ticker in enumerate(tickers):
            status.text(f"Scanning {ticker} ({i+1}/{len(tickers)})...")
            progress.progress((i + 1) / len(tickers))
            result = analyze_ticker(ticker, period=period)
            if result:
                results.append(result)
                if result["score"] >= min_score:
                    alerts.append(result)
        progress.empty()
        status.empty()
        st.success(f"Scan complete — {len(alerts)} alert(s) with score >= {min_score}")
        if alerts:
            st.subheader("Exhaustion Alerts")
            for r in sorted(alerts, key=lambda x: -x["score"]):
                with st.expander(f"{r['ticker']}  |  Score {r['score']}/9  |  ${r['close']}  |  RSI {r['rsi']}", expanded=True):
                    s = r["signals"]
                    st.write(f"Date: {r['date']}  |  ADX: {r['adx']}  |  Vol ratio: {r['vol_ratio']}x")
                    if s["run"]["in_run"]:
                        st.write(f"- Recent run: +{s['run']['pct_change']}% ({s['run']['up_days']} up days)")
                    if s["rsi_div"]["divergence"]:
                        st.write(f"- RSI Bearish Divergence: {s['rsi_div']['details']}")
                    if s["volume"]["climax"]:
                        st.write(f"- Volume climax: {s['volume']['details']}")
                    if s["ma_break"]["break"]:
                        st.write(f"- Short MA break: {s['ma_break']['details']}")
                    if s["overbought"]["signal"]:
                        st.write(f"- Overbought + weakening: {s['overbought']['details']}")
                    if s["candle"]["warning"]:
                        st.write(f"- Candle warning: {s['candle']['pattern']}")
        else:
            st.info("No high-confidence exhaustion signals found with the current settings.")
        if results:
            st.subheader("Full Scan Summary")
            summary = pd.DataFrame([{
                "Ticker": r["ticker"],
                "Score": r["score"],
                "Close": r["close"],
                "RSI": r["rsi"],
                "ADX": r["adx"],
                "Vol": r["vol_ratio"],
            } for r in sorted(results, key=lambda x: -x["score"])])
            st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()
st.caption("Educational tool only. Not financial advice. Data: Yahoo Finance + ApeWisdom")
