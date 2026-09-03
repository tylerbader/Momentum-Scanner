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
    df["Lower_
