import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta # For technical analysis indicators
import time
from datetime import datetime
import pytz # For timezone handling
import telebot # For Telegram alerts (install: pip install pyTelegramBotAPI)
import numpy as np # For numerical operations
import math # For math.floor

# --- Configuration ---
# Replace with your MT5 account details
MT5_LOGIN =  # Your MT5 account login
MT5_PASSWORD =  # Your MT5 account password
MT5_SERVER =  # Your MT5 server name (e.g., "MetaQuotes-Demo")
MT5_PATH = # Path to your MT5 terminal (FIXED SYNTAX)

# Telegram Bot Configuration (Get these from BotFather on Telegram)
TELEGRAM_BOT_TOKEN = # Your Telegram bot token
TELEGRAM_CHAT_ID = # Your chat ID (can be a user ID or group ID)

# Trading Parameters
SYMBOL = "EURUSD" # Trading symbol
LOT_SIZE = 0.01 # Trading volume (in lots)
DEVIATION = 20 # Max price deviation in points for order execution

# Indicator Periods
EMA_SHORT_PERIOD = 20
EMA_LONG_PERIOD = 50
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
RSI_PERIOD = 14
STOCH_RSI_K_PERIOD = 14
STOCH_RSI_D_PERIOD = 3
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_SMOOTH_D = 3
SAR_ACCELERATION = 0.02
SAR_MAX_ACCELERATION = 0.2
ATR_PERIOD = 14 # For dynamic SL/TP

# Multi-timeframe settings
TIMEFRAMES = {
    "4H": mt5.TIMEFRAME_H4,
    "6H": mt5.TIMEFRAME_H6,
    "12H": mt5.TIMEFRAME_H12,
    "1D": mt5.TIMEFRAME_D1,
    "1W": mt5.TIMEFRAME_W1
}

# --- Telegram Bot Initialization ---
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN) # Use the constant here

def send_telegram_message(message):
    """Sends a message to the configured Telegram chat."""
    try:
        bot.send_message(TELEGRAM_CHAT_ID, message) # Use the constant here
        print(f"Telegram message sent: {message}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

# --- MT5 Connection and Data Retrieval ---
def initialize_mt5():
    """Initializes connection to MetaTrader 5 terminal."""
    print(f"Attempting to initialize MT5 from path: {MT5_PATH}") # Added print for debugging
    if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, timeout=120000): # Increased timeout to 120 seconds (120000 ms)
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        send_telegram_message(f"Bot Alert: Failed to initialize MT5: {mt5.last_error()}")
        return False
    print("MetaTrader5 initialized successfully.")
    return True

def shutdown_mt5():
    """Shuts down connection to MetaTrader 5 terminal."""
    mt5.shutdown()
    print("MetaTrader5 shut down.")

def get_ohlc_data(symbol, timeframe, bars=500):
    """Retrieves OHLC data for a given symbol and timeframe."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print(f"No rates data for {symbol} on {timeframe} - {mt5.last_error()}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# --- Technical Indicator Calculations ---
def calculate_indicators(df):
    """Calculates all specified technical indicators for a given DataFrame."""
    if df.empty:
        return pd.DataFrame() # Return empty if input is empty

    # EMA
    df['EMA_Short'] = ta.ema(df['close'], length=EMA_SHORT_PERIOD)
    df['EMA_Long'] = ta.ema(df['close'], length=EMA_LONG_PERIOD)

    # MACD
    macd = df.ta.macd(fast=MACD_FAST_PERIOD, slow=MACD_SLOW_PERIOD, signal=MACD_SIGNAL_PERIOD)
    # print("MACD columns:", macd.columns)  # <--- debug line
    df['MACD_Line'] = macd[macd.columns[0]]
    df['MACD_Histogram'] = macd[macd.columns[1]]
    df['MACD_Signal_Line'] = macd[macd.columns[2]]


    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)

    # Volume Oscillator (using PVO from pandas_ta as a proxy for Volume Oscillator)
    pvo = ta.pvo(df['volume'])
    # print("PVO columns:", pvo.columns)
    df['Volume_Oscillator'] = pvo[pvo.columns[0]] # PVO is Percentage Volume Oscillator, a good proxy

    # SAR (Parabolic SAR)
    sar = ta.psar(df['high'], df['low'], df['close'], af0=SAR_ACCELERATION, afmax=SAR_MAX_ACCELERATION)
    # print("SAR columns:", sar.columns)
    # df['SAR'] = sar[f'PSAR_{SAR_ACCELERATION}_{SAR_MAX_ACCELERATION}']
    df['SAR'] = sar[sar.columns[0]]

    # RSI
    df['RSI'] = ta.rsi(df['close'], length=RSI_PERIOD)

    # Stochastic RSI
    stoch_rsi = ta.stochrsi(df['close'], length=STOCH_RSI_K_PERIOD, rsi_length=RSI_PERIOD, k=STOCH_RSI_SMOOTH_K, d=STOCH_RSI_SMOOTH_D)
    df['StochRSI_K'] = stoch_rsi[f'STOCHRSIk_{RSI_PERIOD}_{STOCH_RSI_K_PERIOD}_{STOCH_RSI_SMOOTH_K}_{STOCH_RSI_SMOOTH_D}']
    df['StochRSI_D'] = stoch_rsi[f'STOCHRSId_{RSI_PERIOD}_{STOCH_RSI_K_PERIOD}_{STOCH_RSI_SMOOTH_K}_{STOCH_RSI_SMOOTH_D}']

    # ATR for dynamic SL/TP
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    # Drop any NaN values that result from indicator calculations
    df.dropna(inplace=True)
    return df

# --- Indicator Alignment Logic --- (Moved to global scope)
# def check_bullish_alignment(df_4h, df_6h, df_12h, df_1d, df_1w):
    # """Checks for bullish alignment across indicators on multiple timeframes."""
    # if df_4h.empty or df_6h.empty or df_12h.empty or df_1d.empty or df_1w.empty:
    #     return False, "Insufficient data for alignment check."

    # # Get the latest values for each timeframe
    # latest_4h = df_4h.iloc[-1]
    # latest_6h = df_6h.iloc[-1]
    # latest_12h = df_12h.iloc[-1]
    # latest_1d = df_1d.iloc[-1]
    # latest_1w = df_1w.iloc[-1]

    # # --- 4H Timeframe Bullish Conditions ---
    # # EMA Crossover: Short EMA above Long EMA
    # ema_4h_bullish = latest_4h['EMA_Short'] > latest_4h['EMA_Long']
    # # MACD: MACD Line above Signal Line and above zero
    # macd_4h_bullish = latest_4h['MACD_Line'] > latest_4h['MACD_Signal_Line'] and latest_4h['MACD_Line'] > 0
    # # Volume Oscillator: Positive (indicating buying pressure)
    # vol_osc_4h_bullish = latest_4h['Volume_Oscillator'] > 0
    # # SAR: Below current price (indicating uptrend)
    # sar_4h_bullish = latest_4h['SAR'] < latest_4h['close']
    # # RSI: Above 50 (indicating bullish momentum)
    # rsi_4h_bullish = latest_4h['RSI'] > 50
    # # Stochastic RSI: K and D lines rising and K above D, and not overbought
    # stoch_rsi_4h_bullish = (latest_4h['StochRSI_K'] > latest_4h['StochRSI_D'] and
    #                         latest_4h['StochRSI_K'] < 80 and latest_4h['StochRSI_D'] < 80)

    # # --- 6H Timeframe Bullish Conditions ---
    # # EMA Crossover: Short EMA above Long EMA
    # ema_6h_bullish = latest_6h['EMA_Short'] > latest_6h['EMA_Long']
    # # MACD: MACD Line above Signal Line and above zero
    # macd_6h_bullish = latest_6h['MACD_Line'] > latest_6h['MACD_Signal_Line'] and latest_6h['MACD_Line'] > 0
    # # Volume Oscillator: Positive (indicating buying pressure)
    # vol_osc_6h_bullish = latest_6h['Volume_Oscillator'] > 0
    # # SAR: Below current price (indicating uptrend)
    # sar_6h_bullish = latest_6h['SAR'] < latest_6h['close']
    # # RSI: Above 50 (indicating bullish momentum)
    # rsi_6h_bullish = latest_6h['RSI'] > 50 # Corrected from latest_4h['RSI']
    # # Stochastic RSI: K and D lines rising and K above D, and not overbought
    # stoch_rsi_6h_bullish = (latest_6h['StochRSI_K'] > latest_6h['StochRSI_D'] and
    #                         latest_6h['StochRSI_K'] < 80 and latest_6h['StochRSI_D'] < 80)

    # # --- 12H Timeframe Bullish Conditions ---
    # # EMA Crossover: Short EMA above Long EMA
    # ema_12h_bullish = latest_12h['EMA_Short'] > latest_12h['EMA_Long']
    # # MACD: MACD Line above Signal Line and above zero
    # macd_12h_bullish = latest_12h['MACD_Line'] > latest_12h['MACD_Signal_Line'] and latest_12h['MACD_Line'] > 0
    # # Volume Oscillator: Positive (indicating buying pressure)
    # vol_osc_12h_bullish = latest_12h['Volume_Oscillator'] > 0
    # # SAR: Below current price (indicating uptrend)
    # sar_12h_bullish = latest_12h['SAR'] < latest_12h['close']
    # # RSI: Above 50 (indicating bullish momentum)
    # rsi_12h_bullish = latest_12h['RSI'] > 50
    # # Stochastic RSI: K and D lines rising and K above D, and not overbought
    # stoch_rsi_12h_bullish = (latest_12h['StochRSI_K'] > latest_12h['StochRSI_D'] and
    #                         latest_12h['StochRSI_K'] < 80 and latest_12h['StochRSI_D'] < 80)

    # # --- 1D Timeframe Bullish Conditions (Confirmation) ---
    # ema_1d_bullish = latest_1d['EMA_Short'] > latest_1d['EMA_Long']
    # macd_1d_bullish = latest_1d['MACD_Line'] > latest_1d['MACD_Signal_Line'] and latest_1d['MACD_Line'] > 0
    # vol_osc_1d_bullish = latest_1d['Volume_Oscillator'] > 0
    # sar_1d_bullish = latest_1d['SAR'] < latest_1d['close']
    # rsi_1d_bullish = latest_1d['RSI'] > 50
    # stoch_rsi_1d_bullish = (latest_1d['StochRSI_K'] > latest_1d['StochRSI_D'] and
    #                         latest_1d['StochRSI_K'] < 80 and latest_1d['StochRSI_D'] < 80)

    # # --- 1W Timeframe Bullish Conditions (Overall Trend) ---
    # ema_1w_bullish = latest_1w['EMA_Short'] > latest_1w['EMA_Long']
    # macd_1w_bullish = latest_1w['MACD_Line'] > latest_1w['MACD_Signal_Line'] and latest_1w['MACD_Line'] > 0

    # # Full Bullish Alignment
    # full_bullish_alignment = (
    #     ema_4h_bullish and macd_4h_bullish and vol_osc_4h_bullish and sar_4h_bullish and rsi_4h_bullish and stoch_rsi_4h_bullish and
    #     ema_6h_bullish and macd_6h_bullish and vol_osc_6h_bullish and sar_6h_bullish and rsi_6h_bullish and stoch_rsi_6h_bullish and # Added 6H
    #     ema_12h_bullish and macd_12h_bullish and vol_osc_12h_bullish and sar_12h_bullish and rsi_12h_bullish and stoch_rsi_12h_bullish and # Added 12H
    #     ema_1d_bullish and macd_1d_bullish and vol_osc_1d_bullish and sar_1d_bullish and rsi_1d_bullish and stoch_rsi_1d_bullish and
    #     ema_1w_bullish and macd_1w_bullish
    # )

    # reason = ""
    # if not full_bullish_alignment:
    #     reason = "Bullish conditions not met: "
    #     if not ema_4h_bullish: reason += "4H EMA, "
    #     if not macd_4h_bullish: reason += "4H MACD, "
    #     if not vol_osc_4h_bullish: reason += "4H Vol Osc, "
    #     if not sar_4h_bullish: reason += "4H SAR, "
    #     if not rsi_4h_bullish: reason += "4H RSI, "
    #     if not stoch_rsi_4h_bullish: reason += "4H StochRSI, "
    #     if not ema_6h_bullish: reason += "6H EMA, " # Added 6H
    #     if not macd_6h_bullish: reason += "6H MACD, " # Added 6H
    #     if not vol_osc_6h_bullish: reason += "6H Vol Osc, " # Added 6H
    #     if not sar_6h_bullish: reason += "6H SAR, " # Added 6H
    #     if not rsi_6h_bullish: reason += "6H RSI, " # Added 6H
    #     if not stoch_rsi_6h_bullish: reason += "6H StochRSI, " # Added 6H
    #     if not ema_12h_bullish: reason += "12H EMA, " # Added 12H
    #     if not macd_12h_bullish: reason += "12H MACD, " # Added 12H
    #     if not vol_osc_12h_bullish: reason += "12H Vol Osc, " # Added 12H
    #     if not sar_12h_bullish: reason += "12H SAR, " # Added 12H
    #     if not rsi_12h_bullish: reason += "12H RSI, " # Added 12H
    #     if not stoch_rsi_12h_bullish: reason += "12H StochRSI, " # Added 12H
    #     if not ema_1d_bullish: reason += "1D EMA, "
    #     if not macd_1d_bullish: reason += "1D MACD, "
    #     if not vol_osc_1d_bullish: reason += "1D Vol Osc, "
    #     if not sar_1d_bullish: reason += "1D SAR, "
    #     if not rsi_1d_bullish: reason += "1D RSI, "
    #     if not stoch_rsi_1d_bullish: reason += "1D StochRSI, "
    #     if not ema_1w_bullish: reason += "1W EMA, "
    #     if not macd_1w_bullish: reason += "1W MACD, "
    #     reason = reason.rstrip(', ') + "."

    # return full_bullish_alignment, reason

def check_bullish_alignment(df_dict):
    """
    df_dict is a dictionary with keys as timeframes ('4h', '6h', etc.)
    and values as the corresponding DataFrames.
    Checks bullish signals for each timeframe independently and gives reasons.
    """
    bullish_results = {}
    bullish_reasons = {}

    for tf, df in df_dict.items():
        if df.empty:
            bullish_results[tf] = False
            bullish_reasons[tf] = f"{Fore.YELLOW}⚠ [{tf}] No data available.{Style.RESET_ALL}"
            continue

        latest = df.iloc[-1]

        ema_bullish = latest['EMA_Short'] > latest['EMA_Long']
        macd_bullish = latest['MACD_Line'] > latest['MACD_Signal_Line'] and latest['MACD_Line'] > 0
        vol_osc_bullish = latest['Volume_Oscillator'] > 0
        sar_bullish = latest['SAR'] < latest['close']
        rsi_bullish = latest['RSI'] > 50
        stoch_rsi_bullish = (
            latest['StochRSI_K'] > latest['StochRSI_D'] and
            latest['StochRSI_K'] < 80 and latest['StochRSI_D'] < 80
        )

        all_bullish = all([
            ema_bullish, macd_bullish, vol_osc_bullish,
            sar_bullish, rsi_bullish, stoch_rsi_bullish
        ])

        bullish_results[tf] = all_bullish

        if all_bullish:
            bullish_reasons[tf] = f"{Fore.GREEN}🟢⬆️🔺 [{tf}] All bullish conditions met!{Style.RESET_ALL}"
        else:
            failed_conditions = []
            if not ema_bullish: failed_conditions.append("EMA")
            if not macd_bullish: failed_conditions.append("MACD")
            if not vol_osc_bullish: failed_conditions.append("Volume Oscillator")
            if not sar_bullish: failed_conditions.append("SAR")
            if not rsi_bullish: failed_conditions.append("RSI")
            if not stoch_rsi_bullish: failed_conditions.append("StochRSI")

            failed_list = f"{Fore.RED}{', '.join(failed_conditions)}{Style.RESET_ALL}"

            bullish_reasons[tf] = (
                f"{Fore.YELLOW}⚠ [{tf}] Bullish conditions NOT met.{Style.RESET_ALL}\n"
                f"{Fore.RED}Missing: {failed_list}{Style.RESET_ALL}"
            )

    return bullish_results, bullish_reasons


# def check_bearish_alignment(df_4h, df_6h, df_12h, df_1d, df_1w): # Added df_6h, df_12h
#     """Checks for bearish alignment across indicators on multiple timeframes."""
#     if df_4h.empty or df_6h.empty or df_12h.empty or df_1d.empty or df_1w.empty:
#         return False, "Insufficient data for alignment check."

#     latest_4h = df_4h.iloc[-1]
#     latest_6h = df_6h.iloc[-1]
#     latest_12h = df_12h.iloc[-1]
#     latest_1d = df_1d.iloc[-1]
#     latest_1w = df_1w.iloc[-1]

#     # --- 4H Timeframe Bearish Conditions ---
#     # EMA Crossover: Short EMA below Long EMA
#     ema_4h_bearish = latest_4h['EMA_Short'] < latest_4h['EMA_Long']
#     # MACD: MACD Line below Signal Line and below zero
#     macd_4h_bearish = latest_4h['MACD_Line'] < latest_4h['MACD_Signal_Line'] and latest_4h['MACD_Line'] < 0
#     # Volume Oscillator: Negative (indicating selling pressure)
#     vol_osc_4h_bearish = latest_4h['Volume_Oscillator'] < 0
#     # SAR: Above current price (indicating downtrend)
#     sar_4h_bearish = latest_4h['SAR'] > latest_4h['close']
#     # RSI: Below 50 (indicating bearish momentum)
#     rsi_4h_bearish = latest_4h['RSI'] < 50
#     # Stochastic RSI: K and D lines falling and K below D, and not oversold
#     stoch_rsi_4h_bearish = (latest_4h['StochRSI_K'] < latest_4h['StochRSI_D'] and
#                             latest_4h['StochRSI_K'] > 20 and latest_4h['StochRSI_D'] > 20)

#     # --- 6H Timeframe Bearish Conditions ---
#     # EMA Crossover: Short EMA below Long EMA
#     ema_6h_bearish = latest_6h['EMA_Short'] < latest_6h['EMA_Long']
#     # MACD: MACD Line below Signal Line and below zero
#     macd_6h_bearish = latest_6h['MACD_Line'] < latest_6h['MACD_Signal_Line'] and latest_6h['MACD_Line'] < 0
#     # Volume Oscillator: Negative (indicating selling pressure)
#     vol_osc_6h_bearish = latest_6h['Volume_Oscillator'] < 0
#     # SAR: Above current price (indicating downtrend)
#     sar_6h_bearish = latest_6h['SAR'] > latest_6h['close']
#     # RSI: Below 50 (indicating bearish momentum)
#     rsi_6h_bearish = latest_6h['RSI'] < 50
#     # Stochastic RSI: K and D lines falling and K below D, and not oversold
#     stoch_rsi_6h_bearish = (latest_6h['StochRSI_K'] < latest_6h['StochRSI_D'] and
#                             latest_6h['StochRSI_K'] > 20 and latest_6h['StochRSI_D'] > 20)

#     # --- 12H Timeframe Bearish Conditions ---
#     # EMA Crossover: Short EMA below Long EMA
#     ema_12h_bearish = latest_12h['EMA_Short'] < latest_12h['EMA_Long']
#     # MACD: MACD Line below Signal Line and below zero
#     macd_12h_bearish = latest_12h['MACD_Line'] < latest_12h['MACD_Signal_Line'] and latest_12h['MACD_Line'] < 0
#     # Volume Oscillator: Negative (indicating selling pressure)
#     vol_osc_12h_bearish = latest_12h['Volume_Oscillator'] < 0
#     # SAR: Above current price (indicating downtrend)
#     sar_12h_bearish = latest_12h['SAR'] > latest_12h['close']
#     # RSI: Below 50 (indicating bearish momentum)
#     rsi_12h_bearish = latest_12h['RSI'] < 50
#     # Stochastic RSI: K and D lines falling and K below D, and not oversold
#     stoch_rsi_12h_bearish = (latest_12h['StochRSI_K'] < latest_12h['StochRSI_D'] and
#                             latest_12h['StochRSI_K'] > 20 and latest_12h['StochRSI_D'] > 20)
    
#     # --- 1D Timeframe Bearish Conditions (Confirmation) ---
#     ema_1d_bearish = latest_1d['EMA_Short'] < latest_1d['EMA_Long']
#     macd_1d_bearish = latest_1d['MACD_Line'] < latest_1d['MACD_Signal_Line'] and latest_1d['MACD_Line'] < 0
#     vol_osc_1d_bearish = latest_1d['Volume_Oscillator'] < 0
#     sar_1d_bearish = latest_1d['SAR'] > latest_1d['close']
#     rsi_1d_bearish = latest_1d['RSI'] < 50
#     stoch_rsi_1d_bearish = (latest_1d['StochRSI_K'] < latest_1d['StochRSI_D'] and
#                             latest_1d['StochRSI_K'] > 20 and latest_1d['StochRSI_D'] > 20)

#     # --- 1W Timeframe Bearish Conditions (Overall Trend) ---
#     ema_1w_bearish = latest_1w['EMA_Short'] < latest_1w['EMA_Long']
#     macd_1w_bearish = latest_1w['MACD_Line'] < latest_1w['MACD_Signal_Line'] and latest_1w['MACD_Line'] < 0

#     # Full Bearish Alignment
#     full_bearish_alignment = (
#         ema_4h_bearish and macd_4h_bearish and vol_osc_4h_bearish and sar_4h_bearish and rsi_4h_bearish and stoch_rsi_4h_bearish and
#         ema_6h_bearish and macd_6h_bearish and vol_osc_6h_bearish and sar_6h_bearish and rsi_6h_bearish and stoch_rsi_6h_bearish and # Added 6H
#         ema_12h_bearish and macd_12h_bearish and vol_osc_12h_bearish and sar_12h_bearish and rsi_12h_bearish and stoch_rsi_12h_bearish and # Added 12H
#         ema_1d_bearish and macd_1d_bearish and vol_osc_1d_bearish and sar_1d_bearish and rsi_1d_bearish and stoch_rsi_1d_bearish and
#         ema_1w_bearish and macd_1w_bearish
#     )

#     reason = ""
#     if not full_bearish_alignment:
#         reason = "Bearish conditions not met: "
#         if not ema_4h_bearish: reason += "4H EMA, "
#         if not macd_4h_bearish: reason += "4H MACD, "
#         if not vol_osc_4h_bearish: reason += "4H Vol Osc, "
#         if not sar_4h_bearish: reason += "4H SAR, "
#         if not rsi_4h_bearish: reason += "4H RSI, "
#         if not stoch_rsi_4h_bearish: reason += "4H StochRSI, "
#         if not ema_6h_bearish: reason += "6H EMA, " # Added 6H
#         if not macd_6h_bearish: reason += "6H MACD, " # Added 6H
#         if not vol_osc_6h_bearish: reason += "6H Vol Osc, " # Added 6H
#         if not sar_6h_bearish: reason += "6H SAR, " # Added 6H
#         if not rsi_6h_bearish: reason += "6H RSI, " # Added 6H
#         if not stoch_rsi_6h_bearish: reason += "6H StochRSI, " # Added 6H
#         if not ema_12h_bearish: reason += "12H EMA, " # Added 12H
#         if not macd_12h_bearish: reason += "12H MACD, " # Added 12H
#         if not vol_osc_12h_bearish: reason += "12H Vol Osc, " # Added 12H
#         if not sar_12h_bearish: reason += "12H SAR, " # Added 12H
#         if not rsi_12h_bearish: reason += "12H RSI, " # Added 12H
#         if not stoch_rsi_12h_bearish: reason += "12H StochRSI, " # Added 12H
#         if not ema_1d_bearish: reason += "1D EMA, "
#         if not macd_1d_bearish: reason += "1D MACD, "
#         if not vol_osc_1d_bearish: reason += "1D Vol Osc, "
#         if not sar_1d_bearish: reason += "1D SAR, "
#         if not rsi_1d_bearish: reason += "1D RSI, "
#         if not stoch_rsi_1d_bearish: reason += "1D StochRSI, "
#         if not ema_1w_bearish: reason += "1W EMA, "
#         if not macd_1w_bearish: reason += "1W MACD, "
#         reason = reason.rstrip(', ') + "."

#     return full_bearish_alignment, reason

def check_bearish_alignment(df_dict):
    """
    df_dict is a dictionary with keys as timeframes ('4h', '6h', etc.)
    and values as the corresponding DataFrames.
    Checks bearish signals for each timeframe independently and gives reasons.
    """
    bearish_results = {}
    bearish_reasons = {}

    for tf, df in df_dict.items():
        if df.empty:
            bearish_results[tf] = False
            bearish_reasons[tf] = "No data"
            continue

        latest = df.iloc[-1]

        ema_bearish = latest['EMA_Short'] < latest['EMA_Long']
        macd_bearish = latest['MACD_Line'] < latest['MACD_Signal_Line'] and latest['MACD_Line'] < 0
        vol_osc_bearish = latest['Volume_Oscillator'] < 0
        sar_bearish = latest['SAR'] > latest['close']
        rsi_bearish = latest['RSI'] < 50
        stoch_rsi_bearish = (
            latest['StochRSI_K'] < latest['StochRSI_D'] and
            latest['StochRSI_K'] > 20 and latest['StochRSI_D'] > 20
        )

        all_bearish = all([
            ema_bearish, macd_bearish, vol_osc_bearish,
            sar_bearish, rsi_bearish, stoch_rsi_bearish
        ])

        bearish_results[tf] = all_bearish

        if all_bearish:
            bearish_reasons[tf] = f"{Fore.RED}🔴⬇️🔻 [{tf}] All bearish conditions met!{Style.RESET_ALL}"
        else:
            failed_conditions = []
            if not ema_bearish: failed_conditions.append("EMA")
            if not macd_bearish: failed_conditions.append("MACD")
            if not vol_osc_bearish: failed_conditions.append("Volume Oscillator")
            if not sar_bearish: failed_conditions.append("SAR")
            if not rsi_bearish: failed_conditions.append("RSI")
            if not stoch_rsi_bearish: failed_conditions.append("StochRSI")

            failed_list = f"{Fore.GREEN}" + ", ".join(failed_conditions) + f"{Style.RESET_ALL}"

            bearish_reasons[tf] = (
                f"{Fore.YELLOW}⚠ [{tf}] Bearish conditions NOT met.\n"
                f"{Fore.GREEN}Missing: {failed_list}{Style.RESET_ALL}"
            )

    return bearish_results, bearish_reasons



# --- Chart Pattern Monitoring (Simplified Example) ---
def detect_chart_patterns(df):
    """
    Simplified chart pattern detection.
    This is a placeholder. Real-world pattern recognition is complex.
    For demonstration, we'll check for simple higher highs/lower lows.
    """
    if len(df) < 3:
        return None # Not enough data for patterns

    # Check for higher high, higher close (bullish trend continuation)
    if df['close'].iloc[-1] > df['close'].iloc[-2] and df['high'].iloc[-1] > df['high'].iloc[-2]:
        return "Bullish Trend Continuation"
    # Check for lower low, lower close (bearish trend continuation)
    if df['close'].iloc[-1] < df['close'].iloc[-2] and df['low'].iloc[-1] < df['low'].iloc[-2]:
        return "Bearish Trend Continuation"
    return None

# --- Trade Management ---
def calculate_sl_tp(current_price, atr_value, trade_type):
    """Calculates dynamic Stop Loss and Take Profit based on ATR."""
    # Multiples can be adjusted based on strategy and risk tolerance
    SL_MULTIPLIER = 1.5
    TP_MULTIPLIER = 3.0

    point = mt5.symbol_info(SYMBOL).point # Get symbol's point value

    if trade_type == "BUY":
        sl = current_price - (atr_value * SL_MULTIPLIER)
        tp = current_price + (atr_value * TP_MULTIPLIER)
        # Adjust to nearest tick for MT5
        sl = round(sl / point) * point
        tp = round(tp / point) * point
    elif trade_type == "SELL":
        sl = current_price + (atr_value * SL_MULTIPLIER)
        tp = current_price - (atr_value * TP_MULTIPLIER)
        # Adjust to nearest tick for MT5
        sl = round(sl / point) * point
        tp = round(tp / point) * point
    else:
        return None, None

    return sl, tp

def open_trade(symbol, trade_type, lot, sl, tp):
    """Opens a buy or sell trade."""
    if trade_type == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif trade_type == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        print("Invalid trade type.")
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": DEVIATION,
        "magic": 20230805, # Unique ID for your bot's trades
        "comment": "Python Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC, # Good Till Cancel
        "type_filling": mt5.ORDER_FILLING_FOK, # Fill Or Kill
    }

    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode} - {mt5.last_error()}")
        send_telegram_message(f"Bot Alert: Order failed for {symbol} ({trade_type}): {result.retcode} - {mt5.last_error()}")
        return None
    else:
        print(f"Order placed successfully: {trade_type} {lot} lots of {symbol} at {price}")
        send_telegram_message(f"Bot Alert: Opened {trade_type} trade for {lot} lots of {symbol}. Ticket: {result.order}")
        return result.order # Return the order ticket

def close_trade(position_ticket):
    """Closes an open position."""
    position = mt5.positions_get(ticket=position_ticket)
    if not position:
        print(f"Position {position_ticket} not found.")
        return False

    position_type = position[0].type
    symbol = position[0].symbol
    volume = position[0].volume

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL if position_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": position_ticket,
        "price": mt5.symbol_info_tick(symbol).bid if position_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
        "deviation": DEVIATION,
        "magic": 20230805,
        "comment": "Python Bot Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Close order failed: {result.retcode} - {mt5.last_error()}")
        send_telegram_message(f"Bot Alert: Close order failed for {symbol} (Ticket: {position_ticket}): {result.retcode} - {mt5.last_error()}")
        return False
    else:
        print(f"Position {position_ticket} closed successfully.")
        send_telegram_message(f"Bot Alert: Closed trade for {symbol}. Ticket: {position_ticket}")
        return True

def monitor_and_exit_trades(open_positions, df_1h_indicators, df_4h_indicators): # Renamed arguments
    """
    Monitors open trades for reversal signs on lower timeframes (1H, 4H)
    and closes them smartly.
    """
    if not open_positions:
        return

    if df_1h_indicators.empty or df_4h_indicators.empty:
        print("Insufficient data for exit monitoring.")
        return

    latest_1h = df_1h_indicators.iloc[-1]
    latest_4h = df_4h_indicators.iloc[-1]

    for pos in open_positions:
        position_ticket = pos.ticket
        position_type = pos.type
        symbol = pos.symbol

        # Check for reversal signs on 1H and 4H
        reversal_detected = False
        reversal_reason = ""

        if position_type == mt5.ORDER_TYPE_BUY: # Currently long, look for bearish reversal
            # 1H Bearish reversal signs
            if (latest_1h['MACD_Line'] < latest_1h['MACD_Signal_Line'] and latest_1h['MACD_Line'] < 0) or \
               (latest_1h['RSI'] < 30) or \
               (latest_1h['StochRSI_K'] < 20 and latest_1h['StochRSI_D'] < 20):
                reversal_detected = True
                reversal_reason = "1H bearish reversal detected."
            # 4H Bearish reversal signs (stronger confirmation)
            if (latest_4h['MACD_Line'] < latest_4h['MACD_Signal_Line'] and latest_4h['MACD_Line'] < 0) or \
               (latest_4h['RSI'] < 30) or \
               (latest_4h['StochRSI_K'] < 20 and latest_4h['StochRSI_D'] < 20):
                reversal_detected = True
                reversal_reason = "4H bearish reversal detected."

        elif position_type == mt5.ORDER_TYPE_SELL: # Currently short, look for bullish reversal
            # 1H Bullish reversal signs
            if (latest_1h['MACD_Line'] > latest_1h['MACD_Signal_Line'] and latest_1h['MACD_Line'] > 0) or \
               (latest_1h['RSI'] > 70) or \
               (latest_1h['StochRSI_K'] > 80 and latest_1h['StochRSI_D'] > 80):
                reversal_detected = True
                reversal_reason = "1H bullish reversal detected."
            # 4H Bullish reversal signs (stronger confirmation)
            if (latest_4h['MACD_Line'] > latest_4h['MACD_Signal_Line'] and latest_4h['MACD_Line'] > 0) or \
               (latest_4h['RSI'] > 70) or \
               (latest_4h['StochRSI_K'] > 80 and latest_4h['StochRSI_D'] > 80):
                reversal_detected = True
                reversal_reason = "4H bullish reversal detected."

        if reversal_detected:
            print(f"Reversal detected for position {position_ticket}: {reversal_reason}. Attempting to close.")
            close_trade(position_ticket)

# --- Main Bot Logic ---
# def run_bot():
    # """Main function to run the trading bot."""
    # if not initialize_mt5():
    #     return

    # # Check if the symbol is available
    # symbol_info = mt5.symbol_info(SYMBOL)
    # if symbol_info is None:
    #     print(f"{SYMBOL} not found, please check the symbol name.")
    #     send_telegram_message(f"Bot Alert: {SYMBOL} not found. Exiting.")
    #     shutdown_mt5()
    #     return

    # if not symbol_info.visible:
    #     print(f"{SYMBOL} is not visible in MarketWatch, trying to select it.")
    #     if not mt5.symbol_select(SYMBOL, True):
    #         print(f"symbol_select({SYMBOL}) failed, exit")
    #         send_telegram_message(f"Bot Alert: Failed to make {SYMBOL} visible. Exiting.")
    #         shutdown_mt5()
    #         return

    # # Get account info
    # account_info = mt5.account_info()
    # if account_info is None:
    #     print(f"Failed to get account info: {mt5.last_error()}")
    #     send_telegram_message(f"Bot Alert: Failed to get account info. Exiting.")
    #     shutdown_mt5()
    #     return
    # print(f"Account: {account_info.login}, Balance: {account_info.balance:.2f} {account_info.currency}")
    # send_telegram_message(f"Bot started! Account: {account_info.login}, Balance: {account_info.balance:.2f} {account_info.currency}")

    # # Main loop
    # while True:
    #     print(f"\n--- Scanning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    #     # Fetch data for all timeframes
    #     data_frames = {}
    #     for tf_name, tf_value in TIMEFRAMES.items():
    #         df = get_ohlc_data(SYMBOL, tf_value, bars=200) # Fetch enough bars for indicators
    #         if not df.empty:
    #             data_frames[tf_name] = calculate_indicators(df)
    #             print(f"Fetched and calculated indicators for {tf_name}. Latest close: {data_frames[tf_name]['close'].iloc[-1]}")
    #         else:
    #             print(f"Could not get data for {tf_name}. Skipping alignment check for this timeframe.")
    #             data_frames[tf_name] = pd.DataFrame() # Ensure it's an empty DataFrame

    #     df_dict = data_frames

    #     # Check for trading signals
    #     bullish_signal, bullish_reason = check_bullish_alignment(df_dict) # Pass all required DFs
    #     bearish_signal, bearish_reason = check_bearish_alignment(df_dict) # Pass all required DFs

    #     # Get current open positions
    #     open_positions = mt5.positions_get(symbol=SYMBOL)

    #     if bullish_signal and not open_positions:
    #         print(f"BULLISH SIGNAL DETECTED! Reason: {bullish_reason}")
    #         send_telegram_message(f"BULLISH SIGNAL DETECTED for {SYMBOL}! Reason: {bullish_reason}")

    #         # Optional: Reinforce with chart patterns (using 4H for entry confidence)
    #         pattern = detect_chart_patterns(df_4h)
    #         if pattern and "Bullish" in pattern:
    #             print(f"Chart pattern reinforcement: {pattern}")
    #             send_telegram_message(f"Chart pattern reinforcement: {pattern}")
                
    #             current_price = mt5.symbol_info_tick(SYMBOL).ask
    #             if not df_4h.empty:
    #                 atr_value = df_4h['ATR'].iloc[-1]
    #                 sl, tp = calculate_sl_tp(current_price, atr_value, "BUY")
    #                 if sl is not None and tp is not None:
    #                     open_trade(SYMBOL, "BUY", LOT_SIZE, sl, tp)
    #                 else:
    #                     print("Could not calculate SL/TP.")
    #             else:
    #                 print("4H data not available for ATR calculation.")
    #         elif not pattern:
    #             print("No reinforcing bullish chart pattern detected.")
    #         else:
    #             print(f"Conflicting chart pattern detected: {pattern}. Skipping trade.")


    #     elif bearish_signal and not open_positions:
    #         print(f"BEARISH SIGNAL DETECTED! Reason: {bearish_reason}")
    #         send_telegram_message(f"BEARISH SIGNAL DETECTED for {SYMBOL}! Reason: {bearish_reason}")

    #         # Optional: Reinforce with chart patterns (using 4H for entry confidence)
    #         pattern = detect_chart_patterns(df_4h)
    #         if pattern and "Bearish" in pattern:
    #             print(f"Chart pattern reinforcement: {pattern}")
    #             send_telegram_message(f"Chart pattern reinforcement: {pattern}")

    #             current_price = mt5.symbol_info_tick(SYMBOL).bid
    #             if not df_4h.empty:
    #                 atr_value = df_4h['ATR'].iloc[-1]
    #                 sl, tp = calculate_sl_tp(current_price, atr_value, "SELL")
    #                 if sl is not None and tp is not None:
    #                     open_trade(SYMBOL, "SELL", LOT_SIZE, sl, tp)
    #                 else:
    #                     print("Could not calculate SL/TP.")
    #             else:
    #                 print("4H data not available for ATR calculation.")
    #         elif not pattern:
    #             print("No reinforcing bearish chart pattern detected.")
    #         else:
    #             print(f"Conflicting chart pattern detected: {pattern}. Skipping trade.")

    #     elif open_positions:
    #         print(f"Currently have {len(open_positions)} open position(s). Monitoring for exits.")
    #         # Fetch 1H data for exit monitoring
    #         df_1h_exit = get_ohlc_data(SYMBOL, mt5.TIMEFRAME_H1, bars=50)
    #         # Pass the already calculated 4H indicators for exit monitoring
    #         monitor_and_exit_trades(open_positions, calculate_indicators(df_1h_exit.copy()), df_4h) # Pass 4H data as well

    #     else:
    #         print(f"No trading signals detected. {bullish_reason} | {bearish_reason}")


    #     # Wait for a defined interval before the next scan
    #     # For a 4H bot, you might check every hour or every 30 minutes.
    #     # For demonstration, we'll use a shorter interval.
    #     time.sleep(300) # Check every 5 minutes (300 seconds)

def run_bot():
    """Main function to run the trading bot."""
    if not initialize_mt5():
        return

    # Check if the symbol is available
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"{SYMBOL} not found, please check the symbol name.")
        send_telegram_message(f"Bot Alert: {SYMBOL} not found. Exiting.")
        shutdown_mt5()
        return

    if not symbol_info.visible:
        print(f"{SYMBOL} is not visible in MarketWatch, trying to select it.")
        if not mt5.symbol_select(SYMBOL, True):
            print(f"symbol_select({SYMBOL}) failed, exit")
            send_telegram_message(f"Bot Alert: Failed to make {SYMBOL} visible. Exiting.")
            shutdown_mt5()
            return

    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        print(f"Failed to get account info: {mt5.last_error()}")
        send_telegram_message(f"Bot Alert: Failed to get account info. Exiting.")
        shutdown_mt5()
        return

    print(f"Account: {account_info.login}, Balance: {account_info.balance:.2f} {account_info.currency}")
    send_telegram_message(f"Bot started! Account: {account_info.login}, Balance: {account_info.balance:.2f} {account_info.currency}")

    # Main loop
    while True:
        print(f"\n--- Scanning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        # Fetch data for all timeframes
        data_frames = {}
        for tf_name, tf_value in TIMEFRAMES.items():
            df = get_ohlc_data(SYMBOL, tf_value, bars=200)
            if not df.empty:
                data_frames[tf_name] = calculate_indicators(df)
                print(f"Fetched and calculated indicators for {tf_name}. Latest close: {data_frames[tf_name]['close'].iloc[-1]}")
            else:
                print(f"Could not get data for {tf_name}. Skipping signal check.")
                data_frames[tf_name] = pd.DataFrame()

        df_dict = data_frames
        df_4h = df_dict.get('4h', pd.DataFrame())

        # Check for trading signals
        bullish_results, bullish_reasons = check_bullish_alignment(df_dict)
        bearish_results, bearish_reasons = check_bearish_alignment(df_dict)

        # Get current open positions
        open_positions = mt5.positions_get(symbol=SYMBOL) or []

        # ✅ Independent signal sending per timeframe
        for tf in TIMEFRAMES.keys():
            # Bullish signal
            if bullish_results.get(tf) and not open_positions:
                print(f"📈 Bullish signal detected on {tf} timeframe! Reason: {bullish_reasons.get(tf, 'N/A')}")
                send_telegram_message(f"📈 Bullish signal detected for {SYMBOL} on {tf} timeframe! {bullish_reasons.get(tf, '')}")

                if tf == '4h':
                    pattern = detect_chart_patterns(df_4h)
                    if pattern and "Bullish" in pattern:
                        send_telegram_message(f"Chart pattern reinforcement: {pattern}")
                    current_price = mt5.symbol_info_tick(SYMBOL).ask
                    if not df_4h.empty:
                        atr_value = df_4h['ATR'].iloc[-1]
                        sl, tp = calculate_sl_tp(current_price, atr_value, "BUY")
                        if sl and tp:
                            open_trade(SYMBOL, "BUY", LOT_SIZE, sl, tp)

            # Bearish signal
            elif bearish_results.get(tf) and not open_positions:
                print(f"📉 Bearish signal detected on {tf} timeframe! Reason: {bearish_reasons.get(tf, 'N/A')}")
                send_telegram_message(f"📉 Bearish signal detected for {SYMBOL} on {tf} timeframe! {bearish_reasons.get(tf, '')}")

                if tf == '4h':
                    pattern = detect_chart_patterns(df_4h)
                    if pattern and "Bearish" in pattern:
                        send_telegram_message(f"Chart pattern reinforcement: {pattern}")
                    current_price = mt5.symbol_info_tick(SYMBOL).bid
                    if not df_4h.empty:
                        atr_value = df_4h['ATR'].iloc[-1]
                        sl, tp = calculate_sl_tp(current_price, atr_value, "SELL")
                        if sl and tp:
                            open_trade(SYMBOL, "SELL", LOT_SIZE, sl, tp)

        # Monitor open positions
        if open_positions:
            print(f"Currently have {len(open_positions)} open position(s). Monitoring for exits.")
            df_1h_exit = get_ohlc_data(SYMBOL, mt5.TIMEFRAME_H1, bars=50)
            monitor_and_exit_trades(open_positions, calculate_indicators(df_1h_exit.copy()), df_4h)
        else:
            print(f"No trading signals detected in this scan.")
            # Print results in color instead of dumping the dict
            print(f"{Fore.CYAN}Bullish Condition Check Results:{Style.RESET_ALL}")
            for tf in sorted(bullish_reasons.keys()):
                print(bullish_reasons[tf])
            print(f"{Fore.CYAN}Bearish Condition Check Results:{Style.RESET_ALL}")
            for tf in sorted(bearish_reasons.keys()):
                print(bearish_reasons[tf])

        # Wait before the next scan
        time.sleep(300)


# --- Run the Bot ---
if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        shutdown_mt5()
