# (FULL FINAL CODE WITH EVERYTHING)

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Smart Investment Advisor", layout="wide")
st.title("📈 Stock Market Technical Analysis and Forecasting Dashboard")
st.markdown("Analyze stock trends, indicators, signals, and future price outlook.")

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------
st.sidebar.header("Input Parameters")

# Dropdown for companies
company_dict = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "HCL Tech": "HCLTECH.NS",
    "L&T": "LT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "Maruti Suzuki": "MARUTI.NS"
}

selected_company = st.sidebar.selectbox("Select Company", list(company_dict.keys()))
ticker = company_dict[selected_company]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
forecast_days = st.sidebar.slider("Forecast Days", 7, 30, 15)

# Financial inputs
st.sidebar.subheader("💰 Your Financial Details")
income = st.sidebar.number_input("Monthly Income (₹)", value=50000.0, step=1000.0)
expenses = st.sidebar.number_input("Monthly Expenses (₹)", value=20000.0, step=1000.0)
savings = st.sidebar.number_input("Total Savings (₹)", value=100000.0, step=1000.0)
risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"])

# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------
@st.cache_data
def load_data(ticker_symbol, start, end):
    data = yf.download(ticker_symbol, start=start, end=end, auto_adjust=False)

    if data.empty:
        return data

    # Flatten MultiIndex columns if returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[required_cols].copy()
    data.dropna(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found. Please check the ticker symbol or date range.")
    st.stop()

# ---------------------------------------------------
# INDICATOR FUNCTIONS
# ---------------------------------------------------
def add_indicators(df):
    df = df.copy()

    # Moving averages
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    # Bollinger Bands
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA20"] + (2 * rolling_std)
    df["BB_Lower"] = df["SMA20"] - (2 * rolling_std)

    # Daily Return
    df["Daily Return"] = df["Close"].pct_change()

    # Volatility (20-day rolling std of returns)
    df["Volatility"] = df["Daily Return"].rolling(window=20).std() * np.sqrt(252)

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

data = add_indicators(data)

# ---------------------------------------------------
# BASIC INFO
# ---------------------------------------------------
latest_close = float(data["Close"].iloc[-1])
latest_sma20 = float(data["SMA20"].iloc[-1]) if not pd.isna(data["SMA20"].iloc[-1]) else None
latest_sma50 = float(data["SMA50"].iloc[-1]) if not pd.isna(data["SMA50"].iloc[-1]) else None
latest_rsi = float(data["RSI"].iloc[-1]) if not pd.isna(data["RSI"].iloc[-1]) else None
latest_macd = float(data["MACD"].iloc[-1]) if not pd.isna(data["MACD"].iloc[-1]) else None
latest_macd_signal = float(data["MACD_Signal"].iloc[-1]) if not pd.isna(data["MACD_Signal"].iloc[-1]) else None
latest_volatility = float(data["Volatility"].iloc[-1]) if not pd.isna(data["Volatility"].iloc[-1]) else None

# ---------------------------------------------------
# SIGNAL LOGIC
# ---------------------------------------------------
def generate_signal(close, sma20, sma50, rsi, macd, macd_signal):
    signal = "HOLD"
    reason = []

    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            reason.append("short-term trend is bullish (SMA20 > SMA50)")
        else:
            reason.append("short-term trend is weak (SMA20 < SMA50)")

    if rsi is not None:
        if rsi < 30:
            reason.append("RSI indicates oversold conditions")
        elif rsi > 70:
            reason.append("RSI indicates overbought conditions")
        else:
            reason.append("RSI is in neutral range")

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            reason.append("MACD is above signal line (bullish momentum)")
        else:
            reason.append("MACD is below signal line (bearish momentum)")

    # Basic rule-based signal
    if sma20 is not None and sma50 is not None and rsi is not None and macd is not None and macd_signal is not None:
        if sma20 > sma50 and macd > macd_signal and rsi < 70:
            signal = "BUY"
        elif sma20 < sma50 and macd < macd_signal and rsi > 30:
            signal = "SELL"
        else:
            signal = "HOLD"

    return signal, "; ".join(reason)

signal, signal_reason = generate_signal(
    latest_close, latest_sma20, latest_sma50, latest_rsi, latest_macd, latest_macd_signal
)

# ---------------------------------------------------
# ML FEATURE ENGINEERING
# ---------------------------------------------------
def create_ml_features(df, forecast_horizon=7):
    df = df.copy()

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)
    df["Lag_5"] = df["Close"].shift(5)
    df["Lag_10"] = df["Close"].shift(10)

    df["Rolling_Mean_5"] = df["Close"].rolling(5).mean()
    df["Rolling_Mean_10"] = df["Close"].rolling(10).mean()
    df["Rolling_STD_5"] = df["Close"].rolling(5).std()

    df["Target"] = df["Close"].shift(-forecast_horizon)

    df.dropna(inplace=True)

    feature_cols = [
        "Lag_1", "Lag_2", "Lag_3", "Lag_5", "Lag_10",
        "Rolling_Mean_5", "Rolling_Mean_10", "Rolling_STD_5",
        "SMA20", "SMA50", "RSI", "MACD", "MACD_Signal", "Volatility"
    ]

    X = df[feature_cols]
    y = df["Target"]

    return df, X, y, feature_cols

ml_df, X, y, feature_cols = create_ml_features(data, forecast_horizon=forecast_days)

if len(ml_df) < 50:
    st.warning("Not enough data for reliable ML forecasting. Try a larger date range.")
    st.stop()

# ---------------------------------------------------
# TRAIN / TEST SPLIT (TIME-BASED)
# ---------------------------------------------------
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ---------------------------------------------------
# FUTURE PREDICTION
# ---------------------------------------------------
latest_features = X.iloc[[-1]]
predicted_price = float(model.predict(latest_features)[0])

current_price = float(data["Close"].iloc[-1])

predicted_direction = "INCREASE" if predicted_price > current_price else "DECREASE"

# ---------------------------------------------------
# INVESTMENT CALCULATION
# ---------------------------------------------------
disposable_income = income - expenses

if disposable_income > 0:
    investment_budget = disposable_income * 0.4
else:
    investment_budget = savings * 0.1

growth_percent = ((predicted_price - current_price) / current_price) * 100

# Risk-based allocation
if risk_level == "Low":
    ratio = 0.2
elif risk_level == "Medium":
    ratio = 0.4
else:
    ratio = 0.6

# Decision logic
if growth_percent > 5:
    decision = "STRONG BUY"
    invest_amount = investment_budget * ratio
elif growth_percent > 0:
    decision = "BUY"
    invest_amount = investment_budget * ratio * 0.7
else:
    decision = "AVOID"
    invest_amount = 0

expected_profit = invest_amount * (growth_percent / 100)

# ---------------------------------------------------
# METRICS DISPLAY
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Latest Close", f"{latest_close:.2f}")
col2.metric("RSI", f"{latest_rsi:.2f}" if latest_rsi is not None else "N/A")
col3.metric("Volatility", f"{latest_volatility:.2%}" if latest_volatility is not None else "N/A")
col4.metric("Signal", signal)

st.markdown("---")

# ---------------------------------------------------
# PRICE + INDICATOR SUMMARY
# ---------------------------------------------------
st.subheader("📋 Analysis Summary")

summary_data = {
    "Ticker": ticker,
    "Current Price": round(current_price, 2),
    "SMA20": round(latest_sma20, 2) if latest_sma20 else None,
    "SMA50": round(latest_sma50, 2) if latest_sma50 else None,
    "RSI": round(latest_rsi, 2) if latest_rsi else None,
    "MACD": round(latest_macd, 2) if latest_macd else None,
    "MACD Signal": round(latest_macd_signal, 2) if latest_macd_signal else None,
    "Signal": signal,
    "Predicted Future Price": round(predicted_price, 2),
    "Predicted Direction": predicted_direction,
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R² Score": round(r2, 4)
}

st.json(summary_data)

st.info(f"**Reason for Signal:** {signal_reason}")

# ---------------------------------------------------
# INVESTMENT RECOMMENDATION DISPLAY
# ---------------------------------------------------
st.subheader("💡 Personalized Investment Recommendation")

colA, colB, colC = st.columns(3)

colA.metric("Investment Budget", f"₹{investment_budget:.2f}")
colB.metric("Recommended Investment", f"₹{invest_amount:.2f}")
colC.metric("Expected Profit", f"₹{expected_profit:.2f}")

st.write(f"📊 Expected Growth: {growth_percent:.2f}%")
st.write(f"📢 Recommendation: {decision}")

# ---------------------------------------------------
# CANDLESTICK CHART
# ---------------------------------------------------
st.subheader("🕯 Candlestick Chart")

# Use last 100 days to avoid mplfinance error
plot_data = data.tail(100).copy()

add_plots = [
    mpf.make_addplot(plot_data["SMA20"], color="blue"),
    mpf.make_addplot(plot_data["SMA50"], color="orange"),
    mpf.make_addplot(plot_data["BB_Upper"], color="green"),
    mpf.make_addplot(plot_data["BB_Lower"], color="red"),
]

fig, _ = mpf.plot(
    plot_data,
    type="candle",
    volume=True,
    addplot=add_plots,
    style="yahoo",
    returnfig=True,
    figsize=(12, 8),
    title=f"{ticker} Candlestick Chart"
)

st.pyplot(fig)

# ---------------------------------------------------
# CLOSE PRICE TREND
# ---------------------------------------------------
st.subheader("📈 Closing Price Trend")

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(data.index, data["Close"], label="Close Price")
ax1.plot(data.index, data["SMA20"], label="SMA20")
ax1.plot(data.index, data["SMA50"], label="SMA50")
ax1.set_title("Close Price with Moving Averages")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# ---------------------------------------------------
# RSI CHART
# ---------------------------------------------------
st.subheader("📉 RSI Chart")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(data.index, data["RSI"], label="RSI")
ax2.axhline(70, linestyle="--")
ax2.axhline(30, linestyle="--")
ax2.set_title("Relative Strength Index (RSI)")
ax2.set_xlabel("Date")
ax2.set_ylabel("RSI")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# ---------------------------------------------------
# MACD CHART
# ---------------------------------------------------
st.subheader("📊 MACD Chart")

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(data.index, data["MACD"], label="MACD")
ax3.plot(data.index, data["MACD_Signal"], label="Signal Line")
ax3.set_title("MACD Indicator")
ax3.set_xlabel("Date")
ax3.set_ylabel("Value")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------
st.subheader("🤖 Actual vs Predicted Prices")

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
}, index=y_test.index)

fig4, ax4 = plt.subplots(figsize=(12, 5))
ax4.plot(comparison_df.index, comparison_df["Actual"], label="Actual")
ax4.plot(comparison_df.index, comparison_df["Predicted"], label="Predicted")
ax4.set_title("Actual vs Predicted Prices")
ax4.set_xlabel("Date")
ax4.set_ylabel("Price")
ax4.legend()
ax4.grid(True)
st.pyplot(fig4)

# ---------------------------------------------------
# RETURNS DISTRIBUTION
# ---------------------------------------------------
st.subheader("📌 Daily Returns Overview")

fig5, ax5 = plt.subplots(figsize=(12, 4))
ax5.hist(data["Daily Return"].dropna(), bins=50)
ax5.set_title("Distribution of Daily Returns")
ax5.set_xlabel("Daily Return")
ax5.set_ylabel("Frequency")
ax5.grid(True)
st.pyplot(fig5)

# ---------------------------------------------------
# RAW DATA
# ---------------------------------------------------
with st.expander("View Raw Data"):
    st.dataframe(data.tail(50))

# ---------------------------------------------------
# FINAL INTERPRETATION
# ---------------------------------------------------
st.subheader("📢 Final Interpretation")

st.write(f"**Current Price:** {current_price:.2f}")
st.write(f"**Predicted Price after {forecast_days} days:** {predicted_price:.2f}")
st.write(f"**Forecast suggests stock may:** {predicted_direction}")
st.write(f"**Trading Signal:** {signal}")
st.write(f"**Model Performance:** MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}")

st.warning(
    "This project is for educational and analytical purposes only. "
    "It should not be treated as financial advice."
)

from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Stock Market!"

# This should be at the very bottom of the file
if __name__ == "__main__":
    app.run()
