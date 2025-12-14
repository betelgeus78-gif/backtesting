import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# 1. Page Config & CSS Styling
# ---------------------------------------------------------
st.set_page_config(page_title="DCA Backtest Simulator", layout="wide")

# CSS to reduce font size for metrics
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 DCA (Dollar Cost Averaging) Backtest Simulator")

# ---------------------------------------------------------
# 2. Predefined Ticker List
# ---------------------------------------------------------
common_tickers = [
    # [US Indices/Sectors]
    "QQQ", "TQQQ", "QLD", "PSQ", "SQQQ", 
    "SPY", "UPRO", "SSO", 
    "SOXX", "SOXL", "SOXS", 
    "TLT", "TMF", "TMV",
    
    # [Big Tech/Individual]
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX",
    "COIN", "MSTR", 
    
    # [Crypto Top 10]
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
    "DOGE-USD", "ADA-USD", "TRX-USD", "AVAX-USD", "SHIB-USD",

    # [Korean Stocks]
    "005930.KS", "000660.KS"
]

# ---------------------------------------------------------
# 3. Sidebar Inputs (Real-time)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # --- Main Asset ---
    selected_ticker = st.selectbox("Select Asset (Main)", common_tickers, index=0)
    
    st.markdown("---")
    
    # --- Comparison Asset Settings ---
    st.subheader("🆚 Comparison Settings")
    
    # Simulate Leverage Checkbox
    use_simulation = st.checkbox("Simulate Leverage (Daily Rebalancing)", value=False)
    
    comparison_ticker = "None"
    leverage_ratio = 1.0
    comp_label_final = "None"

    if use_simulation:
        # Show leverage slider
        leverage_ratio = st.number_input("Target Leverage (1.0x ~ 5.0x)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        comp_label_final = f"Simulated {leverage_ratio:.1f}x ({selected_ticker})"
    else:
        # Show ticker selector with "None" option
        comp_options = ["None"] + common_tickers
        comparison_ticker = st.selectbox("Select Comparison Asset", comp_options, index=0)
        comp_label_final = comparison_ticker

    st.markdown("---")

    # Date Range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    st.markdown("---")
    
    # Capital & Contribution
    initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
    recurring_amount = st.number_input("Recurring Contribution ($)", value=500, step=100)
    
    # Frequency
    frequency = st.selectbox("Contribution Frequency", ["Monthly", "Weekly", "Daily"], index=0)
    
    st.markdown("---")
    
    # Chart Options
    use_log_scale = st.checkbox("Use Log Scale (Portfolio Value)", value=False)

# ---------------------------------------------------------
# 4. Data & Calculation Functions
# ---------------------------------------------------------
@st.cache_data
def get_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Close']]
    except Exception as e:
        return None

def generate_leveraged_data(df_base, leverage):
    """Generates a synthetic price path based on daily rebalancing leverage."""
    df_sim = df_base.copy()
    df_sim['Base_Return'] = df_sim['Close'].pct_change().fillna(0)
    df_sim['Sim_Return'] = df_sim['Base_Return'] * leverage
    
    start_price = df_sim['Close'].iloc[0]
    df_sim['Close'] = start_price * (1 + df_sim['Sim_Return']).cumprod()
    
    return df_sim[['Close']]

def run_dca_backtest(df, initial_cap, recurring_amt, freq):
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    df['Contribution_Day'] = False
    
    if freq == 'Daily':
        df['Contribution_Day'] = True
    elif freq == 'Weekly':
        df['Week_Num'] = df.index.isocalendar().week
        df['Year_Num'] = df.index.isocalendar().year
        contribution_indices = df.groupby(['Year_Num', 'Week_Num']).head(1).index
        df.loc[contribution_indices, 'Contribution_Day'] = True
    elif freq == 'Monthly':
        df['Month_Num'] = df.index.month
        df['Year_Num'] = df.index.year
        contribution_indices = df.groupby(['Year_Num', 'Month_Num']).head(1).index
        df.loc[contribution_indices, 'Contribution_Day'] = True

    closes = df['Close'].values
    is_contrib = df['Contribution_Day'].values
    
    current_shares_arr = []
    total_invested_arr = []
    
    curr_sh = initial_cap / closes[0]
    tot_inv = initial_cap
    
    for i in range(len(df)):
        price = closes[i]
        if is_contrib[i]:
            new_shares = recurring_amt / price
            curr_sh += new_shares
            tot_inv += recurring_amt
            
        current_shares_arr.append(curr_sh)
        total_invested_arr.append(tot_inv)
        
    df['Holdings_Shares'] = current_shares_arr
    df['Total_Invested'] = total_invested_arr
    df['Portfolio_Value'] = df['Holdings_Shares'] * df['Close']
    df['Peak'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Portfolio_Value'] - df['Peak']) / df['Peak']
    
    return df

def calculate_metrics(df):
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    days = (df.index[-1] - df.index[0]).days
    cagr = ((end_price / start_price) ** (365.25 / days) - 1) * 100 if days > 0 else 0.0

    volatility = df['Daily_Return'].std() * np.sqrt(252) * 100

    mean_return = df['Daily_Return'].mean()
    std_return = df['Daily_Return'].std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0.0

    negative_returns = df.loc[df['Daily_Return'] < 0, 'Daily_Return']
    downside_std = negative_returns.std()
    sortino = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std != 0 else 0.0
    
    final_val = df['Portfolio_Value'].iloc[-1]
    total_inv = df['Total_Invested'].iloc[-1]
    profit = final_val - total_inv
    ret_pct = (profit / total_inv) * 100 if total_inv != 0 else 0
    max_dd = df['Drawdown'].min() * 100

    return {
        "final_val": final_val,
        "total_inv": total_inv,
        "profit": profit,
        "ret_pct": ret_pct,
        "max_dd": max_dd,
        "cagr": cagr,
        "vol": volatility,
        "sharpe": sharpe,
        "sortino": sortino
    }

def display_metrics_block(metrics, title
