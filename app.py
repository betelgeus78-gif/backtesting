import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# 1. Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="DCA Backtest Simulator", layout="wide")
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
# 3. Sidebar Inputs (Run Button Removed -> Real-time)
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
        # Add "None" to the options list just for this selectbox
        comp_options = ["None"] + common_tickers
        comparison_ticker = st.selectbox("Select Comparison Asset", comp_options, index=0) # Default index 0 is "None"
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
    use_log_scale = st.checkbox("Use Log Scale (Price & Portfolio)", value=False)

    # No Run Button -> Logic executes immediately below

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
    
    # Reconstruct Price Path
    start_price = df_sim['Close'].iloc[0]
    df_sim['Close'] = start_price * (1 + df_sim['Sim_Return']).cumprod()
    
    return df_sim[['Close']]

def run_dca_backtest(df, initial_cap, recurring_amt, freq):
    df = df.copy()
    
    # Calculate returns for metrics
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    
    # Initialize simulation columns
    df['Contribution_Day'] = False
    
    # Setup resampling rule
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

    # Fast Iteration
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
    
    # Drawdown
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

    return cagr, volatility, sharpe, sortino

# ---------------------------------------------------------
# 5. Plotting Function (Handles None comparison)
# ---------------------------------------------------------
def plot_comparison_charts(df_main, df_comp, name_main, name_comp, log_scale):
    y_axis_type = "log" if log_scale else "linear"
    has_comp = df_comp is not None
    
    # --- 1. Asset Price Chart ---
    fig_price = go.Figure()
    # Main Asset
    fig_price.add_trace(go.Scatter(
        x=df_main.index, y=df_main['Close'],
        mode='lines', name=f'{name_main} Price',
        line=dict(color='black', width=1.5)
    ))
    # Comparison Asset
    if has_comp:
        fig_price.add_trace(go.Scatter(
            x=df_comp.index, y=df_comp['Close'],
            mode='lines', name=f'{name_comp} Price',
            line=dict(color='orange', width=1.5, dash='solid')
        ))
    
    fig_price.update_layout(
        title=f'📊 1. Asset Price Comparison',
        xaxis_title='Date', yaxis_title='Price ($)',
        yaxis_type=y_axis_type, template='plotly_white', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --- 2. Portfolio Value Chart ---
    fig_value = go.Figure()
    
    # Main Portfolio
    fig_value.add_trace(go.Scatter(
        x=df_main.index, y=df_main['Portfolio_Value'],
        mode='lines', name=f'{name_main} Portfolio',
        line=dict(color='red', width=1.5)
    ))
    
    # Comparison Portfolio
    if has_comp:
        fig_value.add_trace(go.Scatter(
            x=df_comp.index, y=df_comp['Portfolio_Value'],
            mode='lines', name=f'{name_comp} Portfolio',
            line=dict(color='orange', width=1.5)
        ))

    # Total Invested (Common Principal)
    fig_value.add_trace(go.Scatter(
        x=df_main.index, y=df_main['Total_Invested'],
        mode='lines', name='Total Invested (Principal)',
        line=dict(color='gray', width=1.0, dash='dash')
    ))

    fig_value.update_layout(
        title=f'💰 2. Portfolio Value Comparison',
        xaxis_title='Date', yaxis_title='Value ($)',
        yaxis_type=y_axis_type, template='plotly_white', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_value, use_container_width=True)

    # --- 3. Drawdown Chart ---
    fig_dd = go.Figure()
    
    # Main Drawdown
    fig_dd.add_trace(go.Scatter(
        x=df_main.index, y=df_main['Drawdown'] * 100,
        mode='lines', name=f'{name_main} DD',
        fill='tozeroy',
        line=dict(color='blue', width=1.0)
    ))
    
    # Comparison Drawdown
    if has_comp:
        fig_dd.add_trace(go.Scatter(
            x=df_comp.index, y=df_comp['Drawdown'] * 100,
            mode='lines', name=f'{name_comp} DD',
            line=dict(color='orange', width=1.0)
        ))
    
    fig_dd.update_layout(
        title='🌊 3. Drawdown Comparison (%)',
        xaxis_title='Date', yaxis_title='Drawdown (%)',
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------
# 6. Main Execution (Real-time)
# ---------------------------------------------------------
# Spinner runs immediately on every rerun
with st.spinner(f'Processing Simulation...'):
    # 1. Fetch Main Data
    df_main = get_data(selected_ticker, start_date, end_date)
    
    # 2. Prepare Comparison Data
    df_comp = None
    res_comp = None
    
    if df_main is not None and not df_main.empty:
        
        # --- Logic for Comparison Data ---
        if use_simulation:
            # Scenario A: Synthetic Leverage
            df_comp = generate_leveraged_data(df_main, leverage_ratio)
        elif comparison_ticker != "None":
            # Scenario B: Real Asset Comparison
            if comparison_ticker == selected_ticker:
                df_comp = df_main.copy()
            else:
                df_comp_raw = get_data(comparison_ticker, start_date, end_date)
                if df_comp_raw is not None and not df_comp_raw.empty:
                    # Reindex to match Main Asset's dates exactly
                    df_comp = df_comp_raw.reindex(df_main.index).ffill().dropna()
                else:
                    st.warning(f"Could not fetch data for {comparison_ticker}. Comparison skipped.")
        
        # --- Run Backtest for Main ---
        res_main = run_dca_backtest(df_main, initial_capital, recurring_amount, frequency)
        
        # --- Run Backtest for Comp (if exists) ---
        if df_comp is not None:
            res_comp = run_dca_backtest(df_comp, initial_capital, recurring_amount, frequency)
            
        # --- Metrics for Main Asset ---
        final_val = res_main['Portfolio_Value'].iloc[-1]
        total_inv = res_main['Total_Invested'].iloc[-1]
        profit = final_val - total_inv
        ret_pct = (profit / total_inv) * 100
        max_dd = res_main['Drawdown'].min() * 100
        cagr, vol, sharpe, sortino = calculate_metrics(res_main)
        
        if res_comp is not None:
            st.success(f"Simulation Complete: {selected_ticker} vs {comp_label_final}")
        else:
            st.success(f"Simulation Complete: {selected_ticker}")
            
        # Show Main Asset Metrics
        st.markdown(f"### 📊 Performance Summary (Main: {selected_ticker})")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Value", f"${final_val:,.0f}", help="최종 자산 평가액입니다.")
        c2.metric("Total Invested", f"${total_inv:,.0f}", help="투자한 원금의 총합입니다.")
        c3.metric("Total Profit", f"${profit:,.0f} ({ret_pct:.1f}%)", help="순이익과 수익률입니다.")
        c4.metric("Max Drawdown", f"{max_dd:.2f}%", help="최고점 대비 가장 많이 하락했던 비율입니다.")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Asset CAGR", f"{cagr:.2f}%", help="연평균 성장률")
        c6.metric("Volatility", f"{vol:.2f}%", help="연간 변동성")
        c7.metric("Sharpe Ratio", f"{sharpe:.2f}", help="위험 대비 수익률 (높을수록 좋음)")
        c8.metric("Sortino Ratio", f"{sortino:.2f}", help="하락 위험 대비 수익률 (높을수록 좋음)")
        
        st.markdown("---")
        
        # --- Plot Comparison ---
        plot_comparison_charts(res_main, res_comp, selected_ticker, comp_label_final, use_log_scale)

        # Optional Data View
        with st.expander("View Detailed Data (Main Asset)"):
            st.dataframe(res_main[['Close', 'Total_Invested', 'Portfolio_Value', 'Drawdown']].style.format("{:.2f}"))

    else:
        st.error("Failed to fetch data for Main Asset. Please check the ticker or date range.")
