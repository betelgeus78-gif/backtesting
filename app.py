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
# 3. Sidebar Inputs
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Ticker Selection
    selected_ticker = st.selectbox("Select Asset", common_tickers, index=0)
    
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

    st.markdown("---")
    
    run_btn = st.button("Run Simulation 🚀")

# ---------------------------------------------------------
# 4. Data & Calculation Functions
# ---------------------------------------------------------
@st.cache_data
def get_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Close']]
    except Exception as e:
        return None

def run_dca_backtest(df, initial_cap, recurring_amt, freq):
    df = df.copy()
    
    # Calculate returns for metrics
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    
    # Initialize simulation columns
    df['Contribution_Day'] = False
    
    # Setup resampling rule based on frequency
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

    # Fast Iteration for DCA
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
    
    # Drawdown Calculation
    df['Peak'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Portfolio_Value'] - df['Peak']) / df['Peak']
    
    return df

def calculate_metrics(df):
    # 1. CAGR (Asset Price Growth)
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    days = (df.index[-1] - df.index[0]).days
    cagr = ((end_price / start_price) ** (365.25 / days) - 1) * 100 if days > 0 else 0.0

    # 2. Volatility (Annualized)
    volatility = df['Daily_Return'].std() * np.sqrt(252) * 100

    # 3. Sharpe Ratio (Annualized)
    mean_return = df['Daily_Return'].mean()
    std_return = df['Daily_Return'].std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0.0

    # 4. Sortino Ratio (Annualized) - Downside Risk only
    negative_returns = df.loc[df['Daily_Return'] < 0, 'Daily_Return']
    downside_std = negative_returns.std()
    sortino = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std != 0 else 0.0

    return cagr, volatility, sharpe, sortino

# ---------------------------------------------------------
# 5. Plotting Function
# ---------------------------------------------------------
def plot_dca_charts(df, ticker, log_scale):
    y_axis_type = "log" if log_scale else "linear"
    
    # --- 1. Asset Price Chart ---
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines',
        name=f'{ticker} Price',
        line=dict(color='black', width=1.0)
    ))
    fig_price.update_layout(
        title=f'📊 1. Asset Price Chart ({ticker})',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis_type=y_axis_type,
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --- 2. Portfolio Value Chart ---
    fig_value = go.Figure()
    
    # Portfolio Value
    fig_value.add_trace(go.Scatter(
        x=df.index, y=df['Portfolio_Value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='red', width=1.0)
    ))
    
    # Total Invested
    fig_value.add_trace(go.Scatter(
        x=df.index, y=df['Total_Invested'],
        mode='lines',
        name='Total Invested',
        line=dict(color='gray', width=1.0, dash='dash')
    ))

    fig_value.update_layout(
        title=f'💰 2. Portfolio Value (Accumulated)',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        yaxis_type=y_axis_type,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_value, use_container_width=True)

    # --- 3. Drawdown Chart ---
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df.index, y=df['Drawdown'] * 100,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='blue', width=1.0)
    ))
    fig_dd.update_layout(
        title='🌊 3. Drawdown (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f'Fetching data for {selected_ticker}...'):
        df = get_data(selected_ticker, start_date, end_date)
        
    if df is not None and not df.empty:
        # Run Backtest
        df = run_dca_backtest(df, initial_capital, recurring_amount, frequency)
        
        # Calculate Financial Metrics
        final_value = df['Portfolio_Value'].iloc[-1]
        total_invested = df['Total_Invested'].iloc[-1]
        profit = final_value - total_invested
        total_return_pct = (profit / total_invested) * 100
        max_dd = df['Drawdown'].min() * 100
        
        # Calculate Technical Metrics
        cagr, volatility, sharpe, sortino = calculate_metrics(df)
        
        st.success(f"Simulation Complete: {selected_ticker}")
        st.markdown("### 📊 Performance Summary")

        # --- Row 1: Financials ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Value", f"${final_value:,.0f}", help="최종 자산 평가액입니다.")
        c2.metric("Total Invested", f"${total_invested:,.0f}", help="투자한 원금의 총합입니다.")
        c3.metric("Total Profit", f"${profit:,.0f} ({total_return_pct:.1f}%)", help="순이익과 수익률입니다.")
        c4.metric("Max Drawdown", f"{max_dd:.2f}%", help="최고점 대비 가장 많이 하락했던 비율입니다. (손실폭)")

        # --- Row 2: Technical Indicators ---
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Asset CAGR", f"{cagr:.2f}%", 
                  help="연평균 성장률입니다. 자산이 매년 평균적으로 얼마나 성장했는지를 나타냅니다.")
        
        c6.metric("Volatility", f"{volatility:.2f}%", 
                  help="이 자산이 1년에 위아래로 평균 몇 % 움직이는지를 보여줍니다. (예: S&P500은 보통 15~20%, 비트코인은 60~80% 수준)")
        
        c7.metric("Sharpe Ratio", f"{sharpe:.2f}", 
                  help="샤프 지수 (위험 대비 수익률). 보통 1.0 이상이면 좋고, 높을수록 훌륭한 전략입니다.")
        
        c8.metric("Sortino Ratio", f"{sortino:.2f}", 
                  help="소티노 지수. 수치가 높을수록 "손실은 적게 보면서 수익을 잘 냈다"는 뜻입니다. (보통 샤프 지수보다 더 실질적인 지표로 칩니다.)"
        
        st.markdown("---")

        # Plot Charts
        plot_dca_charts(df, selected_ticker, use_log_scale)
        
        # Detailed Data
        with st.expander("View Detailed Data"):
            st.dataframe(df[['Close', 'Total_Invested', 'Portfolio_Value', 'Drawdown']].style.format("{:.2f}"))
            
    else:
        st.error("Failed to fetch data. Please check the ticker or date range.")
