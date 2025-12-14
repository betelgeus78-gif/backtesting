import streamlit as st
import yfinance as yf
import pandas as pd
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
    
    # Ticker Selection (No direct input)
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
        # Handle MultiIndex columns if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Close']]
    except Exception as e:
        return None

def run_dca_backtest(df, initial_cap, recurring_amt, freq):
    df = df.copy()
    
    # Calculate returns
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    
    # Initialize simulation columns
    df['Cash_Flow'] = 0.0
    df['Total_Invested'] = 0.0
    df['Holdings_Shares'] = 0.0
    df['Portfolio_Value'] = 0.0
    
    # Setup resampling rule based on frequency
    # We will mark 'True' on days when contribution happens
    df['Contribution_Day'] = False
    
    if freq == 'Daily':
        df['Contribution_Day'] = True
    elif freq == 'Weekly':
        # Contribution on the first available day of the week (Monday or first trading day)
        # Using week number to identify unique weeks
        df['Week_Num'] = df.index.isocalendar().week
        df['Year_Num'] = df.index.isocalendar().year
        # Group by Year/Week and take the first index
        contribution_indices = df.groupby(['Year_Num', 'Week_Num']).head(1).index
        df.loc[contribution_indices, 'Contribution_Day'] = True
    elif freq == 'Monthly':
        # Contribution on the first available day of the month
        df['Month_Num'] = df.index.month
        df['Year_Num'] = df.index.year
        contribution_indices = df.groupby(['Year_Num', 'Month_Num']).head(1).index
        df.loc[contribution_indices, 'Contribution_Day'] = True

    # Iterative calculation (necessary for DCA because shares accumulate)
    # Using a loop is slower but accurate for cash flows. 
    # Optimized approach: Calculate cumulative shares and cash flows.
    
    current_shares = initial_cap / df['Close'].iloc[0]
    total_invested = initial_cap
    
    # Lists to store computed series
    shares_list = []
    invested_list = []
    
    # Fast iteration
    # Create numpy arrays for speed
    closes = df['Close'].values
    is_contrib = df['Contribution_Day'].values
    
    current_shares_arr = []
    total_invested_arr = []
    
    curr_sh = initial_cap / closes[0]
    tot_inv = initial_cap
    
    for i in range(len(df)):
        price = closes[i]
        
        # Add recurring contribution if it's the day (skip first day for recurring, 
        # or handle differently? usually Start Date has Initial, subsequent periods have Recurring)
        # Here we assume Initial Capital is at t=0. 
        # Recurring starts from the first trigger found.
        
        # Logic: If it is a contribution day, buy more shares
        if is_contrib[i]:
            # We assume buying at Close price
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
    
    # Total Invested (Principal)
    fig_value.add_trace(go.Scatter(
        x=df.index, y=df['Total_Invested'],
        mode='lines',
        name='Total Invested (Principal)',
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
        
        # Summary Metrics
        final_value = df['Portfolio_Value'].iloc[-1]
        total_invested = df['Total_Invested'].iloc[-1]
        profit = final_value - total_invested
        total_return_pct = (profit / total_invested) * 100
        max_dd = df['Drawdown'].min() * 100
        
        # Display Results
        st.success(f"Simulation Complete: {selected_ticker}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Value", f"${final_value:,.0f}")
        col2.metric("Total Invested", f"${total_invested:,.0f}")
        col3.metric("Total Profit", f"${profit:,.0f} ({total_return_pct:.2f}%)")
        col4.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        # Plot Charts
        plot_dca_charts(df, selected_ticker, use_log_scale)
        
        # Optional: Show Data
        with st.expander("View Detailed Data"):
            st.dataframe(df[['Close', 'Total_Invested', 'Portfolio_Value', 'Drawdown']].style.format("{:.2f}"))
            
    else:
        st.error("Failed to fetch data. Please check the ticker or date range.")
