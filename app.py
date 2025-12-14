import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ---------------------------------------------------------
# 1. Page Config & CSS Styling
# ---------------------------------------------------------
st.set_page_config(page_title="DCA Backtest Simulator", layout="wide")

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
    "QQQ", "TQQQ", "QLD", "PSQ", "SQQQ", 
    "SPY", "UPRO", "SSO", 
    "SOXX", "SOXL", "SOXS", 
    "TLT", "TMF", "TMV",
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX",
    "COIN", "MSTR", 
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
    "DOGE-USD", "ADA-USD", "TRX-USD", "AVAX-USD", "SHIB-USD",
    "005930.KS", "000660.KS"
]

# ---------------------------------------------------------
# 3. Sidebar Inputs
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Main Asset
    main_ticker = st.selectbox("Select Main Asset", common_tickers, index=0)
    
    st.markdown("---")
    
    # Comparison Assets (Multi-Select)
    st.subheader("🆚 Comparison Settings")
    use_comparison = st.checkbox("Compare with other assets?", value=False)
    
    comp_tickers = []
    if use_comparison:
        # Filter out main ticker from options to avoid duplicate
        options = [t for t in common_tickers if t != main_ticker]
        comp_tickers = st.multiselect(
            "Select Comparison Assets (Max 5)", 
            options, 
            max_selections=5
        )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    st.markdown("---")
    
    initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
    recurring_amount = st.number_input("Recurring Contribution ($)", value=500, step=100)
    frequency = st.selectbox("Contribution Frequency", ["Monthly", "Weekly", "Daily"], index=0)
    
    st.markdown("---")
    
    use_log_scale = st.checkbox("Use Log Scale (Portfolio Value)", value=False)

# ---------------------------------------------------------
# 4. Data Functions
# ---------------------------------------------------------
@st.cache_data
def get_data_multi(tickers, start, end):
    """Fetches data for multiple tickers at once."""
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)['Close']
        # If single ticker, yfinance returns Series, convert to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data
    except:
        return pd.DataFrame()

def run_dca_backtest(series, initial_cap, recurring_amt, freq):
    """
    Runs DCA backtest for a single price series (pd.Series).
    Returns a DataFrame with results.
    """
    df = series.to_frame(name='Close')
    df = df.dropna()
    
    if df.empty: return None

    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    df['Contribution_Day'] = False
    
    if freq == 'Daily': df['Contribution_Day'] = True
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
    
    curr_sh = initial_cap / closes[0]
    tot_inv = initial_cap
    
    current_shares_arr = []
    total_invested_arr = []
    
    for i in range(len(df)):
        if is_contrib[i]:
            new_shares = recurring_amt / closes[i]
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
    sharpe = (mean_return / df['Daily_Return'].std()) * np.sqrt(252) if df['Daily_Return'].std() != 0 else 0.0
    
    neg_ret = df.loc[df['Daily_Return'] < 0, 'Daily_Return']
    down_std = neg_ret.std()
    sortino = (mean_return * 252) / (down_std * np.sqrt(252)) if down_std != 0 else 0.0
    
    final_val = df['Portfolio_Value'].iloc[-1]
    total_inv = df['Total_Invested'].iloc[-1]
    profit = final_val - total_inv
    ret_pct = (profit / total_inv) * 100 if total_inv != 0 else 0
    max_dd = df['Drawdown'].min() * 100

    return {
        "final_val": final_val, "total_inv": total_inv, "profit": profit,
        "ret_pct": ret_pct, "max_dd": max_dd, "cagr": cagr,
        "vol": volatility, "sharpe": sharpe, "sortino": sortino
    }

def display_metrics_block(metrics, title, color_bar):
    st.markdown(f"### {color_bar} {title}")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Final Value", f"${metrics['final_val']:,.0f}", help="최종 자산 평가액입니다.")
        st.metric("Total Profit", f"${metrics['profit']:,.0f} ({metrics['ret_pct']:.1f}%)", help="총 순이익금과 수익률입니다.")
        st.metric("Asset CAGR", f"{metrics['cagr']:.2f}%", help="연평균 성장률(복리)입니다.")
        st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}", help="샤프 지수 (위험 대비 수익률). 1.0 이상이면 우수.")
    with c2:
        st.metric("Total Invested", f"${metrics['total_inv']:,.0f}", help="총 투자된 원금입니다.")
        st.metric("Max Drawdown", f"{metrics['max_dd']:.2f}%", help="최고점 대비 최대 하락폭(MDD)입니다.")
        st.metric("Volatility", f"{metrics['vol']:.2f}%", help="연간 변동성입니다.")
        st.metric("Sortino Ratio", f"{metrics['sortino']:.2f}", help="소티노 지수 (하락 위험 대비 수익률).")

# ---------------------------------------------------------
# 5. Perfect Sync Chart Function (Multi-Asset)
# ---------------------------------------------------------
def plot_charts_synced_multi(results_dict, main_ticker, log_scale):
    y_axis_type = "log" if log_scale else "linear"
    
    # Identify Comparison Tickers
    comp_tickers = [t for t in results_dict.keys() if t != main_ticker]
    has_comp = len(comp_tickers) > 0
    
    # 1. Create Subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True, 
        vertical_spacing=0.03,
        subplot_titles=(
            '💰 Portfolio Value',
            '⚖️ Winning (Diff from Main)',
            '🌊 Drawdown (%)'
        ),
        row_heights=[0.5, 0.25, 0.25]
    )

    # --- Colors for multiple lines ---
    colors = ['orange', 'green', 'purple', 'brown', 'cyan'] # Main is always Red

    # --- Row 1: Portfolio Value ---
    # Main Asset (Red)
    main_df = results_dict[main_ticker]
    fig.add_trace(go.Scatter(
        x=main_df.index, y=main_df['Portfolio_Value'],
        mode='lines', name=f'{main_ticker} (Main)',
        line=dict(color='red', width=2),
        legendgroup='g1'
    ), row=1, col=1)

    # Comparison Assets
    for idx, ticker in enumerate(comp_tickers):
        color = colors[idx % len(colors)]
        comp_df = results_dict[ticker]
        fig.add_trace(go.Scatter(
            x=comp_df.index, y=comp_df['Portfolio_Value'],
            mode='lines', name=f'{ticker}',
            line=dict(color=color, width=1.5),
            legendgroup='g1'
        ), row=1, col=1)

    # Total Invested (Common) - Assuming same logic for all, take main
    fig.add_trace(go.Scatter(
        x=main_df.index, y=main_df['Total_Invested'],
        mode='lines', name='Invested',
        line=dict(color='gray', width=1.0, dash='dash'),
        legendgroup='g1'
    ), row=1, col=1)

    # --- Row 2: Winning (Difference vs Main) ---
    if has_comp:
        for idx, ticker in enumerate(comp_tickers):
            color = colors[idx % len(colors)]
            comp_df = results_dict[ticker]
            # Diff = Comp - Main (or Main - Comp? User usually wants to see if Main is winning)
            # Let's do: Comparison - Main (To show how Comp performs relative to Main baseline 0)
            # OR: Main - Comparison. 
            # Standard: Main Asset Advantage. (Positive = Main is better)
            
            # Align Index
            aligned_comp = comp_df['Portfolio_Value'].reindex(main_df.index, method='ffill')
            diff = main_df['Portfolio_Value'] - aligned_comp
            
            fig.add_trace(go.Scatter(
                x=diff.index, y=diff,
                mode='lines', name=f'Main vs {ticker}',
                line=dict(color=color, width=1.0),
                fill='tozeroy', # Fill to zero
                legendgroup='g2'
            ), row=2, col=1)
            
            # Note: Multiple fills might look messy, but acceptable for comparison.
    else:
        fig.add_annotation(text="Add comparison assets to see winning chart", xref="x2", yref="y2", x=main_df.index[len(main_df)//2], y=0, showarrow=False)

    # --- Row 3: Drawdown ---
    # Main
    fig.add_trace(go.Scatter(
        x=main_df.index, y=main_df['Drawdown'] * 100,
        mode='lines', name=f'{main_ticker} MDD',
        line=dict(color='red', width=1.0),
        legendgroup='g3'
    ), row=3, col=1)
    
    # Comps
    for idx, ticker in enumerate(comp_tickers):
        color = colors[idx % len(colors)]
        comp_df = results_dict[ticker]
        fig.add_trace(go.Scatter(
            x=comp_df.index, y=comp_df['Drawdown'] * 100,
            mode='lines', name=f'{ticker} MDD',
            line=dict(color=color, width=1.0),
            legendgroup='g3'
        ), row=3, col=1)

    # --- Synchronization ---
    fig.update_layout(
        height=900,
        template='plotly_white',
        hovermode='x unified', 
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    )

    fig.update_xaxes(
        showspikes=True, spikemode='across', spikesnap='cursor',
        showline=True, showgrid=True, matches='x'
    )
    
    fig.update_yaxes(type=y_axis_type, row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Diff ($)", row=2, col=1)
    fig.update_yaxes(title_text="MDD (%)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
with st.spinner(f'Processing Simulation...'):
    # 1. Fetch All Data Needed
    tickers_to_fetch = [main_ticker] + comp_tickers
    
    # Fetch all data in one go (optimized)
    data_all = get_data_multi(tickers_to_fetch, start_date, end_date)
    
    results = {}
    
    if not data_all.empty and main_ticker in data_all.columns:
        
        # Run Backtest for Main
        res_main = run_dca_backtest(data_all[main_ticker], initial_capital, recurring_amount, frequency)
        if res_main is not None:
            results[main_ticker] = res_main
        
        # Run Backtest for Comps
        for t in comp_tickers:
            if t in data_all.columns:
                res = run_dca_backtest(data_all[t], initial_capital, recurring_amount, frequency)
                if res is not None:
                    # Align dates with Main if needed (simple reindex)
                    res = res.reindex(results[main_ticker].index).ffill()
                    results[t] = res

        # Display Metrics
        if main_ticker in results:
            st.success(f"Simulation Complete: {main_ticker} vs {len(comp_tickers)} assets")
            
            # 1. Main Asset Metrics (Full Width or Prominent)
            display_metrics_block(calculate_metrics(results[main_ticker]), f"Main: {main_ticker}", "🟦")
            
            # 2. Comparison Assets Metrics (Grid Layout)
            if comp_tickers:
                st.markdown("---")
                st.subheader("🆚 Comparison Assets Performance")
                # Create rows of 2 cols
                cols = st.columns(2)
                for i, t in enumerate(comp_tickers):
                    if t in results:
                        with cols[i % 2]:
                            display_metrics_block(calculate_metrics(results[t]), f"Comp {i+1}: {t}", "🟧")
                            st.markdown("---") # Separator inside column

            st.markdown("---")
            
            # 3. Synced Charts
            plot_charts_synced_multi(results, main_ticker, use_log_scale)
            
            with st.expander("View Detailed Data (Main Asset)"):
                st.dataframe(results[main_ticker][['Close', 'Total_Invested', 'Portfolio_Value', 'Drawdown']].style.format("{:.2f}"))
        
        else:
            st.error("Failed to process Main Asset data.")
    else:
        st.error("Failed to fetch data. Check tickers or date range.")
