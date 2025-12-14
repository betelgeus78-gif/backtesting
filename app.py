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

# Metrics Font Size Adjustment
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
    
    selected_ticker = st.selectbox("Select Asset (Main)", common_tickers, index=0)
    
    st.markdown("---")
    
    st.subheader("🆚 Comparison Settings")
    
    use_simulation = st.checkbox("Simulate Leverage (Daily Rebalancing)", value=False)
    
    comparison_ticker = "None"
    leverage_ratio = 1.0
    comp_label_final = "None"

    if use_simulation:
        leverage_ratio = st.number_input("Target Leverage (1.0x ~ 5.0x)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        comp_label_final = f"Simulated {leverage_ratio:.1f}x ({selected_ticker})"
    else:
        comp_options = ["None"] + common_tickers
        comparison_ticker = st.selectbox("Select Comparison Asset", comp_options, index=0)
        comp_label_final = comparison_ticker

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
def get_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Close']]
    except: return None

def generate_leveraged_data(df_base, leverage):
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
    current_shares_arr, total_invested_arr = [], []
    
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
    std_return = df['Daily_Return'].std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0.0
    
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
# 5. Perfect Sync Chart Function
# ---------------------------------------------------------
def plot_charts_synced(df_main, df_comp, name_main, name_comp, log_scale):
    y_axis_type = "log" if log_scale else "linear"
    has_comp = df_comp is not None
    
    # 1. Create Subplots with SHARED X-AXIS
    # vertical_spacing reduced to bring charts closer
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True, 
        vertical_spacing=0.03,
        subplot_titles=(
            '💰 Portfolio Value',
            f'⚖️ Winning (Diff: {name_main} - {name_comp})' if has_comp else 'Winning Chart',
            '🌊 Drawdown (%)'
        ),
        row_heights=[0.5, 0.25, 0.25]
    )

    # --- Row 1: Value ---
    fig.add_trace(go.Scatter(
        x=df_main.index, y=df_main['Portfolio_Value'],
        mode='lines', name=f'{name_main}',
        line=dict(color='red', width=1.5)
    ), row=1, col=1)

    if has_comp:
        fig.add_trace(go.Scatter(
            x=df_comp.index, y=df_comp['Portfolio_Value'],
            mode='lines', name=f'{name_comp}',
            line=dict(color='orange', width=1.5)
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_main.index, y=df_main['Total_Invested'],
        mode='lines', name='Invested',
        line=dict(color='gray', width=1.0, dash='dash')
    ), row=1, col=1)

    # --- Row 2: Winning ---
    if has_comp:
        diff = df_main['Portfolio_Value'] - df_comp['Portfolio_Value']
        pos = diff.apply(lambda x: x if x > 0 else 0)
        neg = diff.apply(lambda x: x if x < 0 else 0)
        
        fig.add_trace(go.Scatter(
            x=diff.index, y=pos,
            mode='lines', name=f'{name_main} Lead',
            fill='tozeroy', fillcolor='rgba(0, 200, 0, 0.3)',
            line=dict(color='green', width=0.5)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=diff.index, y=neg,
            mode='lines', name=f'{name_comp} Lead',
            fill='tozeroy', fillcolor='rgba(200, 0, 0, 0.3)',
            line=dict(color='red', width=0.5)
        ), row=2, col=1)
    else:
        fig.add_annotation(text="Select Comparison Asset", xref="x2", yref="y2", x=df_main.index[len(df_main)//2], y=0, showarrow=False)

    # --- Row 3: Drawdown ---
    fig.add_trace(go.Scatter(
        x=df_main.index, y=df_main['Drawdown'] * 100,
        mode='lines', name=f'{name_main} MDD',
        fill='tozeroy',
        line=dict(color='blue', width=1.0)
    ), row=3, col=1)
    
    if has_comp:
        fig.add_trace(go.Scatter(
            x=df_comp.index, y=df_comp['Drawdown'] * 100,
            mode='lines', name=f'{name_comp} MDD',
            line=dict(color='orange', width=1.0)
        ), row=3, col=1)

    # --- CRITICAL: Synchronization Settings ---
    # 1. hovermode='x unified': Shows ALL data points for the shared X in one tooltip
    # 2. spikemode='across': Draws the vertical line across ALL subplots
    fig.update_layout(
        height=900,
        template='plotly_white',
        hovermode='x unified', 
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    )

    # Force Spike lines on all X-axes
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showline=True,
        showgrid=True,
        matches='x' # Forces Zoom/Pan sync
    )
    
    # Y-Axis Settings
    fig.update_yaxes(type=y_axis_type, row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Diff ($)", row=2, col=1)
    fig.update_yaxes(title_text="MDD (%)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
with st.spinner(f'Processing Simulation...'):
    df_main = get_data(selected_ticker, start_date, end_date)
    df_comp = None
    
    if df_main is not None and not df_main.empty:
        if use_simulation:
            df_comp = generate_leveraged_data(df_main, leverage_ratio)
        elif comparison_ticker != "None":
            if comparison_ticker == selected_ticker:
                df_comp = df_main.copy()
            else:
                df_comp_raw = get_data(comparison_ticker, start_date, end_date)
                if df_comp_raw is not None and not df_comp_raw.empty:
                    df_comp = df_comp_raw.reindex(df_main.index).ffill().dropna()
                else:
                    st.warning(f"Skipping comparison: No data for {comparison_ticker}")

        res_main = run_dca_backtest(df_main, initial_capital, recurring_amount, frequency)
        metrics_main = calculate_metrics(res_main)
        
        metrics_comp = None
        if df_comp is not None:
            res_comp = run_dca_backtest(df_comp, initial_capital, recurring_amount, frequency)
            metrics_comp = calculate_metrics(res_comp)
            st.success(f"Simulation Complete: {selected_ticker} vs {comp_label_final}")
        else:
            st.success(f"Simulation Complete: {selected_ticker}")

        if metrics_comp:
            c1, c2 = st.columns(2)
            with c1: display_metrics_block(metrics_main, f"Main: {selected_ticker}", "🟦")
            with c2: display_metrics_block(metrics_comp, f"Comp: {comp_label_final}", "🟧")
        else:
            display_metrics_block(metrics_main, f"Main: {selected_ticker}", "🟦")

        st.markdown("---")
        plot_charts_synced(res_main, res_comp, selected_ticker, comp_label_final, use_log_scale)
        
        with st.expander("View Data"):
            st.dataframe(res_main.style.format("{:.2f}"))
    else:
        st.error("No data found.")
