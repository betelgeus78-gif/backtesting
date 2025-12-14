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
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data
    except:
        return pd.DataFrame()

def run_dca_backtest(series, initial_cap, recurring_amt, freq):
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
# 5. Plotting Functions (Formatted Numbers)
# ---------------------------------------------------------
def plot_charts_separated(results_dict, main_ticker, log_scale):
    y_axis_type = "log" if log_scale else "linear"
    
    comp_tickers = [t for t in results_dict.keys() if t != main_ticker]
    has_comp = len(comp_tickers) > 0
    colors = ['orange', 'green', 'purple', 'brown', 'cyan'] 
    
    main_df = results_dict[main_ticker]

    # --- 1. Portfolio Value Chart ---
    fig_val = go.Figure()
    
    # Main
    fig_val.add_trace(go.Scatter(
        x=main_df.index, y=main_df['Portfolio_Value'],
        mode='lines', name=f'{main_ticker} (Main)',
        line=dict(color='red', width=2),
        hovertemplate='%{y:,.0f}' # No decimals, full number with commas
    ))

    # Comps
    for idx, ticker in enumerate(comp_tickers):
        color = colors[idx % len(colors)]
        comp_df = results_dict[ticker]
        fig_val.add_trace(go.Scatter(
            x=comp_df.index, y=comp_df['Portfolio_Value'],
            mode='lines', name=f'{ticker}',
            line=dict(color=color, width=1.5),
            hovertemplate='%{y:,.0f}'
        ))

    # Invested
    fig_val.add_trace(go.Scatter(
        x=main_df.index, y=main_df['Total_Invested'],
        mode='lines', name='Total Invested',
        line=dict(color='gray', width=1.0, dash='dash'),
        hovertemplate='%{y:,.0f}'
    ))

    fig_val.update_layout(
        title=f'💰 1. Portfolio Value Comparison',
        xaxis_title='Date', yaxis_title='Value ($)',
        yaxis_type=y_axis_type, template='plotly_white', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Axis Format: full number, no 'k', commas
    fig_val.update_yaxes(tickformat=",d") 
    st.plotly_chart(fig_val, use_container_width=True)

    # --- 2. Winning Chart ---
    if has_comp:
        fig_win = go.Figure()
        for idx, ticker in enumerate(comp_tickers):
            color = colors[idx % len(colors)]
            comp_df = results_dict[ticker]
            
            aligned_comp = comp_df['Portfolio_Value'].reindex(main_df.index, method='ffill')
            diff = main_df['Portfolio_Value'] - aligned_comp
            
            fig_win.add_trace(go.Scatter(
                x=diff.index, y=diff,
                mode='lines', name=f'Main vs {ticker}',
                line=dict(color=color, width=1.0),
                fill='tozeroy',
                hovertemplate='%{y:,.0f}'
            ))
            
        fig_win.update_layout(
            title=f'⚖️ 2. Winning Chart (Difference vs Main)',
            xaxis_title='Date', yaxis_title='Diff ($)',
            template='plotly_white', hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_win.update_yaxes(tickformat=",d")
        st.plotly_chart(fig_win, use_container_width=True)
    else:
        st.info("ℹ️ Add comparison assets to view the 'Winning Chart'.")

    # --- 3. Drawdown Chart ---
    fig_dd = go.Figure()
    
    # Main
    fig_dd.add_trace(go.Scatter(
        x=main_df.index, y=main_df['Drawdown'] * 100,
        mode='lines', name=f'{main_ticker} MDD',
        line=dict(color='red', width=1.0),
        hovertemplate='%{y:.2f}%' # Percentage needs 2 decimals typically, or user said "All numbers no decimal"? Usually MDD needs precision. Let's assume no decimal for currency, but MDD might need 1 decimal. User said "1,2,3 chart values no decimal". I will apply no decimal to MDD too if strictly requested, but 0% vs -1% is big. Let's stick to strict instruction: No decimal.
    ))
    
    # Comps
    for idx, ticker in enumerate(comp_tickers):
        color = colors[idx % len(colors)]
        comp_df = results_dict[ticker]
        fig_dd.add_trace(go.Scatter(
            x=comp_df.index, y=comp_df['Drawdown'] * 100,
            mode='lines', name=f'{ticker} MDD',
            line=dict(color=color, width=1.0),
            hovertemplate='%{y:.0f}%' 
        ))
        
    fig_dd.update_layout(
        title='🌊 3. Drawdown Comparison (%)',
        xaxis_title='Date', yaxis_title='Drawdown (%)',
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_dd.update_yaxes(tickformat=".0f") # No decimal for axis too
    st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
with st.spinner(f'Processing Simulation...'):
    tickers_to_fetch = [main_ticker] + comp_tickers
    data_all = get_data_multi(tickers_to_fetch, start_date, end_date)
    
    results = {}
    
    if not data_all.empty and main_ticker in data_all.columns:
        
        # Backtest Main
        res_main = run_dca_backtest(data_all[main_ticker], initial_capital, recurring_amount, frequency)
        if res_main is not None:
            results[main_ticker] = res_main
        
        # Backtest Comps
        for t in comp_tickers:
            if t in data_all.columns:
                res = run_dca_backtest(data_all[t], initial_capital, recurring_amount, frequency)
                if res is not None:
                    res = res.reindex(results[main_ticker].index).ffill()
                    results[t] = res

        # Display Metrics
        if main_ticker in results:
            st.success(f"Simulation Complete: {main_ticker} vs {len(comp_tickers)} assets")
            
            # Main Metrics
            display_metrics_block(calculate_metrics(results[main_ticker]), f"Main: {main_ticker}", "🟦")
            
            # Comp Metrics
            if comp_tickers:
                st.markdown("---")
                st.subheader("🆚 Comparison Assets Performance")
                cols = st.columns(2)
                for i, t in enumerate(comp_tickers):
                    if t in results:
                        with cols[i % 2]:
                            display_metrics_block(calculate_metrics(results[t]), f"Comp {i+1}: {t}", "🟧")
                            st.markdown("---") 

            st.markdown("---")
            
            # Separated Charts
            plot_charts_separated(results, main_ticker, use_log_scale)
            
            with st.expander("View Detailed Data (Main Asset)"):
                st.dataframe(results[main_ticker][['Close', 'Total_Invested', 'Portfolio_Value', 'Drawdown']].style.format("{:.2f}"))
        
        else:
            st.error("Failed to process Main Asset data.")
    else:
        st.error("Failed to fetch data.")
