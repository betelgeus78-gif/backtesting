import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# [1] 페이지 설정 & 세션 상태 초기화
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Adaptive DCA Simulator", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .stTable { font-size: 14px !important; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# 시뮬레이션 결과를 저장할 저장소 초기화
if 'sim_result' not in st.session_state:
    st.session_state.sim_result = None

# -----------------------------------------------------------------------------
# [2] 사이드바 설정 (자동완성 기능 추가)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Simulation Settings")

# 1. 주요 티커 리스트 정의 (자동완성 목록)
common_tickers = [
    "QQQ", "TQQQ", "QLD", "SQQQ",  # Nasdaq
    "SPY", "UPRO", "SSO", "VOO",   # S&P 500
    "SOXX", "SOXL", "SOXS",        # Semiconductor
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", # Big Tech
    "BITX", "MSTU", "MSTR", "COIN", # Crypto Related
    "TLT", "TMF", "TMV",           # Bonds
    "SCHD", "JEPI"                 # Dividend
]

# 2. 티커 선택 UI (검색 가능한 Selectbox)
# 사용자가 직접 입력도 가능하게 하려면 text_input과 selectbox를 병행하거나,
# selectbox에 없는 값을 허용하는 서드파티 컴포넌트를 써야 하는데,
# 가장 깔끔한 방법은 '직접 입력' 옵션을 넣는 것입니다.

input_method = st.sidebar.radio("Input Method", ["Select from List", "Type Manually"], horizontal=True, label_visibility="collapsed")

if input_method == "Select from List":
    underlying_ticker = st.sidebar.selectbox("Ticker Symbol", common_tickers, index=0) # QQQ Default
else:
    underlying_ticker = st.sidebar.text_input("Ticker Symbol", value="QQQ").upper() # 직접 입력
    
base_lev_ratio = st.sidebar.number_input("Base Leverage (Normal)", value=1.0, step=0.5)
boost_lev_ratio = st.sidebar.number_input("Boost Leverage (Fear)", value=3.0, step=0.5)
expense_ratio_pct = st.sidebar.number_input("Expense Ratio (%)", value=1.0, step=0.1)
ema_period = st.sidebar.slider("EMA Period", 20, 300, 200, 10)

start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
dca_amount = st.sidebar.number_input("Monthly Investment ($)", value=100)

chart_timeframe = st.sidebar.radio("Chart Timeframe", ["Weekly", "Monthly", "Daily"], index=0)
tf_map = {"Weekly": "W", "Monthly": "ME", "Daily": "D"}
tf_code = tf_map[chart_timeframe]

# -----------------------------------------------------------------------------
# [3] 시뮬레이션 함수 (데이터 계산)
# -----------------------------------------------------------------------------
@st.cache_data
def run_simulation_logic(ticker, start, end, base_lev, boost_lev, expense, ema, amount, tf):
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    
    fetch_year = start.year - (ema // 252 + 2)
    fetch_start = f"{fetch_year}-01-01"
    
    try:
        df_raw = yf.download(ticker, start=fetch_start, end=end_str, auto_adjust=True, progress=False)
    except Exception as e:
        return None, f"Error: {e}"

    if df_raw.empty: return None, "No data found."
    
    price_raw = df_raw['Close'] if 'Close' in df_raw.columns else df_raw.iloc[:, 0]
    price_raw = price_raw.dropna().squeeze()

    if tf == 'W':
        price_base = price_raw.resample('W').last().dropna()
        period_factor = 52
    elif tf == 'ME':
        price_base = price_raw.resample('ME').last().dropna()
        period_factor = 12
    else:
        price_base = price_raw
        period_factor = 252

    period_ret = price_base.pct_change().fillna(0)
    period_cost = expense / 100 / period_factor

    if base_lev == 1.0:
        sim_price_base = price_base
    else:
        sim_price_base = 100 * (1 + (period_ret * base_lev) - period_cost).cumprod()

    sim_price_boost = 100 * (1 + (period_ret * boost_lev) - period_cost).cumprod()

    df_main = pd.DataFrame(index=price_base.index)
    df_main['Raw_Price'] = price_base
    df_main['Base_Asset'] = sim_price_base
    df_main['Boost_Asset'] = sim_price_boost
    df_main[f'EMA{ema}'] = df_main['Raw_Price'].ewm(span=ema, adjust=False).mean()

    df_calc = df_main.loc[start_str:]
    
    df_temp = df_calc.copy()
    df_temp['YM'] = df_temp.index.strftime('%Y-%m')
    buy_dates = set(df_temp.groupby('YM').head(1).index)

    mix_base, mix_boost, inv_mix = 0.0, 0.0, 0.0
    only_boost, inv_boost = 0.0, 0.0
    only_base, inv_base = 0.0, 0.0

    hist_mix, hist_boost, hist_base = [], [], []
    boost_zones = []
    is_boosting = False
    boost_start = None

    for date, row in df_calc.iterrows():
        p_raw = row['Raw_Price']
        p_base = row['Base_Asset']
        p_boost = row['Boost_Asset']
        ema_val = row[f'EMA{ema}']
        
        is_below = p_raw < ema_val
        
        if is_below and not is_boosting:
            is_boosting = True
            boost_start = date
        elif not is_below and is_boosting:
            is_boosting = False
            if boost_start: boost_zones.append((boost_start, date))
                
        if date in buy_dates:
            if is_below: mix_boost += amount / p_boost
            else: mix_base += amount / p_base
            inv_mix += amount
            only_boost += amount / p_boost; inv_boost += amount
            only_base += amount / p_base; inv_base += amount
            
        val_mix = (mix_base * p_base) + (mix_boost * p_boost)
        val_boost = only_boost * p_boost
        val_base = only_base * p_base
        
        hist_mix.append({'Date': date, 'Value': val_mix})
        hist_boost.append({'Date': date, 'Value': val_boost})
        hist_base.append({'Date': date, 'Value': val_base})
        
    if is_boosting and boost_start: boost_zones.append((boost_start, df_calc.index[-1]))

    res_mix = pd.DataFrame(hist_mix).set_index('Date')
    res_boost = pd.DataFrame(hist_boost).set_index('Date')
    res_base = pd.DataFrame(hist_base).set_index('Date')

    def calc_mdd(df):
        peak = df['Value'].cummax()
        dd = (df['Value'] - peak) / peak * 100
        return dd, dd.min()

    res_mix['DD'], mdd_mix = calc_mdd(res_mix)
    res_boost['DD'], mdd_boost = calc_mdd(res_boost)
    res_base['DD'], mdd_base = calc_mdd(res_base)

    return {
        "mix": {"df": res_mix, "inv": inv_mix, "fin": res_mix['Value'].iloc[-1], "mdd": mdd_mix},
        "boost": {"df": res_boost, "inv": inv_boost, "fin": res_boost['Value'].iloc[-1], "mdd": mdd_boost},
        "base": {"df": res_base, "inv": inv_base, "fin": res_base['Value'].iloc[-1], "mdd": mdd_base},
        "df_calc": df_calc, "zones": boost_zones, "factor": period_factor
    }, None

# -----------------------------------------------------------------------------
# [4] 메인 UI 로직
# -----------------------------------------------------------------------------
st.title("🚀 Adaptive DCA Simulator")
st.markdown(f"**Rule:** If {underlying_ticker} < {ema_period} EMA → Buy **{boost_lev_ratio}x** / Else → Buy **{base_lev_ratio}x**")

# [핵심 변경] 버튼 클릭 시에만 계산하고, 결과는 session_state에 저장
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Calculating..."):
        res, err = run_simulation_logic(underlying_ticker, start_date, end_date, base_lev_ratio, boost_lev_ratio, expense_ratio_pct, ema_period, dca_amount, tf_code)
        if err:
            st.error(err)
        else:
            st.session_state.sim_result = res # 결과 저장!

# -----------------------------------------------------------------------------
# [5] 결과 표시 (저장된 데이터가 있을 때만 실행)
# -----------------------------------------------------------------------------
if st.session_state.sim_result is not None:
    res = st.session_state.sim_result
    
    # [A] KPI Table
    def get_kpi(d, name):
        inv, fin = d['inv'], d['fin']
        pnl = fin - inv
        pnl_pct = (pnl / inv * 100) if inv > 0 else 0
        df = d['df']
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = pnl_pct / years if years > 0 else 0
        rets = df['Value'].pct_change().dropna()
        sharpe = (rets.mean()*res['factor'] - 0.02) / (rets.std()*np.sqrt(res['factor'])) if rets.std() != 0 else 0
        return [name, f"${inv:,.0f}", f"${fin:,.0f}", f"${pnl:,.0f}", f"{pnl_pct:,.0f}%", f"{cagr:,.1f}%", f"{d['mdd']:.2f}%", f"{sharpe:.2f}"]

    kpi = [
        get_kpi(res['mix'], f"Mix ({base_lev_ratio}x/{boost_lev_ratio}x)"),
        get_kpi(res['boost'], f"Only {boost_lev_ratio}x"),
        get_kpi(res['base'], f"Only {base_lev_ratio}x")
    ]
    df_kpi = pd.DataFrame(kpi, columns=["Strategy", "Invested", "Final", "PnL $", "PnL %", "CAGR", "MDD", "Sharpe"])
    
    def highlight_mdd(val):
        return 'color: red' if '%' in val and float(val.strip('%')) < -30 else ''

    st.subheader("📊 Performance Summary")
    st.dataframe(df_kpi.style.applymap(highlight_mdd, subset=['MDD']), use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Detailed Charts")

    def create_fig(title, log_y=False):
        f = go.Figure()
        f.update_layout(
            title=title, height=400, template="plotly_white", hovermode="x unified",
            legend=dict(x=1.01, y=1, xanchor='left', yanchor='top'),
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(type="log" if log_y else "linear", fixedrange=False),
            xaxis=dict(fixedrange=False),
            shapes=[dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1, fillcolor="red", opacity=0.1, layer="below", line_width=0) for s, e in res['zones']]
        )
        return f

    # [Chart 1] Portfolio Value
    c1_left, c1_right = st.columns([0.85, 0.15])
    with c1_right:
        st.write("") 
        st.write("") 
        # key="log1" 덕분에 체크 상태가 session_state에 자동 저장됨
        use_log_1 = st.checkbox("Log Scale", value=True, key="log1") 
    
    with c1_left:
        fig1 = create_fig("1. Portfolio Value ($)", log_y=use_log_1)
        fig1.add_trace(go.Scatter(x=res['mix']['df'].index, y=res['mix']['df']['Value'], name='Mix', line=dict(color='blue', width=2)))
        fig1.add_trace(go.Scatter(x=res['boost']['df'].index, y=res['boost']['df']['Value'], name=f'Only {boost_lev_ratio}x', line=dict(color='orange', width=2)))
        fig1.add_trace(go.Scatter(x=res['base']['df'].index, y=res['base']['df']['Value'], name=f'Only {base_lev_ratio}x', line=dict(color='black', width=1, dash='dot')))
        st.plotly_chart(fig1, use_container_width=True)

    # [Chart 2] Value Gap
    c2_left, c2_right = st.columns([0.85, 0.15])
    with c2_right:
        st.write("") 
        st.write("")
        st.caption("ℹ️ Linear Only")
    
    with c2_left:
        fig2 = create_fig("2. Value Gap (Mix - Boost)")
        gap = res['mix']['df']['Value'] - res['boost']['df']['Value']
        fig2.add_trace(go.Scatter(x=gap.index, y=gap.where(gap >= 0, 0), name='Mix Win', mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0,0,255,0.3)'))
        fig2.add_trace(go.Scatter(x=gap.index, y=gap.where(gap < 0, 0), name='Boost Win', mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(255,165,0,0.3)'))
        fig2.add_trace(go.Scatter(x=gap.index, y=gap, name='Gap', line=dict(color='gray', width=1)))
        st.plotly_chart(fig2, use_container_width=True)

    # [Chart 3] Condition
    c3_left, c3_right = st.columns([0.85, 0.15])
    with c3_right:
        st.write("") 
        st.write("")
        use_log_3 = st.checkbox("Log Scale", value=True, key="log3")
    
    with c3_left:
        fig3 = create_fig(f"3. Condition ({underlying_ticker} vs EMA)", log_y=use_log_3)
        df_c = res['df_calc']
        fig3.add_trace(go.Scatter(x=df_c.index, y=df_c['Raw_Price'], name='Price', line=dict(color='black', width=1)))
        fig3.add_trace(go.Scatter(x=df_c.index, y=df_c[f'EMA{ema_period}'], name='EMA', line=dict(color='green', width=1, dash='dash')))
        st.plotly_chart(fig3, use_container_width=True)

    # [Chart 4] Drawdown
    c4_left, c4_right = st.columns([0.85, 0.15])
    with c4_right:
        st.write("") 
        st.write("")
        st.caption("ℹ️ Linear Only")
    
    with c4_left:
        fig4 = create_fig("4. Drawdown (%)")
        fig4.add_trace(go.Scatter(x=res['mix']['df'].index, y=res['mix']['df']['DD'], name='Mix MDD', line=dict(color='blue', width=1)))
        fig4.add_trace(go.Scatter(x=res['boost']['df'].index, y=res['boost']['df']['DD'], name='Boost MDD', line=dict(color='orange', width=1)))
        st.plotly_chart(fig4, use_container_width=True)

else:
    st.info("👈 Please enter parameters in the sidebar and click 'Run Simulation'")
