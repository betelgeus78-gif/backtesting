import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# [1] Streamlit 페이지 설정 (반드시 맨 처음에 와야 함)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive DCA Simulator",
    page_icon="📈",
    layout="wide",  # 전체 너비 사용
    initial_sidebar_state="expanded"
)

# 모바일 호환성을 위한 CSS (테이블 폰트 크기 등)
st.markdown("""
<style>
    .stTable { font-size: 14px !important; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# [2] 사이드바: 사용자 입력 받기
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Simulation Settings")

# 자산 설정
st.sidebar.subheader("Asset & Leverage")
underlying_ticker = st.sidebar.text_input("Ticker Symbol", value="QQQ")
base_lev_ratio = st.sidebar.number_input("Base Leverage (Normal)", value=1.0, step=0.5)
boost_lev_ratio = st.sidebar.number_input("Boost Leverage (Fear)", value=3.0, step=0.5)
expense_ratio_pct = st.sidebar.number_input("Expense Ratio (%)", value=1.0, step=0.1)

# 전략 설정
st.sidebar.subheader("Strategy Logic")
ema_period = st.sidebar.slider("EMA Period", min_value=20, max_value=300, value=200, step=10)

# 기간 및 투자 설정
st.sidebar.subheader("Investment Plan")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
dca_amount = st.sidebar.number_input("Monthly Investment ($)", value=100)

# 차트 옵션
chart_timeframe = st.sidebar.radio("Chart Timeframe", ["Weekly", "Monthly", "Daily"], index=0)
timeframe_map = {"Weekly": "W", "Monthly": "ME", "Daily": "D"}
tf_code = timeframe_map[chart_timeframe]

# -----------------------------------------------------------------------------
# [3] 데이터 처리 및 시뮬레이션 함수 (캐싱 적용)
# -----------------------------------------------------------------------------
@st.cache_data
def run_simulation(ticker, start, end, base_lev, boost_lev, expense, ema, amount, tf):
    # 날짜 처리
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    
    # 데이터 다운로드 (EMA 계산 위해 넉넉히)
    fetch_year = start.year - (ema // 252 + 2)
    fetch_start = f"{fetch_year}-01-01"
    
    try:
        df_raw = yf.download(ticker, start=fetch_start, end=end_str, auto_adjust=True, progress=False)
    except Exception as e:
        return None, f"Error downloading data: {e}"

    if df_raw.empty:
        return None, "No data found for the given ticker."

    if isinstance(df_raw, pd.DataFrame):
        if 'Close' in df_raw.columns:
            price_raw = df_raw['Close']
        else:
            price_raw = df_raw.iloc[:, 0]
    else:
        price_raw = df_raw
    price_raw = price_raw.dropna().squeeze()

    # 리샘플링
    if tf == 'W':
        price_base = price_raw.resample('W').last().dropna()
        period_factor = 52
    elif tf == 'ME':
        price_base = price_raw.resample('ME').last().dropna()
        period_factor = 12
    else:
        price_base = price_raw
        period_factor = 252

    # 가상 자산 생성
    period_ret = price_base.pct_change().fillna(0)
    period_cost = expense / 100 / period_factor

    if base_lev == 1.0:
        sim_price_base = price_base
    else:
        base_ret_net = (period_ret * base_lev) - period_cost
        sim_price_base = 100 * (1 + base_ret_net).cumprod()

    boost_ret_net = (period_ret * boost_lev) - period_cost
    sim_price_boost = 100 * (1 + boost_ret_net).cumprod()

    # 데이터프레임 병합
    df_main = pd.DataFrame(index=price_base.index)
    df_main['Raw_Price']  = price_base       
    df_main['Base_Asset'] = sim_price_base   
    df_main['Boost_Asset'] = sim_price_boost 
    df_main[f'EMA{ema}'] = df_main['Raw_Price'].ewm(span=ema, adjust=False).mean()

    # 시뮬레이션 구간
    df_calc = df_main.loc[start_str:]

    # 매수일 계산 (월초)
    df_temp = df_calc.copy()
    df_temp['YYYY-MM'] = df_temp.index.strftime('%Y-%m')
    df_temp['DateCol'] = df_temp.index
    buy_dates = set(df_temp.groupby('YYYY-MM')['DateCol'].min())

    # 로직 수행
    shares_mix_base, shares_mix_boost = 0.0, 0.0
    invested_mix = 0.0
    shares_only_boost, invested_only_boost = 0.0, 0.0
    shares_only_base, invested_only_base = 0.0, 0.0

    history_mix, history_only_boost, history_only_base = [], [], []
    boost_zones = []
    is_boosting = False
    boost_start = None

    for date, row in df_calc.iterrows():
        p_raw   = row['Raw_Price']
        p_base  = row['Base_Asset']
        p_boost = row['Boost_Asset']
        ema_val = row[f'EMA{ema}']
        
        is_below_ema = p_raw < ema_val
        
        # 구간 기록
        if is_below_ema and not is_boosting:
            is_boosting = True
            boost_start = date
        elif not is_below_ema and is_boosting:
            is_boosting = False
            if boost_start:
                boost_zones.append((boost_start, date))
                boost_start = None

        if date in buy_dates:
            # Mix
            if is_below_ema:
                shares_mix_boost += amount / p_boost 
            else:
                shares_mix_base += amount / p_base   
            invested_mix += amount
            
            # Only
            shares_only_boost += amount / p_boost
            invested_only_boost += amount
            shares_only_base += amount / p_base
            invested_only_base += amount

        # 평가금
        val_mix = (shares_mix_base * p_base) + (shares_mix_boost * p_boost)
        val_only_boost = shares_only_boost * p_boost
        val_only_base  = shares_only_base * p_base
        
        history_mix.append({'Date': date, 'Value': val_mix})
        history_only_boost.append({'Date': date, 'Value': val_only_boost})
        history_only_base.append({'Date': date, 'Value': val_only_base})

    if is_boosting and boost_start:
        boost_zones.append((boost_start, df_calc.index[-1]))

    res_mix = pd.DataFrame(history_mix).set_index('Date')
    res_boost = pd.DataFrame(history_only_boost).set_index('Date')
    res_base  = pd.DataFrame(history_only_base).set_index('Date')

    # MDD 계산
    def calc_mdd(df):
        peak = df['Value'].cummax()
        dd = (df['Value'] - peak) / peak * 100
        return dd, dd.min()

    res_mix['DD'], mdd_mix = calc_mdd(res_mix)
    res_boost['DD'], mdd_boost = calc_mdd(res_boost)
    res_base['DD'], mdd_base   = calc_mdd(res_base)

    # 결과 패키징
    results = {
        "mix": {"df": res_mix, "invested": invested_mix, "final": res_mix['Value'].iloc[-1], "mdd": mdd_mix},
        "boost": {"df": res_boost, "invested": invested_only_boost, "final": res_boost['Value'].iloc[-1], "mdd": mdd_boost},
        "base": {"df": res_base, "invested": invested_only_base, "final": res_base['Value'].iloc[-1], "mdd": mdd_base},
        "df_calc": df_calc,
        "boost_zones": boost_zones,
        "period_factor": period_factor
    }
    return results, None

# -----------------------------------------------------------------------------
# [4] 메인 UI 로직
# -----------------------------------------------------------------------------
st.title("🚀 Adaptive DCA Simulator")
st.markdown(f"**Strategy:** {underlying_ticker} < {ema_period} EMA ? Buy **{boost_lev_ratio}x** : Buy **{base_lev_ratio}x**")

# 실행
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Simulating... Please wait"):
        res, err = run_simulation(underlying_ticker, start_date, end_date, base_lev_ratio, boost_lev_ratio, expense_ratio_pct, ema_period, dca_amount, tf_code)

    if err:
        st.error(err)
    else:
        # 1. Metric Calculation
        def get_kpi(res_dict, name):
            inv = res_dict['invested']
            fin = res_dict['final']
            pnl = fin - inv
            pnl_pct = (pnl / inv) * 100 if inv > 0 else 0
            mdd = res_dict['mdd']
            
            # CAGR & Sharpe
            df = res_dict['df']
            days = (df.index[-1] - df.index[0]).days
            years = days / 365.25
            cagr = pnl_pct / years if years > 0 else 0
            
            returns = df['Value'].pct_change().dropna()
            sharpe = (returns.mean() * res['period_factor'] - 0.02) / (returns.std() * np.sqrt(res['period_factor'])) if returns.std() != 0 else 0
            
            return [name, f"${inv:,.0f}", f"${fin:,.0f}", f"${pnl:,.0f}", f"{pnl_pct:,.2f}%", f"{cagr:,.2f}%", f"{mdd:.2f}%", f"{sharpe:.2f}"]

        kpi_data = [
            get_kpi(res['mix'], f"Mix ({base_lev_ratio}x/{boost_lev_ratio}x)"),
            get_kpi(res['boost'], f"Only {boost_lev_ratio}x"),
            get_kpi(res['base'], f"Only {base_lev_ratio}x")
        ]
        df_kpi = pd.DataFrame(kpi_data, columns=["Strategy", "Invested", "Final Value", "PnL ($)", "PnL (%)", "CAGR", "MDD", "Sharpe"])

        # 2. Display Table (Streamlit Native Table 사용 - 반응형 좋음)
        st.subheader("📊 Performance Summary")
        
        # MDD 컬러링을 위한 스타일링 함수
        def color_mdd_red(val):
            if '%' in val: # MDD 컬럼인지 확인 (단순 문자열 체크)
                try:
                    num = float(val.strip('%'))
                    if num < 0: return 'color: red'
                except: pass
            return ''

        # 스타일 적용하여 데이터프레임 표시
        st.dataframe(df_kpi.style.applymap(color_mdd_red, subset=['MDD']), use_container_width=True)

        # 3. Plotly Charts
        st.subheader("📈 Detailed Analysis")
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08, 
            specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
            row_heights=[0.40, 0.15, 0.20, 0.15],
            subplot_titles=("Portfolio Value (Log)", "Value Gap (Mix - Boost)", f"Condition ({underlying_ticker} vs EMA)", "Drawdown")
        )

        # R1: Value
        fig.add_trace(go.Scatter(x=res['mix']['df'].index, y=res['mix']['df']['Value'], name='Mix', line=dict(color='blue', width=2), legendgroup='1', legend='legend'), row=1, col=1)
        fig.add_trace(go.Scatter(x=res['boost']['df'].index, y=res['boost']['df']['Value'], name=f'Only {boost_lev_ratio}x', line=dict(color='orange', width=2), legendgroup='1', legend='legend'), row=1, col=1)
        fig.add_trace(go.Scatter(x=res['base']['df'].index, y=res['base']['df']['Value'], name=f'Only {base_lev_ratio}x', line=dict(color='black', width=1.5, dash='dot'), legendgroup='1', legend='legend'), row=1, col=1)

        # R2: Gap
        gap = res['mix']['df']['Value'] - res['boost']['df']['Value']
        fig.add_trace(go.Scatter(x=gap.index, y=gap.where(gap >= 0, 0), name='Mix Win', mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0,0,255,0.3)', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=gap.index, y=gap.where(gap < 0, 0), name='Boost Win', mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(255,165,0,0.3)', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=gap.index, y=gap, name='Gap', line=dict(color='gray', width=1), legendgroup='2', legend='legend2'), row=2, col=1)

        # R3: Condition
        df_calc = res['df_calc']
        fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['Raw_Price'], name='Price', line=dict(color='black', width=1), legendgroup='3', legend='legend3'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc[f'EMA{ema_period}'], name='EMA', line=dict(color='green', width=1, dash='dash'), legendgroup='3', legend='legend3'), row=3, col=1)

        # R4: DD
        fig.add_trace(go.Scatter(x=res['mix']['df'].index, y=res['mix']['df']['DD'], name='Mix MDD', line=dict(color='blue', width=1), legendgroup='4', legend='legend4'), row=4, col=1)
        fig.add_trace(go.Scatter(x=res['boost']['df'].index, y=res['boost']['df']['DD'], name='Boost MDD', line=dict(color='orange', width=1), legendgroup='4', legend='legend4'), row=4, col=1)

        # Layout
        fig.update_layout(
            height=1200, # 전체 높이 넉넉하게
            template="plotly_white",
            hovermode="x unified",
            legend=dict(x=1.01, y=0.90, xanchor='left'),
            legend2=dict(x=1.01, y=0.58, xanchor='left'),
            legend3=dict(x=1.01, y=0.38, xanchor='left'),
            legend4=dict(x=1.01, y=0.10, xanchor='left'),
            shapes=[dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1, fillcolor="red", opacity=0.1, layer="below", line_width=0) for s, e in res['boost_zones']]
        )
        
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please adjust settings in the sidebar and click 'Run Simulation'")

