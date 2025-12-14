import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(page_title="나만의 백테스트 앱", layout="wide")
st.title("📈 파인스크립트형 백테스트 시뮬레이터")

# ---------------------------------------------------------
# 2. 티커 리스트 (사용자 요청 반영: 암호화폐 Top 10 포함)
# ---------------------------------------------------------
common_tickers = [
    # [미국 지수/섹터]
    "QQQ", "TQQQ", "QLD", "PSQ", "SQQQ", 
    "SPY", "UPRO", "SSO", 
    "SOXX", "SOXL", "SOXS", 
    "TLT", "TMF", "TMV",
    
    # [미국 빅테크/개별주]
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX",
    "COIN", "MSTR", 
    
    # [암호화폐 Top 10 (Stablecoin 제외)]
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "SOL-USD",   # Solana
    "BNB-USD",   # Binance Coin
    "XRP-USD",   # XRP
    "DOGE-USD",  # Dogecoin
    "ADA-USD",   # Cardano
    "TRX-USD",   # TRON
    "AVAX-USD",  # Avalanche
    "SHIB-USD",  # Shiba Inu

    # [한국 주식 예시]
    "005930.KS", # 삼성전자
    "000660.KS", # SK하이닉스
]

# ---------------------------------------------------------
# 3. 사이드바 설정 (입력)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정 패널")
    
    # 티커 선택 (직접 입력도 가능)
    selected_ticker = st.selectbox("티커 선택", common_tickers, index=0)
    ticker_input = st.text_input("직접 입력 (예: KRW=X)", value="")
    
    final_ticker = ticker_input.upper() if ticker_input else selected_ticker

    # 기간 설정
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("종료일", datetime.now())

    # 초기 자본
    initial_capital = st.number_input("초기 자본 ($)", value=10000, step=1000)

    st.subheader("전략 파라미터 (EMA)")
    ema_short_period = st.number_input("단기 이평선 (Short)", value=20)
    ema_long_period = st.number_input("장기 이평선 (Long)", value=60)

    run_btn = st.button("백테스트 실행 🚀")

# ---------------------------------------------------------
# 4. 데이터 로드 및 계산 함수
# ---------------------------------------------------------
@st.cache_data
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    # 멀티인덱스 컬럼 처리 (yfinance 최신 버전 대응)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def run_backtest(df, short_p, long_p, initial_cap):
    # 지표 계산
    df['EMA_Short'] = df['Close'].ewm(span=short_p, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_p, adjust=False).mean()
    
    # 시그널: 단기 > 장기일 때 매수 (1), 아니면 매도 (0)
    # (여기서는 간단하게 롱 포지션만 잡는 전략으로 가정)
    df['Signal'] = 0
    df.loc[df['EMA_Short'] > df['EMA_Long'], 'Signal'] = 1
    
    # 포지션 변경 확인 (1: 매수 진입, -1: 매도 청산)
    df['Position_Change'] = df['Signal'].diff()

    # 수익률 계산
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 전략 수익률 (전일 시그널 기준)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    df['Strategy_Return'].fillna(0, inplace=True)
    
    # 포트폴리오 가치
    df['Portfolio_Value'] = initial_cap * (1 + df['Strategy_Return']).cumprod()
    df['Buy_Hold_Value'] = initial_cap * (1 + df['Daily_Return']).cumprod()
    
    # 낙폭(MDD) 계산
    df['Peak'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Portfolio_Value'] - df['Peak']) / df['Peak']
    
    return df

# ---------------------------------------------------------
# 5. 차트 그리기 함수 (스타일 수정 적용됨)
# ---------------------------------------------------------
def plot_charts(df, ticker):
    # --- 1. Portfolio Value 차트 ---
    fig_value = go.Figure()
    
    # 전략 성과
    fig_value.add_trace(go.Scatter(
        x=df.index, y=df['Portfolio_Value'],
        mode='lines',
        name='Strategy',
        line=dict(color='red', width=1.0)  # width 1.0
    ))
    
    # Buy & Hold 성과 (검은색 실선 변경)
    fig_value.add_trace(go.Scatter(
        x=df.index, y=df['Buy_Hold_Value'],
        mode='lines',
        name=f'Only 1.0x ({ticker})',
        line=dict(color='black', width=1.0, dash='solid')  # 검은색, 실선, width 1.0
    ))
    
    fig_value.update_layout(
        title=f'💰 Portfolio Value vs Buy & Hold ({ticker})',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_value, use_container_width=True)

    # --- 2. Drawdown 차트 ---
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df.index, y=df['Drawdown'] * 100,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='blue', width=1.0)  # width 1.0
    ))
    fig_dd.update_layout(
        title='🌊 Drawdown (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- 3. Condition (Price & EMA) 차트 ---
    fig_cond = go.Figure()
    
    # 주가
    fig_cond.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='gray', width=1.0)  # width 1.0
    ))
    
    # 단기 EMA (실선 변경)
    fig_cond.add_trace(go.Scatter(
        x=df.index, y=df['EMA_Short'],
        mode='lines',
        name=f'EMA {ema_short_period}',
        line=dict(color='orange', width=1.0, dash='solid')  # 실선, width 1.0
    ))
        
    # 장기 EMA (실선 변경)
    fig_cond.add_trace(go.Scatter(
        x=df.index, y=df['EMA_Long'],
        mode='lines',
        name=f'EMA {ema_long_period}',
        line=dict(color='green', width=1.0, dash='solid')  # 실선, width 1.0
    ))

    # 매수/매도 화살표
    buy_signals = df[df['Position_Change'] == 1]
    sell_signals = df[df['Position_Change'] == -1]

    if not buy_signals.empty:
        fig_cond.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', size=8, color='red')
        ))
    
    if not sell_signals.empty:
        fig_cond.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', size=8, color='blue')
        ))

    fig_cond.update_layout(
        title=f'📊 Price & EMA Condition ({ticker})',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig_cond, use_container_width=True)

# ---------------------------------------------------------
# 6. 메인 실행 로직
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f'{final_ticker} 데이터 불러오는 중...'):
        df = get_data(final_ticker, start_date, end_date)
        
    if df is not None and not df.empty:
        # 백테스트 실행
        df = run_backtest(df, ema_short_period, ema_long_period, initial_capital)
        
        # 결과 요약 계산
        final_value = df['Portfolio_Value'].iloc[-1]
        bh_value = df['Buy_Hold_Value'].iloc[-1]
        
        total_return = (final_value / initial_capital - 1) * 100
        bh_return = (bh_value / initial_capital - 1) * 100
        mdd = df['Drawdown'].min() * 100
        
        # 화면 출력
        st.success(f"백테스트 완료! ({final_ticker})")
        
        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("최종 자산", f"${final_value:,.0f}")
        col2.metric("전략 수익률", f"{total_return:.2f}%")
        col3.metric("단순보유 수익률", f"{bh_return:.2f}%")
        col4.metric("최대 낙폭 (MDD)", f"{mdd:.2f}%")
        
        # 차트 그리기
        plot_charts(df, final_ticker)
        
        # 데이터프레임 보이기 (옵션)
        with st.expander("상세 데이터 보기"):
            st.dataframe(df.style.format("{:.2f}"))
            
    else:
        st.error("데이터를 가져올 수 없습니다. 티커나 날짜를 확인해주세요.")

