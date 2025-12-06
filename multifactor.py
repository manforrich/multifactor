import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# 設定頁面
st.set_page_config(page_title="多因子量化選股工具", layout="wide")
st.title("⚖️ 多因子量化評分系統")
st.markdown("結合 **動能 (Momentum)**、**價值 (Value)** 與 **低波動 (Low Vol)** 三大面向進行評分。")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 參數設定")

# 1. 股票池 (預設美股科技巨頭 + 穩定股 + 大盤ETF)
default_tickers = "NVDA, MSFT, AAPL, GOOGL, AMZN, TSLA, META, AVGO, AMD, QQQ, SPY, V, JPM, KO, PEP"
user_tickers = st.sidebar.text_area("輸入股票代碼 (逗號分隔)", default_tickers, height=100)

# 2. 因子權重設定
st.sidebar.subheader("因子權重 (總和需為 100%)")
w_mom = st.sidebar.slider("動能 (半年漲幅) 權重", 0, 100, 40)
w_val = st.sidebar.slider("價值 (預估本益比) 權重", 0, 100, 30)
w_vol = st.sidebar.slider("低波動 (標準差) 權重", 0, 100, 30)

total_weight = w_mom + w_val + w_vol
if total_weight != 100:
    st.sidebar.error(f"⚠️ 目前權重總和為 {total_weight}%，請調整至 100%")

# --- 核心函數 ---
@st.cache_data(ttl=3600) # 快取 1 小時，避免重複抓取變慢
def get_factor_data(ticker_list):
    data = []
    
    # 建立進度條
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(ticker_list):
        t = ticker.strip().upper()
        if not t: continue
        
        status_text.text(f"正在分析: {t} ...")
        progress_bar.progress((i + 1) / len(ticker_list))
        
        try:
            stock = yf.Ticker(t)
            
            # 1. 抓取歷史股價 (用來算動能和波動)
            hist = stock.history(period="6mo")
            if hist.empty: continue
            
            # 計算動能 (Momentum): 過去6個月報酬率
            momentum = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            
            # 計算波動率 (Volatility): 日報酬率的標準差 (年化)
            daily_ret = hist['Close'].pct_change().dropna()
            volatility = daily_ret.std() * np.sqrt(252) * 100
            
            # 2. 抓取基本面 (用來算價值)
            info = stock.info
            # 優先使用 Forward PE，如果沒有則用 Trailing PE
            pe = info.get('forwardPE', info.get('trailingPE', None))
            
            # 如果真的沒有 PE (例如虧損公司或ETF)，給一個預設值 (例如設為中位數或忽略)
            # 這裡為了展示，若無數據則標記為 NaN
            
            data.append({
                "代碼": t,
                "最新價": round(hist['Close'].iloc[-1], 2),
                "動能(%)": round(momentum, 2),
                "波動率(%)": round(volatility, 2),
                "預估本益比": round(pe, 2) if pe else None,
                "產業": info.get('sector', 'ETF/Unknown')
            })
            
        except Exception as e:
            print(f"Error fetching {t}: {e}")
            
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(data)

# --- 評分邏輯 (正規化) ---
def calculate_scores(df, w_m, w_v, w_l):
    res = df.copy()
    
    # 處理缺失值 (填入平均值，避免計算錯誤)
    res.fillna(res.mean(numeric_only=True), inplace=True)
    
    # 1. 動能分數 (越高越好): 使用 Percentile Rank (0-100分)
    res['動能分數'] = res['動能(%)'].rank(pct=True) * 100
    
    # 2. 價值分數 (PE 越低越好): 反向排序
    # 注意：PE 可能是負的或極端值，這裡簡單處理：PE越小分數越高
    res['價值分數'] = res['預估本益比'].rank(ascending=False, pct=True) * 100
    
    # 3. 低波動分數 (波動越低越好): 反向排序
    res['低波動分數'] = res['波動率(%)'].rank(ascending=False, pct=True) * 100
    
    # 4. 綜合總分
    res['總分'] = (
        res['動能分數'] * (w_m/100) +
        res['價值分數'] * (w_v/100) +
        res['低波動分數'] * (w_l/100)
    )
    
    return res.sort_values(by='總分', ascending=False)

# --- 主程式 ---
if st.button("🚀 開始多因子分析"):
    tickers = user_tickers.split(",")
    if len(tickers) > 0:
        raw_df = get_factor_data(tickers)
        
        if not raw_df.empty:
            final_df = calculate_scores(raw_df, w_mom, w_val, w_vol)
            
            # 顯示結果
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.subheader("🏆 綜合評分排行榜")
                # 格式化顯示 (Color Map)
                st.dataframe(
                    final_df.style.background_gradient(subset=['總分'], cmap='RdYlGn')
                                  .format("{:.2f}", subset=['動能(%)', '波動率(%)', '預估本益比', '動能分數', '價值分數', '低波動分數', '總分']),
                    use_container_width=True,
                    height=500
                )
            
            with c2:
                st.subheader("📊 因子分佈")
                # 散點圖：動能 vs 價值
                fig = px.scatter(
                    final_df, 
                    x="預估本益比", 
                    y="動能(%)", 
                    size="總分", 
                    color="總分",
                    hover_name="代碼", 
                    text="代碼",
                    title="價值 vs 動能 (氣泡大小=總分)",
                    color_continuous_scale='RdYlGn'
                )
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **解讀指南：**
                * **右上角**：動能強且本益比高 (如 NVDA)，適合順勢交易，但要注意反轉。
                * **左上角**：動能強且本益比低 (黃金區)，通常是**高分優質股**。
                * **右下角**：動能弱且本益比高，需避開。
                * **左下角**：動能弱但便宜，可能是價值陷阱或底部股。
                """)

        else:
            st.error("無法抓取數據，請檢查股票代碼是否正確。")
    else:
        st.warning("請輸入股票代碼")
