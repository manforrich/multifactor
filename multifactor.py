import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt # 確保這行有被 import

# 設定頁面
st.set_page_config(page_title="多因子量化選股工具", layout="wide")
st.title("⚖️ 多因子量化評分系統 (台美股通用版)")
st.markdown("結合 **動能 (Momentum)**、**價值 (Value)** 與 **低波動 (Low Vol)** 三大面向進行評分。")

# --- 0050 成分股清單 (硬代碼，可隨時更新) ---

tw0050_list = [
    "2330", "2317", "2454", "2382", "2308", "2881", "2882", "2303", "2886", "2891",
    "3711", "2884", "1216", "2885", "2002", "2412", "5880", "3231", "2892", "2880",
    "3034", "3008", "2603", "2357", "2883", "1101", "2379", "5871", "2345", "3045",
    "5876", "1519", "2890", "2912", "4904", "2887", "3037", "2327", "2408", "2395",
    "3661", "3017", "4938", "1590", "1605", "6669", "6415", "3529", "2301", "3017"
]

# 轉成 Yahoo Finance 格式 (加上 .TW)
tw0050_tickers = [f"{x}.TW" for x in tw0050_list]
tw0050_str = ", ".join(tw0050_tickers)

# --- Session State 初始化 (為了讓按鈕能更新輸入框) ---
if 'ticker_input' not in st.session_state:
    st.session_state['ticker_input'] = "NVDA, MSFT, AAPL, GOOGL, AMZN, TSLA, META, AVGO, AMD, QQQ, SPY"

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 參數設定")

# 快速按鈕區
st.sidebar.subheader("📋 快速帶入清單")
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🇺🇸 美股科技"):
        st.session_state['ticker_input'] = "NVDA, MSFT, AAPL, GOOGL, AMZN, TSLA, META, AVGO, AMD, QQQ, SPY"
with col_btn2:
    if st.button("🇹🇼 台股 0050"):
        st.session_state['ticker_input'] = tw0050_str

# 1. 股票池輸入框 (綁定 key='ticker_input')
user_tickers = st.sidebar.text_area(
    "輸入股票代碼 (逗號分隔)", 
    key='ticker_input', # 這裡綁定 session_state
    height=150
)

# 2. 因子權重設定
st.sidebar.subheader("因子權重 (總和需為 100%)")
w_mom = st.sidebar.slider("動能 (半年漲幅) 權重", 0, 100, 40)
w_val = st.sidebar.slider("價值 (本益比) 權重", 0, 100, 30)
w_vol = st.sidebar.slider("低波動 (標準差) 權重", 0, 100, 30)

total_weight = w_mom + w_val + w_vol
if total_weight != 100:
    st.sidebar.error(f"⚠️ 目前權重總和為 {total_weight}%，請調整至 100%")

# --- 核心函數 ---
@st.cache_data(ttl=3600)
def get_factor_data(ticker_list):
    data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(ticker_list)
    
    for i, ticker in enumerate(ticker_list):
        t = ticker.strip().upper()
        if not t: continue
        
        status_text.text(f"正在分析 ({i+1}/{total_tickers}): {t} ...")
        progress_bar.progress((i + 1) / total_tickers)
        
        try:
            stock = yf.Ticker(t)
            
            # 1. 抓取歷史股價
            hist = stock.history(period="6mo")
            if hist.empty: continue
            
            # 計算動能
            momentum = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            
            # 計算波動率
            daily_ret = hist['Close'].pct_change().dropna()
            volatility = daily_ret.std() * np.sqrt(252) * 100
            
            # 2. 抓取基本面
            info = stock.info
            # 台股有時候抓不到 forwardPE，改用 trailingPE 或是用股價/EPS估算
            pe = info.get('forwardPE')
            if pe is None:
                pe = info.get('trailingPE')
            
            # 產業分類
            sector = info.get('sector', 'Unknown')
            name = info.get('shortName', t) # 嘗試抓取公司簡稱

            data.append({
                "代碼": t,
                "名稱": name,
                "最新價": round(hist['Close'].iloc[-1], 2),
                "動能(%)": round(momentum, 2),
                "波動率(%)": round(volatility, 2),
                "本益比": round(pe, 2) if pe else None,
                "產業": sector
            })
            
        except Exception as e:
            print(f"Error fetching {t}: {e}")
            
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(data)

# --- 評分邏輯 ---
def calculate_scores(df, w_m, w_v, w_l):
    res = df.copy()
    
    # 處理缺失值 (本益比如果是空值，填入該欄位的平均值，以免無法計算)
    res.fillna(res.mean(numeric_only=True), inplace=True)
    
    # 1. 動能分數 (越高越好)
    res['動能分數'] = res['動能(%)'].rank(pct=True) * 100
    
    # 2. 價值分數 (PE 越低越好 -> 反向排序)
    # 這裡加入一個保護：如果本益比是負的(虧損)，視為極差，給予低分
    res['價值分數'] = res['本益比'].rank(ascending=False, pct=True) * 100
    
    # 3. 低波動分數 (波動越低越好 -> 反向排序)
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
    tickers = st.session_state['ticker_input'].split(",") # 改從 session_state 讀取
    if len(tickers) > 0:
        raw_df = get_factor_data(tickers)
        
        if not raw_df.empty:
            final_df = calculate_scores(raw_df, w_mom, w_val, w_vol)
            
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.subheader("🏆 綜合評分排行榜")
                # 為了顯示好看，只選取重要欄位
                display_cols = ['代碼', '名稱', '總分', '最新價', '動能(%)', '波動率(%)', '本益比', '產業']
                
                st.dataframe(
                    final_df[display_cols].style.background_gradient(subset=['總分'], cmap='RdYlGn')
                                  .format("{:.2f}", subset=['動能(%)', '波動率(%)', '本益比', '總分']),
                    use_container_width=True,
                    height=600
                )
            
            with c2:
                st.subheader("📊 價值 vs 動能")
                fig = px.scatter(
                    final_df, 
                    x="本益比", 
                    y="動能(%)", 
                    size="總分", 
                    color="總分",
                    hover_name="名稱", 
                    text="代碼",
                    title="氣泡越大 = 總分越高",
                    color_continuous_scale='RdYlGn',
                    labels={"本益比": "本益比 (越左越便宜)", "動能(%)": "動能 (越上越強)"}
                )
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **台股 0050 觀察重點：**
                * **左上角 (鑽石區)**：動能強、本益比低。這通常是財報剛開出來、市場還沒完全反應的優質股。
                * **台積電 (2330)**：通常在本益比 15-25 倍之間，你可以觀察它的分數變化。
                * **金融股**：通常波動率極低 (適合存股)，在「低波動」權重高時會排在前面。
                """)

        else:
            st.error("無法抓取數據，請檢查代碼或網路連線。")
    else:
        st.warning("請輸入股票代碼")
