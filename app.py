"""
Stock Pulse — Interactive Dashboard
====================================
AI-Powered Stock Recommendation Engine for the Colombo Stock Exchange (CSE).
Launch: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Pulse | CSE Recommendations",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 50%, #415a77 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e1dd !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #e0e1dd !important;
        font-weight: 500;
    }
    
    /* Hero header gradient */
    .hero-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 40%, #415a77 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(13, 27, 42, 0.3);
    }
    .hero-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.3rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0d1b2a;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.2rem;
    }
    
    /* Recommendation cards */
    .rec-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        border-left: 4px solid #00b4d8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .rec-rank {
        font-size: 2rem;
        font-weight: 800;
        color: #415a77;
        opacity: 0.3;
    }
    .rec-stock {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0d1b2a;
    }
    .rec-prob {
        font-size: 1.4rem;
        font-weight: 700;
    }
    .prob-high { color: #2dc653; }
    .prob-med { color: #fb8500; }
    .prob-low { color: #ef233c; }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0d1b2a;
        border-bottom: 3px solid #00b4d8;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        margin-top: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 8px 20px;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#0d1b2a',
    'secondary': '#1b263b',
    'accent': '#415a77',
    'teal': '#00b4d8',
    'green': '#2dc653',
    'red': '#ef233c',
    'orange': '#fb8500',
    'gold': '#ffd60a',
    'purple': '#7209b7',
    'light_bg': '#f8f9fa',
    'grid': '#dee2e6',
}

PLOTLY_TEMPLATE = {
    'layout': {
        'font': {'family': 'Inter, sans-serif', 'color': '#212529'},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(248,249,250,0.5)',
        'xaxis': {'gridcolor': '#dee2e6', 'gridwidth': 0.5},
        'yaxis': {'gridcolor': '#dee2e6', 'gridwidth': 0.5},
        'margin': {'l': 40, 'r': 20, 't': 50, 'b': 40},
    }
}


# ─── Data Loading (Cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading CSE market data...")
def load_engineered_data():
    """Load the engineered features parquet file."""
    paths = [
        os.path.join(r"c:\Users\HP\Documents\Stock_pulse\Dataset\2025 Q4", "engineered_features.parquet"),
        os.path.join(r"c:\Users\HP\Documents\Stock_pulse\Dataset", "engineered_features.parquet"),
        os.path.join(r"c:\Users\HP\Documents\Stock_pulse", "engineered_features.parquet"),
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
    return None


@st.cache_data(ttl=3600, show_spinner="Loading raw price data...")
def load_raw_data():
    """Load the cached consolidated daily prices parquet."""
    paths = [
        os.path.join(r"c:\Users\HP\Documents\Stock_pulse\Dataset\2025 Q4", "consolidated_daily_prices.parquet"),
        os.path.join(r"c:\Users\HP\Documents\Stock_pulse\Dataset", "consolidated_daily_prices.parquet"),
        os.path.join(r"c:\Users\HP\Documents\Stock_pulse", "consolidated_daily_prices.parquet"),
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
    return None


@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """Load the saved XGBoost model."""
    import joblib
    paths = [
        os.path.join(r"c:\Users\HP\Documents\Stock_pulse\Dataset\2025 Q4", "xgboost_3m_model.pkl"),
        os.path.join(r"c:\Users\HP\Documents\Stock_pulse\Dataset", "xgboost_3m_model.pkl"),
        os.path.join(r"c:\Users\HP\Documents\Stock_pulse", "xgboost_3m_model.pkl"),
    ]
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p)
    return None


# ─── Feature columns (must match model training) ─────────────────────────────
FEATURE_COLS = [
    'SMA_20', 'SMA_50', 'SMA_200',
    'Dist_SMA20', 'Dist_SMA50', 'Dist_SMA200',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Lower', 'BB_Width',
    'ATR_14', 'NATR_14', 'ROC_10', 'ROC_21',
    'Vol_21d', 'Vol_63d', 'Volume_Ratio'
]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Stock Pulse")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Stock Explorer", "🤖 AI Recommendations", "📈 Model Performance"],
        index=0,
    )
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:0.75rem; opacity:0.6;'>"
        "Built with ❤️ by Dushan Lakruwan<br>"
        "Colombo Stock Exchange Data (1991-2025)"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    # Hero
    st.markdown(
        '<div class="hero-header">'
        '<h1>📈 Stock Pulse</h1>'
        '<p>AI-Powered Stock Recommendation Engine for the Colombo Stock Exchange</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    df = load_engineered_data()
    
    if df is not None:
        # ── Key Metrics ──────────────────────────────────────────────────
        total_stocks = df['CompanyCode'].nunique()
        total_records = len(df)
        date_range_start = df['Date'].min().strftime('%Y')
        date_range_end = df['Date'].max().strftime('%Y')
        latest_date = df['Date'].max().strftime('%b %d, %Y')
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-icon">🏢</div>'
                f'<div class="metric-value">{total_stocks:,}</div>'
                '<div class="metric-label">Listed Companies</div>'
                '</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-icon">📋</div>'
                f'<div class="metric-value">{total_records:,.0f}</div>'
                '<div class="metric-label">Total Data Points</div>'
                '</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-icon">📅</div>'
                f'<div class="metric-value">{date_range_start}–{date_range_end}</div>'
                '<div class="metric-label">Data Coverage</div>'
                '</div>', unsafe_allow_html=True)
        with c4:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-icon">🔄</div>'
                f'<div class="metric-value">{latest_date}</div>'
                '<div class="metric-label">Latest Data</div>'
                '</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        # ── Market Overview Chart ────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Market Activity Over Time</div>', unsafe_allow_html=True)
        
        # Aggregate daily market turnover
        daily_agg = df.groupby('Date').agg(
            total_turnover=('Turnover', 'sum'),
            avg_close=('Close', 'mean'),
            num_stocks=('CompanyCode', 'nunique')
        ).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=("Average Stock Price (All Companies)", "Daily Market Turnover (LKR)")
        )
        
        fig.add_trace(go.Scatter(
            x=daily_agg['Date'], y=daily_agg['avg_close'],
            mode='lines', name='Avg Price',
            line=dict(color=COLORS['teal'], width=1.5),
            fill='tozeroy', fillcolor='rgba(0,180,216,0.1)',
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=daily_agg['Date'], y=daily_agg['total_turnover'],
            name='Turnover', marker_color=COLORS['accent'],
            opacity=0.6,
        ), row=2, col=1)
        
        fig.update_layout(
            height=500, showlegend=False,
            font=dict(family='Inter', size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,249,250,0.5)',
            margin=dict(l=50, r=20, t=40, b=30),
        )
        fig.update_xaxes(gridcolor=COLORS['grid'], gridwidth=0.5)
        fig.update_yaxes(gridcolor=COLORS['grid'], gridwidth=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ── Sector Distribution ──────────────────────────────────────────
        st.markdown('<div class="section-header">🏭 Sector Distribution</div>', unsafe_allow_html=True)
        
        if 'MainType' in df.columns:
            sector_counts = df.groupby('MainType')['CompanyCode'].nunique().sort_values(ascending=True).tail(15)
            
            fig_sector = go.Figure(go.Bar(
                x=sector_counts.values,
                y=sector_counts.index,
                orientation='h',
                marker=dict(
                    color=sector_counts.values,
                    colorscale='Tealgrn',
                    line=dict(width=0),
                ),
                text=sector_counts.values,
                textposition='outside',
            ))
            fig_sector.update_layout(
                height=450,
                font=dict(family='Inter', size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,249,250,0.5)',
                margin=dict(l=150, r=40, t=20, b=30),
                xaxis_title="Number of Companies",
            )
            fig_sector.update_xaxes(gridcolor=COLORS['grid'])
            fig_sector.update_yaxes(gridcolor=COLORS['grid'])
            st.plotly_chart(fig_sector, use_container_width=True)
        
        # ── How it works ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">🧠 How Stock Pulse Works</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        steps = [
            ("📥", "Data Ingestion", "34 years of CSE historical data from ZIP archives"),
            ("🧹", "Data Cleaning", "Stock split adjustment, calendar alignment, outlier removal"),
            ("📐", "Feature Engineering", "20+ technical indicators: RSI, MACD, Bollinger Bands, ATR..."),
            ("🤖", "AI Prediction", "XGBoost model trained on 22 years of data, tested on 3 years"),
        ]
        for col, (icon, title, desc) in zip([col1, col2, col3, col4], steps):
            with col:
                st.markdown(
                    f'<div class="metric-card" style="text-align:left;">'
                    f'<div style="font-size:2rem;">{icon}</div>'
                    f'<div style="font-weight:700; font-size:1rem; margin:0.5rem 0 0.3rem;">{title}</div>'
                    f'<div style="font-size:0.85rem; color:#6c757d;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("⚠️ Engineered data not found. Please run the Phase 2 notebook first to generate `engineered_features.parquet`.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: STOCK EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Stock Explorer":
    st.markdown(
        '<div class="hero-header">'
        '<h1>📊 Stock Explorer</h1>'
        '<p>Deep-dive into any CSE-listed company with interactive charts and technical indicators</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    df = load_engineered_data()
    
    if df is not None:
        stocks = sorted(df['CompanyCode'].unique())
        
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_stock = st.selectbox("Select Company", stocks, index=0)
        with col_sel2:
            period = st.selectbox("Time Period", ["All Time", "Last 5 Years", "Last 2 Years", "Last 1 Year", "Last 6 Months"], index=2)
        
        stock_df = df[df['CompanyCode'] == selected_stock].copy().sort_values('Date')
        
        # Apply time filter
        if period == "Last 5 Years":
            stock_df = stock_df[stock_df['Date'] >= stock_df['Date'].max() - pd.Timedelta(days=5*365)]
        elif period == "Last 2 Years":
            stock_df = stock_df[stock_df['Date'] >= stock_df['Date'].max() - pd.Timedelta(days=2*365)]
        elif period == "Last 1 Year":
            stock_df = stock_df[stock_df['Date'] >= stock_df['Date'].max() - pd.Timedelta(days=365)]
        elif period == "Last 6 Months":
            stock_df = stock_df[stock_df['Date'] >= stock_df['Date'].max() - pd.Timedelta(days=183)]
        
        if len(stock_df) > 0:
            latest = stock_df.iloc[-1]
            
            # ── Key Stats Cards ──────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            
            current_price = latest['Close']
            high_52w = stock_df.tail(252)['High'].max() if len(stock_df) >= 252 else stock_df['High'].max()
            low_52w = stock_df.tail(252)['Low'].min() if len(stock_df) >= 252 else stock_df['Low'].min()
            rsi_val = latest.get('RSI_14', None)
            vol_val = latest.get('Vol_63d', None)
            
            with c1:
                st.metric("Current Price", f"Rs. {current_price:,.2f}")
            with c2:
                st.metric("52-Week High", f"Rs. {high_52w:,.2f}")
            with c3:
                st.metric("52-Week Low", f"Rs. {low_52w:,.2f}")
            with c4:
                rsi_display = f"{rsi_val:.1f}" if pd.notna(rsi_val) else "N/A"
                st.metric("RSI (14)", rsi_display)
            with c5:
                vol_display = f"{vol_val*100:.1f}%" if pd.notna(vol_val) else "N/A"
                st.metric("Volatility (63d)", vol_display)
            
            # ── Candlestick Chart ────────────────────────────────────────
            st.markdown('<div class="section-header">🕯️ Price Chart</div>', unsafe_allow_html=True)
            
            show_indicators = st.multiselect(
                "Overlay Indicators",
                ["SMA 20", "SMA 50", "SMA 200", "Bollinger Bands"],
                default=["SMA 20", "SMA 50"],
            )
            
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.55, 0.25, 0.20],
                subplot_titles=(f"{selected_stock} — Price", "Volume", "RSI (14)")
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=stock_df['Date'],
                open=stock_df['Open'], high=stock_df['High'],
                low=stock_df['Low'], close=stock_df['Close'],
                name='OHLC',
                increasing_line_color=COLORS['green'],
                decreasing_line_color=COLORS['red'],
            ), row=1, col=1)
            
            # Indicator overlays
            if "SMA 20" in show_indicators and 'SMA_20' in stock_df.columns:
                fig.add_trace(go.Scatter(
                    x=stock_df['Date'], y=stock_df['SMA_20'],
                    name='SMA 20', line=dict(color=COLORS['orange'], width=1.2, dash='dot'),
                ), row=1, col=1)
            
            if "SMA 50" in show_indicators and 'SMA_50' in stock_df.columns:
                fig.add_trace(go.Scatter(
                    x=stock_df['Date'], y=stock_df['SMA_50'],
                    name='SMA 50', line=dict(color=COLORS['teal'], width=1.2, dash='dot'),
                ), row=1, col=1)
            
            if "SMA 200" in show_indicators and 'SMA_200' in stock_df.columns:
                fig.add_trace(go.Scatter(
                    x=stock_df['Date'], y=stock_df['SMA_200'],
                    name='SMA 200', line=dict(color=COLORS['purple'], width=1.5),
                ), row=1, col=1)
            
            if "Bollinger Bands" in show_indicators:
                if 'BB_Upper' in stock_df.columns and 'BB_Lower' in stock_df.columns:
                    fig.add_trace(go.Scatter(
                        x=stock_df['Date'], y=stock_df['BB_Upper'],
                        name='BB Upper', line=dict(color='rgba(119,141,169,0.5)', width=1),
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=stock_df['Date'], y=stock_df['BB_Lower'],
                        name='BB Lower', line=dict(color='rgba(119,141,169,0.5)', width=1),
                        fill='tonexty', fillcolor='rgba(119,141,169,0.08)',
                    ), row=1, col=1)
            
            # Volume bars
            vol_colors = [COLORS['green'] if c >= o else COLORS['red'] 
                          for c, o in zip(stock_df['Close'], stock_df['Open'])]
            fig.add_trace(go.Bar(
                x=stock_df['Date'], y=stock_df['ShareVolume'],
                name='Volume', marker_color=vol_colors, opacity=0.7,
            ), row=2, col=1)
            
            # RSI
            if 'RSI_14' in stock_df.columns:
                fig.add_trace(go.Scatter(
                    x=stock_df['Date'], y=stock_df['RSI_14'],
                    name='RSI 14', line=dict(color=COLORS['teal'], width=1.5),
                ), row=3, col=1)
                # Overbought/Oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color=COLORS['red'], opacity=0.5, row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color=COLORS['green'], opacity=0.5, row=3, col=1)
            
            fig.update_layout(
                height=750,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                font=dict(family='Inter', size=11),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,249,250,0.5)',
                margin=dict(l=50, r=20, t=40, b=30),
                xaxis_rangeslider_visible=False,
            )
            fig.update_xaxes(gridcolor=COLORS['grid'], gridwidth=0.5)
            fig.update_yaxes(gridcolor=COLORS['grid'], gridwidth=0.5)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ── MACD Chart ───────────────────────────────────────────────
            if 'MACD' in stock_df.columns:
                st.markdown('<div class="section-header">📉 MACD Analysis</div>', unsafe_allow_html=True)
                
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=stock_df['Date'], y=stock_df['MACD'],
                    name='MACD', line=dict(color=COLORS['teal'], width=1.5),
                ))
                fig_macd.add_trace(go.Scatter(
                    x=stock_df['Date'], y=stock_df['MACD_Signal'],
                    name='Signal', line=dict(color=COLORS['orange'], width=1.5),
                ))
                
                hist_colors = [COLORS['green'] if v >= 0 else COLORS['red'] 
                               for v in stock_df['MACD_Hist']]
                fig_macd.add_trace(go.Bar(
                    x=stock_df['Date'], y=stock_df['MACD_Hist'],
                    name='Histogram', marker_color=hist_colors, opacity=0.5,
                ))
                
                fig_macd.update_layout(
                    height=300,
                    font=dict(family='Inter', size=11),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(248,249,250,0.5)',
                    margin=dict(l=50, r=20, t=20, b=30),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                )
                fig_macd.update_xaxes(gridcolor=COLORS['grid'])
                fig_macd.update_yaxes(gridcolor=COLORS['grid'])
                
                st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.warning("⚠️ Data not available. Please run notebooks first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: AI RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Recommendations":
    st.markdown(
        '<div class="hero-header">'
        '<h1>🤖 AI Recommendations</h1>'
        '<p>XGBoost-powered stock picks ranked by predicted uptrend probability</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    df = load_engineered_data()
    model = load_model()
    
    if df is not None and model is not None:
        # ── Controls ─────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            min_price = st.slider("Minimum Price (LKR)", 1, 100, 5, help="Filter out penny stocks below this price")
        with col2:
            max_vol = st.slider("Max Volatility (%)", 20, 100, 80, help="Filter out stocks with annualized volatility above this") / 100
        with col3:
            top_n = st.slider("Number of Recommendations", 5, 25, 10)
        
        # Get latest date per stock
        latest_date = df['Date'].max()
        df_current = df[df['Date'] == latest_date].copy()
        
        st.info(f"📅 Generating recommendations based on market data as of **{latest_date.strftime('%B %d, %Y')}**")
        
        # Apply filters
        df_filtered = df_current[
            (df_current['Close'] >= min_price) &
            (df_current['Vol_63d'] <= max_vol)
        ].copy()
        df_filtered = df_filtered.dropna(subset=FEATURE_COLS)
        
        if len(df_filtered) > 0:
            # Run model
            X = df_filtered[FEATURE_COLS]
            probabilities = model.predict_proba(X)[:, 1]
            df_filtered = df_filtered.copy()
            df_filtered['Uptrend_Probability'] = probabilities
            
            recommendations = df_filtered.sort_values('Uptrend_Probability', ascending=False).head(top_n)
            
            # ── Display Recommendations ──────────────────────────────────
            st.markdown('<div class="section-header">🏆 Top Recommendations</div>', unsafe_allow_html=True)
            
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                rank = idx + 1
                prob = row['Uptrend_Probability']
                prob_pct = prob * 100
                
                if prob >= 0.7:
                    prob_class = "prob-high"
                    signal = "🟢 STRONG BUY"
                elif prob >= 0.5:
                    prob_class = "prob-med"
                    signal = "🟡 BUY"
                else:
                    prob_class = "prob-low"
                    signal = "🔴 WEAK"
                
                rsi_str = f"{row['RSI_14']:.1f}" if pd.notna(row.get('RSI_14')) else "N/A"
                vol_str = f"{row['Vol_63d']*100:.1f}%" if pd.notna(row.get('Vol_63d')) else "N/A"
                
                c1, c2, c3, c4, c5 = st.columns([0.5, 2, 1.5, 1.5, 1.5])
                with c1:
                    st.markdown(f"<div style='font-size:1.8rem; font-weight:800; color:#415a77; opacity:0.4;'>#{rank}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{row['CompanyCode']}**")
                    st.caption(f"Rs. {row['Close']:,.2f}")
                with c3:
                    st.markdown(f"<span class='{prob_class}' style='font-size:1.3rem; font-weight:700;'>{prob_pct:.1f}%</span>", unsafe_allow_html=True)
                    st.caption("Uptrend Probability")
                with c4:
                    st.markdown(f"{signal}")
                    st.caption(f"RSI: {rsi_str}")
                with c5:
                    st.caption(f"Volatility: {vol_str}")
                
                st.divider()
            
            # ── Probability Distribution ─────────────────────────────────
            st.markdown('<div class="section-header">📊 Probability Distribution (All Filtered Stocks)</div>', unsafe_allow_html=True)
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=df_filtered['Uptrend_Probability'] * 100,
                nbinsx=30,
                marker_color=COLORS['teal'],
                opacity=0.75,
                name='Stocks',
            ))
            fig_dist.add_vline(x=50, line_dash="dash", line_color=COLORS['red'], 
                              annotation_text="50% threshold", annotation_position="top")
            fig_dist.update_layout(
                height=350,
                xaxis_title="Predicted Uptrend Probability (%)",
                yaxis_title="Number of Stocks",
                font=dict(family='Inter', size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,249,250,0.5)',
                margin=dict(l=50, r=20, t=20, b=50),
            )
            fig_dist.update_xaxes(gridcolor=COLORS['grid'])
            fig_dist.update_yaxes(gridcolor=COLORS['grid'])
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # ── Why Follow Our Recommendations ──────────────────────────
            st.markdown('<div class="section-header">💡 Why Follow Our Recommendations?</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### 🧪 Scientifically Rigorous
                - **No future data leakage**: Strict time-series split (trained on 2001-2022, tested on 2023-2025)
                - **20+ technical indicators**: RSI, MACD, Bollinger Bands, ATR, Moving Averages — the same tools used by professional quants
                - **Precision-optimized**: We'd rather miss a good stock than recommend a bad one that loses you money
                
                #### 🛡️ Built-In Risk Management
                - **Penny stock filter**: Automatically excludes low-price, high-manipulation-risk stocks
                - **Volatility cap**: Filters out hyper-volatile stocks prone to sudden crashes
                - **Multi-indicator consensus**: The model cross-references 20+ signals, not just one indicator
                """)
            with col2:
                st.markdown("""
                #### 📊 Backed by Data
                - **2.4M+ data points** from 34 years of CSE trading history
                - **300+ companies** analyzed simultaneously 
                - XGBoost ensemble of 100 decision trees, each learning different market patterns
                
                #### ⚡ Probabilistic Ranking
                Unlike binary "buy/sell" signals, Stock Pulse provides a **probability score** for each stock. This lets you:
                - Focus on the highest-confidence picks
                - Diversify across probability tiers
                - Adjust your risk tolerance dynamically
                """)
            
            # Download button
            st.markdown("---")
            csv = recommendations[['CompanyCode', 'Close', 'Uptrend_Probability', 'RSI_14', 'Vol_63d']].to_csv(index=False)
            st.download_button(
                "📥 Download Recommendations (CSV)",
                csv,
                "stock_pulse_recommendations.csv",
                "text/csv",
            )
        else:
            st.warning("No stocks passed the current filters. Try adjusting the parameters.")
    
    elif model is None:
        st.warning("⚠️ AI model not found. Please run the Phase 3 notebook (`03_Recommendation_System.ipynb`) first to train and save the XGBoost model.")
    else:
        st.warning("⚠️ Data not available. Please run the Phase 2 notebook first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown(
        '<div class="hero-header">'
        '<h1>📈 Model Performance</h1>'
        '<p>Evaluation metrics, feature importance, and test-period analysis</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    df = load_engineered_data()
    model = load_model()
    
    if df is not None and model is not None:
        # Reproduce the train/test split
        target = 'Is_Uptrend_3M'
        cols_to_check = FEATURE_COLS + [target]
        df_clean = df.dropna(subset=cols_to_check).copy()
        
        train_df = df_clean[df_clean['Date'].dt.year <= 2022]
        test_df = df_clean[df_clean['Date'].dt.year > 2022]
        
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[target]
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # ── Metrics Cards ────────────────────────────────────────────────
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics_data = [
            ("🎯", "Precision", f"{precision:.3f}", c1),
            ("🔍", "Recall", f"{recall:.3f}", c2),
            ("⚖️", "F1-Score", f"{f1:.3f}", c3),
            ("📈", "ROC-AUC", f"{roc_auc:.3f}", c4),
            ("✅", "Accuracy", f"{accuracy:.3f}", c5),
        ]
        for icon, label, value, col in metrics_data:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-icon">{icon}</div>'
                    f'<div class="metric-value">{value}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        # ── Two columns: Feature Importance + Confusion Matrix ───────────
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown('<div class="section-header">🏆 Feature Importance</div>', unsafe_allow_html=True)
            
            importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
            
            fig_imp = go.Figure(go.Bar(
                x=importances.values,
                y=importances.index,
                orientation='h',
                marker=dict(
                    color=importances.values,
                    colorscale='Tealgrn',
                ),
                text=[f'{v:.3f}' for v in importances.values],
                textposition='outside',
            ))
            fig_imp.update_layout(
                height=500,
                font=dict(family='Inter', size=11),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,249,250,0.5)',
                margin=dict(l=100, r=60, t=10, b=30),
                xaxis_title="Importance Score",
            )
            fig_imp.update_xaxes(gridcolor=COLORS['grid'])
            fig_imp.update_yaxes(gridcolor=COLORS['grid'])
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
            
            cm = confusion_matrix(y_test, y_pred)
            labels = ['No Uptrend', 'Uptrend']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels, y=labels,
                colorscale='Blues',
                text=cm, texttemplate="%{text}",
                textfont=dict(size=18, color='white'),
                showscale=False,
            ))
            fig_cm.update_layout(
                height=400,
                font=dict(family='Inter', size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=10, b=30),
                xaxis_title="Predicted",
                yaxis_title="Actual",
                yaxis=dict(autorange='reversed'),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # ── ROC Curve ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📉 ROC Curve</div>', unsafe_allow_html=True)
        
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
        
        col_roc, col_pr = st.columns(2)
        
        with col_roc:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'XGBoost (AUC={roc_auc:.3f})',
                line=dict(color=COLORS['teal'], width=2.5),
                fill='tozeroy', fillcolor='rgba(0,180,216,0.1)',
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name='Random Baseline',
                line=dict(color=COLORS['grid'], width=1, dash='dash'),
            ))
            fig_roc.update_layout(
                height=380,
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                font=dict(family='Inter', size=11),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,249,250,0.5)',
                margin=dict(l=50, r=20, t=40, b=50),
                legend=dict(x=0.5, y=0.05),
            )
            fig_roc.update_xaxes(gridcolor=COLORS['grid'])
            fig_roc.update_yaxes(gridcolor=COLORS['grid'])
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col_pr:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=rec_vals, y=prec_vals, mode='lines',
                name='Precision-Recall',
                line=dict(color=COLORS['orange'], width=2.5),
                fill='tozeroy', fillcolor='rgba(251,133,0,0.1)',
            ))
            fig_pr.update_layout(
                height=380,
                title="Precision-Recall Curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
                font=dict(family='Inter', size=11),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,249,250,0.5)',
                margin=dict(l=50, r=20, t=40, b=50),
            )
            fig_pr.update_xaxes(gridcolor=COLORS['grid'])
            fig_pr.update_yaxes(gridcolor=COLORS['grid'])
            st.plotly_chart(fig_pr, use_container_width=True)
        
        # ── Train/Test Split Visualization ────────────────────────────────
        st.markdown('<div class="section-header">📅 Train/Test Split</div>', unsafe_allow_html=True)
        
        train_size = len(train_df)
        test_size = len(test_df)
        
        fig_split = go.Figure()
        fig_split.add_trace(go.Bar(
            x=['Training Set (2001-2022)', 'Test Set (2023-2025)'],
            y=[train_size, test_size],
            marker_color=[COLORS['teal'], COLORS['orange']],
            text=[f'{train_size:,} samples', f'{test_size:,} samples'],
            textposition='outside',
            textfont=dict(size=14, family='Inter'),
        ))
        fig_split.update_layout(
            height=300,
            font=dict(family='Inter', size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,249,250,0.5)',
            margin=dict(l=50, r=20, t=20, b=50),
            yaxis_title="Number of Samples",
        )
        fig_split.update_xaxes(gridcolor=COLORS['grid'])
        fig_split.update_yaxes(gridcolor=COLORS['grid'])
        st.plotly_chart(fig_split, use_container_width=True)
        
        st.markdown("""
        > **Why Time-Series Split?** Unlike random splits that would leak future data into training 
        > (a critical flaw in many stock prediction projects), we train strictly on historical data 
        > (2001-2022) and test on unseen future data (2023-2025). This gives an honest estimate of 
        > how the model would perform in real-world trading.
        """)
    
    elif model is None:
        st.warning("⚠️ AI model not found. Please run the Phase 3 notebook first.")
    else:
        st.warning("⚠️ Data not available. Please run the notebooks first.")
