import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment × Alpha Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
  html, body, [class*="css"] { font-family: 'Space Mono', monospace; background-color: #0a0a0f; color: #e8e8f0; }
  .block-container { padding: 2rem 2rem 2rem 2rem; max-width: 1400px; }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #ffffff !important; }
  .stMetric { background: #16161f; border: 1px solid #2a2a3a; border-radius: 10px; padding: 1rem; }
  .stMetric label { color: #6b6b85 !important; font-size: 11px !important; letter-spacing: 1px; }
  .stMetric [data-testid="metric-container"] { color: #fff; }
  div[data-testid="metric-container"] > div { color: #fff !important; }
  .insight-box { border-left: 3px solid #f0b429; padding: 12px 16px; margin-bottom: 12px;
                 background: #1a1a25; border-radius: 0 8px 8px 0; }
  .insight-title { color: #fff; font-size: 13px; font-weight: 700; margin-bottom: 4px; }
  .insight-text  { color: #6b6b85; font-size: 12px; line-height: 1.7; }
  .header-band   { background: #111118; border: 1px solid #2a2a3a; border-radius: 12px;
                   padding: 1.5rem 2rem; margin-bottom: 1.5rem; }
  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
SENT_COLORS = {
    'Extreme Fear' : '#ff4d6d',
    'Fear'         : '#ff8c69',
    'Neutral'      : '#a8b3cf',
    'Greed'        : '#4dd9ac',
    'Extreme Greed': '#00f5d4'
}
PLOT_THEME = dict(
    paper_bgcolor='#0a0a0f',
    plot_bgcolor='#111118',
    font=dict(color='#a8b3cf', family='Space Mono'),
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    fg = pd.read_csv('fear_greed_index.csv')
   ht = pd.read_csv(
        "https://drive.google.com/uc?id=1esf6loq0F85lrqy3jZay4o_Mcx_j8LH9",
        engine='python',
        on_bad_lines='skip'
    )


    fg['date'] = pd.to_datetime(fg['date'])
    ht['trade_date'] = pd.to_datetime(
        ht['Timestamp IST'], format='%d-%m-%Y %H:%M'
    ).dt.normalize()

    merged = ht.merge(
        fg[['date', 'value', 'classification']],
        left_on='trade_date', right_on='date', how='left'
    )

    merged['month'] = merged['trade_date'].dt.to_period('M')
    merged['fg_bucket'] = pd.cut(
        merged['value'], bins=[0, 25, 45, 55, 75, 100],
        labels=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    )
    return fg, merged

fg, merged = load_data()

# ── Metrics ───────────────────────────────────────────────────────────────────
pnl_sent = merged.groupby('classification').agg(
    total_pnl   = ('Closed PnL', 'sum'),
    avg_pnl     = ('Closed PnL', 'mean'),
    trade_count = ('Closed PnL', 'count'),
    win_trades  = ('Closed PnL', lambda x: (x > 0).sum()),
).reset_index()
pnl_sent['win_rate'] = pnl_sent['win_trades'] / pnl_sent['trade_count'] * 100
pnl_sent['classification'] = pd.Categorical(pnl_sent['classification'], categories=SENT_ORDER, ordered=True)
pnl_sent = pnl_sent.sort_values('classification').reset_index(drop=True)

buysell = merged.groupby(['classification', 'Side'])['Closed PnL'].mean().reset_index()
buysell.columns = ['classification', 'Side', 'avg_pnl']
buysell['classification'] = pd.Categorical(buysell['classification'], categories=SENT_ORDER, ordered=True)
buysell = buysell.sort_values('classification').reset_index(drop=True)

monthly = merged.groupby('month')['Closed PnL'].sum().reset_index()
monthly['month_str'] = monthly['month'].astype(str)

top_coins = merged.groupby('Coin')['Closed PnL'].sum().nlargest(5).index.tolist()
coin_sent = (
    merged[merged['Coin'].isin(top_coins)]
    .groupby(['Coin', 'classification'])['Closed PnL']
    .mean().reset_index()
)

trader_pnl = merged.groupby('Account').agg(
    total_pnl   = ('Closed PnL', 'sum'),
    trade_count = ('Closed PnL', 'count'),
    win_trades  = ('Closed PnL', lambda x: (x > 0).sum())
).reset_index()
trader_pnl['win_rate']   = trader_pnl['win_trades'] / trader_pnl['trade_count'] * 100
trader_pnl['short_addr'] = trader_pnl['Account'].str[:6] + '...' + trader_pnl['Account'].str[-4:]

size_sent = merged.groupby('classification').agg(
    avg_size_usd = ('Size USD', 'mean'),
    median_size  = ('Size USD', 'median')
).reset_index()
size_sent['classification'] = pd.Categorical(size_sent['classification'], categories=SENT_ORDER, ordered=True)
size_sent = size_sent.sort_values('classification').reset_index(drop=True)

contrarian_vals = {
    'Ext.Fear BUY'  : merged[(merged['classification']=='Extreme Fear')  & (merged['Side']=='BUY') ]['Closed PnL'].mean(),
    'Fear BUY'      : merged[(merged['classification']=='Fear')           & (merged['Side']=='BUY') ]['Closed PnL'].mean(),
    'Greed SELL'    : merged[(merged['classification']=='Greed')          & (merged['Side']=='SELL')]['Closed PnL'].mean(),
    'Ext.Greed SELL': merged[(merged['classification']=='Extreme Greed')  & (merged['Side']=='SELL')]['Closed PnL'].mean(),
}

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-band">
  <h1 style="font-size:28px;margin:0 0 4px">📊 Sentiment × Alpha</h1>
  <p style="color:#6b6b85;margin:0;font-size:12px;letter-spacing:1px">
    BITCOIN FEAR & GREED INDEX × HYPERLIQUID TRADER INTELLIGENCE DASHBOARD
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total PnL",      f"${merged['Closed PnL'].sum()/1e6:.2f}M")
k2.metric("Total Trades",   f"{len(merged):,}")
k3.metric("Unique Traders", f"{merged['Account'].nunique()}")
k4.metric("Coins Traded",   f"{merged['Coin'].nunique()}")
k5.metric("Best Avg/Trade", f"${pnl_sent['avg_pnl'].max():.1f}",  delta=pnl_sent.loc[pnl_sent['avg_pnl'].idxmax(),'classification'])
k6.metric("Leverage Used",  f"{merged['Crossed'].mean()*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# ROW 1 — PnL by Sentiment + BUY vs SELL
# ════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    fig1 = make_subplots(specs=[[{'secondary_y': True}]])
    sent_labels = pnl_sent['classification'].astype(str).tolist()
    fig1.add_trace(go.Bar(
        x=sent_labels, y=pnl_sent['total_pnl'],
        name='Total PnL ($)',
        marker_color=[SENT_COLORS[s] for s in sent_labels],
        marker_line_width=0,
        text=['${:,.0f}'.format(v) for v in pnl_sent['total_pnl']],
        textposition='outside', textfont=dict(color='#fff', size=10)
    ), secondary_y=False)
    fig1.add_trace(go.Scatter(
        x=sent_labels, y=pnl_sent['win_rate'],
        name='Win Rate (%)', mode='lines+markers',
        marker=dict(size=9, color='#f0b429', symbol='diamond'),
        line=dict(color='#f0b429', width=2, dash='dot')
    ), secondary_y=True)
    fig1.update_layout(
        title=dict(text='Total PnL & Win Rate by Sentiment Zone', font=dict(size=14, color='#fff')),
        height=380, margin=dict(t=50, b=40, l=10, r=10), bargap=0.35,
        legend=dict(bgcolor='#16161f', bordercolor='#2a2a3a', borderwidth=1, font=dict(size=10)),
        **PLOT_THEME
    )
    fig1.update_yaxes(title_text='Total PnL ($)', gridcolor='#1e1e2e', tickformat='$,.0f', secondary_y=False)
    fig1.update_yaxes(title_text='Win Rate (%)', gridcolor='#1e1e2e', range=[30, 52], secondary_y=True)
    fig1.update_xaxes(gridcolor='#1e1e2e')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    buys  = buysell[buysell['Side'] == 'BUY']
    sells = buysell[buysell['Side'] == 'SELL']
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=buys['classification'].astype(str), y=buys['avg_pnl'],
        name='BUY avg PnL', marker_color='#4dd9ac', marker_line_width=0,
        text=['${:.1f}'.format(v) for v in buys['avg_pnl']],
        textposition='outside', textfont=dict(color='#4dd9ac', size=10)
    ))
    fig2.add_trace(go.Bar(
        x=sells['classification'].astype(str), y=sells['avg_pnl'],
        name='SELL avg PnL', marker_color='#f0b429', marker_line_width=0,
        text=['${:.1f}'.format(v) for v in sells['avg_pnl']],
        textposition='outside', textfont=dict(color='#f0b429', size=10)
    ))
    fig2.add_annotation(
        x='Extreme Greed', y=122,
        text='SELL 11x > BUY here',
        showarrow=True, arrowhead=2, arrowcolor='#f0b429',
        font=dict(color='#f0b429', size=10),
        bgcolor='#1a1a25', bordercolor='#f0b429', borderwidth=1
    )
    fig2.update_layout(
        title=dict(text='BUY vs SELL Avg PnL per Trade by Sentiment', font=dict(size=14, color='#fff')),
        barmode='group', height=380, margin=dict(t=50, b=40, l=10, r=10),
        legend=dict(bgcolor='#16161f', bordercolor='#2a2a3a', borderwidth=1, font=dict(size=10)),
        bargap=0.25, bargroupgap=0.1,
        yaxis=dict(title='Avg PnL ($)', gridcolor='#1e1e2e', tickprefix='$'),
        xaxis=dict(gridcolor='#1e1e2e'),
        **PLOT_THEME
    )
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# ROW 2 — Monthly PnL + Contrarian Signals
# ════════════════════════════════════════════════════════════════
col3, col4 = st.columns([3, 2])

with col3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=monthly['month_str'], y=monthly['Closed PnL'],
        mode='lines+markers',
        line=dict(color='#f0b429', width=2.5),
        marker=dict(size=7, color='#f0b429', line=dict(color='#0a0a0f', width=2)),
        fill='tozeroy', fillcolor='rgba(240,180,41,0.08)',
        hovertemplate='%{x}<br>PnL: $%{y:,.0f}<extra></extra>'
    ))
    fig3.add_hline(y=0, line_color='#ff4d6d', line_dash='dot', line_width=1)
    peak_month = monthly.loc[monthly['Closed PnL'].idxmax(), 'month_str']
    peak_val   = monthly['Closed PnL'].max()
    fig3.add_annotation(
        x=peak_month, y=peak_val,
        text=f'Peak: ${peak_val/1e6:.2f}M',
        showarrow=True, arrowhead=2, arrowcolor='#00f5d4',
        font=dict(color='#00f5d4', size=10),
        bgcolor='#1a1a25', bordercolor='#00f5d4', borderwidth=1, ay=-40
    )
    fig3.update_layout(
        title=dict(text='Monthly Total PnL — All Traders', font=dict(size=14, color='#fff')),
        height=370, margin=dict(t=50, b=70, l=10, r=10),
        xaxis=dict(tickangle=45, gridcolor='#1e1e2e'),
        yaxis=dict(title='PnL ($)', gridcolor='#1e1e2e', tickformat='$,.0f'),
        showlegend=False,
        **PLOT_THEME
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    c_labels = list(contrarian_vals.keys())
    c_values = list(contrarian_vals.values())
    c_colors = ['#ff4d6d', '#ff8c69', '#4dd9ac', '#00f5d4']
    fig5 = go.Figure(go.Bar(
        x=c_values, y=c_labels, orientation='h',
        marker_color=c_colors, marker_line_width=0,
        text=['${:.1f}'.format(v) for v in c_values],
        textposition='outside', textfont=dict(color='#fff', size=11),
        hovertemplate='%{y}<br>$%{x:.2f}/trade<extra></extra>'
    ))
    fig5.update_layout(
        title=dict(text='Contrarian Signal Alpha', font=dict(size=14, color='#fff')),
        height=370, margin=dict(t=50, b=40, l=10, r=80),
        xaxis=dict(title='Avg PnL/Trade ($)', gridcolor='#1e1e2e', tickprefix='$'),
        yaxis=dict(autorange='reversed', gridcolor='#1e1e2e'),
        showlegend=False,
        **PLOT_THEME
    )
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# ROW 3 — Coin Heatmap + Trader Bubbles
# ════════════════════════════════════════════════════════════════
col5, col6 = st.columns(2)

with col5:
    pivot = coin_sent.pivot(index='Coin', columns='classification', values='Closed PnL')
    cols_present = [c for c in SENT_ORDER if c in pivot.columns]
    pivot = pivot[cols_present]
    fig4 = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[
            [0.0, '#ff4d6d'], [0.3, '#ff8c69'],
            [0.5, '#1e1e2e'], [0.7, '#4dd9ac'], [1.0, '#00f5d4']
        ],
        text=[[('${:,.0f}'.format(v) if not np.isnan(v) else 'N/A') for v in row] for row in pivot.values],
        texttemplate='%{text}', textfont=dict(size=11, color='#fff'),
        hovertemplate='Coin: %{y}<br>Sentiment: %{x}<br>Avg PnL: $%{z:,.1f}<extra></extra>',
        colorbar=dict(title='Avg PnL', tickprefix='$', tickfont=dict(color='#a8b3cf'))
    ))
    fig4.update_layout(
        title=dict(text='Top 5 Coins × Sentiment Heatmap (Avg PnL/Trade)', font=dict(size=14, color='#fff')),
        height=360, margin=dict(t=50, b=50, l=10, r=10),
        xaxis=dict(side='bottom'), **PLOT_THEME
    )
    st.plotly_chart(fig4, use_container_width=True)

with col6:
    fig6 = go.Figure(go.Scatter(
        x=trader_pnl['win_rate'], y=trader_pnl['total_pnl'],
        mode='markers+text',
        marker=dict(
            size=np.sqrt(trader_pnl['trade_count']) * 1.8,
            color=trader_pnl['total_pnl'],
            colorscale=[[0,'#ff4d6d'],[0.3,'#f0b429'],[0.7,'#4dd9ac'],[1,'#00f5d4']],
            showscale=True,
            colorbar=dict(title='PnL ($)', tickprefix='$', tickfont=dict(color='#a8b3cf')),
            line=dict(color='#0a0a0f', width=1)
        ),
        text=trader_pnl['short_addr'],
        textposition='top center', textfont=dict(size=8, color='#6b6b85'),
        hovertemplate='<b>%{text}</b><br>Win Rate: %{x:.1f}%<br>PnL: $%{y:,.0f}<extra></extra>'
    ))
    fig6.add_hline(y=trader_pnl['total_pnl'].mean(), line_color='#2a2a3a', line_dash='dot', line_width=1)
    fig6.add_vline(x=trader_pnl['win_rate'].mean(),  line_color='#2a2a3a', line_dash='dot', line_width=1)
    fig6.update_layout(
        title=dict(text='Trader Universe — PnL vs Win Rate (bubble = volume)', font=dict(size=14, color='#fff')),
        height=360, margin=dict(t=50, b=40, l=10, r=10),
        xaxis=dict(title='Win Rate (%)', gridcolor='#1e1e2e'),
        yaxis=dict(title='Total PnL ($)', gridcolor='#1e1e2e', tickformat='$,.0f'),
        showlegend=False, **PLOT_THEME
    )
    st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# ROW 4 — Trade Size + FG Distribution
# ════════════════════════════════════════════════════════════════
col7, col8 = st.columns(2)

with col7:
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(
        x=size_sent['classification'].astype(str), y=size_sent['avg_size_usd'],
        name='Avg Size (USD)',
        marker_color=[SENT_COLORS[s] for s in size_sent['classification'].astype(str)],
        marker_line_width=0,
        text=['${:,.0f}'.format(v) for v in size_sent['avg_size_usd']],
        textposition='outside', textfont=dict(color='#fff', size=10)
    ))
    fig8.add_trace(go.Scatter(
        x=size_sent['classification'].astype(str), y=size_sent['median_size'],
        name='Median Size', mode='lines+markers',
        marker=dict(size=8, color='#a855f7'),
        line=dict(color='#a855f7', width=2, dash='dot')
    ))
    fig8.update_layout(
        title=dict(text='Avg & Median Trade Size by Sentiment', font=dict(size=14, color='#fff')),
        height=360, margin=dict(t=50, b=40, l=10, r=10),
        yaxis=dict(title='Trade Size (USD)', gridcolor='#1e1e2e', tickprefix='$'),
        xaxis=dict(gridcolor='#1e1e2e'), bargap=0.35,
        legend=dict(bgcolor='#16161f', bordercolor='#2a2a3a', borderwidth=1, font=dict(size=10)),
        **PLOT_THEME
    )
    st.plotly_chart(fig8, use_container_width=True)

with col8:
    fg_monthly = fg.copy()
    fg_monthly['month'] = fg_monthly['date'].dt.to_period('M').astype(str)
    fg_dist = fg_monthly.groupby(['month', 'classification']).size().reset_index(name='count')
    fig7 = go.Figure()
    for sent in SENT_ORDER:
        d = fg_dist[fg_dist['classification'] == sent]
        fig7.add_trace(go.Bar(
            x=d['month'], y=d['count'],
            name=sent, marker_color=SENT_COLORS[sent], marker_line_width=0
        ))
    fig7.update_layout(
        title=dict(text='Fear & Greed Distribution Over Time', font=dict(size=14, color='#fff')),
        barmode='stack', height=360, margin=dict(t=50, b=70, l=10, r=10),
        xaxis=dict(tickangle=45, gridcolor='#1e1e2e', nticks=15),
        yaxis=dict(title='Days', gridcolor='#1e1e2e'),
        legend=dict(bgcolor='#16161f', bordercolor='#2a2a3a', borderwidth=1, font=dict(size=9)),
        **PLOT_THEME
    )
    st.plotly_chart(fig7, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# INSIGHTS
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h3 style='color:#f0b429;letter-spacing:2px;font-size:15px'>🔍 KEY INSIGHTS</h3>", unsafe_allow_html=True)

insights = [
    ("#f0b429", "Extreme Greed = Highest Alpha ($67.9/trade, 46.5% win rate)",
     "Best zone for overall profitability, but driven by aggressive SELL activity — not buying the market."),
    ("#00f5d4", "Contrarian SELL in Greed outperforms BUY by 11x",
     "Extreme Greed SELL = $114.6/trade vs $10.5 for BUY. When the market is euphoric, smart money sells into it."),
    ("#ff8c69", "Fear-zone BUYs ($64) beat Extreme Fear BUYs ($34) by 88%",
     "Don't catch the falling knife. Buy the early recovery signal (FGI 25–45), not the absolute panic bottom."),
    ("#38bdf8", "ETH & SOL massively outperform BTC in sentiment extremes",
     "ETH earns $237/trade in Fear. SOL earns $285/trade in Greed. BTC lags across all sentiment zones."),
    ("#a855f7", "HYPE token is a natural Fear hedge — anti-correlated with market mood",
     "$840K PnL in Fear vs only $160K in Greed zones. Rare asset that thrives when the market panics."),
]

i_cols = st.columns(len(insights))
for col, (color, title, text) in zip(i_cols, insights):
    col.markdown(f"""
    <div style="border-left:3px solid {color};padding:12px;background:#1a1a25;border-radius:0 8px 8px 0;height:100%">
      <div style="color:#fff;font-size:11px;font-weight:700;margin-bottom:6px">{title}</div>
      <div style="color:#6b6b85;font-size:10px;line-height:1.7">{text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><p style='color:#2a2a3a;font-size:10px;text-align:center'>Data: Hyperliquid Trader History × Bitcoin Fear & Greed Index | May 2023 – May 2025 | 211,224 trades | 32 traders</p>", unsafe_allow_html=True)
