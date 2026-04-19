import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Model Benchmark · AG News",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

/* ── Root & Base ── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #111827;
    --bg-card:       #161d2e;
    --bg-card-hover: #1c2540;
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-violet: #8b5cf6;
    --accent-green:  #10b981;
    --accent-amber:  #f59e0b;
    --accent-rose:   #f43f5e;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --border:        #1e293b;
    --border-bright: #334155;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-bright) !important;
}

/* ── Sidebar Nav Buttons ── */
.stButton > button {
    width: 100%;
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border-bright);
    border-radius: 8px;
    padding: 10px 16px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px;
    font-weight: 500;
    text-align: left;
    transition: all 0.2s ease;
    margin-bottom: 4px;
}
.stButton > button:hover {
    background: var(--bg-card-hover);
    color: var(--text-primary);
    border-color: var(--accent-blue);
    transform: translateX(4px);
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}
.metric-card:hover { border-color: var(--accent-blue); }
.metric-card .label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
}
.metric-card .delta {
    font-size: 12px;
    color: var(--accent-green);
    font-weight: 500;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-bright);
}
.section-header .title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
}
.section-header .badge {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet));
    color: white;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
}

/* ── Page Title ── */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    font-weight: 700;
    background: linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 4px;
}
.page-subtitle {
    color: var(--text-secondary);
    font-size: 15px;
    font-weight: 400;
    margin-bottom: 32px;
}

/* ── Model Chips ── */
.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* ── Info Box ── */
.info-box {
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
    color: var(--text-secondary);
    font-size: 14px;
    line-height: 1.6;
}

/* ── Table Styling ── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Sidebar Logo ── */
.sidebar-logo {
    text-align: center;
    padding: 20px 0 24px 0;
    border-bottom: 1px solid var(--border-bright);
    margin-bottom: 20px;
}
.sidebar-logo .logo-icon {
    font-size: 36px;
    display: block;
    margin-bottom: 6px;
}
.sidebar-logo .logo-title {
    font-family: 'Playfair Display', serif;
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
}
.sidebar-logo .logo-sub {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
}

/* ── Winner Card ── */
.winner-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(59,130,246,0.15));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}

/* ── Hide Streamlit defaults ── */
#MainMenu, footer, { visibility: hidden; }
.stAppDeployButton { display: none; } 
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA — Hasil evaluasi model
# ─────────────────────────────────────────────
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
MODEL_COLORS = {
    "Naive Bayes":  "#06b6d4",
    "SVM":          "#f59e0b",
    "BERT":         "#8b5cf6",
}
MODEL_ICONS = {
    "Naive Bayes": "📊",
    "SVM":         "⚡",
    "BERT":        "🧠",
}

# Overall metrics
METRICS = {
    "Naive Bayes": {
        "accuracy":  0.8917,
        "precision": 0.8912,
        "recall":    0.8917,
        "f1":        0.8912,
    },
    "SVM": {
        "accuracy":  0.9130,
        "precision": 0.9130,
        "recall":    0.9130,
        "f1":        0.9129,
    },
    "BERT": {
        "accuracy":  0.9450,
        "precision": 0.9452,
        "recall":    0.9450,
        "f1":        0.9450,
    },
}

# Per-class F1 scores
CLASS_F1 = {
    "Naive Bayes": {
        "World":    0.90,
        "Sports":   0.96,
        "Business": 0.85,
        "Sci/Tech": 0.86,
    },
    "SVM": {
        "World":    0.92,
        "Sports":   0.97,
        "Business": 0.88,
        "Sci/Tech": 0.89,
    },
    "BERT": {
        "World":    0.96,
        "Sports":   0.99,
        "Business": 0.92,
        "Sci/Tech": 0.92,
    },
}

# Confusion matrices (rows=actual, cols=predicted)
CM = {
    "Naive Bayes": np.array([
        [1690,  69,  90,  51],
        [  22, 1857,   8,  13],
        [  75,  24, 1582, 219],
        [  75,  32,  145, 1648],
    ]),
    "SVM": np.array([
        [1713,  57,   80,  50],
        [   14, 1865,   12,  9],
        [  51,  18, 1670, 161],
        [  55,  17,  137, 1691],
    ]),
    "BERT": np.array([
        [1812,  10,   42,  36],
        [   12, 1875,    6,   7],
        [  34,   7, 1723,  136],
        [  26,   9,   93, 1772],
    ]),
}

# Training info
TRAINING_INFO = {
    "Naive Bayes": {
        "algorithm":    "Multinomial Naive Bayes",
        "vectorizer":   "TF-IDF (max_features=10,000)",
        "params":       "alpha=1.0 (Laplace smoothing)",
        "train_time":   "~5 seconds",
        "inference":    "< 1ms / sample",
        "train_size":   "120,000 samples",
        "test_size":    "7,600 samples",
    },
    "SVM": {
        "algorithm":    "Linear SVM (LinearSVC)",
        "vectorizer":   "TF-IDF (max_features=10,000)",
        "params":       "C=1.0, max_iter=1,000",
        "train_time":   "~15 seconds",
        "inference":    "< 1ms / sample",
        "train_size":   "120,000 samples",
        "test_size":    "7,600 samples",
    },
    "BERT": {
        "algorithm":    "BERT-base-uncased (Fine-tuned)",
        "vectorizer":   "WordPiece Tokenizer (max_len=128)",
        "params":       "lr=2e-5, epochs=2, batch=16",
        "train_time":   "~60 minutes (GPU)",
        "inference":    "~12ms / sample",
        "train_size":   "120,000 samples",
        "test_size":    "7,600 samples",
    },
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#94a3b8"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)

def axis_style(grid=True):
    d = dict(
        gridcolor="#1e293b" if grid else "rgba(0,0,0,0)",
        zerolinecolor="#334155",
        linecolor="#334155",
        tickfont=dict(size=11, color="#64748b"),
    )
    return d

def fmt(v): return f"{v*100:.2f}%"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🧠</span>
        <div class="logo-title">NLP Benchmark</div>
        <div class="logo-sub">AG News · Text Classification</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")

    pages = {
        "🏠  Overview":           "overview",
        "📊  Metrics Comparison":  "metrics",
        "🔢  Confusion Matrix":    "confusion",
        "📈  Per-Class Analysis":  "perclass",
        "🔍  Model Details":       "details",
        "ℹ️  About":               "about",
    }

    if "page" not in st.session_state:
        st.session_state.page = "overview"

    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}"):
            st.session_state.page = key

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#475569; line-height:1.6;">
        <strong style="color:#64748b;">Dataset</strong><br>
        AG News Corpus<br>
        120K train · 7.6K test<br>
        4 categories<br><br>
        <strong style="color:#64748b;">Project</strong><br>
        NLP Final Project<br>
        Text Classification
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════
if page == "overview":
    st.markdown('<div class="page-title">Model Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AG News Text Classification · Naive Bayes vs SVM vs BERT</div>', unsafe_allow_html=True)

    # ── Best model callout ──
    col_w, col_s, col_b = st.columns(3)
    with col_w:
        st.markdown("""
        <div class="winner-card">
            <div style="font-size:32px; margin-bottom:8px;">🏆</div>
            <div style="font-size:11px; letter-spacing:1px; text-transform:uppercase; color:#8b5cf6; font-weight:600; margin-bottom:4px;">Best Overall</div>
            <div style="font-size:22px; font-weight:700; color:#f1f5f9;">BERT</div>
            <div style="font-size:28px; font-weight:700; color:#8b5cf6; margin-top:4px;">94.50%</div>
            <div style="font-size:12px; color:#64748b; margin-top:2px;">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col_s:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:24px; margin-bottom:6px;">⚡</div>
            <div style="font-size:11px; letter-spacing:1px; text-transform:uppercase; color:#f59e0b; font-weight:600; margin-bottom:4px;">Runner-up</div>
            <div style="font-size:20px; font-weight:700; color:#f1f5f9;">SVM</div>
            <div style="font-size:26px; font-weight:700; color:#f59e0b; margin-top:4px;">91.30%</div>
            <div style="font-size:12px; color:#64748b; margin-top:2px;">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:24px; margin-bottom:6px;">📊</div>
            <div style="font-size:11px; letter-spacing:1px; text-transform:uppercase; color:#06b6d4; font-weight:600; margin-bottom:4px;">Baseline</div>
            <div style="font-size:20px; font-weight:700; color:#f1f5f9;">Naive Bayes</div>
            <div style="font-size:26px; font-weight:700; color:#06b6d4; margin-top:4px;">89.17%</div>
            <div style="font-size:12px; color:#64748b; margin-top:2px;">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar chart ──
    st.markdown('<div class="section-header"><div class="title">Holistic Performance Radar</div><div class="badge">4 Metrics</div></div>', unsafe_allow_html=True)

    metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig_radar = go.Figure()
    for model, color in MODEL_COLORS.items():
        vals = [METRICS[model][k] for k in ["accuracy", "precision", "recall", "f1"]]
        vals += [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=metrics_labels + [metrics_labels[0]],
            fill='toself',
            name=model,
            line_color=color,
            # fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in color else color + "1a",
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
            line_width=2,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0.88, 0.96], tickformat=".0%", gridcolor="#1e293b", linecolor="#334155"),
            angularaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
        ),
        **PLOTLY_LAYOUT,
        height=380,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Summary table ──
    st.markdown('<div class="section-header"><div class="title">Summary Table</div><div class="badge">All Models</div></div>', unsafe_allow_html=True)

    rows = []
    for model in ["Naive Bayes", "SVM", "BERT"]:
        m = METRICS[model]
        rows.append({
            "Model": f"{MODEL_ICONS[model]} {model}",
            "Accuracy":  fmt(m["accuracy"]),
            "Precision": fmt(m["precision"]),
            "Recall":    fmt(m["recall"]),
            "F1-Score":  fmt(m["f1"]),
            "Δ vs Baseline": f"+{(m['accuracy']-METRICS['Naive Bayes']['accuracy'])*100:.2f}%" if model != "Naive Bayes" else "—",
        })
    df_summary = pd.DataFrame(rows)
    st.dataframe(df_summary.set_index("Model"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        💡 <strong>Key Takeaway:</strong> BERT outperforms classical ML models by a significant margin (+5.33% over Naive Bayes),
        demonstrating the power of pre-trained transformer representations. However, SVM offers an excellent speed–accuracy
        trade-off, achieving 91.30% accuracy in just seconds of training.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2 — METRICS COMPARISON
# ══════════════════════════════════════════════
elif page == "metrics":
    st.markdown('<div class="page-title">Metrics Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Side-by-side evaluation across all four metrics</div>', unsafe_allow_html=True)

    # ── Grouped bar ──
    st.markdown('<div class="section-header"><div class="title">Grouped Bar Chart</div><div class="badge">4 Metrics</div></div>', unsafe_allow_html=True)

    metrics_keys   = ["accuracy", "precision", "recall", "f1"]
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig_bar = go.Figure()
    for model, color in MODEL_COLORS.items():
        fig_bar.add_trace(go.Bar(
            name=model,
            x=metrics_labels,
            y=[METRICS[model][k] * 100 for k in metrics_keys],
            marker_color=color,
            marker_line_width=0,
            text=[f"{METRICS[model][k]*100:.2f}%" for k in metrics_keys],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
        ))

    fig_bar.update_layout(
        barmode="group",
        yaxis=dict(range=[87, 97], ticksuffix="%", **axis_style()),
        xaxis=axis_style(grid=False),
        **PLOTLY_LAYOUT,
        height=420,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Individual metric cards ──
    st.markdown('<div class="section-header"><div class="title">Metric Deep-Dive</div></div>', unsafe_allow_html=True)

    selected_metric = st.selectbox(
        "Select metric to explore",
        options=["Accuracy", "Precision", "Recall", "F1-Score"],
        index=3,
    )
    mk = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1-Score": "f1"}[selected_metric]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    models = ["Naive Bayes", "SVM", "BERT"]
    for i, model in enumerate(models):
        val = METRICS[model][mk]
        diff = val - METRICS["Naive Bayes"][mk]
        delta_str = f"+{diff*100:.2f}% vs Baseline" if model != "Naive Bayes" else "Baseline model"
        color = MODEL_COLORS[model]
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{color};"></div>
                <div class="label">{MODEL_ICONS[model]} {model}</div>
                <div class="value">{val*100:.2f}<span style="font-size:16px;color:#64748b;">%</span></div>
                <div class="delta" style="color:{'#10b981' if diff > 0 else '#94a3b8'};">{delta_str}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Line / progress view ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><div class="title">Model Score Progression</div></div>', unsafe_allow_html=True)

    fig_line = go.Figure()
    for mk_k, mk_l in zip(metrics_keys, metrics_labels):
        for model, color in MODEL_COLORS.items():
            fig_line.add_trace(go.Scatter(
                x=[model],
                y=[METRICS[model][mk_k] * 100],
                mode="markers",
                marker=dict(size=12, color=color),
                showlegend=False,
            ))

    fig_dot = go.Figure()
    x_vals = metrics_labels
    for model, color in MODEL_COLORS.items():
        y_vals = [METRICS[model][mk_k] * 100 for mk_k in metrics_keys]
        fig_dot.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name=model,
            line=dict(color=color, width=2),
            marker=dict(size=10, color=color, line=dict(width=2, color="#0a0e1a")),
        ))
    fig_dot.update_layout(
        yaxis=dict(range=[88, 97], ticksuffix="%", **axis_style()),
        xaxis=axis_style(grid=False),
        **PLOTLY_LAYOUT,
        height=350,
    )
    st.plotly_chart(fig_dot, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 3 — CONFUSION MATRIX
# ══════════════════════════════════════════════
elif page == "confusion":
    st.markdown('<div class="page-title">Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Predicted vs Actual class distributions for each model</div>', unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Select model",
        options=["Naive Bayes", "SVM", "BERT"],
        index=2,
    )

    cm = CM[selected_model]
    color = MODEL_COLORS[selected_model]

    # Normalize
    view_mode = st.radio("View mode", ["Raw Counts", "Normalized (%)"], horizontal=True)

    if view_mode == "Normalized (%)":
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        text_fmt = [[f"{v:.1f}%" for v in row] for row in cm_display]
        zmax = 100
        colorscale_label = "Percentage"
    else:
        cm_display = cm.astype(float)
        text_fmt = [[f"{int(v):,}" for v in row] for row in cm_display]
        zmax = cm.max()
        colorscale_label = "Count"

    fig_cm = ff.create_annotated_heatmap(
        z=cm_display,
        x=[f"Pred: {c}" for c in CLASSES],
        y=[f"True: {c}" for c in CLASSES],
        annotation_text=text_fmt,
        colorscale=[
            [0.0,  "#0a0e1a"],
            [0.25, "#1a2436"],
            [0.5,  "#1e3a6e" if color == "#3b82f6" else "#2d1f4e" if color == "#8b5cf6" else "#1f3a2d"],
            [1.0,  color],
        ],
        showscale=True,
        zmax=zmax,
        zmin=0,
    )

    fig_cm.update_layout(**PLOTLY_LAYOUT)
    
    fig_cm.update_layout(
        height=480,
        margin=dict(l=100, r=10, t=40, b=100),
        xaxis=dict(side="bottom", tickfont=dict(size=12, color="#94a3b8")),
        yaxis=dict(tickfont=dict(size=12, color="#94a3b8")),
    )

    for ann in fig_cm.layout.annotations:
        ann.font.size = 14
        ann.font.color = "#f1f5f9"

    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Derived stats ──
    st.markdown('<div class="section-header"><div class="title">Per-Class Statistics</div></div>', unsafe_allow_html=True)

    rows = []
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        rows.append({
            "Class": cls,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%",
        })

    df_stats = pd.DataFrame(rows).set_index("Class")
    st.dataframe(df_stats, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 4 — PER-CLASS ANALYSIS
# ══════════════════════════════════════════════
elif page == "perclass":
    st.markdown('<div class="page-title">Per-Class Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">F1-Score breakdown across all four AG News categories</div>', unsafe_allow_html=True)

    # ── Grouped bar per class ──
    st.markdown('<div class="section-header"><div class="title">F1-Score by Category</div><div class="badge">Per Class</div></div>', unsafe_allow_html=True)

    fig_cls = go.Figure()
    for model, color in MODEL_COLORS.items():
        fig_cls.add_trace(go.Bar(
            name=model,
            x=CLASSES,
            y=[CLASS_F1[model][c] * 100 for c in CLASSES],
            marker_color=color,
            text=[f"{CLASS_F1[model][c]*100:.2f}%" for c in CLASSES],
            textposition="outside",
            textfont=dict(size=11),
        ))

    fig_cls.update_layout(
        barmode="group",
        yaxis=dict(range=[83, 102], ticksuffix="%", **axis_style()),
        xaxis=axis_style(grid=False),
        **PLOTLY_LAYOUT,
        height=420,
    )
    st.plotly_chart(fig_cls, use_container_width=True)

    # ── Heatmap: models × classes ──
    st.markdown('<div class="section-header"><div class="title">F1-Score Heatmap</div></div>', unsafe_allow_html=True)

    z_data = [[CLASS_F1[m][c] * 100 for c in CLASSES] for m in ["Naive Bayes", "SVM", "BERT"]]
    text_data = [[f"{v:.2f}%" for v in row] for row in z_data]

    fig_heat = ff.create_annotated_heatmap(
        z=z_data,
        x=CLASSES,
        y=["Naive Bayes", "SVM", "BERT"],
        annotation_text=text_data,
        colorscale=[
            [0.0, "#1a1535"],
            [0.5, "#312e81"],
            [1.0, "#8b5cf6"],
        ],
        showscale=True,
    )
    # fig_heat.update_layout(
    #     **PLOTLY_LAYOUT,
    #     height=300,
    #     margin=dict(l=120, r=10, t=20, b=10),
    # )

    fig_heat.update_layout(**PLOTLY_LAYOUT)

    fig_heat.update_layout(
        height=300,
        margin=dict(l=120, r=10, t=20, b=10),
    )
    for ann in fig_heat.layout.annotations:
        ann.font.size = 14
        ann.font.color = "#f1f5f9"

    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Class-specific view ──
    st.markdown('<div class="section-header"><div class="title">Class Deep-Dive</div></div>', unsafe_allow_html=True)

    cls_select = st.selectbox("Select category", CLASSES, index=2)
    col1, col2, col3 = st.columns(3)
    for i, (model, col) in enumerate(zip(["Naive Bayes", "SVM", "BERT"], [col1, col2, col3])):
        val = CLASS_F1[model][cls_select]
        color = MODEL_COLORS[model]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{color};"></div>
                <div class="label">{MODEL_ICONS[model]} {model}</div>
                <div style="font-size:13px;color:#64748b;margin-bottom:8px;">F1-Score · {cls_select}</div>
                <div class="value">{val*100:.2f}<span style="font-size:14px;color:#64748b;">%</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Observation ──
    hardest = min(CLASSES, key=lambda c: CLASS_F1["Naive Bayes"][c])
    easiest = max(CLASSES, key=lambda c: CLASS_F1["BERT"][c])
    st.markdown(f"""
    <div class="info-box">
        🔍 <strong>Observations:</strong><br>
        • <strong>{easiest}</strong> is the easiest category to classify across all models (highest F1 with BERT: {CLASS_F1['BERT'][easiest]*100:.2f}%).<br>
        • <strong>{hardest}</strong> is the most challenging for the baseline (Naive Bayes F1: {CLASS_F1['Naive Bayes'][hardest]*100:.2f}%), 
          but BERT significantly closes the gap ({CLASS_F1['BERT'][hardest]*100:.2f}%).<br>
        • BERT shows the most balanced performance across all categories, suggesting stronger generalization.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 5 — MODEL DETAILS
# ══════════════════════════════════════════════
elif page == "details":
    st.markdown('<div class="page-title">Model Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Architecture, hyperparameters, and training configuration</div>', unsafe_allow_html=True)

    tabs = st.tabs(["📊 Naive Bayes", "⚡ SVM", "🧠 BERT"])

    for tab, model in zip(tabs, ["Naive Bayes", "SVM", "BERT"]):
        with tab:
            info = TRAINING_INFO[model]
            color = MODEL_COLORS[model]

            col_info, col_metric = st.columns([1, 1])

            with col_info:
                st.markdown(f"""
                <div class="metric-card" style="border-top-color:{color};">
                    <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{color};"></div>
                    <div style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:16px;">{MODEL_ICONS[model]} {model}</div>
                    <table style="width:100%;border-collapse:collapse;font-size:13px;">
                        <tr><td style="color:#475569;padding:6px 0;width:40%;">Algorithm</td><td style="color:#cbd5e1;font-weight:500;">{info['algorithm']}</td></tr>
                        <tr><td style="color:#475569;padding:6px 0;">Vectorizer</td><td style="color:#cbd5e1;font-weight:500;">{info['vectorizer']}</td></tr>
                        <tr><td style="color:#475569;padding:6px 0;">Parameters</td><td style="color:#cbd5e1;font-weight:500;">{info['params']}</td></tr>
                        <tr><td style="color:#475569;padding:6px 0;">Train Time</td><td style="color:#cbd5e1;font-weight:500;">{info['train_time']}</td></tr>
                        <tr><td style="color:#475569;padding:6px 0;">Inference</td><td style="color:#cbd5e1;font-weight:500;">{info['inference']}</td></tr>
                        <tr><td style="color:#475569;padding:6px 0;">Train Set</td><td style="color:#cbd5e1;font-weight:500;">{info['train_size']}</td></tr>
                        <tr><td style="color:#475569;padding:6px 0;">Test Set</td><td style="color:#cbd5e1;font-weight:500;">{info['test_size']}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with col_metric:
                m = METRICS[model]
                for label, key in [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1-Score", "f1")]:
                    val = m[key]
                    bar_pct = int((val - 0.85) / (0.97 - 0.85) * 100)
                    st.markdown(f"""
                    <div style="margin-bottom:14px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-size:12px;color:#64748b;font-weight:500;">{label}</span>
                            <span style="font-size:13px;color:#f1f5f9;font-weight:700;">{val*100:.2f}%</span>
                        </div>
                        <div style="background:#1e293b;border-radius:4px;height:6px;overflow:hidden;">
                            <div style="width:{bar_pct}%;height:100%;background:{color};border-radius:4px;transition:width 0.6s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Speed vs Accuracy scatter ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><div class="title">Speed vs Accuracy Trade-off</div></div>', unsafe_allow_html=True)

    speed_data = {
        "Naive Bayes": {"accuracy": 90.11, "train_sec": 3,    "size": 20},
        "SVM":         {"accuracy": 92.38, "train_sec": 45,   "size": 30},
        "BERT":        {"accuracy": 94.61, "train_sec": 2700, "size": 50},
    }

    fig_scatter = go.Figure()
    for model, d in speed_data.items():
        color = MODEL_COLORS[model]
        fig_scatter.add_trace(go.Scatter(
            x=[d["train_sec"]],
            y=[d["accuracy"]],
            mode="markers+text",
            name=model,
            text=[f"  {model}"],
            textposition="middle right",
            textfont=dict(size=13, color=color),
            marker=dict(size=d["size"], color=color, opacity=0.85, line=dict(width=2, color="#0a0e1a")),
        ))
    fig_scatter.update_layout(
        xaxis=dict(title="Training Time (seconds)", type="log", **axis_style()),
        yaxis=dict(title="Accuracy (%)", range=[88, 96], **axis_style()),
        **PLOTLY_LAYOUT,
        height=380,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        ⏱️ <strong>Speed vs. Accuracy:</strong> The log scale reveals a dramatic training time difference.
        Naive Bayes (~5s) and SVM (~15s) are orders of magnitude faster than BERT (~60 min), yet BERT delivers
        the highest accuracy. For production deployments, the right choice depends on latency constraints,
        available GPU resources, and accuracy requirements.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════
elif page == "about":
    st.markdown('<div class="page-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">NLP Final Project — Text Classification on AG News Dataset</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:16px;">📚 Project Overview</div>
            <p style="color:#94a3b8;line-height:1.7;font-size:14px;">
                This project benchmarks three classical and modern NLP models on the AG News corpus,
                a widely-used benchmark for text classification tasks. The objective is to evaluate
                model accuracy, generalization, and efficiency across four news categories.
            </p>
            <p style="color:#94a3b8;line-height:1.7;font-size:14px;">
                The pipeline covers full pre-processing (tokenization, TF-IDF / WordPiece),
                model training, hyperparameter tuning, and evaluation with standard classification metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <div style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:16px;">🗂️ Dataset — AG News</div>
            <table style="width:100%;font-size:13px;border-collapse:collapse;">
                <tr><td style="color:#475569;padding:6px 0;width:40%;">Source</td><td style="color:#cbd5e1;">Academic Torrents / Hugging Face</td></tr>
                <tr><td style="color:#475569;padding:6px 0;">Train Set</td><td style="color:#cbd5e1;">120,000 samples (30K × 4 classes)</td></tr>
                <tr><td style="color:#475569;padding:6px 0;">Test Set</td><td style="color:#cbd5e1;">7,600 samples (1,900 × 4 classes)</td></tr>
                <tr><td style="color:#475569;padding:6px 0;">Classes</td><td style="color:#cbd5e1;">World · Sports · Business · Sci/Tech</td></tr>
                <tr><td style="color:#475569;padding:6px 0;">Balance</td><td style="color:#cbd5e1;">Perfectly balanced (25% each)</td></tr>
                <tr><td style="color:#475569;padding:6px 0;">Input</td><td style="color:#cbd5e1;">Title + Description concatenated</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:16px;">🔬 Models</div>
        """, unsafe_allow_html=True)

        for model, color in MODEL_COLORS.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;padding:10px 12px;background:#0a0e1a;border-radius:8px;border-left:3px solid {color};">
                <span style="font-size:20px;">{MODEL_ICONS[model]}</span>
                <div>
                    <div style="color:#f1f5f9;font-weight:600;font-size:13px;">{model}</div>
                    <div style="color:#475569;font-size:11px;">{TRAINING_INFO[model]['algorithm']}</div>
                </div>
                <div style="margin-left:auto;font-size:18px;font-weight:700;color:{color};">{METRICS[model]['accuracy']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <div style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:16px;">📏 Evaluation Metrics</div>
            <ul style="color:#94a3b8;font-size:13px;line-height:2;padding-left:16px;margin:0;">
                <li><strong style="color:#f1f5f9;">Accuracy</strong> — Overall correct predictions</li>
                <li><strong style="color:#f1f5f9;">Precision</strong> — Exactness of positive predictions</li>
                <li><strong style="color:#f1f5f9;">Recall</strong> — Coverage of actual positives</li>
                <li><strong style="color:#f1f5f9;">F1-Score</strong> — Harmonic mean of P & R</li>
                <li><strong style="color:#f1f5f9;">Confusion Matrix</strong> — Full prediction breakdown</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#334155;font-size:12px;padding:20px 0;border-top:1px solid #1e293b;">
        NLP Final Project · Text Classification · AG News Dataset · Naive Bayes · SVM · BERT
    </div>
    """, unsafe_allow_html=True)
