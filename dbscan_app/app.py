import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import (
    make_blobs, make_moons, make_circles, make_classification
)
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

# ---- Page Config ----
st.set_page_config(
    page_title="DBSCAN Explorer",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0f0500 0%, #1a0a00 50%, #100400 100%);
    color: #fff0e6;
}
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    background: linear-gradient(135deg, rgba(230,80,0,0.38), rgba(160,40,0,0.28));
    border: 1px solid rgba(255,140,0,0.28);
    border-radius: 22px;
    padding: 34px 32px 26px;
    text-align: center;
    margin-bottom: 26px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 44px rgba(200,60,0,0.22);
}
.hero h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #fb923c, #fbbf24, #f97316, #fb923c);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px !important;
    animation: shimmer 4s linear infinite;
}
.hero p { color: rgba(255,220,170,0.8); font-size: 0.95rem; margin: 0; }
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,140,0,0.18) !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,200,130,0.7) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #92400e, #c2410c) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(194,65,12,0.5) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* Glass */
.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,140,0,0.14);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.sec-title {
    font-size: 0.88rem; font-weight: 700; color: #fb923c;
    text-transform: uppercase; letter-spacing: 1.6px; margin-bottom: 14px;
}

/* Metrics */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; }
.metric-chip {
    flex: 1; min-width: 110px;
    background: rgba(251,146,60,0.1);
    border: 1px solid rgba(251,146,60,0.22);
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
}
.chip-label { font-size: 0.7rem; color: rgba(255,190,100,0.6); text-transform: uppercase; letter-spacing: 1px; }
.chip-val   { font-size: 1.3rem; font-weight: 800; color: #fbbf24; margin-top: 4px; }
.chip-bad   { color: #f87171 !important; }

/* Info box */
.info-box {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.2);
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.86rem;
    color: rgba(180,230,255,0.85);
    margin-bottom: 16px;
    line-height: 1.55;
}
.warn-box {
    background: rgba(251,146,60,0.08);
    border: 1px solid rgba(251,146,60,0.25);
    border-left: 4px solid #fb923c;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.86rem;
    color: rgba(255,210,150,0.85);
    margin-bottom: 16px;
    line-height: 1.55;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c2d12, #c2410c) !important;
    color: white !important;
    font-size: 1rem !important; font-weight: 700 !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.8em 2em !important; width: 100%;
    box-shadow: 0 6px 20px rgba(194,65,12,0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

/* Sliders, selects */
.stSlider label, .stSelectbox label, .stRadio label {
    color: #fcd34d !important; font-weight: 600 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0300, #060200) !important;
    border-right: 1px solid rgba(255,140,0,0.12) !important;
}
[data-testid="stSidebar"] * { color: #fcd34d !important; }
[data-testid="stSidebar"] h3 { color: #fb923c !important; }
.sb-card {
    background: rgba(251,146,60,0.07);
    border: 1px solid rgba(251,146,60,0.18);
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 0.83rem;
    line-height: 1.65;
    margin-bottom: 12px;
}
.sb-step {
    display: flex; gap: 10px; align-items: flex-start; margin-bottom: 10px;
}
.sb-num {
    background: linear-gradient(135deg, #92400e, #c2410c);
    color: white; width: 24px; height: 24px;
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 0.78rem; font-weight: 700;
    flex-shrink: 0; margin-top: 1px;
}
.sb-text { font-size: 0.82rem; color: rgba(255,210,140,0.8); line-height: 1.5; }

hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 16px 0 !important; }
.footer {
    text-align: center; padding: 16px;
    color: rgba(255,160,60,0.3); font-size: 0.78rem;
    margin-top: 34px; border-top: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🟠 DBSCAN")
    st.markdown("""<div class="sb-card">
    <b>DBSCAN</b> (Density-Based Spatial Clustering of Applications with Noise) 
    finds clusters of arbitrary shape without needing to specify K upfront.
    </div>""", unsafe_allow_html=True)

    st.markdown("### ⚙️ DBSCAN Parameters")
    eps      = st.slider("Epsilon (ε) — Neighbourhood Radius", 0.05, 3.0, 0.5, 0.05)
    min_samp = st.slider("Min Samples — Core Point Threshold", 2, 30, 5)

    st.markdown("---")
    st.markdown("### 🧪 Dataset")
    dataset_choice = st.selectbox("Choose Dataset Shape", [
        "Blobs (Convex)",
        "Moons (Non-convex)",
        "Circles (Ring shape)",
        "Anisotropic Blobs",
        "Noisy Scatter"
    ])
    n_samples = st.slider("Number of Points", 100, 800, 300, 50)
    noise_level = st.slider("Noise / Scatter", 0.0, 0.5, 0.1, 0.02)
    rand_seed = st.slider("Random Seed", 0, 99, 42)

    st.markdown("---")
    st.markdown("### 📖 How It Works")
    st.markdown("""
    <div class="sb-step"><div class="sb-num">1</div><div class="sb-text">For each point, find neighbours within radius <b>ε</b></div></div>
    <div class="sb-step"><div class="sb-num">2</div><div class="sb-text">If ≥ MinSamples neighbours → <b>Core Point</b></div></div>
    <div class="sb-step"><div class="sb-num">3</div><div class="sb-text">Connect core points within ε into one cluster</div></div>
    <div class="sb-step"><div class="sb-num">4</div><div class="sb-text">Border points (near core) join the cluster</div></div>
    <div class="sb-step"><div class="sb-num">5</div><div class="sb-text">Remaining points = <b style="color:#f87171;">Noise</b> (label −1)</div></div>
    """, unsafe_allow_html=True)

# ============================================================
# GENERATE DATA
# ============================================================
np.random.seed(rand_seed)
if dataset_choice == "Blobs (Convex)":
    X_raw, y_true = make_blobs(n_samples=n_samples, centers=4, cluster_std=noise_level+0.5, random_state=rand_seed)
elif dataset_choice == "Moons (Non-convex)":
    X_raw, y_true = make_moons(n_samples=n_samples, noise=noise_level + 0.05, random_state=rand_seed)
elif dataset_choice == "Circles (Ring shape)":
    X_raw, y_true = make_circles(n_samples=n_samples, noise=noise_level + 0.04, factor=0.4, random_state=rand_seed)
elif dataset_choice == "Anisotropic Blobs":
    X_raw, _ = make_blobs(n_samples=n_samples, centers=3, random_state=rand_seed)
    transform = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X_raw = X_raw @ transform + np.random.normal(0, noise_level, X_raw.shape)
    y_true = _
else:
    X_raw = np.random.randn(n_samples, 2) * 2
    y_true = np.zeros(n_samples, dtype=int)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ============================================================
# RUN DBSCAN
# ============================================================
db = DBSCAN(eps=eps, min_samples=min_samp)
db_labels = db.fit_predict(X)

n_clusters  = len(set(db_labels)) - (1 if -1 in db_labels else 0)
noise_pts   = int(np.sum(db_labels == -1))
core_pts    = len(db.core_sample_indices_)
sil_score   = silhouette_score(X, db_labels) if n_clusters > 1 and noise_pts < n_samples - 2 else None

PALETTE = ['#fb923c','#f472b6','#34d399','#60a5fa','#fbbf24','#a78bfa','#4ade80','#e879f9','#38bdf8','#f87171']

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🟠 DBSCAN Clustering Explorer</h1>
    <p>Density-Based Spatial Clustering of Applications with Noise — No K required!</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# METRICS ROW
# ============================================================
sil_txt = f"{sil_score:.3f}" if sil_score is not None else "N/A"
sil_cls = "" if sil_score is not None else "chip-bad"

st.markdown(f"""<div class="metric-row" style="margin-bottom:22px;">
    <div class="metric-chip">
        <div class="chip-label">Clusters Found</div>
        <div class="chip-val">{n_clusters}</div>
    </div>
    <div class="metric-chip">
        <div class="chip-label">Noise Points</div>
        <div class="chip-val {'chip-bad' if noise_pts > 0 else ''}">{noise_pts}</div>
    </div>
    <div class="metric-chip">
        <div class="chip-label">Core Points</div>
        <div class="chip-val">{core_pts}</div>
    </div>
    <div class="metric-chip">
        <div class="chip-label">Silhouette Score</div>
        <div class="chip-val {sil_cls}">{sil_txt}</div>
    </div>
    <div class="metric-chip">
        <div class="chip-label">ε · MinSamples</div>
        <div class="chip-val">{eps} · {min_samp}</div>
    </div>
</div>""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Cluster Result", "🔎 Point Types", "📈 ε Tuning (k-NN)", "📋 Analysis"
])


def dark_fig(w=13, h=5.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0a0300')
    ax.set_facecolor('#120500')
    ax.tick_params(colors='#8b5e3c')
    for s in ax.spines.values(): s.set_color('#2d1000')
    return fig, ax

def dark_figs(ncols, w=14, h=5.5):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    fig.patch.set_facecolor('#0a0300')
    for ax in axes:
        ax.set_facecolor('#120500')
        ax.tick_params(colors='#8b5e3c')
        for s in ax.spines.values(): s.set_color('#2d1000')
    return fig, axes


# ──── TAB 1: Cluster Result ────────────────────────────────
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">📊 DBSCAN Result — {dataset_choice}</div>', unsafe_allow_html=True)

    if n_clusters == 0:
        st.markdown('<div class="warn-box">⚠️ No clusters formed. All points are noise. Try <b>increasing ε</b> or <b>decreasing Min Samples</b>.</div>', unsafe_allow_html=True)
    elif noise_pts > n_samples * 0.4:
        st.markdown('<div class="warn-box">⚠️ More than 40% points are noise. Try <b>increasing ε</b> or <b>decreasing Min Samples</b>.</div>', unsafe_allow_html=True)

    fig, axes = dark_figs(2, 14, 5.5)

    # Left: DBSCAN labels
    unique_labels = np.unique(db_labels)
    for i, lbl in enumerate(unique_labels):
        m = db_labels == lbl
        if lbl == -1:
            axes[0].scatter(X[m, 0], X[m, 1], s=14, color='#444455', alpha=0.6,
                           marker='x', linewidths=0.8, label='Noise', zorder=2)
        else:
            axes[0].scatter(X[m, 0], X[m, 1], s=24, color=PALETTE[i % len(PALETTE)],
                           alpha=0.85, label=f'Cluster {lbl}', zorder=3)

    # Mark core points with a ring
    if len(db.core_sample_indices_) > 0:
        core = X[db.core_sample_indices_]
        axes[0].scatter(core[:, 0], core[:, 1], s=60, facecolors='none',
                       edgecolors='#fbbf24', linewidths=0.7, alpha=0.4, zorder=4, label='Core pts')

    axes[0].set_title(f'DBSCAN · ε={eps} · MinSamples={min_samp}\n{n_clusters} cluster(s) · {noise_pts} noise pts',
                     color='#fb923c', fontweight='bold', fontsize=10)
    axes[0].legend(fontsize=7, labelcolor='white', facecolor='#120500', edgecolor='#2d1000', loc='upper right')

    # Right: True labels
    unique_true = np.unique(y_true)
    for i, lbl in enumerate(unique_true):
        m = y_true == lbl
        axes[1].scatter(X[m, 0], X[m, 1], s=22, color=PALETTE[i % len(PALETTE)], alpha=0.8, label=f'Class {lbl}')
    axes[1].set_title('True Labels (Reference)', color='#fb923c', fontweight='bold', fontsize=10)
    axes[1].legend(fontsize=7, labelcolor='white', facecolor='#120500', edgecolor='#2d1000')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# ──── TAB 2: Point Types ───────────────────────────────────
with tab2:
    st.markdown('<div class="info-box">DBSCAN classifies each point as <b>Core</b> (dense region center), <b>Border</b> (near core but not dense enough), or <b>Noise</b> (isolated outlier).</div>', unsafe_allow_html=True)

    # Categorise points
    core_mask   = np.zeros(len(X), dtype=bool)
    core_mask[db.core_sample_indices_] = True
    noise_mask  = db_labels == -1
    border_mask = (~core_mask) & (~noise_mask)

    fig, ax = dark_fig(12, 5.5)

    ax.scatter(X[noise_mask,  0], X[noise_mask,  1], s=20, color='#f87171', alpha=0.7, marker='x', linewidths=1, label=f'Noise ({noise_mask.sum()})', zorder=2)
    ax.scatter(X[border_mask, 0], X[border_mask, 1], s=22, color='#fbbf24', alpha=0.8, label=f'Border ({border_mask.sum()})', zorder=3)
    ax.scatter(X[core_mask,   0], X[core_mask,   1], s=28, color='#34d399', alpha=0.9, label=f'Core ({core_mask.sum()})', zorder=4)

    ax.set_title('Point Classification: Core · Border · Noise', color='#fb923c', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9, labelcolor='white', facecolor='#120500', edgecolor='#2d1000')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Donut stats
    counts = {
        "🟢 Core Points":   int(core_mask.sum()),
        "🟡 Border Points": int(border_mask.sum()),
        "🔴 Noise Points":  int(noise_mask.sum()),
    }
    cols = st.columns(3)
    for col, (label, count) in zip(cols, counts.items()):
        pct = count / n_samples * 100
        col.markdown(f"""<div class="glass" style="text-align:center;">
            <div style="font-size:0.8rem; color:rgba(255,200,120,0.6); text-transform:uppercase;">{label}</div>
            <div style="font-size:2rem; font-weight:800; color:#fbbf24;">{count}</div>
            <div style="font-size:0.85rem; color:rgba(255,200,100,0.6);">{pct:.1f}% of dataset</div>
        </div>""", unsafe_allow_html=True)


# ──── TAB 3: ε Tuning ──────────────────────────────────────
with tab3:
    st.markdown('<div class="info-box">The <b>k-NN distance plot</b> helps choose the optimal ε. Sort points by distance to k-th nearest neighbour. The <b>"elbow"</b> of the curve is a good ε value.</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📈 k-NN Distance Plot (Elbow Method for ε)</div>', unsafe_allow_html=True)

    k_nn = min_samp
    nbrs = NearestNeighbors(n_neighbors=k_nn).fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth_dist = np.sort(distances[:, -1])[::-1]

    fig, ax = dark_fig(12, 4.5)
    ax.plot(kth_dist, color='#fb923c', linewidth=2.5)
    ax.axhline(eps, color='#fbbf24', linestyle='--', linewidth=2, alpha=0.8, label=f'Current ε = {eps}')
    ax.fill_between(range(len(kth_dist)), kth_dist, alpha=0.12, color='#fb923c')
    ax.set_xlabel(f'Points sorted by {k_nn}-NN distance', color='#8b5e3c', fontsize=9)
    ax.set_ylabel(f'{k_nn}th Nearest Neighbour Distance', color='#8b5e3c', fontsize=9)
    ax.set_title(f'k-NN Distance Plot (k = MinSamples = {k_nn}) · Look for the "Elbow"',
                color='#fb923c', fontweight='bold', fontsize=10)
    ax.legend(fontsize=9, labelcolor='white', facecolor='#120500', edgecolor='#2d1000')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Silhouette across epsilon values
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📉 Silhouette Score vs Epsilon</div>', unsafe_allow_html=True)

    eps_range = np.round(np.arange(0.1, 2.05, 0.1), 2)
    sil_values, cluster_counts = [], []
    for e in eps_range:
        lbl = DBSCAN(eps=e, min_samples=min_samp).fit_predict(X)
        nc  = len(set(lbl)) - (1 if -1 in lbl else 0)
        cluster_counts.append(nc)
        if nc > 1 and sum(lbl == -1) < len(X) - 2:
            sil_values.append(silhouette_score(X, lbl))
        else:
            sil_values.append(0)

    fig, axes = dark_figs(2, 14, 4.5)

    axes[0].plot(eps_range, sil_values, 'o-', color='#34d399', linewidth=2.5, markersize=6)
    axes[0].axvline(eps, color='#fbbf24', linestyle='--', alpha=0.8, label=f'Current ε={eps}')
    best_eps = eps_range[np.argmax(sil_values)]
    axes[0].axvline(best_eps, color='#f472b6', linestyle=':', alpha=0.8, label=f'Best ε={best_eps}')
    axes[0].set_xlabel('ε', color='#8b5e3c', fontsize=9)
    axes[0].set_ylabel('Silhouette Score', color='#8b5e3c', fontsize=9)
    axes[0].set_title('Silhouette Score vs ε', color='#fb923c', fontweight='bold', fontsize=10)
    axes[0].legend(fontsize=8, labelcolor='white', facecolor='#120500', edgecolor='#2d1000')

    axes[1].plot(eps_range, cluster_counts, 's-', color='#60a5fa', linewidth=2.5, markersize=6)
    axes[1].axvline(eps, color='#fbbf24', linestyle='--', alpha=0.8, label=f'Current ε={eps}')
    axes[1].set_xlabel('ε', color='#8b5e3c', fontsize=9)
    axes[1].set_ylabel('Number of Clusters', color='#8b5e3c', fontsize=9)
    axes[1].set_title('Clusters Found vs ε', color='#fb923c', fontweight='bold', fontsize=10)
    axes[1].legend(fontsize=8, labelcolor='white', facecolor='#120500', edgecolor='#2d1000')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if sil_values and max(sil_values) > 0:
        st.markdown(f"""<div class="warn-box">
        💡 <b>Suggested optimal ε = {best_eps}</b> (highest silhouette score = {max(sil_values):.3f})<br>
        Your current ε = {eps}. {"✅ This is the optimal value!" if best_eps == eps else f"Try adjusting to ε = {best_eps} for better separation."}
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ──── TAB 4: Analysis ──────────────────────────────────────
with tab4:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📋 Cluster Analysis Table</div>', unsafe_allow_html=True)

    rows = []
    for lbl in np.unique(db_labels):
        m    = db_labels == lbl
        pts  = int(m.sum())
        cx   = X[m, 0].mean()
        cy   = X[m, 1].mean()
        spread = X[m].std()
        kind = "Noise" if lbl == -1 else f"Cluster {lbl}"
        rows.append({
            "Label":       kind,
            "Type":        "Outliers / Noise" if lbl == -1 else "Dense Cluster",
            "Points":      pts,
            "Share (%)":   f"{pts/n_samples*100:.1f}%",
            "Center X":    "−" if lbl == -1 else f"{cx:.3f}",
            "Center Y":    "−" if lbl == -1 else f"{cy:.3f}",
            "Spread (std)":"−" if lbl == -1 else f"{spread:.3f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Parameter comparison
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🔄 DBSCAN vs K-Means Summary</div>', unsafe_allow_html=True)
    st.markdown("""
    | Feature | DBSCAN | K-Means |
    |---|---|---|
    | Specify K upfront | ❌ No | ✅ Yes (required) |
    | Handles noise/outliers | ✅ Yes (noise label) | ❌ No |
    | Arbitrary cluster shapes | ✅ Yes | ❌ No (convex only) |
    | Sensitive to density | ✅ Yes | ❌ No |
    | Parameters | ε, MinSamples | K, n_init |
    | Best for | Geospatial, anomaly detection | Uniform-size, convex clusters |
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Raw data preview
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🗂️ Data Preview (with DBSCAN labels)</div>', unsafe_allow_html=True)
    preview = pd.DataFrame(X, columns=["Feature 1 (scaled)", "Feature 2 (scaled)"])
    preview["DBSCAN Label"] = db_labels
    preview["Point Type"]   = ["Core" if i in set(db.core_sample_indices_)
                                 else ("Noise" if db_labels[i] == -1 else "Border")
                                 for i in range(len(X))]
    st.dataframe(preview.head(50), use_container_width=True, hide_index=True)
    st.markdown(f"*Showing first 50 of {n_samples} points*")
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
st.markdown("""
<div class="footer">
    🟠 DBSCAN — Density-Based Spatial Clustering of Applications with Noise &nbsp;|&nbsp;
    Scikit-Learn · Streamlit
</div>
""", unsafe_allow_html=True)
