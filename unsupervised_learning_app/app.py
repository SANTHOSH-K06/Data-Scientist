import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, load_iris, load_digits
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

# ---- Page Config ----
st.set_page_config(
    page_title="Unsupervised Learning Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Premium CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #03000f 0%, #080020 50%, #020010 100%);
    color: #ece8ff;
}
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    background: linear-gradient(135deg, rgba(90,0,190,0.4), rgba(40,0,140,0.3));
    border: 1px solid rgba(160,80,255,0.28);
    border-radius: 22px;
    padding: 34px 32px 26px;
    text-align: center;
    margin-bottom: 26px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 44px rgba(100,0,220,0.25);
}
.hero h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #a78bfa, #e879f9, #38bdf8, #a78bfa);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px !important;
    animation: shimmer 4s linear infinite;
}
.hero p { color: rgba(200,180,255,0.8); font-size: 0.95rem; margin: 0; }

@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

/* Tab bar */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(160,80,255,0.18) !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(200,170,255,0.7) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6d28d9, #9333ea) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(147,51,234,0.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(160,80,255,0.14);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.sec-title {
    font-size: 0.88rem; font-weight: 700; color: #a78bfa;
    text-transform: uppercase; letter-spacing: 1.6px; margin-bottom: 14px;
}

/* Metric boxes */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; }
.metric-chip {
    flex: 1; min-width: 110px;
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.22);
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
}
.chip-label { font-size: 0.7rem; color: rgba(190,160,255,0.6); text-transform: uppercase; letter-spacing: 1px; }
.chip-val   { font-size: 1.35rem; font-weight: 800; color: #c4b5fd; margin-top: 4px; }

/* Info banner */
.info-box {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.87rem;
    color: rgba(180,230,255,0.85);
    margin-bottom: 16px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4c1d95, #7c3aed) !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8em 2em !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 28px rgba(124,58,237,0.55) !important; }

/* Sliders & selects */
.stSlider label, .stSelectbox label, .stRadio label, .stCheckbox label {
    color: #c4b5fd !important; font-weight: 600 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #04000e, #020009) !important;
    border-right: 1px solid rgba(160,80,255,0.12) !important;
}
[data-testid="stSidebar"] * { color: #c4b5fd !important; }
[data-testid="stSidebar"] h3 { color: #a78bfa !important; }

.sb-card {
    background: rgba(167,139,250,0.07);
    border: 1px solid rgba(167,139,250,0.18);
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 0.83rem;
    line-height: 1.65;
    margin-bottom: 12px;
}

hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 16px 0 !important; }
.footer { text-align: center; padding: 16px; color: rgba(160,100,255,0.3); font-size: 0.78rem; margin-top: 34px; border-top: 1px solid rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🔍 About")
    st.markdown("""<div class="sb-card">
    An interactive explorer for <strong>Unsupervised Learning</strong> algorithms — 
    covering clustering and dimensionality reduction.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📚 Topics Covered")
    st.markdown("""<div class="sb-card">
    <b>Tab 1:</b> 🟣 K-Means Clustering<br>
    <b>Tab 2:</b> 🟠 DBSCAN Clustering<br>
    <b>Tab 3:</b> 🔵 PCA (Dimensionality Reduction)<br>
    <b>Tab 4:</b> 🟢 Hierarchical Clustering
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Dataset Settings")
    n_samples   = st.slider("Num Samples", 100, 800, 350, 50)
    n_centers   = st.slider("True Centers", 2, 8, 4)
    cluster_std = st.slider("Cluster Spread", 0.3, 2.5, 0.85, 0.05)
    rand_seed   = st.slider("Random Seed", 0, 99, 42)

    st.markdown("---")
    st.markdown("### 📖 Key Concepts")
    st.markdown("""<div class="sb-card">
    <b>Silhouette Score:</b> Measures how well-separated clusters are. Closer to 1 = better.<br><br>
    <b>Inertia (KMeans):</b> Within-cluster sum of squares. Lower = tighter clusters.<br><br>
    <b>DBSCAN ε:</b> Neighbourhood radius for density estimation.<br><br>
    <b>PCA:</b> Projects high-dim data to lower-dim while retaining max variance.
    </div>""", unsafe_allow_html=True)

# ============================================================
# SHARED DATA GENERATION
# ============================================================
X_raw, y_true = make_blobs(
    n_samples=n_samples, centers=n_centers,
    cluster_std=cluster_std, random_state=rand_seed
)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🔍 Unsupervised Learning Explorer</h1>
    <p>Interactively explore K-Means · DBSCAN · PCA · Hierarchical Clustering</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🟣 K-Means", "🟠 DBSCAN", "🔵 PCA", "🟢 Hierarchical"
])

# ─── HELPER: dark figure ────────────────────────────────────
def dark_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#05001a')
    ax.set_facecolor('#09002a')
    ax.tick_params(colors='#8b7aaa')
    for s in ax.spines.values(): s.set_color('#2a1460')
    return fig, ax

def dark_figs(ncols, w=14, h=5):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    fig.patch.set_facecolor('#05001a')
    for ax in axes:
        ax.set_facecolor('#09002a')
        ax.tick_params(colors='#8b7aaa')
        for s in ax.spines.values(): s.set_color('#2a1460')
    return fig, axes

PALETTE = ['#a78bfa','#f472b6','#34d399','#fbbf24','#60a5fa','#fb923c','#e879f9','#4ade80']

# ============================
# TAB 1 — K-MEANS
# ============================
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">⚙️ K-Means Configuration</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        k = st.slider("Number of Clusters (K)", 2, 10, n_centers, key="km_k")
    with c2:
        init_method = st.selectbox("Initialization Method", ["k-means++", "random"], key="km_init")

    st.markdown("</div>", unsafe_allow_html=True)

    # Train
    km = KMeans(n_clusters=k, init=init_method, n_init=10, random_state=42)
    km_labels = km.fit_predict(X)
    inertia = km.inertia_
    sil = silhouette_score(X, km_labels) if k > 1 else 0.0

    # Metrics
    st.markdown(f"""<div class="metric-row">
        <div class="metric-chip"><div class="chip-label">Clusters (K)</div><div class="chip-val">{k}</div></div>
        <div class="metric-chip"><div class="chip-label">Silhouette Score</div><div class="chip-val">{sil:.3f}</div></div>
        <div class="metric-chip"><div class="chip-label">Inertia</div><div class="chip-val">{inertia:.1f}</div></div>
        <div class="metric-chip"><div class="chip-label">Data Points</div><div class="chip-val">{n_samples}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Plot: clusters + elbow
    fig, axes = dark_figs(2, 14, 5)

    # Left: cluster scatter
    for i in range(k):
        m = km_labels == i
        axes[0].scatter(X[m, 0], X[m, 1], s=22, color=PALETTE[i % len(PALETTE)], alpha=0.8, label=f'C{i}')
    axes[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                    s=220, marker='*', color='#fbbf24', zorder=5, label='Centroids')
    axes[0].set_title(f'K-Means (K={k}) · Silhouette={sil:.3f}', color='#a78bfa', fontweight='bold')
    axes[0].legend(fontsize=7, labelcolor='white', facecolor='#09002a', edgecolor='#2a1460')

    # Right: Elbow curve
    inertias = []
    ks = range(2, min(11, n_samples))
    for ki in ks:
        inertias.append(KMeans(n_clusters=ki, n_init=5, random_state=42).fit(X).inertia_)
    axes[1].plot(list(ks), inertias, 'o-', color='#a78bfa', linewidth=2.5, markersize=7)
    axes[1].axvline(k, color='#fbbf24', linestyle='--', alpha=0.7, label=f'K={k}')
    axes[1].set_xlabel('K', color='#8b7aaa')
    axes[1].set_ylabel('Inertia', color='#8b7aaa')
    axes[1].set_title('Elbow Curve', color='#a78bfa', fontweight='bold')
    axes[1].legend(fontsize=8, labelcolor='white', facecolor='#09002a', edgecolor='#2a1460')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Cluster table
    st.markdown('<div class="glass" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📋 Cluster Summary</div>', unsafe_allow_html=True)
    rows = []
    for i in range(k):
        pts = int(np.sum(km_labels == i))
        cx, cy = km.cluster_centers_[i]
        rows.append({"Cluster": f"Cluster {i}", "Points": pts, "Share": f"{pts/n_samples*100:.1f}%",
                     "Center X": f"{cx:.3f}", "Center Y": f"{cy:.3f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================
# TAB 2 — DBSCAN
# ============================
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">⚙️ DBSCAN Configuration</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">DBSCAN finds clusters of arbitrary shape and marks low-density points as <b>Noise</b> (label = -1). It does NOT require K upfront.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        eps      = st.slider("Epsilon (ε) — Neighbourhood Radius", 0.1, 3.0, 0.6, 0.05, key="db_eps")
    with c2:
        min_samp = st.slider("Min Samples (core point threshold)", 2, 20, 5, key="db_min")
    st.markdown("</div>", unsafe_allow_html=True)

    db = DBSCAN(eps=eps, min_samples=min_samp)
    db_labels = db.fit_predict(X)
    n_found  = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    noise_ct = int(np.sum(db_labels == -1))
    sil_db   = silhouette_score(X, db_labels) if n_found > 1 and noise_ct < n_samples - 1 else 0.0

    st.markdown(f"""<div class="metric-row">
        <div class="metric-chip"><div class="chip-label">Clusters Found</div><div class="chip-val">{n_found}</div></div>
        <div class="metric-chip"><div class="chip-label">Noise Points</div><div class="chip-val" style="color:#f87171;">{noise_ct}</div></div>
        <div class="metric-chip"><div class="chip-label">Silhouette</div><div class="chip-val">{sil_db:.3f}</div></div>
        <div class="metric-chip"><div class="chip-label">ε</div><div class="chip-val">{eps}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    fig, axes = dark_figs(2, 14, 5)

    # Left: DBSCAN result
    unique_l = np.unique(db_labels)
    for i, lbl in enumerate(unique_l):
        m = db_labels == lbl
        color = '#555566' if lbl == -1 else PALETTE[i % len(PALETTE)]
        name  = 'Noise' if lbl == -1 else f'Cluster {lbl}'
        axes[0].scatter(X[m, 0], X[m, 1], s=22, color=color, alpha=0.85, label=name)
    axes[0].set_title(f'DBSCAN · ε={eps} · minSamples={min_samp}', color='#fb923c', fontweight='bold')
    axes[0].legend(fontsize=7, labelcolor='white', facecolor='#09002a', edgecolor='#2a1460')

    # Right: True labels for comparison
    for i in range(n_centers):
        m = y_true == i
        axes[1].scatter(X[m, 0], X[m, 1], s=22, color=PALETTE[i % len(PALETTE)], alpha=0.8, label=f'True {i}')
    axes[1].set_title('True Labels (Reference)', color='#fb923c', fontweight='bold')
    axes[1].legend(fontsize=7, labelcolor='white', facecolor='#09002a', edgecolor='#2a1460')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Noise/cluster breakdown
    if n_found > 0:
        st.markdown('<div class="glass" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">📋 Density Cluster Summary</div>', unsafe_allow_html=True)
        rows = []
        for lbl in [l for l in np.unique(db_labels) if l != -1]:
            pts = int(np.sum(db_labels == lbl))
            rows.append({"Cluster": f"Cluster {lbl}", "Points": pts, "Share": f"{pts/n_samples*100:.1f}%", "Type": "Dense Region"})
        if noise_ct > 0:
            rows.append({"Cluster": "Noise", "Points": noise_ct, "Share": f"{noise_ct/n_samples*100:.1f}%", "Type": "Outlier / Low-density"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ============================
# TAB 3 — PCA
# ============================
with tab3:
    st.markdown('<div class="info-box">PCA (Principal Component Analysis) reduces high-dimensional data to fewer dimensions while retaining the most variance. Here we apply it to the <b>Iris dataset</b> (4 features → 2D).</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">⚙️ PCA Configuration</div>', unsafe_allow_html=True)

    iris = __import__('sklearn.datasets', fromlist=['load_iris']).load_iris()
    X_iris = iris.data
    y_iris = iris.target
    target_names = iris.target_names

    scaler_pca = StandardScaler()
    X_iris_s   = scaler_pca.fit_transform(X_iris)

    n_components = st.slider("Number of PCA Components", 1, 4, 2, key="pca_nc")
    st.markdown("</div>", unsafe_allow_html=True)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_iris_s)

    explained = pca.explained_variance_ratio_
    total_var = explained.sum() * 100

    st.markdown(f"""<div class="metric-row">
        <div class="metric-chip"><div class="chip-label">Components</div><div class="chip-val">{n_components}</div></div>
        <div class="metric-chip"><div class="chip-label">Total Variance Explained</div><div class="chip-val">{total_var:.1f}%</div></div>
        <div class="metric-chip"><div class="chip-label">PC1 Variance</div><div class="chip-val">{explained[0]*100:.1f}%</div></div>
        {"<div class='metric-chip'><div class='chip-label'>PC2 Variance</div><div class='chip-val'>" + f"{explained[1]*100:.1f}%" + "</div></div>" if n_components >= 2 else ""}
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    fig, axes = dark_figs(2, 14, 5)

    # Left: 2D PCA scatter (PC1 vs PC2 if available)
    colors_pca = ['#60a5fa', '#f472b6', '#34d399']
    pc1 = X_pca[:, 0]
    pc2 = X_pca[:, 1] if n_components >= 2 else np.zeros(len(X_pca))

    for i, name in enumerate(target_names):
        m = y_iris == i
        axes[0].scatter(pc1[m], pc2[m], s=30, color=colors_pca[i], alpha=0.85, label=name.capitalize())

    axes[0].set_xlabel(f'PC1 ({explained[0]*100:.1f}% var)', color='#8b7aaa', fontsize=9)
    axes[0].set_ylabel(f'PC2 ({explained[1]*100:.1f}% var)' if n_components >= 2 else 'PC2 (0%)', color='#8b7aaa', fontsize=9)
    axes[0].set_title('Iris Dataset — PCA 2D Projection', color='#38bdf8', fontweight='bold')
    axes[0].legend(fontsize=9, labelcolor='white', facecolor='#09002a', edgecolor='#2a1460')

    # Right: Explained Variance Bar
    axes[1].bar(
        [f'PC{i+1}' for i in range(n_components)],
        explained * 100,
        color=[PALETTE[i % len(PALETTE)] for i in range(n_components)],
        edgecolor='#2a1460', linewidth=0.8
    )
    cumulative = np.cumsum(explained) * 100
    ax_right = axes[1].twinx()
    ax_right.plot([f'PC{i+1}' for i in range(n_components)], cumulative,
                  'o--', color='#fbbf24', linewidth=2, markersize=8, label='Cumulative')
    ax_right.set_ylabel('Cumulative %', color='#fbbf24', fontsize=9)
    ax_right.tick_params(colors='#fbbf24')
    ax_right.set_facecolor('#09002a')
    axes[1].set_ylabel('Variance Explained (%)', color='#8b7aaa', fontsize=9)
    axes[1].set_title('Explained Variance per Component', color='#38bdf8', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Loadings table
    st.markdown('<div class="glass" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📊 PCA Component Loadings</div>', unsafe_allow_html=True)
    loading_df = pd.DataFrame(
        pca.components_.T,
        index=iris.feature_names,
        columns=[f'PC{i+1}' for i in range(n_components)]
    ).round(3)
    st.dataframe(loading_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================
# TAB 4 — HIERARCHICAL
# ============================
with tab4:
    st.markdown('<div class="info-box">Hierarchical (Agglomerative) Clustering builds a tree of clusters bottom-up. The <b>dendrogram</b> visualizes how individual points merge into clusters.</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">⚙️ Hierarchical Configuration</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        h_k       = st.slider("Number of Clusters", 2, 8, n_centers, key="hc_k")
        linkage_m = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], key="hc_link")
    with c2:
        affinity  = "euclidean" if linkage_m == "ward" else st.selectbox("Affinity", ["euclidean", "manhattan", "cosine"], key="hc_aff")

    st.markdown("</div>", unsafe_allow_html=True)

    # Subset for dendrogram (max 80 pts for readability)
    X_sub = X[:min(80, n_samples)]

    hc = AgglomerativeClustering(n_clusters=h_k, linkage=linkage_m)
    hc_labels = hc.fit_predict(X)
    sil_hc = silhouette_score(X, hc_labels) if h_k > 1 else 0.0

    st.markdown(f"""<div class="metric-row">
        <div class="metric-chip"><div class="chip-label">Clusters</div><div class="chip-val">{h_k}</div></div>
        <div class="metric-chip"><div class="chip-label">Silhouette</div><div class="chip-val">{sil_hc:.3f}</div></div>
        <div class="metric-chip"><div class="chip-label">Linkage</div><div class="chip-val">{linkage_m}</div></div>
        <div class="metric-chip"><div class="chip-label">Points</div><div class="chip-val">{n_samples}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Figure: scatter + dendrogram
    fig, axes = dark_figs(2, 14, 5)

    # Left: scatter
    for i in range(h_k):
        m = hc_labels == i
        axes[0].scatter(X[m, 0], X[m, 1], s=22, color=PALETTE[i % len(PALETTE)], alpha=0.85, label=f'Cluster {i}')
    axes[0].set_title(f'Hierarchical Clustering (k={h_k}, {linkage_m})', color='#4ade80', fontweight='bold')
    axes[0].legend(fontsize=7, labelcolor='white', facecolor='#09002a', edgecolor='#2a1460')

    # Right: dendrogram
    Z = linkage(X_sub, method=linkage_m)
    dendrogram(
        Z, ax=axes[1],
        color_threshold=0,
        above_threshold_color='#a78bfa',
        leaf_font_size=6,
        no_labels=True
    )
    # Color dendrogram lines
    for coll in axes[1].collections:
        coll.set_color('#a78bfa')
    axes[1].set_title(f'Dendrogram (first {len(X_sub)} pts, {linkage_m} linkage)', color='#4ade80', fontweight='bold')
    axes[1].set_xlabel('Samples', color='#8b7aaa', fontsize=8)
    axes[1].set_ylabel('Distance', color='#8b7aaa', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Cluster table
    st.markdown('<div class="glass" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📋 Cluster Summary</div>', unsafe_allow_html=True)
    rows = []
    for i in range(h_k):
        pts = int(np.sum(hc_labels == i))
        rows.append({"Cluster": f"Cluster {i}", "Points": pts, "Share (%)": f"{pts/n_samples*100:.1f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    🔍 Unsupervised Learning Explorer &nbsp;|&nbsp; K-Means · DBSCAN · PCA · Hierarchical Clustering &nbsp;|&nbsp; 
    Scikit-Learn · Streamlit
</div>
""", unsafe_allow_html=True)
