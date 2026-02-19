import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---- Page Config ----
st.set_page_config(
    page_title="Clustering Explorer | KMeans & DBSCAN",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #001020 0%, #001828 50%, #000c18 100%);
    color: #e0f4ff;
}
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, rgba(0,150,200,0.35), rgba(0,100,160,0.25));
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 40px rgba(0,150,200,0.2);
}
.hero h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #22d3ee, #38bdf8, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px !important;
}
.hero p { color: rgba(150,220,255,0.75); font-size: 0.95rem; margin: 0; }

.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(56,189,248,0.14);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}
.sec-title {
    font-size: 0.9rem; font-weight: 700; color: #22d3ee;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px;
}
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; }
.stat-box {
    flex: 1; min-width: 100px;
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}
.stat-lab { font-size: 0.72rem; color: rgba(100,220,255,0.6); text-transform: uppercase; letter-spacing: 1px; }
.stat-val { font-size: 1.3rem; font-weight: 800; color: #22d3ee; margin-top: 4px; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0c4a6e, #0e7490) !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85em 2em !important;
    box-shadow: 0 6px 20px rgba(14,116,144,0.45) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

.stSlider label, .stSelectbox label, .stRadio label { color: #67e8f9 !important; font-weight: 600 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #000c18, #000810) !important;
    border-right: 1px solid rgba(34,211,238,0.12) !important;
}
[data-testid="stSidebar"] * { color: #67e8f9 !important; }
.sidebar-card {
    background: rgba(34,211,238,0.07);
    border: 1px solid rgba(34,211,238,0.18);
    border-radius: 12px;
    padding: 13px;
    font-size: 0.84rem;
    line-height: 1.65;
    margin-bottom: 12px;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 16px 0 !important; }
.footer { text-align: center; padding: 16px; color: rgba(100,200,255,0.3); font-size: 0.78rem; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🔵 About")
    st.markdown("""<div class="sidebar-card">
    Visualize <strong>K-Means</strong> and <strong>DBSCAN</strong> clustering 
    on synthetic 2D data. Adjust parameters to explore cluster formation.
    </div>""", unsafe_allow_html=True)

    algo = st.radio("Algorithm", ["K-Means", "DBSCAN"])
    st.markdown("---")

    if algo == "K-Means":
        n_clusters = st.slider("Number of Clusters (K)", 2, 10, 4)
        n_init = st.slider("n_init", 5, 20, 10)
    else:
        eps   = st.slider("Epsilon (ε)", 0.1, 3.0, 0.7, 0.05)
        min_s = st.slider("Min Samples", 2, 20, 5)

    st.markdown("---")
    st.markdown("### 🧪 Dataset Settings")
    n_samples  = st.slider("Num Samples", 100, 1000, 300, 50)
    n_centers  = st.slider("True Centers", 2, 8, 4)
    cluster_std = st.slider("Cluster Std Dev", 0.3, 3.0, 0.9, 0.1)
    random_seed = st.slider("Random Seed", 0, 99, 42)

# ---- Generate Data ----
X_raw, y_true = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=cluster_std, random_state=random_seed)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ---- Cluster ----
if algo == "K-Means":
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    labels = model.fit_predict(X)
    inertia = model.inertia_
    n_found = n_clusters
else:
    model = DBSCAN(eps=eps, min_samples=min_s)
    labels = model.fit_predict(X)
    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    inertia = None

noise_pts = np.sum(labels == -1) if -1 in labels else 0

# ---- Hero ----
st.markdown("""
<div class="hero">
    <h1>🔵 Clustering Explorer</h1>
    <p>Visualize K-Means and DBSCAN clustering on interactive 2D datasets</p>
</div>
""", unsafe_allow_html=True)

# ---- Stats ----
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="glass"><div class="sec-title">Algorithm</div><div style="font-size:1rem;font-weight:800;color:#22d3ee;">{algo}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="glass"><div class="sec-title">Clusters Found</div><div style="font-size:1.8rem;font-weight:800;color:#22d3ee;">{n_found}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="glass"><div class="sec-title">Data Points</div><div style="font-size:1.8rem;font-weight:800;color:#22d3ee;">{n_samples}</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="glass"><div class="sec-title">Noise Points</div><div style="font-size:1.8rem;font-weight:800;color:#f87171 if noise_pts>0 else #22d3ee;">{noise_pts}</div></div>', unsafe_allow_html=True)

# ---- Plot ----
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown(f'<div class="sec-title">📊 {algo} Cluster Visualization</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#001020')

palette = cm.get_cmap('tab10', max(n_found, 1) + 1)

for ax, (lbl, title) in zip(axes, [(labels, f'{algo} Labels'), (y_true, 'True Labels')]):
    ax.set_facecolor('#001828')
    unique = np.unique(lbl)
    for i, u in enumerate(unique):
        mask = lbl == u
        color = '#555555' if u == -1 else palette(i % 10)
        label_name = 'Noise' if u == -1 else f'Cluster {u}'
        ax.scatter(X[mask, 0], X[mask, 1], s=22, color=color, alpha=0.8, label=label_name)
    ax.set_title(title, color='#67e8f9', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#4b8fa8')
    for spine in ax.spines.values():
        spine.set_color('#1e4a6e')
    ax.legend(loc='upper right', fontsize=7, labelcolor='white', facecolor='#001828', edgecolor='#1e4a6e')

plt.tight_layout()
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Cluster Summary ----
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">📋 Cluster Summary</div>', unsafe_allow_html=True)

unique_labels = [l for l in np.unique(labels) if l != -1]
summary_data = []
for l in unique_labels:
    pts = np.sum(labels == l)
    pct = pts / n_samples * 100
    cx, cy = X[labels == l, 0].mean(), X[labels == l, 1].mean()
    summary_data.append({"Cluster": f"Cluster {l}", "Points": pts, "Percentage": f"{pct:.1f}%", "Center X": f"{cx:.3f}", "Center Y": f"{cy:.3f}"})

if noise_pts > 0:
    summary_data.append({"Cluster": "Noise", "Points": noise_pts, "Percentage": f"{noise_pts/n_samples*100:.1f}%", "Center X": "N/A", "Center Y": "N/A"})

if summary_data:
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

if algo == "K-Means" and inertia is not None:
    st.markdown(f'<div style="font-size:0.85rem;color:rgba(100,200,255,0.6);margin-top:8px;">📉 KMeans Inertia (within-cluster sum of squares): <strong style="color:#22d3ee;">{inertia:.2f}</strong></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>🔵 Synthetic Blobs · K-Means & DBSCAN · Scikit-Learn · Streamlit</div>", unsafe_allow_html=True)
