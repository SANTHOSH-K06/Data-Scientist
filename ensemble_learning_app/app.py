import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Ensemble Learning Explorer",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #010f06 0%, #011a0a 50%, #010c04 100%);
    color: #edfff4;
}
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    background: linear-gradient(135deg, rgba(5,120,60,0.4), rgba(0,80,30,0.3));
    border: 1px solid rgba(52,211,153,0.28);
    border-radius: 22px;
    padding: 34px 32px 26px;
    text-align: center;
    margin-bottom: 26px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 44px rgba(5,150,60,0.22);
}
.hero h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #4ade80, #34d399, #fbbf24, #4ade80);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px !important;
    animation: shimmer 4s linear infinite;
}
.hero p { color: rgba(180,255,210,0.75); font-size: 0.95rem; margin: 0; }
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important; gap: 4px !important;
    border: 1px solid rgba(52,211,153,0.18) !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(150,255,200,0.7) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 0.87rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #064e3b, #065f46) !important;
    color: #4ade80 !important;
    box-shadow: 0 4px 14px rgba(5,100,50,0.5) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* Glass */
.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(52,211,153,0.13);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.sec-title {
    font-size: 0.88rem; font-weight: 700; color: #34d399;
    text-transform: uppercase; letter-spacing: 1.6px; margin-bottom: 14px;
}

/* Metric chips */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; }
.metric-chip {
    flex: 1; min-width: 110px;
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 12px;
    padding: 14px 12px; text-align: center;
}
.chip-label { font-size: 0.7rem; color: rgba(100,255,180,0.55); text-transform: uppercase; letter-spacing: 1px; }
.chip-val   { font-size: 1.3rem; font-weight: 800; color: #6ee7b7; margin-top: 4px; }

/* Info boxes */
.info-box {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.22);
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 12px 16px; font-size: 0.86rem;
    color: rgba(180,230,255,0.85); margin-bottom: 16px; line-height: 1.55;
}
.rf-box {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.22);
    border-left: 4px solid #34d399;
    border-radius: 10px;
    padding: 12px 16px; font-size: 0.86rem;
    color: rgba(180,255,210,0.85); margin-bottom: 16px; line-height: 1.55;
}
.gb-box {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.22);
    border-left: 4px solid #fbbf24;
    border-radius: 10px;
    padding: 12px 16px; font-size: 0.86rem;
    color: rgba(255,240,180,0.85); margin-bottom: 16px; line-height: 1.55;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #064e3b, #059669) !important;
    color: white !important; font-size: 1rem !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 12px !important; padding: 0.8em 2em !important;
    width: 100%; box-shadow: 0 6px 20px rgba(5,150,105,0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

/* Inputs */
.stSlider label, .stSelectbox label, .stRadio label, .stCheckbox label {
    color: #86efac !important; font-weight: 600 !important;
}
.stNumberInput label { color: #86efac !important; font-weight: 600 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #010a04, #010702) !important;
    border-right: 1px solid rgba(52,211,153,0.12) !important;
}
[data-testid="stSidebar"] * { color: #86efac !important; }
[data-testid="stSidebar"] h3 { color: #34d399 !important; }
.sb-card {
    background: rgba(52,211,153,0.06);
    border: 1px solid rgba(52,211,153,0.17);
    border-radius: 12px; padding: 12px 14px;
    font-size: 0.83rem; line-height: 1.65; margin-bottom: 12px;
}

hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 16px 0 !important; }
.footer {
    text-align: center; padding: 16px;
    color: rgba(60,200,120,0.3); font-size: 0.78rem;
    margin-top: 34px; border-top: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌲 About")
    st.markdown("""<div class="sb-card">
    Compare <strong>4 Ensemble methods</strong>: Random Forest, Gradient Boosting, 
    AdaBoost, and Bagging on real classification datasets.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📂 Dataset")
    dataset = st.selectbox("Choose Dataset", ["Breast Cancer", "Wine Quality"])

    st.markdown("---")
    st.markdown("### 🌲 Random Forest")
    rf_n    = st.slider("n_estimators (RF)", 10, 300, 100, 10)
    rf_depth= st.slider("max_depth (RF)", 1, 20, 5)

    st.markdown("### 📈 Gradient Boosting")
    gb_n    = st.slider("n_estimators (GB)", 10, 300, 100, 10)
    gb_lr   = st.slider("Learning Rate (GB)", 0.01, 1.0, 0.1, 0.01)
    gb_depth= st.slider("max_depth (GB)", 1, 8, 3)

    st.markdown("---")
    st.markdown("### 📋 Ensemble Methods")
    st.markdown("""<div class="sb-card">
    🌲 <b>Random Forest:</b> Bagging + feature randomness<br>
    📈 <b>Gradient Boosting:</b> Sequential error correction<br>
    🔁 <b>AdaBoost:</b> Re-weights misclassified samples<br>
    📦 <b>Bagging:</b> Bootstrap aggregation of base learners
    </div>""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────
@st.cache_data
def get_data(name):
    if name == "Breast Cancer":
        d = load_breast_cancer()
    else:
        d = load_wine()
    X_tr, X_te, y_tr, y_te = train_test_split(d.data, d.target, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return X_tr, X_te, y_tr, y_te, d.feature_names, d.target_names, sc, d

X_tr, X_te, y_tr, y_te, feat_names, target_names, sc, raw_data = get_data(dataset)
is_binary = len(target_names) == 2

# ── Train Models ─────────────────────────────────────────────
@st.cache_data
def train_all(rf_n, rf_d, gb_n, gb_lr, gb_d, dataset):
    X_tr, X_te, y_tr, y_te, _, _, _, _ = get_data(dataset)
    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d,  random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=gb_n, learning_rate=gb_lr, max_depth=gb_d, random_state=42),
        "AdaBoost":           AdaBoostClassifier(n_estimators=100, random_state=42),
        "Bagging":            BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42, n_jobs=-1),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        acc  = accuracy_score(y_te, y_pred)
        cv   = cross_val_score(m, X_tr, y_tr, cv=5).mean()
        results[name] = {"model": m, "acc": acc, "cv": cv, "pred": y_pred}
    return results

with st.spinner("🌲 Training all ensemble models..."):
    results = train_all(rf_n, rf_depth, gb_n, gb_lr, gb_depth, dataset)

rf_model = results["Random Forest"]["model"]
gb_model = results["Gradient Boosting"]["model"]

# ── Helpers ──────────────────────────────────────────────────
def dark_fig(w=13, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#010f06')
    ax.set_facecolor('#011408')
    ax.tick_params(colors='#3a7a50')
    for s in ax.spines.values(): s.set_color('#0a3020')
    return fig, ax

def dark_figs(ncols, w=14, h=5):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    fig.patch.set_facecolor('#010f06')
    for ax in (axes if ncols > 1 else [axes]):
        ax.set_facecolor('#011408')
        ax.tick_params(colors='#3a7a50')
        for s in ax.spines.values(): s.set_color('#0a3020')
    return fig, (axes if ncols > 1 else [axes])

COLORS = {'Random Forest': '#34d399', 'Gradient Boosting': '#fbbf24',
          'AdaBoost': '#f472b6', 'Bagging': '#60a5fa'}

# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌲 Ensemble Learning Explorer</h1>
    <p>Random Forest · Gradient Boosting · AdaBoost · Bagging — Compare them all</p>
</div>
""", unsafe_allow_html=True)

# ── Summary Metrics ──────────────────────────────────────────
chips = "".join([
    f"""<div class="metric-chip">
        <div class="chip-label">{name}</div>
        <div class="chip-val">{r['acc']*100:.1f}%</div>
    </div>""" for name, r in results.items()
])
st.markdown(f'<div class="metric-row" style="margin-bottom:22px;">{chips}</div>', unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Comparison", "🌲 Random Forest", "📈 Gradient Boosting",
    "🔁 AdaBoost & Bagging", "🎯 Predict"
])


# ──────────────────────────────────────────────
# TAB 1 — COMPARISON
# ──────────────────────────────────────────────
with tab1:
    # Bar chart: accuracy + CV score
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📊 Model Comparison — Accuracy & Cross-Validation</div>', unsafe_allow_html=True)

    fig, axes = dark_figs(2, 14, 5)
    names = list(results.keys())
    accs  = [results[n]["acc"] * 100 for n in names]
    cvs   = [results[n]["cv"]  * 100 for n in names]
    colors = [COLORS[n] for n in names]
    x = np.arange(len(names))

    axes[0].bar(names, accs, color=colors, edgecolor='#0a3020', linewidth=0.8, width=0.55)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, color='white', fontweight='bold')
    axes[0].set_ylim(min(accs) - 3, 101)
    axes[0].set_ylabel('Test Accuracy (%)', color='#3a7a50', fontsize=9)
    axes[0].set_title('Test Accuracy', color='#34d399', fontweight='bold', fontsize=10)
    axes[0].tick_params(axis='x', labelsize=8)
    for lbl in axes[0].get_xticklabels(): lbl.set_color('white')

    axes[1].bar(names, cvs, color=colors, edgecolor='#0a3020', linewidth=0.8, width=0.55)
    for i, v in enumerate(cvs):
        axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, color='white', fontweight='bold')
    axes[1].set_ylim(min(cvs) - 3, 101)
    axes[1].set_ylabel('CV Score (%)', color='#3a7a50', fontsize=9)
    axes[1].set_title('5-Fold Cross-Validation Score', color='#34d399', fontweight='bold', fontsize=10)
    axes[1].tick_params(axis='x', labelsize=8)
    for lbl in axes[1].get_xticklabels(): lbl.set_color('white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Summary table
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📋 Results Table</div>', unsafe_allow_html=True)
    df_cmp = pd.DataFrame([
        {"Model": n, "Test Accuracy": f"{r['acc']*100:.2f}%",
         "CV Score (5-fold)": f"{r['cv']*100:.2f}%",
         "Type": {"Random Forest":"Bagging","Gradient Boosting":"Boosting","AdaBoost":"Boosting","Bagging":"Bagging"}[n]}
        for n, r in results.items()
    ])
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ROC curves (binary only)
    if is_binary:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">📈 ROC Curves</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(10, 5)
        for name, r in results.items():
            try:
                proba = r["model"].predict_proba(X_te)[:, 1]
                fpr, tpr, _ = roc_curve(y_te, proba)
                auc = roc_auc_score(y_te, proba)
                ax.plot(fpr, tpr, color=COLORS[name], linewidth=2.2, label=f'{name} (AUC={auc:.3f})')
            except Exception:
                pass
        ax.plot([0,1],[0,1],'--', color='#555', linewidth=1)
        ax.set_xlabel('False Positive Rate', color='#3a7a50', fontsize=9)
        ax.set_ylabel('True Positive Rate', color='#3a7a50', fontsize=9)
        ax.set_title('ROC Curves — All Models', color='#34d399', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8, labelcolor='white', facecolor='#011408', edgecolor='#0a3020')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TAB 2 — RANDOM FOREST
# ──────────────────────────────────────────────
with tab2:
    st.markdown('<div class="rf-box">🌲 <b>Random Forest</b> = many Decision Trees trained on random data subsets + random feature subsets, then <b>majority voted</b>. Reduces variance (overfitting) through diversity.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Feature importance
    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">⭐ Feature Importance (Top 15)</div>', unsafe_allow_html=True)
        top_n = 15
        importances = rf_model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        top_feat   = [feat_names[i] for i in idx]
        top_imp    = importances[idx]

        fig, ax = dark_fig(6, 5)
        bars = ax.barh(range(top_n), top_imp[::-1], color='#34d399', edgecolor='#0a3020', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f[:20] for f in top_feat[::-1]], fontsize=8, color='white')
        ax.set_xlabel('Importance', color='#3a7a50', fontsize=8)
        ax.set_title('Random Forest Feature Importance', color='#34d399', fontweight='bold', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Confusion matrix
    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        cm_rf = confusion_matrix(y_te, results["Random Forest"]["pred"])
        fig, ax = dark_fig(5.5, 4.5)
        im = ax.imshow(cm_rf, interpolation='nearest', cmap='Greens')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(target_names))); ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, fontsize=8, color='white')
        ax.set_yticklabels(target_names, fontsize=8, color='white')
        thresh = cm_rf.max() / 2
        for i in range(cm_rf.shape[0]):
            for j in range(cm_rf.shape[1]):
                ax.text(j, i, cm_rf[i, j], ha='center', va='center', fontsize=11, fontweight='bold',
                        color='black' if cm_rf[i, j] > thresh else 'white')
        ax.set_xlabel('Predicted', color='#3a7a50', fontsize=9)
        ax.set_ylabel('Actual', color='#3a7a50', fontsize=9)
        ax.set_title('Confusion Matrix — Random Forest', color='#34d399', fontweight='bold', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # n_estimators effect
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📉 Accuracy vs Number of Trees</div>', unsafe_allow_html=True)
    ns = [10, 20, 30, 50, 75, 100, 150, 200, 300]
    rf_accs = [RandomForestClassifier(n_estimators=n, max_depth=rf_depth, random_state=42, n_jobs=-1)
               .fit(X_tr, y_tr).score(X_te, y_te) * 100 for n in ns]
    fig, ax = dark_fig(11, 4)
    ax.plot(ns, rf_accs, 'o-', color='#34d399', linewidth=2.5, markersize=7)
    ax.axvline(rf_n, color='#fbbf24', linestyle='--', alpha=0.8, label=f'Current n={rf_n}')
    ax.fill_between(ns, rf_accs, min(rf_accs) - 1, alpha=0.1, color='#34d399')
    ax.set_xlabel('n_estimators', color='#3a7a50', fontsize=9)
    ax.set_ylabel('Test Accuracy (%)', color='#3a7a50', fontsize=9)
    ax.set_title('Effect of Number of Trees on Accuracy', color='#34d399', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, labelcolor='white', facecolor='#011408', edgecolor='#0a3020')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TAB 3 — GRADIENT BOOSTING
# ──────────────────────────────────────────────
with tab3:
    st.markdown('<div class="gb-box">📈 <b>Gradient Boosting</b> builds trees <b>sequentially</b>: each tree corrects the errors of the previous. Uses gradient descent to minimise loss. Reduces both bias and variance.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">⭐ Feature Importance (Top 15)</div>', unsafe_allow_html=True)
        imp_gb = gb_model.feature_importances_
        idx_gb = np.argsort(imp_gb)[::-1][:15]
        fig, ax = dark_fig(6, 5)
        ax.barh(range(15), imp_gb[idx_gb][::-1], color='#fbbf24', edgecolor='#0a3020', linewidth=0.5)
        ax.set_yticks(range(15))
        ax.set_yticklabels([feat_names[i][:20] for i in idx_gb[::-1]], fontsize=8, color='white')
        ax.set_xlabel('Importance', color='#3a7a50', fontsize=8)
        ax.set_title('Gradient Boosting Feature Importance', color='#fbbf24', fontweight='bold', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        cm_gb = confusion_matrix(y_te, results["Gradient Boosting"]["pred"])
        fig, ax = dark_fig(5.5, 4.5)
        im2 = ax.imshow(cm_gb, interpolation='nearest', cmap='YlOrBr')
        plt.colorbar(im2, ax=ax)
        ax.set_xticks(range(len(target_names))); ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, fontsize=8, color='white')
        ax.set_yticklabels(target_names, fontsize=8, color='white')
        thresh2 = cm_gb.max() / 2
        for i in range(cm_gb.shape[0]):
            for j in range(cm_gb.shape[1]):
                ax.text(j, i, cm_gb[i, j], ha='center', va='center', fontsize=11, fontweight='bold',
                        color='black' if cm_gb[i, j] > thresh2 else 'white')
        ax.set_xlabel('Predicted', color='#3a7a50', fontsize=9)
        ax.set_ylabel('Actual', color='#3a7a50', fontsize=9)
        ax.set_title('Confusion Matrix — Gradient Boosting', color='#fbbf24', fontweight='bold', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Learning rate effect
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📉 Training Loss per Iteration (Deviance)</div>', unsafe_allow_html=True)
    # We retrain a GB to get staged scores
    gb_staged = GradientBoostingClassifier(n_estimators=gb_n, learning_rate=gb_lr, max_depth=gb_depth, random_state=42)
    gb_staged.fit(X_tr, y_tr)
    train_scores = [s for s in gb_staged.staged_score(X_tr, y_tr)]
    test_scores  = [s for s in gb_staged.staged_score(X_te, y_te)]
    fig, ax = dark_fig(11, 4)
    ax.plot(train_scores, color='#34d399', linewidth=2, label='Train Accuracy')
    ax.plot(test_scores,  color='#fbbf24', linewidth=2, label='Test Accuracy')
    ax.set_xlabel('Boosting Iteration', color='#3a7a50', fontsize=9)
    ax.set_ylabel('Accuracy', color='#3a7a50', fontsize=9)
    ax.set_title(f'Learning Curve — Gradient Boosting (lr={gb_lr})', color='#fbbf24', fontweight='bold', fontsize=10)
    ax.legend(fontsize=9, labelcolor='white', facecolor='#011408', edgecolor='#0a3020')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TAB 4 — AdaBoost & Bagging
# ──────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">🔁 AdaBoost — Confusion Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">AdaBoost re-weights misclassified samples, forcing the next tree to focus on hard examples. Uses <b>Decision Stumps</b> (depth=1) as base learners.</div>', unsafe_allow_html=True)
        cm_ab = confusion_matrix(y_te, results["AdaBoost"]["pred"])
        fig, ax = dark_fig(5.5, 4)
        im3 = ax.imshow(cm_ab, cmap='RdPu')
        plt.colorbar(im3, ax=ax)
        ax.set_xticks(range(len(target_names))); ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, fontsize=8, color='white')
        ax.set_yticklabels(target_names, fontsize=8, color='white')
        thr = cm_ab.max() / 2
        for i in range(cm_ab.shape[0]):
            for j in range(cm_ab.shape[1]):
                ax.text(j, i, cm_ab[i, j], ha='center', va='center', fontsize=11, fontweight='bold',
                        color='black' if cm_ab[i, j] > thr else 'white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(f"""<div style="text-align:center; margin-top:10px;">
            <span style="font-size:0.85rem; color:#86efac;">Test Accuracy: <strong style="color:#f472b6;">{results['AdaBoost']['acc']*100:.2f}%</strong></span>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">📦 Bagging — Confusion Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Bagging trains multiple models on <b>bootstrap samples</b> (random subsets with replacement) and aggregates predictions by majority vote.</div>', unsafe_allow_html=True)
        cm_bg = confusion_matrix(y_te, results["Bagging"]["pred"])
        fig, ax = dark_fig(5.5, 4)
        im4 = ax.imshow(cm_bg, cmap='Blues')
        plt.colorbar(im4, ax=ax)
        ax.set_xticks(range(len(target_names))); ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, fontsize=8, color='white')
        ax.set_yticklabels(target_names, fontsize=8, color='white')
        thr2 = cm_bg.max() / 2
        for i in range(cm_bg.shape[0]):
            for j in range(cm_bg.shape[1]):
                ax.text(j, i, cm_bg[i, j], ha='center', va='center', fontsize=11, fontweight='bold',
                        color='black' if cm_bg[i, j] > thr2 else 'white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(f"""<div style="text-align:center; margin-top:10px;">
            <span style="font-size:0.85rem; color:#86efac;">Test Accuracy: <strong style="color:#60a5fa;">{results['Bagging']['acc']*100:.2f}%</strong></span>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Method summary table
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📖 Ensemble Method Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    | Method | Strategy | Base Learner | Parallelisable | Handles Noise |
    |---|---|---|---|---|
    | 🌲 **Random Forest** | Bagging + Feature Randomness | Decision Trees | ✅ Yes | ✅ Yes |
    | 📈 **Gradient Boosting** | Sequential Boosting | Shallow Trees | ❌ No | ⚠️ Sensitive |
    | 🔁 **AdaBoost** | Adaptive Weighting | Stumps (depth=1) | ❌ No | ❌ Sensitive |
    | 📦 **Bagging** | Bootstrap Aggregation | Any estimator | ✅ Yes | ✅ Yes |
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TAB 5 — PREDICT
# ──────────────────────────────────────────────
with tab5:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🎯 Live Prediction — Enter Feature Values</div>', unsafe_allow_html=True)
    st.info(f"Adjust the top 10 features below. Values default to dataset mean. Dataset: **{dataset}**")

    means = raw_data.data.mean(axis=0)
    stds  = raw_data.data.std(axis=0)
    feat_n = list(feat_names)

    cols_a, cols_b = st.columns(2)
    user_vals = []
    for i in range(min(10, len(feat_n))):
        c = cols_a if i % 2 == 0 else cols_b
        with c:
            v = st.number_input(feat_n[i], value=float(round(means[i], 4)),
                               format="%.4f", step=float(max(stds[i]*0.05, 1e-5)), key=f"feat_{i}")
            user_vals.append(v)

    full_input = list(user_vals) + list(means[len(user_vals):])
    st.markdown("</div>", unsafe_allow_html=True)

    model_choice = st.selectbox("🌲 Choose Model for Prediction", list(results.keys()))

    if st.button("🎯  Run Prediction"):
        raw_arr   = np.array([full_input])
        scaled    = sc.transform(raw_arr)
        m         = results[model_choice]["model"]
        pred      = m.predict(scaled)[0]
        proba     = m.predict_proba(scaled)[0]
        confidence= proba[pred] * 100
        class_name= target_names[pred]

        color = COLORS[model_choice]
        st.markdown(f"""
        <div class="glass" style="border-left: 5px solid {color}; margin-top:18px; animation: fadeIn 0.5s ease;">
            <div style="font-size:1.6rem; font-weight:800; color:{color};">
                ✅ Predicted: {class_name}
            </div>
            <div style="font-size:0.9rem; color:rgba(180,255,210,0.7); margin-top:6px;">
                Model: <strong>{model_choice}</strong> · Confidence: <strong style="color:{color};">{confidence:.1f}%</strong>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:50px; height:10px; margin-top:14px; overflow:hidden;">
                <div style="width:{confidence:.1f}%; height:100%; background:{color}; border-radius:50px;"></div>
            </div>
            <div style="font-size:0.78rem; color:rgba(150,255,180,0.5); margin-top:8px;">
                {"  ·  ".join([f"{target_names[i]}: {proba[i]*100:.1f}%" for i in range(len(target_names))])}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    🌲 Ensemble Learning — Random Forest · Gradient Boosting · AdaBoost · Bagging &nbsp;|&nbsp; Scikit-Learn · Streamlit
</div>
""", unsafe_allow_html=True)
