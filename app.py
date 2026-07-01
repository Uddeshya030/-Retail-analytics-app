"""
Online Retail — Customer Segmentation & Product Recommendation
Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.75; font-size: 1rem; }

    .metric-card {
        background: #f8f9fa;
        border-left: 5px solid #4C72B0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-card.green { border-color: #55A868; }
    .metric-card.red   { border-color: #C44E52; }
    .metric-card.purple{ border-color: #8172B2; }
    .metric-card h3 { margin: 0; font-size: 1.6rem; }
    .metric-card p  { margin: 0; color: #666; font-size: 0.85rem; }

    .rec-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .rec-rank {
        background: #4C72B0;
        color: white;
        border-radius: 50%;
        width: 2rem; height: 2rem;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 0.85rem; flex-shrink: 0;
    }

    .segment-badge {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .badge-high  { background:#ffeaea; color:#c0392b; border:2px solid #c0392b; }
    .badge-reg   { background:#eaf1ff; color:#2566b8; border:2px solid #2566b8; }
    .badge-occ   { background:#eafff1; color:#1a7a3e; border:2px solid #1a7a3e; }
    .badge-risk  { background:#fff7ea; color:#b86a00; border:2px solid #b86a00; }

    div[data-testid="stSidebar"] { background: #f0f2f6; }
    .stButton>button {
        width: 100%; padding: 0.65rem;
        font-size: 1rem; font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────────
BASE = "."

@st.cache_resource
def load_models():
    scaler   = joblib.load(f"{BASE}/scaler.pkl")
    kmeans   = joblib.load(f"{BASE}/kmeans_model.pkl")
    sim_df   = joblib.load(f"{BASE}/similarity_df.pkl")
    if True:
        label_map = joblib.load(f"{BASE}/cluster_labels_map.pkl")
    rfm = pd.read_csv(f"{BASE}/rfm_segmented.csv")
    return scaler, kmeans, sim_df, label_map, rfm

scaler, kmeans, sim_df, label_map, rfm = load_models()

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛍️ Online Retail Analytics</h1>
    <p>Customer Segmentation · Product Recommendations · RFM Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=70)
    st.markdown("## Navigation")
    page = st.radio("", [
        "📊 Dashboard Overview",
        "🎯 Product Recommendations",
        "👤 Customer Segmentation",
        "📈 RFM Analysis",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Dataset Summary**")
    total_customers = rfm["CustomerID"].nunique()
    st.metric("Customers", f"{total_customers:,}")
    st.metric("Segments", "4")
    st.metric("Products", f"{len(sim_df.columns):,}")

# ═══════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    st.markdown("## 📈 Dashboard Overview")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    seg_counts = rfm["Segment"].value_counts().to_dict()
    with c1:
        st.metric("🏆 High-Value", seg_counts.get('High-Value', 0))
    with c2:
        st.metric("🔄 Regular", seg_counts.get('Regular', 0))
    with c3:
        st.metric("🛒 Occasional", seg_counts.get('Occasional', 0))
    with c4:
        st.metric("⚠️ At-Risk", seg_counts.get('At-Risk', 0))

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Segment Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = {"High-Value": "#C44E52", "Regular": "#4C72B0",
                  "Occasional": "#55A868", "At-Risk": "#8172B2"}
        seg_counts_sorted = pd.Series(seg_counts).sort_values(ascending=False)
        bars = ax.bar(seg_counts_sorted.index, seg_counts_sorted.values,
                      color=[colors[s] for s in seg_counts_sorted.index],
                      edgecolor="white", linewidth=1.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    str(int(bar.get_height())), ha="center", fontsize=11, fontweight="bold")
        ax.set_title("Customers per Segment", fontweight="bold", fontsize=13)
        ax.set_ylabel("Customer Count")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### RFM Profile by Segment")
        fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
        for ax, metric in zip(axes, ["Recency", "Frequency", "Monetary"]):
            data = rfm.groupby("Segment")[metric].mean().sort_values()
            ax.barh(data.index, data.values,
                    color=[colors.get(s, "gray") for s in data.index])
            ax.set_xlabel(metric, fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            for i, (val, name) in enumerate(zip(data.values, data.index)):
                ax.text(val * 0.02, i, f"{val:.1f}", va="center", fontsize=8)
        axes[0].set_title("Average RFM by Segment", fontweight="bold", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # RFM distributions
    st.markdown("### RFM Value Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_colors = {"Recency": "#C44E52", "Frequency": "#4C72B0", "Monetary": "#55A868"}
    for ax, (metric, color) in zip(axes, metric_colors.items()):
        for seg, grp in rfm.groupby("Segment"):
            ax.hist(grp[metric], bins=25, alpha=0.6,
                    color=colors.get(seg, "gray"), label=seg, edgecolor="white")
        ax.set_title(f"{metric} Distribution by Segment", fontweight="bold")
        ax.set_xlabel(metric)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Saved EDA images
    st.markdown("---")
    st.markdown("### 📸 Analysis Highlights")
    imgs = [
        ("eda_plots.png", "EDA Overview"),
        ("rfm_distributions.png", "RFM Distributions"),
        ("elbow_curve.png", "Elbow Curve"),
        ("cluster_profiles.png", "Cluster Profiles"),
        ("cluster_scatter.png", "Cluster Scatter"),
        ("similarity_heatmap.png", "Similarity Heatmap"),
    ]
    img_cols = st.columns(2)
    for i, (fname, caption) in enumerate(imgs):
        path = f"{BASE}/{fname}"
        if os.path.exists(path):
            img_cols[i % 2].image(path, caption=caption, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 2: PRODUCT RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════
elif page == "🎯 Product Recommendations":
    st.markdown("## 🎯 Product Recommendation System")
    st.markdown(
        "Uses **Item-Based Collaborative Filtering** (cosine similarity) "
        "to recommend the 5 most similar products based on shared purchase history."
    )

    all_products = sorted(sim_df.columns.tolist())

    col1, col2 = st.columns([3, 1])
    with col1:
        product_input = st.selectbox(
            "Select a Product", options=all_products,
            help="Start typing to search products"
        )
    with col2:
        top_n = st.selectbox("# Recommendations", [3, 5, 7, 10], index=1)

    if st.button("🔍  Get Recommendations", type="primary"):
        if product_input:
            scores = sim_df[product_input].drop(product_input).sort_values(ascending=False)
            recommendations = scores.head(top_n)

            st.markdown(f"### Recommendations for: *{product_input}*")

            for rank, (prod, score) in enumerate(recommendations.items(), 1):
                badge_color = "#4C72B0" if score > 0.5 else "#8172B2" if score > 0.3 else "#aaa"
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-rank" style="background:{badge_color};">{rank}</div>
                    <div style="flex:1">
                        <strong>{prod}</strong>
                    </div>
                    <div style="color:#666; font-size:0.85rem;">
                        Similarity: <strong>{score:.3f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Bar chart of scores
            st.markdown("#### Similarity Scores")
            fig, ax = plt.subplots(figsize=(9, 4))
            bars = ax.barh(recommendations.index[::-1],
                           recommendations.values[::-1],
                           color="#4C72B0", edgecolor="white")
            for bar in bars:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{bar.get_width():.3f}", va="center", fontsize=9)
            ax.set_xlabel("Cosine Similarity Score")
            ax.set_title("Product Similarity Scores", fontweight="bold")
            ax.set_xlim(0, min(1.1, recommendations.max() * 1.3))
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Please select a product.")

    # Product explorer
    with st.expander("📦 Browse All Products"):
        st.dataframe(pd.DataFrame({"Product Name": all_products}), height=300)

# ═══════════════════════════════════════════════════════════════════
# PAGE 3: CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════
elif page == "👤 Customer Segmentation":
    st.markdown("## 👤 Customer Segmentation Predictor")
    st.markdown(
        "Enter a customer's **RFM values** to predict their segment using "
        "the trained **K-Means clustering model**."
    )

    # Segment reference table
    with st.expander("📖 Segment Reference Guide"):
        guide = pd.DataFrame({
            "Segment":     ["High-Value 🏆", "Regular 🔄", "Occasional 🛒", "At-Risk ⚠️"],
            "Recency":     ["Very recent (<30 days)", "Moderate (30–90 days)",
                            "Longer ago (>90 days)", "Very old (>180 days)"],
            "Frequency":   ["Very high (>30 orders)", "Moderate (10–20 orders)",
                            "Low (5–10 orders)", "Low (<5 orders)"],
            "Monetary":    ["High spender (>£400)", "Medium (£200–£400)",
                            "Low (£100–£200)", "Minimal (<£100)"],
            "Strategy":    ["Reward & retain", "Upsell & cross-sell",
                            "Re-engage campaigns", "Win-back urgently"],
        })
        st.dataframe(guide, hide_index=True, use_container_width=True)

    st.markdown("---")
    col_inp, col_res = st.columns([1, 1])

    with col_inp:
        st.markdown("### Enter RFM Values")

        recency = st.number_input(
            "📅 Recency (days since last purchase)",
            min_value=1, max_value=730, value=30, step=1,
            help="Smaller = more recent = better"
        )
        frequency = st.number_input(
            "🔁 Frequency (number of transactions)",
            min_value=1, max_value=200, value=10, step=1,
            help="How many times they've purchased"
        )
        monetary = st.number_input(
            "💰 Monetary (total spend in £)",
            min_value=1.0, max_value=50000.0, value=300.0, step=10.0,
            help="Total revenue generated by customer"
        )

        predict_btn = st.button("🚀  Predict Customer Segment", type="primary")

    with col_res:
        st.markdown("### Prediction Result")

        if predict_btn:
            X = np.array([[recency, frequency, monetary]])
            X_scaled = scaler.transform(X)
            cluster_id = int(kmeans.predict(X_scaled)[0])
            segment = label_map.get(cluster_id, "Unknown")

            badge_class = {
                "High-Value": "badge-high",
                "Regular":    "badge-reg",
                "Occasional": "badge-occ",
                "At-Risk":    "badge-risk",
            }.get(segment, "badge-reg")

            emoji = {"High-Value":"🏆","Regular":"🔄","Occasional":"🛒","At-Risk":"⚠️"}.get(segment,"")

            st.markdown(f"""
            <div style="text-align:center; padding:2rem; background:#f8f9fa;
                        border-radius:12px; border:1px solid #dee2e6;">
                <div style="font-size:3.5rem; margin-bottom:0.5rem;">{emoji}</div>
                <div style="font-size:1rem; color:#666; margin-bottom:0.5rem;">
                    Predicted Cluster: <strong>#{cluster_id}</strong>
                </div>
                <span class="segment-badge {badge_class}">{segment}</span>
            </div>
            """, unsafe_allow_html=True)

            # Strategy card
            strategies = {
                "High-Value":  ("💎 VIP Treatment", "#ffeaea",
                    "Offer exclusive loyalty rewards, early access to new products, "
                    "and dedicated account management. These customers drive maximum revenue."),
                "Regular":     ("📈 Growth Opportunity", "#eaf1ff",
                    "Encourage upsells with personalized recommendations. "
                    "Send curated bundles and loyalty points to increase basket size."),
                "Occasional":  ("📢 Re-Engagement", "#eafff1",
                    "Launch targeted email campaigns with promotions. "
                    "Highlight new products and offer a welcome-back discount."),
                "At-Risk":     ("🚨 Win-Back Campaign", "#fff7ea",
                    "Act urgently with a compelling win-back offer. "
                    "Survey to understand churn reason and address pain points."),
            }
            title, bg, desc = strategies.get(segment, ("", "#fff", ""))
            st.markdown(f"""
            <div style="background:{bg}; padding:1.2rem; border-radius:10px;
                        margin-top:1.2rem; border:1px solid #dee2e6;">
                <strong>{title}</strong><br>
                <span style="color:#444; font-size:0.9rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

            # Gauge / radar: how this customer compares to cluster averages
            st.markdown("#### How You Compare to Cluster Averages")
            cluster_avgs = rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean()
            if segment in cluster_avgs.index:
                avg = cluster_avgs.loc[segment]
                compare = pd.DataFrame({
                    "Metric":  ["Recency","Frequency","Monetary"],
                    "You":     [recency, frequency, monetary],
                    "Cluster Avg": [avg["Recency"], avg["Frequency"], avg["Monetary"]],
                })
                fig, axes = plt.subplots(1, 3, figsize=(10, 3))
                for ax, row in zip(axes, compare.itertuples()):
                    vals = [row.You, getattr(row, "Cluster Avg")]
                    bars = ax.bar(["You", "Avg"], vals,
                                  color=["#4C72B0", "#aab7cc"], edgecolor="white")
                    ax.set_title(row.Metric, fontweight="bold", fontsize=10)
                    for b in bars:
                        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                                f"{b.get_height():.0f}", ha="center", fontsize=9)
                    ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.info("👈 Fill in the RFM values and click **Predict Customer Segment**")

    # Segment explorer
    st.markdown("---")
    st.markdown("### 🔍 Explore Existing Customers")
    selected_seg = st.selectbox("Filter by Segment",
                                ["All"] + sorted(rfm["Segment"].unique().tolist()))
    filtered = rfm if selected_seg == "All" else rfm[rfm["Segment"] == selected_seg]
    st.dataframe(
        filtered[["CustomerID","Recency","Frequency","Monetary","Segment"]]
        .sort_values("Monetary", ascending=False)
        .reset_index(drop=True),
        height=300,
        use_container_width=True,
    )
    st.caption(f"Showing {len(filtered):,} customers")
    # ═══════════════════════════════════════════════════════════
# PAGE 4: RFM ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "📈 RFM Analysis":
    st.markdown("## 📈 RFM Analysis")
    st.markdown("*Recency, Frequency, Monetary Analysis by Segment*")

    # RFM Summary by Segment
    st.markdown("### 📊 RFM Summary by Segment")
    seg_summary = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].agg(['mean','min','max']).round(2)
    st.dataframe(seg_summary, use_container_width=True)

    # Segment wise counts
    st.markdown("### 👥 Customer Count by Segment")
    seg_counts = rfm['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    st.dataframe(seg_counts, use_container_width=True)

    # RFM Distribution plots
    st.markdown("### 📉 RFM Distributions")
    if os.path.exists(f"{BASE}/rfm_distributions.png"):
        st.image(f"{BASE}/rfm_distributions.png", use_container_width=True)
# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Online Retail Analytics Dashboard · RFM Segmentation + Collaborative Filtering"
    "</div>",
    unsafe_allow_html=True,
)
