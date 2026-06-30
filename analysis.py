"""
Online Retail Customer Segmentation & Product Recommendation
Complete Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD & UNDERSTAND DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_csv('online_retail.csv', parse_dates=['InvoiceDate'])
print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")
print(f"\nSample:\n{df.head(3)}")

# ─────────────────────────────────────────────
# STEP 2: DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

raw_size = len(df)

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])
print(f"After removing missing CustomerID: {len(df)} rows (removed {raw_size - len(df)})")

# Exclude cancelled invoices
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"After removing cancellations: {len(df)} rows")

# Remove negative/zero quantities and prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
print(f"After removing invalid qty/price: {len(df)} rows")

# Add TotalAmount column
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(str)

print(f"\nFinal clean dataset: {df.shape}")
print(df.describe())

# ─────────────────────────────────────────────
# STEP 3: EDA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

fig = plt.figure(figsize=(20, 24))
fig.suptitle('Online Retail - Exploratory Data Analysis', fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# 1. Transaction Volume by Country (top 10)
ax1 = fig.add_subplot(gs[0, 0])
country_counts = df['Country'].value_counts().head(10)
bars = ax1.barh(country_counts.index[::-1], country_counts.values[::-1], color=sns.color_palette('Blues_r', 10))
ax1.set_title('Top 10 Countries by Transaction Volume', fontweight='bold')
ax1.set_xlabel('Number of Transactions')
for bar, val in zip(bars, country_counts.values[::-1]):
    ax1.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=8)

# 2. Top 10 Products by Revenue
ax2 = fig.add_subplot(gs[0, 1])
top_products = df.groupby('Description')['TotalAmount'].sum().sort_values(ascending=False).head(10)
bars2 = ax2.barh(top_products.index[::-1], top_products.values[::-1], color=sns.color_palette('Greens_r', 10))
ax2.set_title('Top 10 Products by Revenue', fontweight='bold')
ax2.set_xlabel('Total Revenue (£)')
for bar, val in zip(bars2, top_products.values[::-1]):
    ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f'£{val:,.0f}', va='center', fontsize=7)

# 3. Monthly Purchase Trend
ax3 = fig.add_subplot(gs[1, :])
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
monthly = df.groupby('YearMonth').agg(Transactions=('InvoiceNo','nunique'), Revenue=('TotalAmount','sum')).reset_index()
monthly['YearMonth'] = monthly['YearMonth'].astype(str)
ax3_twin = ax3.twinx()
ax3.bar(monthly['YearMonth'], monthly['Transactions'], alpha=0.6, color='steelblue', label='Transactions')
ax3_twin.plot(monthly['YearMonth'], monthly['Revenue'], color='darkorange', marker='o', linewidth=2, label='Revenue')
ax3.set_title('Monthly Purchase Trends: Transactions & Revenue', fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Number of Transactions', color='steelblue')
ax3_twin.set_ylabel('Revenue (£)', color='darkorange')
ax3.tick_params(axis='x', rotation=45)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 4. Monetary Distribution per Transaction
ax4 = fig.add_subplot(gs[2, 0])
trans_monetary = df.groupby('InvoiceNo')['TotalAmount'].sum()
ax4.hist(trans_monetary.clip(upper=trans_monetary.quantile(0.95)), bins=50, color='mediumpurple', edgecolor='white')
ax4.set_title('Monetary Distribution per Transaction\n(clipped at 95th percentile)', fontweight='bold')
ax4.set_xlabel('Transaction Value (£)')
ax4.set_ylabel('Count')

# 5. Monetary Distribution per Customer
ax5 = fig.add_subplot(gs[2, 1])
cust_monetary = df.groupby('CustomerID')['TotalAmount'].sum()
ax5.hist(cust_monetary.clip(upper=cust_monetary.quantile(0.95)), bins=50, color='salmon', edgecolor='white')
ax5.set_title('Monetary Distribution per Customer\n(clipped at 95th percentile)', fontweight='bold')
ax5.set_xlabel('Total Spend (£)')
ax5.set_ylabel('Count')

# 6. Top Products by Quantity
ax6 = fig.add_subplot(gs[3, 0])
top_qty = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(8)
ax6.pie(top_qty.values, labels=[l[:20] for l in top_qty.index], autopct='%1.1f%%',
        colors=sns.color_palette('Set3', 8), startangle=90)
ax6.set_title('Top 8 Products by Quantity Sold', fontweight='bold')

# 7. Day of Week pattern
ax7 = fig.add_subplot(gs[3, 1])
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_rev = df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(day_order)
ax7.bar(day_order, day_rev.values, color=sns.color_palette('coolwarm', 7))
ax7.set_title('Revenue by Day of Week', fontweight='bold')
ax7.set_xlabel('Day')
ax7.set_ylabel('Total Revenue (£)')
ax7.tick_params(axis='x', rotation=30)

plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA plots saved.")

# ─────────────────────────────────────────────
# STEP 4: RFM FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: RFM FEATURE ENGINEERING")
print("=" * 60)

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Snapshot date: {snapshot_date.date()}")

rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalAmount', 'sum')
).reset_index()

print(f"\nRFM shape: {rfm.shape}")
print(rfm.describe().round(2))

# RFM Distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('RFM Distributions', fontsize=15, fontweight='bold')

for ax, col, color in zip(axes, ['Recency','Frequency','Monetary'], ['steelblue','seagreen','tomato']):
    ax.hist(rfm[col].clip(upper=rfm[col].quantile(0.95)), bins=40, color=color, edgecolor='white')
    ax.set_title(f'{col} Distribution', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.axvline(rfm[col].median(), color='black', linestyle='--', linewidth=1.5, label=f'Median: {rfm[col].median():.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig('rfm_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("RFM distribution plots saved.")

# ─────────────────────────────────────────────
# STEP 5: CLUSTERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: CLUSTERING")
print("=" * 60)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

# Elbow curve + Silhouette scores
inertia = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Cluster Selection Metrics', fontsize=14, fontweight='bold')

axes[0].plot(list(K_range), inertia, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Chosen K=4')
axes[0].set_title('Elbow Method (Inertia)', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].legend()

axes[1].plot(list(K_range), sil_scores, 'rs-', linewidth=2, markersize=8)
axes[1].axvline(x=4, color='blue', linestyle='--', alpha=0.7, label='Chosen K=4')
axes[1].set_title('Silhouette Score', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Best Silhouette Score (K=4): {sil_scores[2]:.4f}")

# Final model with K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Analyze cluster profiles
cluster_summary = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(2)
print("\nCluster RFM Averages:")
print(cluster_summary)

# Label clusters based on RFM averages
def label_cluster(row):
    r, f, m = row['Recency'], row['Frequency'], row['Monetary']
    r_med = rfm['Recency'].median()
    f_med = rfm['Frequency'].median()
    m_med = rfm['Monetary'].median()
    
    if r < r_med and f >= f_med and m >= m_med:
        return 'High-Value'
    elif f >= f_med and m >= m_med:
        return 'Regular'
    elif r > r_med and f < f_med:
        return 'At-Risk'
    else:
        return 'Occasional'

# Map cluster numbers to labels via averages
cluster_labels_map = {}
for cluster_id in range(4):
    subset = rfm[rfm['Cluster'] == cluster_id]
    avg = subset[['Recency','Frequency','Monetary']].mean()
    r_med = rfm['Recency'].median()
    f_med = rfm['Frequency'].median()
    m_med = rfm['Monetary'].median()
    
    if avg['Recency'] < r_med and avg['Frequency'] >= f_med and avg['Monetary'] >= m_med:
        cluster_labels_map[cluster_id] = 'High-Value'
    elif avg['Frequency'] >= f_med and avg['Monetary'] >= m_med:
        cluster_labels_map[cluster_id] = 'Regular'
    elif avg['Recency'] > r_med and avg['Frequency'] < f_med:
        cluster_labels_map[cluster_id] = 'At-Risk'
    else:
        cluster_labels_map[cluster_id] = 'Occasional'

rfm['Segment'] = rfm['Cluster'].map(cluster_labels_map)
print("\nCluster → Segment mapping:", cluster_labels_map)
print("\nSegment distribution:")
print(rfm['Segment'].value_counts())

# Cluster profile plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Customer Cluster Profiles (RFM Averages by Segment)', fontsize=14, fontweight='bold')

colors = {'High-Value': '#2ecc71', 'Regular': '#3498db', 'Occasional': '#f39c12', 'At-Risk': '#e74c3c'}
seg_summary = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean()

for ax, metric in zip(axes, ['Recency', 'Frequency', 'Monetary']):
    seg_colors = [colors.get(s, 'gray') for s in seg_summary.index]
    bars = ax.bar(seg_summary.index, seg_summary[metric], color=seg_colors, edgecolor='white', linewidth=1.5)
    ax.set_title(f'Average {metric} by Segment', fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, seg_summary[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()

# 2D PCA scatter of clusters
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
for seg, color in colors.items():
    mask = rfm['Segment'] == seg
    ax.scatter(rfm_pca[mask, 0], rfm_pca[mask, 1], c=color, label=seg, alpha=0.7, s=40, edgecolors='none')
ax.set_title('Customer Segments — PCA Scatter Plot', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.legend(title='Segment', fontsize=10)
plt.tight_layout()
plt.savefig('cluster_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Cluster plots saved.")

# ─────────────────────────────────────────────
# STEP 6: RECOMMENDATION SYSTEM
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: RECOMMENDATION SYSTEM (Item-Based Collaborative Filtering)")
print("=" * 60)

# Create Customer-Product matrix
pivot = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)
print(f"Customer-Product matrix shape: {pivot.shape}")

# Compute cosine similarity between products (columns)
from sklearn.metrics.pairwise import cosine_similarity
product_matrix = pivot.T  # Products as rows
similarity = cosine_similarity(product_matrix)
similarity_df = pd.DataFrame(similarity, index=product_matrix.index, columns=product_matrix.index)

# Test recommendation
def get_recommendations(product_name, sim_df, top_n=5):
    if product_name not in sim_df.index:
        return []
    sim_scores = sim_df[product_name].drop(product_name).sort_values(ascending=False)
    return list(sim_scores.head(top_n).index)

test_product = list(similarity_df.index)[0]
recs = get_recommendations(test_product, similarity_df)
print(f"\nTest recommendations for '{test_product}':")
for i, r in enumerate(recs, 1):
    print(f"  {i}. {r}")

# Product similarity heatmap (top 10 products)
top10 = df['Description'].value_counts().head(10).index.tolist()
sim_top10 = similarity_df.loc[top10, top10]

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(sim_top10, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            xticklabels=[l[:25] for l in top10], yticklabels=[l[:25] for l in top10],
            linewidths=0.5, cbar_kws={'label': 'Cosine Similarity'})
ax.set_title('Product Similarity Heatmap (Top 10 Products)', fontsize=13, fontweight='bold')
plt.xticks(rotation=40, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('similarity_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Similarity heatmap saved.")

# ─────────────────────────────────────────────
# SAVE MODELS & ARTIFACTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING MODELS & ARTIFACTS")
print("=" * 60)

joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(similarity_df, 'similarity_df.pkl')
joblib.dump(cluster_labels_map, 'cluster_labels_map.pkl')

# Save RFM with segments
rfm.to_csv('rfm_segmented.csv', index=False)

# Save product list for app
product_list = list(similarity_df.index)
joblib.dump(product_list, 'product_list.pkl')

print("All models and artifacts saved!")
print("\n✅ ANALYSIS COMPLETE")
print(f"  • Customers segmented: {len(rfm)}")
print(f"  • Products in recommender: {len(product_list)}")
print(f"  • Silhouette score (K=4): {sil_scores[2]:.4f}")
