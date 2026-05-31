import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data and isolate features (exclude player names from mathematical grouping)
df = pd.read_csv('kmeans_ready_data.csv')
X = df.drop(columns=['batter'])

# Group players into 4 archetypes using K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Calculate Variance Explained to evaluate clustering quality
kmeans_1 = KMeans(n_clusters=1, random_state=42)
kmeans_1.fit(X)
total_ss = kmeans_1.inertia_
within_ss = kmeans.inertia_
variance_explained = (total_ss - within_ss) / total_ss

# Use a Random Forest classifier to verify cluster separation and stability
X_train, X_test, y_train, y_test = train_test_split(X, df['cluster'], test_size=0.2, random_state=42)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)
separation_accuracy = classifier.score(X_test, y_test)

# Reduce stats to 2 dimensions for X/Y scatter plot visualization
pca = PCA(n_components=2, random_state=42)
pca_features = pca.fit_transform(X)
df['pca_x'] = pca_features[:, 0]
df['pca_y'] = pca_features[:, 1]

# Map numerical clusters to readable labels
cluster_labels = {
    2: 'Consistent Elite Players',
    3: 'Power-Hitters (Aggressive)',
    0: 'Dependable Anchors',
    1: 'Bowling All-Rounders'
}
df['label'] = df['cluster'].map(cluster_labels)

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = {
    'Consistent Elite Players': '#2ca02c',
    'Power-Hitters (Aggressive)': '#d62728',
    'Dependable Anchors': '#1f77b4',
    'Bowling All-Rounders': '#ff7f0e'
}

for label, group in df.groupby('label'):
    ax.scatter(group['pca_x'], group['pca_y'], label=label, color=colors[label], alpha=0.7, edgecolors='w', s=60)

# Annotate notable players on the graph for context
notable_players = ['V Kohli', 'MS Dhoni', 'CH Gayle', 'SP Narine', 'J Fraser-McGurk', 'AM Rahane', 'KD Karthik', 'R Ashwin', 'PP Chawla']
for player in notable_players:
    player_data = df[df['batter'] == player]
    if not player_data.empty:
        ax.annotate(player,
                    (player_data['pca_x'].values[0], player_data['pca_y'].values[0]),
                    textcoords="offset points",
                    xytext=(5,5),
                    ha='center', fontsize=9, fontweight='bold')

# Chart formatting
ax.set_title('IPL Player Grouping using K-Means Clustering (2008-2025)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Principal Component 1 (Overall Playstyle/Impact)', fontsize=10)
ax.set_ylabel('Principal Component 2 (Role/Batting Position Context)', fontsize=10)
ax.legend(title="Player Archetypes", loc='upper right', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)

# Overlay evaluation metrics on the bottom-left of the chart
textstr = '\n'.join((
    f'Cluster Separation Accuracy: {separation_accuracy * 100:.2f}%',
    f'Variance Explained Accuracy: {variance_explained * 100:.2f}%'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='bottom', bbox=props)

# Export graph and assigned archetypes
plt.tight_layout()
plt.savefig('player_grouping_clusters.png', dpi=100)
plt.clf()
plt.close('all')

df[['batter', 'label']].to_csv('player_assigned_archetypes.csv', index=False)
print("✅ K-Means Clustering, Accuracy Evaluations, and Visualizations complete!")
print(f"Results Generated -> Separation Stability: {separation_accuracy * 100:.2f}% | Model Variance Explained: {variance_explained * 100:.2f}%")

# Utility function for terminal lookups
def check_player_group(player_name):
    player_row = df[df['batter'] == player_name]
    print("\n================ PLAYER ARCHETYPE LOOKUP ================")
    if not player_row.empty:
        assigned_group = player_row['label'].values[0]
        print(f"Player Name: {player_name}")
        print(f"Assigned Group: {assigned_group}")
    else:
        print(f"Result: Player '{player_name}' was not found in this dataset.")
    print("=========================================================")


check_player_group('KD Karthik')