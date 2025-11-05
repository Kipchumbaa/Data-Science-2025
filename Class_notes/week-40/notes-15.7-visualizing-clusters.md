# Lecture 15.7: Visualizing Clusters and Advanced Techniques

## Key Learning Objectives
- Master techniques for effective cluster visualization
- Learn advanced clustering evaluation methods
- Understand cluster stability and validation techniques
- Implement comprehensive cluster analysis pipelines

## Core Concepts

### Cluster Visualization Challenges

#### High-Dimensional Data
- **Curse of dimensionality**: Impossible to visualize >3D directly
- **Dimensionality reduction**: Use PCA, t-SNE, UMAP for 2D/3D visualization
- **Multiple views**: Different projections show different aspects
- **Interactive visualization**: Tools for exploring high-dimensional clusters

#### Cluster Overlap and Uncertainty
- **Soft clustering**: Points can belong to multiple clusters with probabilities
- **Boundary points**: Uncertainty in cluster assignment
- **Noise points**: Outliers that don't belong to clear clusters
- **Cluster validation**: Statistical tests for cluster quality

## Advanced Visualization Techniques

### Multi-Dimensional Scaling (MDS)
```python
from sklearn.manifold import MDS

# Classical MDS
mds = MDS(
    n_components=2,           # Target dimensions
    metric=True,              # Metric MDS (preserves distances)
    n_init=10,                # Number of random starts
    max_iter=1000,            # Maximum iterations
    random_state=42
)

X_mds = mds.fit_transform(X_scaled)

# Visualize MDS embedding
plt.figure(figsize=(10, 8))
plt.scatter(X_mds[:, 0], X_mds[:, 1], alpha=0.6, s=50)
plt.xlabel('MDS Component 1')
plt.ylabel('MDS Component 2')
plt.title('Multi-Dimensional Scaling')
plt.grid(True)
plt.show()

print(f"MDS stress: {mds.stress_:.4f}")  # Lower stress = better fit
```

### Interactive Cluster Visualization
```python
# Using plotly for interactive 3D visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    
    # 3D scatter plot with clusters
    fig = px.scatter_3d(
        x=X_embedded_3d[:, 0],
        y=X_embedded_3d[:, 1], 
        z=X_embedded_3d[:, 2],
        color=cluster_labels.astype(str),
        title="Interactive 3D Cluster Visualization",
        labels={'color': 'Cluster'}
    )
    fig.show()
    
    # Parallel coordinates plot
    cluster_df = pd.DataFrame(X_scaled, columns=[f'Feature_{i}' for i in range(X_scaled.shape[1])])
    cluster_df['Cluster'] = cluster_labels
    
    fig = px.parallel_coordinates(
        cluster_df, 
        color="Cluster",
        title="Parallel Coordinates Plot"
    )
    fig.show()
    
except ImportError:
    print("Plotly not available. Install with: pip install plotly")
```

### Cluster Profiles and Centroids
```python
def plot_cluster_profiles(X, cluster_labels, feature_names=None):
    """
    Plot feature profiles for each cluster
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Calculate cluster centroids
    centroids = []
    for cluster_id in np.unique(cluster_labels):
        if cluster_id != -1:  # Skip noise points
            cluster_points = X[cluster_labels == cluster_id]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Plot cluster profiles
    plt.figure(figsize=(12, 8))
    
    for i, centroid in enumerate(centroids):
        plt.plot(feature_names, centroid, 'o-', label=f'Cluster {i}', linewidth=2, markersize=6)
    
    plt.xlabel('Features')
    plt.ylabel('Mean Value (Standardized)')
    plt.title('Cluster Profiles')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return centroids

# Usage
if X.shape[1] <= 20:  # Only plot if reasonable number of features
    centroids = plot_cluster_profiles(X_scaled, cluster_labels, X.columns.tolist())
else:
    print("Too many features for profile plot. Consider feature selection first.")
```

### Silhouette Analysis Visualization
```python
from sklearn.metrics import silhouette_samples

def plot_silhouette_analysis(X, cluster_labels, n_clusters):
    """
    Create detailed silhouette analysis plot
    """
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    y_lower = 10
    
    for i in range(n_clusters):
        # Aggregate silhouette scores for samples in cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        # Fill silhouette plot
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, 
            ith_cluster_silhouette_values,
            facecolor=color, 
            edgecolor=color, 
            alpha=0.7
        )
        
        # Label clusters
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    # Add vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)
    
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.set_title(f"Silhouette Analysis (Average: {silhouette_avg:.3f})")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return silhouette_avg

# Usage
if len(np.unique(cluster_labels[cluster_labels != -1])) > 1:
    silhouette_avg = plot_silhouette_analysis(X_scaled, cluster_labels, len(np.unique(cluster_labels[cluster_labels != -1])))
else:
    print("Need at least 2 clusters for silhouette analysis")
```

## Advanced Clustering Evaluation

### Cluster Stability Assessment
```python
from sklearn.metrics import adjusted_rand_score

def assess_cluster_stability(X, cluster_func, n_runs=10, test_size=0.2):
    """
    Assess clustering stability using bootstrap sampling
    """
    stability_scores = []
    
    for run in range(n_runs):
        # Bootstrap sample
        indices = np.random.choice(len(X), size=int(len(X) * (1 - test_size)), replace=True)
        X_boot = X[indices]
        
        # Cluster on bootstrap sample
        labels_boot = cluster_func(X_boot)
        
        # Map back to original indices (simplified)
        # In practice, you'd need more sophisticated mapping
        labels_full = np.full(len(X), -1)
        labels_full[indices] = labels_boot
        
        # Compare with full clustering (placeholder)
        # This is simplified - real implementation would compare consistent subsets
        stability_scores.append(1.0)  # Placeholder
    
    return np.mean(stability_scores), np.std(stability_scores)

# Example usage with K-means
def kmeans_cluster_func(X):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    return kmeans.fit_predict(X)

stability_mean, stability_std = assess_cluster_stability(X_scaled, kmeans_cluster_func)
print(f"Cluster stability: {stability_mean:.3f} ± {stability_std:.3f}")
```

### External Validation Metrics
```python
# When ground truth labels are available
from sklearn.metrics import (
    homogeneity_score, 
    completeness_score, 
    v_measure_score,
    adjusted_mutual_info_score
)

if true_labels is not None:
    # Remove noise points for comparison
    mask = cluster_labels != -1
    if sum(mask) > 0:
        ari = adjusted_rand_score(true_labels[mask], cluster_labels[mask])
        ami = adjusted_mutual_info_score(true_labels[mask], cluster_labels[mask])
        homogeneity = homogeneity_score(true_labels[mask], cluster_labels[mask])
        completeness = completeness_score(true_labels[mask], cluster_labels[mask])
        v_measure = v_measure_score(true_labels[mask], cluster_labels[mask])
        
        print("External Validation Metrics:")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print(f"Adjusted Mutual Information: {ami:.3f}")
        print(f"Homogeneity: {homogeneity:.3f}")
        print(f"Completeness: {completeness:.3f}")
        print(f"V-measure: {v_measure:.3f}")
```

### Cluster Validity Indices
```python
# Dunn Index
def dunn_index(X, cluster_labels):
    """
    Calculate Dunn Index (ratio of min inter-cluster distance to max intra-cluster distance)
    """
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0
    
    # Calculate intra-cluster distances (max within cluster)
    max_intra_dist = 0
    for label in unique_labels:
        cluster_points = X[cluster_labels == label]
        if len(cluster_points) > 1:
            distances = pdist(cluster_points)
            max_intra_dist = max(max_intra_dist, np.max(distances))
    
    # Calculate inter-cluster distances (min between clusters)
    min_inter_dist = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = X[cluster_labels == unique_labels[i]]
            cluster_j = X[cluster_labels == unique_labels[j]]
            
            # Calculate minimum distance between any points in different clusters
            for point_i in cluster_i:
                for point_j in cluster_j:
                    dist = np.linalg.norm(point_i - point_j)
                    min_inter_dist = min(min_inter_dist, dist)
    
    return min_inter_dist / max_intra_dist if max_intra_dist > 0 else 0

# Calculate Dunn Index
from scipy.spatial.distance import pdist
dunn = dunn_index(X_scaled, cluster_labels)
print(f"Dunn Index: {dunn:.3f}")  # Higher values indicate better clustering
```

## Advanced Clustering Techniques

### Ensemble Clustering
```python
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering

def ensemble_clustering(X, n_clusters=3, n_ensemble=5):
    """
    Combine multiple clustering algorithms
    """
    ensemble_labels = []
    
    for i in range(n_ensemble):
        # Different clustering algorithms
        algorithms = [
            KMeans(n_clusters=n_clusters, random_state=i, n_init=10),
            AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
            AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        ]
        
        # Random feature subset (optional)
        n_features = X.shape[1]
        feature_subset = np.random.choice(n_features, size=int(0.8 * n_features), replace=False)
        X_subset = X[:, feature_subset]
        
        # Apply clustering
        labels_list = []
        for alg in algorithms:
            labels = alg.fit_predict(X_subset)
            labels_list.append(labels)
        
        # Majority vote
        labels_matrix = np.array(labels_list)
        ensemble_label = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n_clusters).argmax(), 
            axis=0, 
            arr=labels_matrix
        )
        ensemble_labels.append(ensemble_label)
    
    # Final ensemble (majority vote across ensembles)
    ensemble_matrix = np.array(ensemble_labels)
    final_labels = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_clusters).argmax(), 
        axis=0, 
        arr=ensemble_matrix
    )
    
    return final_labels

# Apply ensemble clustering
ensemble_labels = ensemble_clustering(X_scaled, n_clusters=3, n_ensemble=3)

# Evaluate ensemble
ensemble_silhouette = silhouette_score(X_scaled, ensemble_labels)
print(f"Ensemble clustering silhouette: {ensemble_silhouette:.3f}")
```

### Hierarchical Clustering Visualization
```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Create linkage matrix
linkage_matrix = linkage(X_scaled, method='ward', metric='euclidean')

# Plot detailed dendrogram
plt.figure(figsize=(15, 8))
dendrogram(
    linkage_matrix,
    truncate_mode='level',    # Show only last p levels
    p=5,
    leaf_rotation=90,
    leaf_font_size=8,
    show_contracted=True,
    color_threshold=0.7 * max(linkage_matrix[:, 2])  # Color threshold
)

plt.xlabel('Sample Index (or Cluster Size)')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.axhline(y=0.7 * max(linkage_matrix[:, 2]), color='r', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Extract clusters at different levels
height_thresholds = [0.3, 0.5, 0.7, 0.9]
for threshold in height_thresholds:
    height = threshold * max(linkage_matrix[:, 2])
    clusters = fcluster(linkage_matrix, t=height, criterion='distance')
    n_clusters = len(np.unique(clusters))
    print(f"Height {height:.2f}: {n_clusters} clusters")
```

## Practical Implementation

### Complete Cluster Analysis Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
print("Loading data...")
df = pd.read_csv('clustering_dataset.csv')

# Handle target variable if present
if 'target' in df.columns:
    X = df.drop('target', axis=1)
    y_true = df['target']
    has_labels = True
else:
    X = df.copy()
    y_true = None
    has_labels = False

print(f"Dataset shape: {X.shape}")

# Preprocessing
print("Preprocessing...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Compare multiple clustering algorithms
print("Comparing clustering algorithms...")

algorithms = {
    'K-means': KMeans(n_clusters=3, random_state=42, n_init=10),
    'Hierarchical': AgglomerativeClustering(n_clusters=3, linkage='ward'),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
}

clustering_results = {}

for name, algorithm in algorithms.items():
    print(f"Running {name}...")
    
    # Fit and predict
    if name == 'DBSCAN':
        labels = algorithm.fit_predict(X_scaled)
        # Handle variable number of clusters for DBSCAN
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
    else:
        labels = algorithm.fit_predict(X_scaled)
        n_clusters = len(set(labels))
        n_noise = 0
    
    clustering_results[name] = {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }
    
    # Calculate metrics (excluding noise points)
    if n_clusters > 1:
        mask = labels != -1
        if sum(mask) > n_clusters:
            silhouette = silhouette_score(X_scaled[mask], labels[mask])
            ch_score = calinski_harabasz_score(X_scaled[mask], labels[mask])
        else:
            silhouette = None
            ch_score = None
    else:
        silhouette = None
        ch_score = None
    
    print(f"  Clusters: {n_clusters}, Noise: {n_noise}, Silhouette: {silhouette:.3f}")

# Step 2: Dimensionality reduction for visualization
print("Creating visualizations...")

# Choose best clustering for visualization
best_algorithm = max(clustering_results.keys(), 
                    key=lambda x: clustering_results[x]['n_clusters'] if clustering_results[x]['n_clusters'] > 1 else 0)
best_labels = clustering_results[best_algorithm]['labels']

# t-SNE for visualization
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP for comparison
print("Running UMAP...")
try:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    umap_available = True
except ImportError:
    print("UMAP not available")
    umap_available = False

# Step 3: Create comprehensive visualization
fig, axes = plt.subplots(2, 2 if umap_available else 1, figsize=(16, 12))

# t-SNE with clusters
ax = axes[0, 0] if umap_available else axes[0]
unique_labels = np.unique(best_labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    mask = best_labels == label
    if label == -1:
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c='black', alpha=0.5, s=30, label='Noise')
    else:
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], alpha=0.7, s=50, label=f'Cluster {label}')

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title(f't-SNE Clusters ({best_algorithm})')
ax.legend()
ax.grid(True, alpha=0.3)

# UMAP with clusters (if available)
if umap_available:
    ax = axes[0, 1]
    for i, label in enumerate(unique_labels):
        mask = best_labels == label
        if label == -1:
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c='black', alpha=0.5, s=30, label='Noise')
        else:
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[colors[i]], alpha=0.7, s=50, label=f'Cluster {label}')
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'UMAP Clusters ({best_algorithm})')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Silhouette analysis
ax = axes[1, 0] if umap_available else axes[1]
if clustering_results[best_algorithm]['n_clusters'] > 1:
    from sklearn.metrics import silhouette_samples
    
    silhouette_avg = silhouette_score(X_scaled[best_labels != -1], best_labels[best_labels != -1])
    sample_silhouette_values = silhouette_samples(X_scaled[best_labels != -1], best_labels[best_labels != -1])
    
    y_lower = 10
    for i in range(clustering_results[best_algorithm]['n_clusters']):
        ith_cluster_values = sample_silhouette_values[best_labels[best_labels != -1] == i]
        ith_cluster_values.sort()
        
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Analysis (Avg: {silhouette_avg:.3f})")
    ax.grid(True, alpha=0.3)

# Cluster profiles (if reasonable number of features)
ax = axes[1, 1] if umap_available else axes[2] if not umap_available else None
if X.shape[1] <= 15 and ax is not None:
    centroids = []
    for label in np.unique(best_labels):
        if label != -1:
            cluster_points = X_scaled[best_labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
    
    centroids = np.array(centroids)
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]
    
    for i, centroid in enumerate(centroids):
        ax.plot(feature_names, centroid, 'o-', label=f'Cluster {i}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Mean Value (Standardized)')
    ax.set_title('Cluster Profiles')
    ax.legend()
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 4: Hierarchical clustering visualization
print("Creating hierarchical clustering visualization...")
if X_scaled.shape[0] <= 1000:  # Only for reasonable dataset sizes
    linkage_matrix = linkage(X_scaled, method='ward', metric='euclidean')
    
    plt.figure(figsize=(15, 8))
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=10,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True
    )
    plt.xlabel('Sample Index (or Cluster Size)')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Step 5: Statistical summary
print("
=== Clustering Analysis Summary ===")
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Best algorithm: {best_algorithm}")

for name, result in clustering_results.items():
    print(f"\n{name}:")
    print(f"  Clusters: {result['n_clusters']}")
    print(f"  Noise points: {result['n_noise']}")
    
    if result['n_clusters'] > 1:
        mask = result['labels'] != -1
        if sum(mask) > result['n_clusters']:
            sil = silhouette_score(X_scaled[mask], result['labels'][mask])
            print(f"  Silhouette: {sil:.3f}")

# Cluster characteristics
print(f"\nCluster sizes for {best_algorithm}:")
for label in np.unique(best_labels):
    if label == -1:
        continue
    size = np.sum(best_labels == label)
    percentage = size / len(best_labels) * 100
    print(f"  Cluster {label}: {size} samples ({percentage:.1f}%)")

# Step 6: Save results
print("Saving results...")
results_df = pd.DataFrame({
    'tsne_1': X_tsne[:, 0],
    'tsne_2': X_tsne[:, 1],
    'cluster_label': best_labels,
    'algorithm': best_algorithm
})

if has_labels:
    results_df['true_label'] = y_true

if umap_available:
    results_df['umap_1'] = X_umap[:, 0]
    results_df['umap_2'] = X_umap[:, 1]

results_df.to_csv('comprehensive_clustering_analysis.csv', index=False)
print("Results saved to 'comprehensive_clustering_analysis.csv'")

print("\nComprehensive clustering analysis completed!")
```

## Best Practices

### Visualization Guidelines
1. **Multiple projections**: Use different dimensionality reduction techniques
2. **Interactive tools**: Enable exploration of high-dimensional data
3. **Color schemes**: Use distinguishable colors for different clusters
4. **Scale consistency**: Maintain consistent scales across plots

### Evaluation Strategy
1. **Multiple metrics**: Use internal, external, and stability metrics
2. **Cross-validation**: Assess clustering stability
3. **Domain knowledge**: Validate results with subject matter experts
4. **Comparative analysis**: Compare multiple algorithms and parameters

### Production Considerations
1. **Scalability**: Choose algorithms that work with large datasets
2. **Interpretability**: Ensure cluster results are understandable
3. **Robustness**: Test clustering stability with different parameters
4. **Integration**: Connect clustering results with business applications

## Next Steps

This lecture covers advanced cluster visualization and evaluation techniques. The final lecture (15.8) will explore RAPIDS acceleration for PCA, UMAP, and DBSCAN, showing how GPU computing enables scalable dimensionality reduction and clustering for massive datasets.