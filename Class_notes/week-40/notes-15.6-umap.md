# Lecture 15.6: UMAP for Dimensionality Reduction

## Key Learning Objectives
- Understand UMAP algorithm and its mathematical foundation
- Learn to apply UMAP for scalable dimensionality reduction
- Master parameter tuning and result interpretation
- Compare UMAP with t-SNE and other dimensionality reduction methods

## Core Concepts

### What is UMAP?

#### Uniform Manifold Approximation and Projection
- **UMAP**: Uniform Manifold Approximation and Projection
- **Purpose**: Non-linear dimensionality reduction for visualization and analysis
- **Key innovation**: Combines ideas from manifold learning and topological data analysis
- **Advantages**: Faster than t-SNE, preserves more global structure

#### Why UMAP?
- **Speed**: Significantly faster than t-SNE, especially for large datasets
- **Scalability**: Better performance on large datasets
- **Flexibility**: Preserves both local and global structure
- **Mathematics**: Rigorous mathematical foundation

### Mathematical Foundation

#### Manifold Learning
- **Manifold hypothesis**: High-dimensional data lies on low-dimensional manifold
- **Local structure**: Data points have similar neighborhoods in high and low dimensions
- **Global structure**: Overall shape and relationships are preserved

#### Fuzzy Topological Representation
UMAP constructs a high-dimensional graph where:
- **Nodes**: Data points
- **Edges**: Connections based on local neighborhoods
- **Weights**: Similarity measures between points

#### Optimization
- **Low-dimensional embedding**: Find 2D/3D representation that preserves topology
- **Cross-entropy loss**: Minimize difference between high and low-dimensional similarities
- **Stochastic gradient descent**: Efficient optimization for large datasets

## UMAP Algorithm

### Algorithm Overview
1. **Construct graph**: Build weighted graph of data points using k-nearest neighbors
2. **Optimize embedding**: Find low-dimensional coordinates that preserve graph structure
3. **Refine layout**: Use force-directed layout to improve embedding quality

### Key Parameters

#### n_neighbors
```python
# Number of neighbors for local manifold approximation
# Smaller values: Focus on local structure, more clusters
# Larger values: Focus on global structure, fewer clusters

n_neighbors_values = [5, 15, 50, 100]

for n_n in n_neighbors_values:
    reducer = umap.UMAP(n_neighbors=n_n, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    # Evaluate embedding quality
```

#### min_dist
```python
# Minimum distance between points in low-dimensional space
# Smaller values: Tighter clusters, more detail
# Larger values: More spread out, less detail

min_dist_values = [0.0, 0.1, 0.5, 0.99]

for md in min_dist_values:
    reducer = umap.UMAP(min_dist=md, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
```

#### n_components
```python
# Number of dimensions for embedding
# 2: Standard visualization
# 3: 3D visualization
# Higher: For further processing

reducer_2d = umap.UMAP(n_components=2, random_state=42)
reducer_3d = umap.UMAP(n_components=3, random_state=42)

X_2d = reducer_2d.fit_transform(X_scaled)
X_3d = reducer_3d.fit_transform(X_scaled)
```

## Implementation with UMAP

### Basic UMAP
```python
import umap
import matplotlib.pyplot as plt

# Create UMAP reducer
reducer = umap.UMAP(
    n_neighbors=15,       # Local neighborhood size
    n_components=2,       # Dimensions of embedding
    metric='euclidean',   # Distance metric
    min_dist=0.1,         # Minimum distance in embedding
    random_state=42,      # Reproducibility
    verbose=True          # Print progress
)

# Fit and transform
X_embedded = reducer.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.6, s=50)
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('UMAP Visualization')
plt.grid(True)
plt.show()
```

### UMAP with Class Labels
```python
# Color points by class labels
plt.figure(figsize=(12, 8))

unique_labels = np.unique(y)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    mask = y == label
    plt.scatter(
        X_embedded[mask, 0], 
        X_embedded[mask, 1],
        c=[colors[i]], 
        label=f'Class {label}',
        alpha=0.7,
        s=50
    )

plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('UMAP Visualization with Class Labels')
plt.legend()
plt.grid(True)
plt.show()
```

### 3D UMAP
```python
from mpl_toolkits.mplot3d import Axes3D

# 3D UMAP
reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, random_state=42)
X_embedded_3d = reducer_3d.fit_transform(X_scaled)

# 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_embedded_3d[:, 0], 
    X_embedded_3d[:, 1], 
    X_embedded_3d[:, 2],
    c=y, 
    cmap='tab10', 
    alpha=0.7, 
    s=50
)

ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')
ax.set_zlabel('UMAP Component 3')
ax.set_title('3D UMAP Visualization')
plt.colorbar(scatter)
plt.show()
```

## Parameter Optimization

### Finding Optimal n_neighbors
```python
def evaluate_umap_neighbors(X, n_neighbors_list, n_components=2):
    """
    Evaluate different n_neighbors values
    """
    results = []
    
    for n_n in n_neighbors_list:
        reducer = umap.UMAP(
            n_neighbors=n_n, 
            n_components=n_components, 
            random_state=42,
            verbose=False
        )
        
        embedding = reducer.fit_transform(X)
        
        # Simple quality metrics
        # (More sophisticated metrics exist but require additional libraries)
        results.append({
            'n_neighbors': n_n,
            'embedding': embedding,
            'trustworthiness': None,  # Would need additional computation
            'continuity': None       # Would need additional computation
        })
    
    return results

# Test different n_neighbors
n_neighbors_list = [5, 10, 15, 20, 30, 50]
neighbors_results = evaluate_umap_neighbors(X_scaled, n_neighbors_list)

# Visualize different embeddings
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, result in enumerate(neighbors_results):
    axes[i].scatter(
        result['embedding'][:, 0], 
        result['embedding'][:, 1], 
        alpha=0.6, 
        s=30
    )
    axes[i].set_title(f'n_neighbors = {result["n_neighbors"]}')
    axes[i].grid(True)

plt.tight_layout()
plt.show()
```

### Optimizing min_dist
```python
def evaluate_umap_min_dist(X, min_dist_list, n_neighbors=15):
    """
    Evaluate different min_dist values
    """
    results = []
    
    for md in min_dist_list:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=md,
            random_state=42,
            verbose=False
        )
        
        embedding = reducer.fit_transform(X)
        
        results.append({
            'min_dist': md,
            'embedding': embedding
        })
    
    return results

# Test different min_dist values
min_dist_list = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
min_dist_results = evaluate_umap_min_dist(X_scaled, min_dist_list)

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, result in enumerate(min_dist_results):
    axes[i].scatter(
        result['embedding'][:, 0], 
        result['embedding'][:, 1], 
        alpha=0.6, 
        s=30
    )
    axes[i].set_title(f'min_dist = {result["min_dist"]}')
    axes[i].grid(True)

plt.tight_layout()
plt.show()
```

## Comparison with Other Methods

### UMAP vs t-SNE
```python
import time

# t-SNE
tsne_start = time.time()
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
X_tsne = tsne.fit_transform(X_scaled)
tsne_time = time.time() - tsne_start

# UMAP
umap_start = time.time()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
umap_time = time.time() - umap_start

# Compare visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# t-SNE plot
if hasattr(y, '__len__') and len(np.unique(y)) <= 20:
    scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter1, ax=ax1)
else:
    ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)

ax1.set_title(f't-SNE (Time: {tsne_time:.2f}s)')
ax1.set_xlabel('t-SNE 1')
ax1.set_ylabel('t-SNE 2')
ax1.grid(True)

# UMAP plot
if hasattr(y, '__len__') and len(np.unique(y)) <= 20:
    scatter2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter2, ax=ax2)
else:
    ax2.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.7)

ax2.set_title(f'UMAP (Time: {umap_time:.2f}s)')
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.grid(True)

plt.suptitle(f'UMAP vs t-SNE Comparison\nUMAP is {tsne_time/umap_time:.1f}x faster')
plt.tight_layout()
plt.show()

print(f"t-SNE time: {tsne_time:.2f}s")
print(f"UMAP time: {umap_time:.2f}s")
print(f"Speedup: {tsne_time/umap_time:.1f}x")
```

### UMAP vs PCA
```python
from sklearn.decomposition import PCA

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Compare
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# PCA plot
if hasattr(y, '__len__') and len(np.unique(y)) <= 20:
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter1, ax=ax1)
else:
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

ax1.set_title('PCA (Linear)')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.grid(True)

# UMAP plot
if hasattr(y, '__len__') and len(np.unique(y)) <= 20:
    scatter2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter2, ax=ax2)
else:
    ax2.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.7)

ax2.set_title('UMAP (Non-linear)')
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.grid(True)

plt.suptitle('Linear vs Non-linear Dimensionality Reduction')
plt.tight_layout()
plt.show()

# Explained variance for PCA
explained_var = pca.explained_variance_ratio_
print(f"PCA explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
```

## Advanced UMAP Features

### Custom Distance Metrics
```python
# Different distance metrics
metrics = ['euclidean', 'manhattan', 'cosine', 'correlation']

for metric in metrics:
    try:
        reducer = umap.UMAP(n_neighbors=15, metric=metric, random_state=42)
        X_embedded = reducer.fit_transform(X_scaled)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.6, s=50)
        plt.title(f'UMAP with {metric} distance')
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Error with {metric}: {e}")
```

### Supervised UMAP
```python
# Use labels to guide the embedding
supervised_reducer = umap.UMAP(
    n_neighbors=15,
    random_state=42
)

# Fit with labels for supervised dimension reduction
X_supervised = supervised_reducer.fit_transform(X_scaled, y=y)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_supervised[:, 0], X_supervised[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.colorbar(scatter)
plt.title('Supervised UMAP')
plt.grid(True)
plt.show()
```

### Semi-supervised UMAP
```python
# Use partial labels
n_labeled = len(y) // 10  # Label only 10% of data
labeled_indices = np.random.choice(len(y), n_labeled, replace=False)

y_partial = np.full(len(y), -1)  # -1 for unlabeled
y_partial[labeled_indices] = y[labeled_indices]

semi_supervised_reducer = umap.UMAP(
    n_neighbors=15,
    random_state=42
)

X_semi_supervised = semi_supervised_reducer.fit_transform(X_scaled, y=y_partial)

plt.figure(figsize=(10, 8))
# Plot unlabeled points
mask_unlabeled = y_partial == -1
plt.scatter(X_semi_supervised[mask_unlabeled, 0], X_semi_supervised[mask_unlabeled, 1], 
           c='lightgray', alpha=0.5, s=30, label='Unlabeled')

# Plot labeled points
for label in np.unique(y_partial[y_partial != -1]):
    mask = y_partial == label
    plt.scatter(X_semi_supervised[mask, 0], X_semi_supervised[mask, 1], 
               alpha=0.8, s=50, label=f'Class {label}')

plt.title('Semi-supervised UMAP')
plt.legend()
plt.grid(True)
plt.show()
```

## Practical Implementation

### Complete UMAP Pipeline
```python
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time

# Load and prepare data
print("Loading data...")
df = pd.read_csv('high_dim_data.csv')

# Handle target variable
if 'target' in df.columns:
    X = df.drop('target', axis=1)
    y = df['target']
    has_labels = True
else:
    X = df.copy()
    y = None
    has_labels = False

print(f"Dataset shape: {X.shape}")

# Preprocessing
print("Preprocessing...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle any NaN values
X_scaled = np.nan_to_num(X_scaled, nan=0.0)

# Step 1: Parameter optimization
print("Optimizing UMAP parameters...")

# Test different n_neighbors
n_neighbors_list = [5, 10, 15, 20, 30, 50]
n_neighbors_results = []

for n_n in n_neighbors_list:
    start_time = time.time()
    reducer = umap.UMAP(n_neighbors=n_n, random_state=42, verbose=False)
    embedding = reducer.fit_transform(X_scaled)
    elapsed = time.time() - start_time
    
    n_neighbors_results.append({
        'n_neighbors': n_n,
        'embedding': embedding,
        'time': elapsed
    })

print("n_neighbors optimization completed")

# Test different min_dist
min_dist_list = [0.0, 0.1, 0.25, 0.5, 0.8]
min_dist_results = []

for md in min_dist_list:
    start_time = time.time()
    reducer = umap.UMAP(min_dist=md, random_state=42, verbose=False)
    embedding = reducer.fit_transform(X_scaled)
    elapsed = time.time() - start_time
    
    min_dist_results.append({
        'min_dist': md,
        'embedding': embedding,
        'time': elapsed
    })

print("min_dist optimization completed")

# Step 2: Final UMAP with optimized parameters
print("Running final UMAP...")
# Choose reasonable defaults (can be further optimized)
optimal_n_neighbors = 15
optimal_min_dist = 0.1

final_reducer = umap.UMAP(
    n_neighbors=optimal_n_neighbors,
    min_dist=optimal_min_dist,
    n_components=2,
    random_state=42,
    verbose=True
)

start_time = time.time()
X_embedded_final = final_reducer.fit_transform(X_scaled)
umap_time = time.time() - start_time

print(f"UMAP completed in {umap_time:.2f} seconds")

# Step 3: Visualization and analysis
plt.figure(figsize=(16, 12))

# Main UMAP plot
plt.subplot(2, 2, 1)
if has_labels and len(np.unique(y)) <= 20:
    scatter = plt.scatter(
        X_embedded_final[:, 0], 
        X_embedded_final[:, 1],
        c=y, 
        cmap='tab20', 
        alpha=0.7, 
        s=50
    )
    plt.colorbar(scatter)
    plt.title('UMAP Visualization (Colored by Labels)')
else:
    plt.scatter(
        X_embedded_final[:, 0], 
        X_embedded_final[:, 1],
        alpha=0.7, 
        s=50,
        c='blue'
    )
    plt.title('UMAP Visualization')

plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.grid(True)

# Parameter comparison - n_neighbors
plt.subplot(2, 2, 2)
for i, result in enumerate(n_neighbors_results[:4]):  # Show first 4
    plt.scatter(
        result['embedding'][:, 0], 
        result['embedding'][:, 1],
        alpha=0.3, 
        s=20,
        label=f'n={result["n_neighbors"]}'
    )
plt.title('n_neighbors Comparison')
plt.legend()
plt.grid(True)

# Parameter comparison - min_dist
plt.subplot(2, 2, 3)
colors = plt.cm.viridis(np.linspace(0, 1, len(min_dist_results)))
for i, result in enumerate(min_dist_results):
    plt.scatter(
        result['embedding'][:, 0], 
        result['embedding'][:, 1],
        alpha=0.3, 
        s=20,
        c=[colors[i]],
        label=f'md={result["min_dist"]}'
    )
plt.title('min_dist Comparison')
plt.legend()
plt.grid(True)

# Point density analysis
plt.subplot(2, 2, 4)
plt.hist2d(X_embedded_final[:, 0], X_embedded_final[:, 1], bins=50, cmap='Blues')
plt.colorbar()
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('Point Density Distribution')

plt.tight_layout()
plt.show()

# Step 4: Clustering on UMAP embedding
print("Analyzing clusters in UMAP space...")
cluster_range = range(2, 11)
cluster_scores = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_embedded_final)
    
    silhouette = silhouette_score(X_embedded_final, cluster_labels)
    cluster_scores.append(silhouette)

# Find best clustering
best_n_clusters = cluster_range[np.argmax(cluster_scores)]
best_silhouette = max(cluster_scores)

print(f"Best clustering: {best_n_clusters} clusters (Silhouette: {best_silhouette:.3f})")

# Final clustering
final_kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
final_cluster_labels = final_kmeans.fit_predict(X_embedded_final)

# Visualize clustering results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    X_embedded_final[:, 0], 
    X_embedded_final[:, 1],
    c=final_cluster_labels, 
    cmap='tab10', 
    alpha=0.7, 
    s=50
)

# Plot cluster centers
centers = final_kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidth=3, label='Centers')

plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title(f'UMAP Clusters (k={best_n_clusters})')
plt.colorbar(scatter)
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Compare with t-SNE (optional)
try:
    print("Comparing with t-SNE...")
    from sklearn.manifold import TSNE
    
    tsne_start = time.time()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500, verbose=0)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_time = time.time() - tsne_start
    
    print(f"UMAP time: {umap_time:.2f} seconds")
    print(f"t-SNE time: {tsne_time:.2f} seconds")
    print(f"UMAP is {tsne_time/umap_time:.1f}x faster than t-SNE")
    
    # Quick comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.scatter(X_embedded_final[:, 0], X_embedded_final[:, 1], alpha=0.6, s=30)
    ax1.set_title('UMAP')
    ax1.grid(True)
    
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=30)
    ax2.set_title('t-SNE')
    ax2.grid(True)
    
    plt.suptitle('UMAP vs t-SNE Comparison')
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("t-SNE not available for comparison")

# Step 6: Save results
print("Saving results...")
results_df = pd.DataFrame({
    'umap_1': X_embedded_final[:, 0],
    'umap_2': X_embedded_final[:, 1],
    'cluster_label': final_cluster_labels
})

if has_labels:
    results_df['original_label'] = y

results_df.to_csv('umap_analysis_results.csv', index=False)
print("Results saved to 'umap_analysis_results.csv'")

# Summary statistics
print("
=== UMAP Analysis Summary ===")
print(f"Original dimensions: {X.shape[1]}")
print(f"Embedded dimensions: 2")
print(f"n_neighbors used: {optimal_n_neighbors}")
print(f"min_dist used: {optimal_min_dist}")
print(f"Clusters found: {best_n_clusters}")
print(f"Clustering silhouette: {best_silhouette:.3f}")
print(f"Computation time: {umap_time:.2f} seconds")

print("\nCluster sizes:")
for cluster_id in range(best_n_clusters):
    size = np.sum(final_cluster_labels == cluster_id)
    percentage = size / len(final_cluster_labels) * 100
    print(f"Cluster {cluster_id}: {size} points ({percentage:.1f}%)")

print("\nUMAP analysis completed!")
```

## Best Practices

### Parameter Selection
1. **n_neighbors**: Start with 15-50, smaller values for more clusters
2. **min_dist**: Start with 0.1, smaller values for tighter clusters
3. **n_components**: 2 for visualization, 3 for 3D, higher for preprocessing
4. **metric**: 'euclidean' default, choose based on data type

### Data Preparation
1. **Scaling**: Essential for distance-based methods
2. **Missing values**: Handle NaN values before UMAP
3. **Categorical data**: Convert to numerical representations
4. **High dimensions**: Consider PCA preprocessing for very high dimensions

### Performance Optimization
1. **n_neighbors**: Smaller values are faster
2. **min_dist**: Larger values can be faster
3. **Batch processing**: Process large datasets in chunks
4. **Parallel processing**: Use multiple cores when available

### Interpretation Guidelines
1. **Distances**: Focus on relative positions, not absolute distances
2. **Clusters**: Clear separations often indicate real structure
3. **Density**: Point density indicates confidence in structure
4. **Stability**: Run multiple times to check consistency

## Next Steps

This lecture covers UMAP for scalable non-linear dimensionality reduction. The next lecture (15.7) will explore visualizing clusters and advanced techniques for cluster analysis and interpretation.