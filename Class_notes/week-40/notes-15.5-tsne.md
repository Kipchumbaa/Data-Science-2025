# Lecture 15.5: t-SNE for Dimensionality Reduction

## Key Learning Objectives
- Understand t-SNE algorithm and its mathematical foundation
- Learn to apply t-SNE for high-dimensional data visualization
- Master parameter tuning and interpretation of t-SNE results
- Compare t-SNE with other dimensionality reduction techniques

## Core Concepts

### What is t-SNE?

#### Stochastic Neighbor Embedding
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding
- **Purpose**: Reduce high-dimensional data to 2D/3D for visualization
- **Key innovation**: Uses t-distribution in low-dimensional space
- **Focus**: Preserves local structure and neighborhoods

#### Why t-SNE?
- **Non-linear**: Can capture complex, non-linear relationships
- **Local structure**: Excellent at preserving local neighborhoods
- **Visualization**: Produces beautiful, interpretable 2D/3D plots
- **Clustering**: Often reveals natural clusters in data

### Mathematical Foundation

#### High-Dimensional Similarities
For points x_i and x_j in high-dimensional space:
```
p_{j|i} = exp(-||x_i - x_j||² / 2σ_i²) / Σ_{k≠i} exp(-||x_i - x_k||² / 2σ_i²)
```
- **Conditional probability**: Probability that j would pick i as neighbor
- **Gaussian kernel**: Uses Gaussian distribution for similarities
- **Perplexity**: Controls effective number of neighbors (user parameter)

#### Low-Dimensional Similarities
For points y_i and y_j in low-dimensional space:
```
q_{ij} = (1 + ||y_i - y_j||²)⁻¹ / Σ_{k≠l} (1 + ||y_k - y_l||²)⁻¹
```
- **t-distribution**: Uses t-distribution (heavy tails) instead of Gaussian
- **Student's t**: Allows for better modeling of far-apart points
- **Normalization**: Symmetric probabilities for computational efficiency

#### Cost Function
```
C = KL(P||Q) = Σ_i Σ_j p_{ij} log(p_{ij}/q_{ij})
```
- **Kullback-Leibler divergence**: Measures difference between distributions
- **Asymmetric**: P is conditional, Q is symmetric
- **Optimization**: Gradient descent to minimize KL divergence

## t-SNE Algorithm

### Algorithm Steps
1. **Compute similarities**: Calculate conditional probabilities p_{j|i}
2. **Initialize embedding**: Randomly place points in 2D/3D space
3. **Optimize**: Use gradient descent to minimize KL divergence
4. **Early exaggeration**: Initially emphasize close points (Phase 1)
5. **Normal optimization**: Standard optimization (Phase 2)

### Key Parameters

#### Perplexity
```python
# Perplexity controls effective number of neighbors
# Typical range: 5-50
# Rule of thumb: perplexity ≈ sqrt(n_samples)

perplexity_values = [5, 10, 30, 50, 100]

for perp in perplexity_values:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_embedded = tsne.fit_transform(X)
    # Evaluate embedding quality
```

#### Learning Rate
```python
# Learning rate for gradient descent
# Default: 200.0 for 2D, 200.0 for 3D
# Too high: Oscillations, instability
# Too low: Slow convergence

learning_rates = [10, 50, 200, 500, 1000]

for lr in learning_rates:
    tsne = TSNE(learning_rate=lr, random_state=42)
    X_embedded = tsne.fit_transform(X)
```

#### Number of Iterations
```python
# Total iterations for optimization
# Default: 1000
# Early exaggeration: First 250 iterations (by default)
# Normal optimization: Remaining iterations

tsne = TSNE(
    n_iter=1000,           # Total iterations
    n_iter_without_progress=300,  # Stop if no progress
    min_grad_norm=1e-7     # Minimum gradient for convergence
)
```

## Implementation in Scikit-learn

### Basic t-SNE
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Create t-SNE model
tsne = TSNE(
    n_components=2,        # Dimensions of embedding
    perplexity=30.0,       # Effective number of neighbors
    early_exaggeration=12.0,  # Early exaggeration factor
    learning_rate=200.0,   # Learning rate
    n_iter=1000,           # Maximum iterations
    random_state=42,       # Reproducibility
    init='pca',            # Initialization method
    verbose=1              # Print progress
)

# Fit and transform
X_embedded = tsne.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.6, s=50)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization')
plt.grid(True)
plt.show()
```

### t-SNE with Class Labels
```python
# Color points by class labels
plt.figure(figsize=(12, 8))

# Get unique classes
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

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization with Class Labels')
plt.legend()
plt.grid(True)
plt.show()
```

### 3D t-SNE
```python
from mpl_toolkits.mplot3d import Axes3D

# 3D t-SNE
tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42)
X_embedded_3d = tsne_3d.fit_transform(X_scaled)

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

ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
ax.set_title('3D t-SNE Visualization')
plt.colorbar(scatter)
plt.show()
```

## Parameter Optimization

### Perplexity Selection
```python
def evaluate_tsne_perplexity(X, perplexities, n_components=2):
    """
    Evaluate different perplexity values
    """
    results = []
    
    for perp in perplexities:
        tsne = TSNE(
            n_components=n_components, 
            perplexity=perp, 
            random_state=42,
            n_iter=500  # Fewer iterations for speed
        )
        
        X_embedded = tsne.fit_transform(X)
        kl_divergence = tsne.kl_divergence_
        
        results.append({
            'perplexity': perp,
            'kl_divergence': kl_divergence,
            'embedding': X_embedded
        })
    
    return results

# Test different perplexities
perplexities = [5, 10, 20, 30, 50]
perp_results = evaluate_tsne_perplexity(X_scaled, perplexities)

# Plot KL divergence vs perplexity
kl_values = [r['kl_divergence'] for r in perp_results]
plt.plot(perplexities, kl_values, 'bo-')
plt.xlabel('Perplexity')
plt.ylabel('KL Divergence')
plt.title('t-SNE KL Divergence vs Perplexity')
plt.grid(True)
plt.show()
```

### Learning Rate Optimization
```python
def find_optimal_learning_rate(X, perplexity=30):
    """
    Find optimal learning rate for t-SNE
    """
    learning_rates = [10, 50, 100, 200, 500, 1000]
    results = []
    
    for lr in learning_rates:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=lr,
            random_state=42,
            n_iter=500
        )
        
        X_embedded = tsne.fit_transform(X)
        
        results.append({
            'learning_rate': lr,
            'kl_divergence': tsne.kl_divergence_,
            'embedding': X_embedded
        })
    
    return results

# Optimize learning rate
lr_results = find_optimal_learning_rate(X_scaled)

# Plot results
kl_values = [r['kl_divergence'] for r in lr_results]
lrs = [r['learning_rate'] for r in lr_results]

plt.plot(lrs, kl_values, 'ro-')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('KL Divergence')
plt.title('t-SNE KL Divergence vs Learning Rate')
plt.grid(True)
plt.show()
```

## Comparison with Other Methods

### t-SNE vs PCA
```python
from sklearn.decomposition import PCA

# PCA for comparison
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Compare visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# PCA plot
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
ax1.set_title('PCA Visualization')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
plt.colorbar(scatter1, ax=ax1)

# t-SNE plot
scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
ax2.set_title('t-SNE Visualization')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.show()

# Explained variance for PCA
explained_var = pca.explained_variance_ratio_
print(f"PCA explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
```

### t-SNE vs UMAP
```python
# UMAP for comparison (if available)
try:
    import umap
    
    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Compare
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    ax1.set_title('t-SNE Visualization')
    
    scatter2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
    ax2.set_title('UMAP Visualization')
    
    plt.show()
    
except ImportError:
    print("UMAP not available. Install with: pip install umap-learn")
```

## Advanced t-SNE Features

### Barnes-Hut Approximation
```python
# Barnes-Hut approximation for speed (default for N > 1200)
tsne_fast = TSNE(
    n_components=2,
    perplexity=30,
    method='barnes_hut',    # Approximation method
    angle=0.5,             # Trade-off between speed and accuracy
    random_state=42
)

X_embedded_fast = tsne_fast.fit_transform(X_scaled)
```

### Initialization Methods
```python
# Different initialization strategies
init_methods = ['random', 'pca']

for init_method in init_methods:
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        init=init_method, 
        random_state=42
    )
    X_embedded = tsne.fit_transform(X_scaled)
    
    print(f"Init method: {init_method}, KL divergence: {tsne.kl_divergence_:.4f}")
```

### Monitoring Convergence
```python
# Monitor KL divergence over iterations
tsne = TSNE(
    n_components=2,
    perplexity=30,
    verbose=2,  # Print progress every 50 iterations
    random_state=42
)

X_embedded = tsne.fit_transform(X_scaled)
print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")
```

## Practical Implementation

### Complete t-SNE Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import time

# Load and prepare data
print("Loading data...")
df = pd.read_csv('high_dim_data.csv')

# Handle target variable if present
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

# Remove any NaN values
X_scaled = np.nan_to_num(X_scaled, nan=0.0)

# Step 1: Parameter optimization
print("Optimizing t-SNE parameters...")

# Test different perplexities
perplexities = [5, 10, 20, 30, 50]
perp_results = []

for perp in perplexities:
    if perp < len(X_scaled) / 3:  # Perplexity should be much smaller than n_samples
        tsne = TSNE(
            n_components=2, 
            perplexity=perp, 
            random_state=42,
            n_iter=500,  # Fewer iterations for parameter search
            verbose=0
        )
        
        start_time = time.time()
        X_embedded = tsne.fit_transform(X_scaled)
        elapsed = time.time() - start_time
        
        perp_results.append({
            'perplexity': perp,
            'kl_divergence': tsne.kl_divergence_,
            'time': elapsed,
            'embedding': X_embedded
        })

# Find best perplexity
best_perp_result = min(perp_results, key=lambda x: x['kl_divergence'])
best_perplexity = best_perp_result['perplexity']

print(f"Best perplexity: {best_perplexity} (KL: {best_perp_result['kl_divergence']:.4f})")

# Step 2: Final t-SNE with optimized parameters
print("Running final t-SNE...")
final_tsne = TSNE(
    n_components=2,
    perplexity=best_perplexity,
    early_exaggeration=12.0,
    learning_rate=200.0,
    n_iter=1000,
    n_iter_without_progress=300,
    min_grad_norm=1e-7,
    random_state=42,
    init='pca',  # Better initialization
    verbose=1
)

start_time = time.time()
X_embedded_final = final_tsne.fit_transform(X_scaled)
tsne_time = time.time() - start_time

print(f"t-SNE completed in {tsne_time:.2f} seconds")
print(f"Final KL divergence: {final_tsne.kl_divergence_:.4f}")

# Step 3: Visualization and analysis
plt.figure(figsize=(16, 12))

# Main t-SNE plot
plt.subplot(2, 2, 1)
if has_labels and len(np.unique(y)) <= 20:  # Color by labels if available
    scatter = plt.scatter(
        X_embedded_final[:, 0], 
        X_embedded_final[:, 1],
        c=y, 
        cmap='tab20', 
        alpha=0.7, 
        s=50
    )
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization (Colored by Labels)')
else:
    plt.scatter(
        X_embedded_final[:, 0], 
        X_embedded_final[:, 1],
        alpha=0.7, 
        s=50,
        c='blue'
    )
    plt.title('t-SNE Visualization')

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)

# KL divergence vs perplexity
plt.subplot(2, 2, 2)
perp_values = [r['perplexity'] for r in perp_results]
kl_values = [r['kl_divergence'] for r in perp_results]
plt.plot(perp_values, kl_values, 'bo-')
plt.axvline(x=best_perplexity, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_perplexity}')
plt.xlabel('Perplexity')
plt.ylabel('KL Divergence')
plt.title('Perplexity Optimization')
plt.legend()
plt.grid(True)

# Distribution analysis
plt.subplot(2, 2, 3)
plt.hist2d(X_embedded_final[:, 0], X_embedded_final[:, 1], bins=50, cmap='Blues')
plt.colorbar()
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Point Density Distribution')

# Nearest neighbor preservation (simplified)
plt.subplot(2, 2, 4)
# Calculate preservation of local structure (simplified metric)
from sklearn.neighbors import NearestNeighbors

# Find nearest neighbors in original space
nn_orig = NearestNeighbors(n_neighbors=10, metric='euclidean')
nn_orig.fit(X_scaled)
distances_orig, indices_orig = nn_orig.kneighbors(X_scaled)

# Find nearest neighbors in embedded space
nn_embed = NearestNeighbors(n_neighbors=10, metric='euclidean')
nn_embed.fit(X_embedded_final)
distances_embed, indices_embed = nn_embed.kneighbors(X_embedded_final)

# Calculate preservation score (simplified)
preservation_scores = []
for i in range(min(100, len(X_scaled))):  # Sample for speed
    orig_neighbors = set(indices_orig[i])
    embed_neighbors = set(indices_embed[i])
    overlap = len(orig_neighbors.intersection(embed_neighbors))
    preservation_scores.append(overlap / 10)

plt.hist(preservation_scores, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Neighbor Preservation Score')
plt.ylabel('Frequency')
plt.title('Local Structure Preservation')
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 4: Clustering on t-SNE embedding
print("Analyzing clusters in t-SNE space...")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Try different numbers of clusters
cluster_range = range(2, 11)
cluster_scores = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_embedded_final)
    
    silhouette = silhouette_score(X_embedded_final, cluster_labels)
    cluster_scores.append(silhouette)

# Find best clustering
best_n_clusters = cluster_range[np.argmax(cluster_scores)]
best_silhouette = max(cluster_scores)

print(f"Best clustering: {best_n_clusters} clusters (Silhouette: {best_silhouette:.3f})")

# Final clustering
final_kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
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

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title(f't-SNE Clusters (k={best_n_clusters})')
plt.colorbar(scatter)
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Save results
print("Saving results...")
results_df = pd.DataFrame({
    'tsne_1': X_embedded_final[:, 0],
    'tsne_2': X_embedded_final[:, 1],
    'cluster_label': final_cluster_labels
})

if has_labels:
    results_df['original_label'] = y

results_df.to_csv('tsne_analysis_results.csv', index=False)
print("Results saved to 'tsne_analysis_results.csv'")

# Summary statistics
print("
=== t-SNE Analysis Summary ===")
print(f"Original dimensions: {X.shape[1]}")
print(f"Embedded dimensions: 2")
print(f"Perplexity used: {best_perplexity}")
print(f"Final KL divergence: {final_tsne.kl_divergence_:.4f}")
print(f"Clusters found: {best_n_clusters}")
print(f"Clustering silhouette: {best_silhouette:.3f}")
print(f"Computation time: {tsne_time:.2f} seconds")

print("\nCluster sizes:")
for cluster_id in range(best_n_clusters):
    size = np.sum(final_cluster_labels == cluster_id)
    percentage = size / len(final_cluster_labels) * 100
    print(f"Cluster {cluster_id}: {size} points ({percentage:.1f}%)")

print("\nt-SNE analysis completed!")
```

## Best Practices

### Parameter Selection
1. **Perplexity**: Start with 5-50, use sqrt(n_samples) as rule of thumb
2. **Learning rate**: Default 200 usually works, adjust if optimization unstable
3. **Iterations**: Use at least 1000, monitor KL divergence convergence
4. **Initialization**: Use 'pca' for better starting point

### Data Preparation
1. **Scaling**: Essential for distance-based methods
2. **Missing values**: Handle NaN values before t-SNE
3. **Sparsity**: Dense representations work better than sparse
4. **Dimensionality**: Can handle high dimensions but may be slow

### Interpretation Guidelines
1. **Distances**: Don't interpret absolute distances, focus on relative positions
2. **Clusters**: Clear clusters often indicate real data structure
3. **Outliers**: Isolated points may be noise or anomalies
4. **Stability**: Run multiple times to check consistency

## Next Steps

This lecture covers t-SNE for non-linear dimensionality reduction and visualization. The next lecture (15.6) will introduce UMAP, another powerful non-linear dimensionality reduction technique that is often faster and more scalable than t-SNE.