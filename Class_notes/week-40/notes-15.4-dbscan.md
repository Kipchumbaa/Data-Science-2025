# Lecture 15.4: DBSCAN Clustering

## Key Learning Objectives
- Understand density-based clustering concepts
- Master DBSCAN algorithm implementation
- Learn to handle clusters of arbitrary shapes
- Compare DBSCAN with partitioning methods

## Core Concepts

### Density-Based Clustering

#### Why Density-Based Methods?
- **Arbitrary shapes**: Can find clusters of any shape (not just spherical)
- **Noise handling**: Automatically identifies and labels noise points
- **No k parameter**: Doesn't require specifying number of clusters
- **Parameter sensitivity**: Sensitive to distance and density parameters

#### DBSCAN Fundamentals
- **Core points**: Points with sufficient neighbors within ε distance
- **Border points**: Points within ε of core points but have insufficient neighbors
- **Noise points**: Points that are neither core nor border points
- **Direct density-reachability**: Points connected through core points

### DBSCAN Algorithm

#### Algorithm Steps
1. **Label all points as unvisited**
2. **For each unvisited point**:
   - If it's a core point, start a new cluster
   - Find all density-reachable points from this core point
   - Assign all found points to the current cluster
3. **Repeat until all points are visited**
4. **Points not assigned to any cluster are noise**

#### Key Parameters
- **ε (epsilon)**: Maximum distance between two points to be neighbors
- **MinPts**: Minimum number of points required to form a dense region
- **Distance metric**: Usually Euclidean, but can be others

### Mathematical Foundation

#### Core Point Definition
A point p is a core point if:
```
|N_ε(p)| ≥ MinPts
```
Where N_ε(p) is the ε-neighborhood of p

#### Density-Reachability
A point q is density-reachable from p if there exists a chain of points p₁, p₂, ..., pₙ where:
- p₁ = p, pₙ = q
- Each p_{i+1} is directly density-reachable from p_i
- All points in the chain (except q) are core points

#### Direct Density-Reachability
A point q is directly density-reachable from p if:
- q ∈ N_ε(p) (q is within ε distance of p)
- |N_ε(p)| ≥ MinPts (p is a core point)

## DBSCAN Implementation

### Scikit-learn DBSCAN
```python
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Create DBSCAN model
dbscan = DBSCAN(
    eps=0.5,           # Maximum distance between points
    min_samples=5,     # Minimum samples in neighborhood
    metric='euclidean', # Distance metric
    algorithm='auto',   # Algorithm for nearest neighbors
    n_jobs=-1          # Use all CPU cores
)

# Fit and predict
cluster_labels = dbscan.fit_predict(X_scaled)

# Analyze results
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Evaluate clustering (excluding noise points)
if n_clusters > 1:
    mask = cluster_labels != -1
    if sum(mask) > 1:
        silhouette = silhouette_score(X_scaled[mask], cluster_labels[mask])
        print(f"Silhouette score (excluding noise): {silhouette:.3f}")
```

### Parameter Selection

#### Choosing ε (Epsilon)
```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Find optimal eps using k-distance plot
def find_optimal_eps(X, min_samples=5):
    """
    Find optimal eps using k-distance plot
    """
    # Compute k-nearest neighbors distances
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    
    # Sort distances to k-th nearest neighbor
    k_distances = np.sort(distances[:, -1])
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-th nearest neighbor distance')
    plt.title('K-distance Graph for DBSCAN')
    plt.grid(True)
    plt.show()
    
    return k_distances

# Find optimal eps
k_distances = find_optimal_eps(X_scaled, min_samples=5)

# The "elbow" point suggests good eps value
# Look for the point where the curve starts to rise sharply
```

#### Choosing MinPts
```python
# General guidelines for MinPts:
# - MinPts ≥ D + 1 where D is dimensionality
# - For 2D data: MinPts = 4
# - For higher dimensions: MinPts = 2*D
# - Larger datasets can use larger MinPts

def test_minpts_values(X, eps, minpts_range):
    """
    Test different MinPts values
    """
    results = []
    
    for minpts in minpts_range:
        dbscan = DBSCAN(eps=eps, min_samples=minpts)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        results.append({
            'minpts': minpts,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(X)
        })
    
    return pd.DataFrame(results)

# Test different MinPts values
minpts_range = [3, 5, 7, 10, 15]
eps_value = 0.5  # From k-distance analysis
minpts_results = test_minpts_values(X_scaled, eps_value, minpts_range)

print(minpts_results)
```

## Advantages and Limitations

### Advantages
- **Arbitrary shapes**: Can find clusters of any shape
- **No k parameter**: Automatically determines number of clusters
- **Noise detection**: Identifies and labels noise points
- **Robust to outliers**: Less affected by noise than centroid-based methods
- **Deterministic**: Same results for same parameters

### Limitations
- **Parameter sensitivity**: Performance depends heavily on ε and MinPts
- **High dimensions**: Struggles with high-dimensional data (curse of dimensionality)
- **Varying densities**: Can't handle clusters with different densities well
- **Border points**: Ambiguous assignment of border points
- **Computational cost**: O(n²) in worst case, though optimized versions exist

## Comparison with Other Methods

### DBSCAN vs K-means
| Aspect | DBSCAN | K-means |
|--------|--------|---------|
| **Cluster shapes** | Arbitrary | Spherical |
| **Number of clusters** | Auto-detected | Must specify |
| **Noise handling** | Explicit noise | All points assigned |
| **Parameter complexity** | Two parameters | One parameter + initialization |
| **Scalability** | Good with optimizations | Very scalable |
| **High dimensions** | Poor | Good |

### DBSCAN vs Hierarchical Clustering
| Aspect | DBSCAN | Hierarchical |
|--------|--------|--------------|
| **Cluster shapes** | Arbitrary | Arbitrary |
| **Number of clusters** | Auto-detected | Post-hoc selection |
| **Noise handling** | Explicit noise | All points in some cluster |
| **Computational cost** | O(n log n) optimized | O(n²) |
| **Memory usage** | Moderate | High |
| **Interpretability** | Good | Excellent (dendrogram) |

## Advanced DBSCAN Variants

### OPTICS (Ordering Points To Identify Clustering Structure)
```python
from sklearn.cluster import OPTICS

# OPTICS clustering
optics = OPTICS(
    min_samples=5,
    xi=0.05,          # Steepness threshold
    min_cluster_size=0.1  # Minimum cluster size
)

optics_labels = optics.fit_predict(X_scaled)

# OPTICS provides reachability plot
plt.figure(figsize=(10, 6))
plt.plot(optics.reachability_[optics.ordering_])
plt.xlabel('Points in cluster order')
plt.ylabel('Reachability distance')
plt.title('OPTICS Reachability Plot')
plt.show()
```

### HDBSCAN (Hierarchical DBSCAN)
```python
# HDBSCAN provides hierarchical clustering with variable density
# Requires hdbscan library: pip install hdbscan
try:
    import hdbscan
    
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_epsilon=0.5
    )
    
    hdb_labels = hdb.fit_predict(X_scaled)
    
    print(f"HDBSCAN found {len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)} clusters")
    
except ImportError:
    print("HDBSCAN not available. Install with: pip install hdbscan")
```

## Handling Different Data Types

### Categorical Data
```python
# For mixed data types, use Gower distance or other appropriate metrics
from sklearn.preprocessing import OneHotEncoder

# One-hot encode categorical variables
encoder = OneHotEncoder()
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Combine with numerical features
from scipy.sparse import hstack
X_mixed = hstack([X_numerical_scaled, X_categorical_encoded])

# Use cosine or Jaccard distance for mixed data
# Note: sklearn DBSCAN doesn't support all distance metrics directly
```

### High-Dimensional Data
```python
# For high-dimensional data, consider dimensionality reduction first
from sklearn.decomposition import PCA

# Reduce dimensionality
pca = PCA(n_components=50, random_state=42)  # Adjust based on data
X_reduced = pca.fit_transform(X_scaled)

# Apply DBSCAN on reduced data
dbscan_hd = DBSCAN(eps=0.5, min_samples=5)
labels_hd = dbscan_hd.fit_predict(X_reduced)

print(f"Clusters in reduced space: {len(set(labels_hd)) - (1 if -1 in labels_hd else 0)}")
```

## Evaluation Metrics

### Internal Metrics (No Ground Truth)
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Remove noise points for evaluation
mask = cluster_labels != -1
if sum(mask) > 1 and len(set(cluster_labels[mask])) > 1:
    X_eval = X_scaled[mask]
    labels_eval = cluster_labels[mask]
    
    silhouette = silhouette_score(X_eval, labels_eval)
    ch_score = calinski_harabasz_score(X_eval, labels_eval)
    db_score = davies_bouldin_score(X_eval, labels_eval)
    
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Score: {ch_score:.3f}")
    print(f"Davies-Bouldin Score: {db_score:.3f}")
```

### External Metrics (With Ground Truth)
```python
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

if true_labels is not None:
    # Remove noise points for comparison
    mask = cluster_labels != -1
    ari = adjusted_rand_score(true_labels[mask], cluster_labels[mask])
    ami = adjusted_mutual_info_score(true_labels[mask], cluster_labels[mask])
    
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Adjusted Mutual Information: {ami:.3f}")
```

## Practical Implementation

### Complete DBSCAN Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('clustering_data.csv')
X = df.drop(['id', 'target'], axis=1, errors='ignore')

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape: {X_scaled.shape}")

# Step 1: Find optimal eps using k-distance plot
print("Finding optimal eps...")
min_samples = 5
neigh = NearestNeighbors(n_neighbors=min_samples)
neigh.fit(X_scaled)
distances, indices = neigh.kneighbors(X_scaled)

# Sort k-distances
k_distances = np.sort(distances[:, -1])

# Plot k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{min_samples}-th nearest neighbor distance')
plt.title('K-distance Graph for DBSCAN')
plt.grid(True)
plt.show()

# Choose eps at the "elbow" point (visual inspection needed)
# For automation, find the knee point
from kneed import KneeLocator
kneedle = KneeLocator(range(len(k_distances)), k_distances, S=1.0, curve='convex', direction='increasing')
optimal_eps = k_distances[kneedle.knee]
print(f"Suggested eps: {optimal_eps:.3f}")

# Step 2: Test different parameter combinations
eps_range = [optimal_eps * 0.5, optimal_eps * 0.75, optimal_eps, optimal_eps * 1.25, optimal_eps * 1.5]
min_samples_range = [3, 5, 7, 10]

param_results = []

for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(X_scaled)
        
        # Calculate silhouette if possible
        silhouette = None
        if n_clusters > 1:
            mask = labels != -1
            if sum(mask) > n_clusters:
                try:
                    silhouette = silhouette_score(X_scaled[mask], labels[mask])
                except:
                    silhouette = None
        
        param_results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette': silhouette
        })

# Display parameter tuning results
param_df = pd.DataFrame(param_results)
print("\nParameter tuning results:")
print(param_df.sort_values(['silhouette', 'noise_ratio'], ascending=[False, True]).head(10))

# Step 3: Choose best parameters
best_params = param_df.loc[param_df['silhouette'].idxmax()]
best_eps = best_params['eps']
best_min_samples = best_params['min_samples']

print(f"\nBest parameters: eps={best_eps:.3f}, min_samples={int(best_min_samples)}")

# Step 4: Final DBSCAN clustering
print("Performing final clustering...")
final_dbscan = DBSCAN(eps=best_eps, min_samples=int(best_min_samples))
final_labels = final_dbscan.fit_predict(X_scaled)

final_n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
final_n_noise = list(final_labels).count(-1)
final_noise_ratio = final_n_noise / len(X_scaled)

print(f"Final clustering results:")
print(f"Number of clusters: {final_n_clusters}")
print(f"Number of noise points: {final_n_noise} ({final_noise_ratio:.1%})")

# Step 5: Evaluate clustering quality
if final_n_clusters > 1:
    mask = final_labels != -1
    if sum(mask) > final_n_clusters:
        final_silhouette = silhouette_score(X_scaled[mask], final_labels[mask])
        print(f"Silhouette score: {final_silhouette:.3f}")

# Step 6: Visualize results (if 2D or reducible to 2D)
if X_scaled.shape[1] > 2:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained_var:.1%}")
else:
    X_2d = X_scaled

# Create visualization
plt.figure(figsize=(12, 8))

# Plot clusters
unique_labels = set(final_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black for noise
        col = 'black'
        label = 'Noise'
    else:
        label = f'Cluster {k}'
    
    class_member_mask = (final_labels == k)
    xy = X_2d[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.6, s=50, label=label)

plt.xlabel('PC1' if X_scaled.shape[1] > 2 else 'Feature 1')
plt.ylabel('PC2' if X_scaled.shape[1] > 2 else 'Feature 2')
plt.title(f'DBSCAN Clustering Results\nε={best_eps:.3f}, MinPts={int(best_min_samples)}, Clusters={final_n_clusters}')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Cluster analysis
print("\nCluster Analysis:")
cluster_sizes = []
for cluster_id in sorted(set(final_labels)):
    if cluster_id == -1:
        continue
    size = np.sum(final_labels == cluster_id)
    percentage = size / len(final_labels) * 100
    cluster_sizes.append(size)
    print(f"Cluster {cluster_id}: {size} points ({percentage:.1f}%)")

if cluster_sizes:
    print(f"Cluster size statistics:")
    print(f"Mean cluster size: {np.mean(cluster_sizes):.1f}")
    print(f"Std cluster size: {np.std(cluster_sizes):.1f}")
    print(f"Min cluster size: {np.min(cluster_sizes)}")
    print(f"Max cluster size: {np.max(cluster_sizes)}")

# Step 8: Save results
results_df = pd.DataFrame({
    'original_index': range(len(final_labels)),
    'cluster_label': final_labels
})

results_df.to_csv('dbscan_clustering_results.csv', index=False)
print("\nResults saved to 'dbscan_clustering_results.csv'")

print("\nDBSCAN clustering analysis completed!")
```

## Best Practices

### Parameter Selection
1. **Start with k-distance plot**: Find natural density thresholds
2. **Test parameter combinations**: Use grid search approach
3. **Consider domain knowledge**: Incorporate expert insights
4. **Validate results**: Check if clusters make sense

### Data Preparation
1. **Scale features**: Essential for distance-based methods
2. **Handle outliers**: DBSCAN is robust but extreme outliers can affect results
3. **Dimensionality**: Consider reduction for high-dimensional data
4. **Distance metrics**: Choose appropriate metric for data type

### Performance Optimization
1. **Algorithm selection**: Use 'ball_tree' or 'kd_tree' for low dimensions
2. **Parallel processing**: Use n_jobs parameter for CPU parallelization
3. **Memory management**: Process large datasets in batches if needed
4. **GPU acceleration**: Consider RAPIDS for very large datasets

## Next Steps

This lecture covers DBSCAN, a powerful density-based clustering algorithm. The next lecture (15.5) will introduce t-SNE, a popular non-linear dimensionality reduction technique for visualizing high-dimensional data in 2D or 3D.