# Lecture 15.2: K-means and Hierarchical Clustering

## Key Learning Objectives
- Understand partitioning and hierarchical clustering approaches
- Master K-means algorithm implementation and optimization
- Learn hierarchical clustering methods and dendrograms
- Compare clustering algorithms and their applications

## Core Concepts

### Clustering Taxonomy

#### Partitioning Methods
- **K-means**: Divides data into k non-overlapping clusters
- **K-medoids**: Similar to K-means but uses medoids (actual points)
- **Characteristics**: Each point belongs to exactly one cluster
- **Efficiency**: Generally faster, suitable for large datasets

#### Hierarchical Methods
- **Agglomerative**: Bottom-up, start with individual points
- **Divisive**: Top-down, start with one cluster
- **Characteristics**: Produces nested cluster hierarchy
- **Visualization**: Dendrograms show clustering process

## K-means Clustering

### Algorithm Overview

#### Basic K-means Steps
1. **Initialize centroids**: Choose k initial cluster centers
2. **Assignment step**: Assign each point to nearest centroid
3. **Update step**: Recalculate centroids as mean of assigned points
4. **Convergence check**: Repeat until centroids don't change
5. **Termination**: Stop when convergence or max iterations reached

#### Mathematical Foundation
- **Objective**: Minimize within-cluster sum of squares (WCSS)
- **WCSS formula**: Σ Σ ||x - μ_i||² for all points in cluster i
- **Optimization**: Iterative refinement using expectation-maximization

### K-means Implementation

#### Scikit-learn Implementation
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create K-means model
kmeans = KMeans(
    n_clusters=3,        # Number of clusters
    init='k-means++',    # Smart initialization
    n_init=10,           # Number of random initializations
    max_iter=300,        # Maximum iterations
    random_state=42
)

# Fit and predict
cluster_labels = kmeans.fit_predict(X_scaled)

# Get cluster centers
centroids = kmeans.cluster_centers_

# Evaluate clustering quality
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")
```

#### Manual K-means Implementation
```python
import numpy as np

def kmeans_manual(X, k, max_iter=100, tol=1e-4):
    """
    Manual K-means implementation
    """
    # Random initialization
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for iteration in range(max_iter):
        # Assignment step
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update step
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
            
        centroids = new_centroids
    
    return labels, centroids
```

### Choosing Optimal K

#### Elbow Method
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Calculate WCSS for different k values
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(k_range, wcss, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
```

#### Silhouette Analysis
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.grid(True)
plt.show()

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal k: {optimal_k}")
```

### Initialization Methods

#### Random Initialization
- **Simple approach**: Randomly select k points
- **Problem**: Can lead to poor convergence or local optima
- **Use case**: Small datasets, quick experiments

#### K-means++ Initialization
- **Smart initialization**: Probabilistically select initial centroids
- **Algorithm**: First centroid random, subsequent based on distance
- **Advantage**: Better convergence, fewer iterations
- **Default in scikit-learn**: `init='k-means++'`

## Hierarchical Clustering

### Agglomerative Clustering

#### Algorithm Steps
1. **Start**: Each point is its own cluster
2. **Find closest**: Identify two closest clusters
3. **Merge**: Combine them into single cluster
4. **Update distances**: Recalculate distances between new cluster and others
5. **Repeat**: Until desired number of clusters reached

#### Linkage Methods

##### Single Linkage (Minimum)
- **Distance between clusters**: Minimum distance between any two points
- **Characteristics**: Can create long, chain-like clusters
- **Sensitivity**: Sensitive to noise and outliers

##### Complete Linkage (Maximum)
- **Distance between clusters**: Maximum distance between any two points
- **Characteristics**: Tends to create compact, spherical clusters
- **Robustness**: Less sensitive to outliers than single linkage

##### Average Linkage
- **Distance between clusters**: Average distance between all pairs of points
- **Characteristics**: Balance between single and complete linkage
- **Performance**: Good general-purpose choice

##### Ward's Linkage
- **Criterion**: Minimize within-cluster variance (similar to K-means)
- **Characteristics**: Tends to create equally sized clusters
- **Similarity to K-means**: Often produces similar results

### Dendrograms and Visualization

#### Creating Dendrograms
```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Create linkage matrix
linkage_matrix = linkage(X_scaled, method='ward', metric='euclidean')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # Show only last p clusters
    p=10,
    leaf_rotation=90,
    leaf_font_size=8,
    show_contracted=True
)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
```

#### Cutting Dendrograms
```python
from scipy.cluster.hierarchy import fcluster

# Cut dendrogram at specific height
clusters_height = fcluster(linkage_matrix, t=5, criterion='distance')

# Cut to get specific number of clusters
clusters_k = fcluster(linkage_matrix, t=3, criterion='maxclust')

print(f"Clusters by height: {np.unique(clusters_height)}")
print(f"Clusters by count: {np.unique(clusters_k)}")
```

### Scikit-learn Hierarchical Clustering
```python
from sklearn.cluster import AgglomerativeClustering

# Create hierarchical clustering model
hierarchical = AgglomerativeClustering(
    n_clusters=3,           # Number of clusters
    affinity='euclidean',   # Distance metric
    linkage='ward'          # Linkage method
)

# Fit and predict
cluster_labels = hierarchical.fit_predict(X_scaled)

# Get children of hierarchical tree
from scipy.cluster.hierarchy import linkage
linkage_matrix = linkage(X_scaled, method='ward')
print("Linkage matrix shape:", linkage_matrix.shape)
```

## Comparison of Methods

### K-means vs Hierarchical

| Aspect | K-means | Hierarchical |
|--------|---------|--------------|
| **Speed** | Fast (linear) | Slow (quadratic) |
| **Scalability** | Good for large n | Poor for large n |
| **Clusters** | Spherical, equal size | Any shape/size |
| **K requirement** | Must specify k | Can choose k post-hoc |
| **Deterministic** | Depends on initialization | Deterministic |
| **Memory** | Low memory usage | High memory usage |

### When to Use Each Method

#### Use K-means when:
- Large datasets (n > 1000)
- Spherical, well-separated clusters
- Approximately equal cluster sizes
- Need fast, scalable solution

#### Use Hierarchical when:
- Small to medium datasets
- Need to explore different numbers of clusters
- Want to understand cluster relationships
- Clusters have complex shapes or sizes

## Evaluation Metrics

### Internal Metrics

#### Within-Cluster Sum of Squares (WCSS)
```python
def calculate_wcss(X, labels, centroids):
    """
    Calculate within-cluster sum of squares
    """
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

wcss = calculate_wcss(X_scaled, cluster_labels, kmeans.cluster_centers_)
print(f"WCSS: {wcss:.3f}")
```

#### Silhouette Coefficient
```python
from sklearn.metrics import silhouette_samples, silhouette_score

# Overall silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)

# Individual silhouette scores
sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

# Visualize silhouette plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10
for i in range(k):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, ith_cluster_silhouette_values,
                     alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_xlabel("Silhouette coefficient values")
ax.set_ylabel("Cluster label")
ax.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.show()
```

#### Calinski-Harabasz Index
```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
print(f"Calinski-Harabasz Score: {ch_score:.3f}")
```

## Practical Implementation

### Complete Clustering Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('clustering_data.csv')
X = df.drop(['id', 'target'], axis=1, errors='ignore')

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape: {X_scaled.shape}")

# K-means clustering
k_values = range(2, 11)
kmeans_results = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    wcss = kmeans.inertia_
    
    kmeans_results.append({
        'k': k,
        'silhouette': silhouette,
        'ch_score': ch_score,
        'wcss': wcss
    })

# Find optimal k
optimal_k = max(kmeans_results, key=lambda x: x['silhouette'])['k']
print(f"Optimal k (silhouette): {optimal_k}")

# Final K-means with optimal k
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(
    n_clusters=optimal_k,
    affinity='euclidean',
    linkage='ward'
)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Compare clustering results
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)

print("
Clustering Comparison:")
print(f"K-means Silhouette: {kmeans_silhouette:.3f}")
print(f"Hierarchical Silhouette: {hierarchical_silhouette:.3f}")

# Visualize clusters (if 2D or can be reduced)
if X_scaled.shape[1] > 2:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
else:
    X_2d = X_scaled

# Plot clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# K-means clusters
scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
ax1.set_title(f'K-means Clustering (k={optimal_k})')
ax1.set_xlabel('PC1' if X_scaled.shape[1] > 2 else 'Feature 1')
ax1.set_ylabel('PC2' if X_scaled.shape[1] > 2 else 'Feature 2')
plt.colorbar(scatter1, ax=ax1)

# Hierarchical clusters
scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.6)
ax2.set_title(f'Hierarchical Clustering (k={optimal_k})')
ax2.set_xlabel('PC1' if X_scaled.shape[1] > 2 else 'Feature 1')
ax2.set_ylabel('PC2' if X_scaled.shape[1] > 2 else 'Feature 2')
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.show()

# Create linkage matrix for dendrogram (sample for large datasets)
if X_scaled.shape[0] > 100:
    sample_indices = np.random.choice(X_scaled.shape[0], 100, replace=False)
    X_sample = X_scaled[sample_indices]
else:
    X_sample = X_scaled

linkage_matrix = linkage(X_sample, method='ward', metric='euclidean')

# Plot dendrogram
plt.figure(figsize=(12, 8))
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
plt.title('Hierarchical Clustering Dendrogram (Sample)')
plt.show()

# Cluster analysis
print("
Cluster Analysis:")
for cluster_id in range(optimal_k):
    cluster_size = np.sum(kmeans_labels == cluster_id)
    print(f"Cluster {cluster_id}: {cluster_size} samples ({cluster_size/len(kmeans_labels)*100:.1f}%)")

print("
Clustering completed successfully!")
```

## Best Practices

### K-means Optimization
1. **Scale features**: Essential for distance-based clustering
2. **Choose k wisely**: Use elbow method and silhouette analysis
3. **Smart initialization**: Use k-means++ (default in sklearn)
4. **Multiple runs**: Set n_init > 1 to avoid local optima

### Hierarchical Clustering Tips
1. **Linkage method**: Ward's for similar cluster sizes, complete for compact clusters
2. **Distance metric**: Euclidean for continuous, Manhattan for mixed data
3. **Dendrogram inspection**: Cut at meaningful heights
4. **Computational cost**: Use on smaller datasets or samples

### General Clustering Advice
1. **Data preprocessing**: Handle missing values, scale features
2. **Domain knowledge**: Incorporate expert insights for k selection
3. **Multiple algorithms**: Try different methods and compare
4. **Validation**: Use internal and external metrics when possible

## Next Steps

This lecture covers fundamental clustering algorithms. The next lecture (15.3) will explore RAPIDS acceleration for K-means clustering, showing how GPU computing dramatically speeds up clustering for large datasets.