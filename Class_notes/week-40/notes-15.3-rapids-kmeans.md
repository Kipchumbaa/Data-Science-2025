# Lecture 15.3: RAPIDS Acceleration for K-means Clustering

## Key Learning Objectives
- Understand GPU acceleration benefits for K-means clustering
- Learn RAPIDS cuML K-means implementation
- Compare CPU vs GPU performance for clustering
- Implement scalable K-means with distributed computing

## Core Concepts

### Why GPU for K-means?

#### Computational Characteristics
- **Distance calculations**: Highly parallelizable matrix operations
- **Centroid updates**: Parallel aggregation across data points
- **Multiple iterations**: Each iteration benefits from parallelism
- **Large datasets**: Can handle datasets that don't fit in CPU memory

#### GPU Advantages
- **Massive parallelism**: Thousands of CUDA cores for simultaneous distance calculations
- **Memory bandwidth**: High-speed data transfer between GPU memory and cores
- **Scalability**: Handle larger datasets and more clusters
- **Performance gains**: 10-50x speedup depending on dataset size and k

### RAPIDS cuML K-means Architecture

#### Core Optimizations
- **GPU memory layout**: Optimized data structures for GPU access patterns
- **Parallel distance computation**: Batched distance calculations
- **Atomic operations**: Thread-safe centroid updates
- **Memory coalescing**: Efficient GPU memory access

#### Algorithm Adaptations
- **GPU-specific initialization**: Optimized centroid initialization
- **Batched processing**: Process data in GPU memory blocks
- **Asynchronous operations**: Overlap computation and data transfer
- **Precision handling**: Support for different floating-point precisions

## Implementation with RAPIDS

### Basic GPU K-means
```python
import cudf
import cupy as cp
from cuml.cluster import KMeans
from cuml.metrics.cluster import silhouette_score

# Load data into GPU memory
X_train_gpu = cudf.DataFrame.from_pandas(X_train)
X_test_gpu = cudf.DataFrame.from_pandas(X_test)

# Create GPU K-means model
gpu_kmeans = KMeans(
    n_clusters=3,         # Number of clusters
    init='k-means++',     # Initialization method
    n_init=10,            # Number of random initializations
    max_iter=300,         # Maximum iterations
    tol=1e-4,             # Tolerance for convergence
    random_state=42
)

# Fit on GPU
gpu_kmeans.fit(X_train_gpu)

# Predict cluster labels
train_labels = gpu_kmeans.predict(X_train_gpu)
test_labels = gpu_kmeans.predict(X_test_gpu)

# Get cluster centers
centroids = gpu_kmeans.cluster_centers_

# Evaluate clustering quality
silhouette = silhouette_score(X_train_gpu, train_labels)
print(f"GPU K-means Silhouette Score: {silhouette:.3f}")
print(f"Inertia (WCSS): {gpu_kmeans.inertia_:.3f}")
```

### Advanced GPU K-means Features
```python
# GPU K-means with custom parameters
gpu_kmeans_advanced = KMeans(
    n_clusters=5,
    init='random',        # Random initialization
    n_init=20,            # More initialization attempts
    max_iter=500,         # More iterations
    tol=1e-6,             # Tighter convergence
    verbose=True,         # Print progress
    random_state=42
)

# Fit with progress monitoring
gpu_kmeans_advanced.fit(X_train_gpu)

# Access additional attributes
print(f"Number of iterations: {gpu_kmeans_advanced.n_iter_}")
print(f"Converged: {gpu_kmeans_advanced.converged_}")
print(f"Final inertia: {gpu_kmeans_advanced.inertia_}")
```

## Performance Comparison

### CPU vs GPU Benchmarking

#### Comprehensive Benchmark
```python
import time
from sklearn.cluster import KMeans as SklearnKMeans
from cuml.cluster import KMeans as CumlKMeans

def benchmark_kmeans(X_train, y_train, k_values, n_runs=3):
    """
    Compare CPU and GPU K-means performance across different k values
    """
    results = []
    
    for k in k_values:
        cpu_times = []
        gpu_times = []
        cpu_inertias = []
        gpu_inertias = []
        
        for run in range(n_runs):
            # CPU K-means
            cpu_start = time.time()
            cpu_kmeans = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
            cpu_kmeans.fit(X_train)
            cpu_time = time.time() - cpu_start
            
            cpu_times.append(cpu_time)
            cpu_inertias.append(cpu_kmeans.inertia_)
            
            # GPU K-means
            gpu_start = time.time()
            gpu_kmeans = CumlKMeans(n_clusters=k, random_state=42, n_init=10)
            gpu_kmeans.fit(X_train_gpu)
            gpu_time = time.time() - gpu_start
            
            gpu_times.append(gpu_time)
            gpu_inertias.append(float(gpu_kmeans.inertia_))
        
        # Average results
        results.append({
            'k': k,
            'cpu_time_avg': np.mean(cpu_times),
            'gpu_time_avg': np.mean(gpu_times),
            'speedup': np.mean(cpu_times) / np.mean(gpu_times),
            'cpu_inertia_avg': np.mean(cpu_inertias),
            'gpu_inertia_avg': np.mean(gpu_inertias)
        })
    
    return pd.DataFrame(results)

# Run benchmark
k_values = [2, 5, 10, 15, 20, 25]
benchmark_results = benchmark_kmeans(X_train, y_train, k_values)

print(benchmark_results)

# Visualize results
plt.figure(figsize=(15, 5))

# Training time comparison
plt.subplot(1, 3, 1)
plt.plot(benchmark_results['k'], benchmark_results['cpu_time_avg'], 'b-o', label='CPU')
plt.plot(benchmark_results['k'], benchmark_results['gpu_time_avg'], 'r-o', label='GPU')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.legend()
plt.grid(True)

# Speedup
plt.subplot(1, 3, 2)
plt.plot(benchmark_results['k'], benchmark_results['speedup'], 'g-o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Speedup (CPU/GPU)')
plt.title('GPU Speedup')
plt.grid(True)

# Inertia comparison
plt.subplot(1, 3, 3)
plt.plot(benchmark_results['k'], benchmark_results['cpu_inertia_avg'], 'b-o', label='CPU')
plt.plot(benchmark_results['k'], benchmark_results['gpu_inertia_avg'], 'r-o', label='GPU')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Final Inertia (WCSS)')
plt.title('Clustering Quality Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### Expected Performance Gains
- **Small datasets (< 10K samples)**: 3-8x speedup
- **Medium datasets (10K-100K samples)**: 8-20x speedup
- **Large datasets (> 100K samples)**: 20-50x speedup
- **High-dimensional data**: Even larger speedups due to matrix operations

## Memory Management

### GPU Memory Considerations

#### Memory Requirements
- **Dataset storage**: Must fit in GPU memory
- **Centroid storage**: k × d floating-point values
- **Distance matrices**: Temporary storage during computation
- **Multiple initializations**: Memory usage scales with n_init

#### Memory Optimization Strategies
```python
# Check GPU memory before clustering
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(f"GPU Memory: {info.used/1024**3:.1f}GB used / {info.total/1024**3:.1f}GB total")
print(f"Available: {info.free/1024**3:.1f}GB")

# Estimate memory requirements
n_samples, n_features = X_train.shape
estimated_memory = (n_samples * n_features * 4) / 1024**3  # float32
centroid_memory = (k * n_features * 4) / 1024**3
total_estimated = estimated_memory + centroid_memory

print(f"Estimated dataset memory: {estimated_memory:.1f}GB")
print(f"Estimated centroid memory: {centroid_memory:.1f}GB")
print(f"Total estimated: {total_estimated:.1f}GB")
```

### Handling Large Datasets

#### Data Chunking
```python
def gpu_kmeans_chunked(X, k, chunk_size=50000):
    """
    Perform GPU K-means on large datasets using chunking
    """
    n_samples = len(X)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    # Initialize with first chunk
    start_idx = 0
    end_idx = min(chunk_size, n_samples)
    X_chunk = cudf.DataFrame.from_pandas(X.iloc[start_idx:end_idx])
    
    kmeans = CumlKMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_chunk)
    
    # Process remaining chunks (simplified)
    all_labels = []
    all_labels.extend(kmeans.predict(X_chunk).values_host)
    
    for i in range(1, n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        X_chunk = cudf.DataFrame.from_pandas(X.iloc[start_idx:end_idx])
        
        # Predict using existing model
        chunk_labels = kmeans.predict(X_chunk)
        all_labels.extend(chunk_labels.values_host)
    
    return np.array(all_labels), kmeans.cluster_centers_
```

## Choosing Optimal K on GPU

### GPU-Accelerated Elbow Method
```python
def gpu_elbow_method(X_gpu, k_range):
    """
    Calculate WCSS for different k values using GPU
    """
    wcss_values = []
    
    for k in k_range:
        kmeans = CumlKMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_gpu)
        wcss_values.append(float(kmeans.inertia_))
    
    return wcss_values

# Calculate WCSS for different k
k_range = range(1, 16)
wcss_values = gpu_elbow_method(X_train_gpu, k_range)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss_values, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('GPU K-means Elbow Method')
plt.grid(True)
plt.show()

# Find elbow point (simplified)
diffs = np.diff(wcss_values)
elbow_k = np.argmin(diffs) + 2  # +2 because diff reduces length by 1, +1 for 1-indexing
print(f"Suggested elbow at k={elbow_k}")
```

### GPU Silhouette Analysis
```python
from cuml.metrics.cluster import silhouette_score

def gpu_silhouette_analysis(X_gpu, k_range):
    """
    Calculate silhouette scores for different k values
    """
    silhouette_scores = []
    
    for k in k_range:
        kmeans = CumlKMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_gpu)
        score = silhouette_score(X_gpu, labels)
        silhouette_scores.append(float(score))
    
    return silhouette_scores

# Calculate silhouette scores
k_range = range(2, 11)
silhouette_scores = gpu_silhouette_analysis(X_train_gpu, k_range)

# Plot silhouette analysis
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('GPU K-means Silhouette Analysis')
plt.grid(True)
plt.show()

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal k by silhouette: {optimal_k}")
```

## Integration with Dask

### Distributed GPU K-means
```python
import dask
import dask_cudf
from dask.distributed import Client, LocalCluster
from cuml.dask.cluster import KMeans as DaskCumlKMeans

# Start Dask cluster
cluster = LocalCluster(n_workers=2, threads_per_worker=1)
client = Client(cluster)

# Load data with dask-cudf
X_dask = dask_cudf.read_csv('large_clustering_dataset.csv')
X_dask = X_dask.drop(['id'], axis=1, errors='ignore')  # Drop ID columns

# Distributed GPU K-means
dask_gpu_kmeans = DaskCumlKMeans(
    n_clusters=10,
    random_state=42
)

# Fit on distributed data
dask_gpu_kmeans.fit(X_dask)

# Predict cluster labels
labels_dask = dask_gpu_kmeans.predict(X_dask)

# Compute results
final_labels = labels_dask.compute()
centroids = dask_gpu_kmeans.cluster_centers_.compute()

print(f"Distributed clustering completed on {len(X_dask)} samples")
print(f"Cluster centers shape: {centroids.shape}")
```

## Practical Implementation

### Complete GPU K-means Pipeline
```python
import pandas as pd
import numpy as np
import cudf
import cupy as cp
from cuml.cluster import KMeans
from cuml.metrics.cluster import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

# Load and prepare data
print("Loading data...")
df = pd.read_csv('large_clustering_dataset.csv')

# Handle potential ID columns
id_columns = [col for col in df.columns if 'id' in col.lower() or 'index' in col.lower()]
if id_columns:
    X = df.drop(id_columns, axis=1)
else:
    X = df.copy()

print(f"Dataset shape: {X.shape}")

# Preprocessing
print("Preprocessing...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Dimensionality reduction for visualization
if X_scaled.shape[1] > 50:
    print("Reducing dimensionality for visualization...")
    pca = PCA(n_components=50, random_state=42)
    X_scaled = pca.fit_transform(X_scaled)
    print(f"Reduced to {X_scaled.shape[1]} dimensions")

# Convert to GPU dataframes
print("Converting to GPU dataframes...")
X_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_scaled))

# GPU K-means parameter optimization
print("Optimizing k parameter...")
k_range = range(2, 16)
gpu_wcss = []
gpu_silhouettes = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_gpu)
    
    wcss_temp = float(kmeans_temp.inertia_)
    gpu_wcss.append(wcss_temp)
    
    if k > 1:  # Silhouette requires at least 2 clusters
        sil_temp = float(silhouette_score(X_gpu, labels_temp))
        gpu_silhouettes.append(sil_temp)

# Plot optimization results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Elbow method
ax1.plot(k_range, gpu_wcss, 'bx-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
ax1.set_title('GPU K-means Elbow Method')
ax1.grid(True)

# Silhouette analysis
ax2.plot(k_range[1:], gpu_silhouettes, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('GPU K-means Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Choose optimal k (combine elbow and silhouette)
elbow_k = 5  # Example - in practice, analyze the plots
optimal_k = k_range[np.argmax(gpu_silhouettes) + 1]  # +1 because silhouettes start from k=2

print(f"Elbow method suggests k around: {elbow_k}")
print(f"Silhouette analysis suggests k: {optimal_k}")

# Use silhouette-optimal k
final_k = optimal_k

# Final GPU K-means clustering
print(f"Performing final clustering with k={final_k}...")
start_time = time.time()

final_kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=20,  # More initializations for better results
    verbose=True
)

final_labels = final_kmeans.fit_predict(X_gpu)
clustering_time = time.time() - start_time

# Get results
final_centroids = final_kmeans.cluster_centers_
final_inertia = float(final_kmeans.inertia_)

print(f"Clustering completed in {clustering_time:.2f} seconds")
print(f"Final inertia: {final_inertia:.3f}")

# Evaluate clustering quality
final_silhouette = float(silhouette_score(X_gpu, final_labels))
print(f"Final silhouette score: {final_silhouette:.3f}")

# Cluster analysis
cluster_sizes = np.bincount(final_labels.values_host)
print("
Cluster sizes:")
for i, size in enumerate(cluster_sizes):
    percentage = size / len(final_labels) * 100
    print(f"Cluster {i}: {size} samples ({percentage:.1f}%)")

# Visualize clusters (if 2D or can be reduced to 2D)
if X_scaled.shape[1] > 2:
    print("Creating 2D visualization...")
    pca_viz = PCA(n_components=2, random_state=42)
    X_2d = pca_viz.fit_transform(X_scaled)
else:
    X_2d = X_scaled

# Plot clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=final_labels.values_host, 
                     cmap='tab10', alpha=0.6, s=50)

# Plot centroids
centroids_2d = pca_viz.transform(final_centroids.values_host) if X_scaled.shape[1] > 2 else final_centroids.values_host
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200, 
           linewidth=3, label='Centroids')

plt.xlabel('PC1' if X_scaled.shape[1] > 2 else 'Feature 1')
plt.ylabel('PC2' if X_scaled.shape[1] > 2 else 'Feature 2')
plt.title(f'GPU K-means Clustering Results (k={final_k})')
plt.colorbar(scatter)
plt.legend()
plt.grid(True)
plt.show()

# Performance comparison with CPU (optional)
try:
    print("Comparing with CPU implementation...")
    from sklearn.cluster import KMeans as SklearnKMeans
    
    cpu_kmeans = SklearnKMeans(n_clusters=final_k, random_state=42, n_init=10)
    
    start_time = time.time()
    cpu_labels = cpu_kmeans.fit_predict(X_scaled)
    cpu_time = time.time() - start_time
    
    cpu_inertia = cpu_kmeans.inertia_
    cpu_silhouette = silhouette_score(X_scaled, cpu_labels)
    
    print(f"CPU clustering time: {cpu_time:.2f} seconds")
    print(f"GPU clustering time: {clustering_time:.2f} seconds")
    print(f"Speedup: {cpu_time/clustering_time:.1f}x")
    print(f"CPU inertia: {cpu_inertia:.3f}, GPU inertia: {final_inertia:.3f}")
    print(f"CPU silhouette: {cpu_silhouette:.3f}, GPU silhouette: {final_silhouette:.3f}")
    
except ImportError:
    print("Scikit-learn not available for comparison")

# Save results
print("Saving clustering results...")
results_df = pd.DataFrame({
    'original_index': range(len(final_labels)),
    'cluster_label': final_labels.values_host
})

results_df.to_csv('gpu_kmeans_results.csv', index=False)
print("Results saved to 'gpu_kmeans_results.csv'")

print("
GPU K-means clustering analysis completed!")
```

## Best Practices

### GPU K-means Optimization
1. **Memory monitoring**: Track GPU memory usage throughout clustering
2. **Optimal k selection**: Use GPU-accelerated elbow and silhouette methods
3. **Initialization**: Use multiple random initializations (n_init > 10)
4. **Convergence**: Monitor convergence and adjust tolerance if needed

### Performance Tuning
1. **Data preprocessing**: Scale features and handle missing values
2. **Dimensionality**: Consider PCA for high-dimensional data
3. **Batch processing**: Process large datasets in chunks if needed
4. **GPU utilization**: Monitor GPU usage with nvidia-smi

### Production Considerations
1. **Model persistence**: Save trained GPU models
2. **Batch inference**: Process new data efficiently
3. **Resource management**: Monitor GPU resources in production
4. **Fallback options**: Have CPU implementation for edge cases

## Troubleshooting

### Common GPU Issues
- **Out of memory**: Reduce n_init, use data chunking, or reduce dataset size
- **Slow convergence**: Adjust tolerance, increase max_iter, or improve initialization
- **Poor clustering**: Check data preprocessing, feature scaling, and k selection
- **Installation issues**: Ensure proper CUDA and RAPIDS versions

### Performance Debugging
```python
# Monitor GPU usage during clustering
import subprocess

def monitor_gpu_usage(duration=10):
    """Monitor GPU usage for specified duration"""
    for _ in range(duration):
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        print(result.stdout.strip())
        time.sleep(1)

# Time different phases
start = time.time()
kmeans = KMeans(n_clusters=10, random_state=42)
data_transfer_time = time.time() - start

start = time.time()
kmeans.fit(X_gpu)
training_time = time.time() - start

start = time.time()
labels = kmeans.predict(X_gpu)
prediction_time = time.time() - start

print(f"Data transfer time: {data_transfer_time:.3f}s")
print(f"Training time: {training_time:.3f}s")
print(f"Prediction time: {prediction_time:.3f}s")
```

## Connection to Scalable Computing

### Integration with Previous Modules
- **Dask**: Distributed GPU clustering for massive datasets
- **Spark**: GPU acceleration within Spark ML pipelines
- **HBase**: Fast feature retrieval for clustering
- **Module 14**: Feature engineering from supervised learning

### Enterprise Applications
- **Real-time clustering**: Low-latency GPU inference
- **Large-scale customer segmentation**: Distributed clustering
- **Anomaly detection**: GPU-accelerated outlier identification
- **Recommendation systems**: User clustering at scale

## Next Steps

This lecture demonstrates GPU acceleration for K-means clustering. The next lecture (15.4) will introduce DBSCAN, a density-based clustering algorithm that can discover clusters of arbitrary shapes, complementing the partitioning approaches covered so far.